# run_experiments.py

import asyncio
import logging
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    get_all_game_configs,
    get_challenger_models,
    get_defender_model,
    get_model_display_name,
    get_experiments_dir,
    get_simulation_count,
    get_data_dir,
    load_config  # Import load_config to dynamically get game names
)
from games import create_game
from agents import create_agent, AgentResponse
from metrics.metric_utils import GameResult, create_game_result

# Helper function for NPV calculation
def calculate_npv(profit_stream: List[float], discount_factor: float) -> float:
    """Calculates the Net Present Value of a stream of profits."""
    return sum(profit * (discount_factor ** t) for t, profit in enumerate(profit_stream))

class Competition:
    """Orchestrates a series of game simulations between LLM agents."""

    def __init__(self, challenger_models: List[str], defender_model: str, mock_mode: bool = False):
        self.challenger_models = challenger_models
        self.defender_model = defender_model
        self.mock_mode = mock_mode
        self.output_dir = get_experiments_dir()
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            dataset_path = get_data_dir() / "master_datasets.json"
            with open(dataset_path, 'r') as f:
                self.master_datasets = json.load(f)
            self.logger.info("Successfully loaded master datasets for reproducible experiments.")
        except FileNotFoundError:
            self.logger.critical(f"❌ CRITICAL ERROR: Input file not found at '{dataset_path}'.")
            self.logger.critical("Please run the data generation script first by executing: python generate_data.py")
            sys.exit(1)

    async def run_all_experiments(self):
        """
        Runs the full suite of experiments as defined in the config, parallelizing
        all models within each game condition for maximum efficiency.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING EXPERIMENT SUITE")
        if self.mock_mode: self.logger.info("🎭 RUNNING IN MOCK MODE 🎭")
        self.logger.info("=" * 80)

        # --- UPDATED LOGIC: Dynamically get game names from config ---
        all_game_names = list(load_config().get('game_configs', {}).keys())
        self.logger.info(f"Found the following games to run in config.json: {all_game_names}")

        for game_name in all_game_names:
            game_configs = get_all_game_configs(game_name)
            for config in game_configs:
                self.logger.info(f"--- Preparing to run all models for: [{game_name}]-[{config.condition_name}] ---")

                model_tasks = []
                for challenger_model in self.challenger_models:
                    model_task = self._run_and_save_model_simulations(challenger_model, config)
                    model_tasks.append(model_task)
                
                await asyncio.gather(*model_tasks)
                self.logger.info(f"--- Completed all models for: [{game_name}]-[{config.condition_name}] ---")

    async def _run_and_save_model_simulations(self, challenger_model: str, config):
        """A helper coroutine to run all simulations for a single model and save the result."""
        self.logger.info(f"  -> Launching simulations for Challenger: [{get_model_display_name(challenger_model)}]")
        
        num_simulations = get_simulation_count(config.experiment_type)
        simulation_tasks = [self.run_single_simulation(challenger_model, config, sim_id) for sim_id in range(num_simulations)]
        
        simulation_results = await asyncio.gather(*simulation_tasks)
        
        self._save_competition_result(challenger_model, config, simulation_results)

    async def run_single_simulation(self, challenger_model: str, config, sim_id: int):
        """Runs a single simulation of a game."""
        game = create_game(config.game_name)
        game_state = game.initialize_game_state(config, sim_id)
        game_state['simulation_id'] = sim_id

        dataset_key = f"{config.game_name}_{config.experiment_type}_{config.condition_name}"
        if dataset_key in self.master_datasets:
            dataset = self.master_datasets[dataset_key]
            
            if config.game_name == 'spulber' and 'player_private_costs' in dataset:
                costs_data = dataset.get('player_private_costs', {})
                sim_costs = {}
                for player, costs_list in costs_data.items():
                    if isinstance(costs_list, list) and len(costs_list) > sim_id:
                        sim_costs[player] = costs_list[sim_id]
                    else:
                        sim_costs[player] = costs_list
                game_state['player_private_costs'] = sim_costs
            
            else:
                sim_specific_data = {}
                for key, value in dataset.items():
                    if isinstance(value, list) and len(value) > sim_id:
                        sim_specific_data[key] = value[sim_id]
                    elif isinstance(value, dict):
                        player_data = {}
                        for player, data in value.items():
                            if isinstance(data, list) and len(data) > sim_id:
                                player_data[player] = data[sim_id]
                            else:
                                player_data[player] = data
                        sim_specific_data[key] = player_data
                game_state['predefined_sequences'] = sim_specific_data


        agents = {
            'challenger': create_agent(challenger_model, 'challenger', agent_type='experiment', mock_mode=self.mock_mode),
            **{f'defender_{i+1}': create_agent(self.defender_model, f'defender_{i+1}', agent_type='experiment', mock_mode=self.mock_mode)
               for i in range(config.constants['number_of_players'] - 1)}
        }

        if hasattr(game, 'update_game_state'):
            return await self._run_dynamic_game(game, agents, config, game_state, challenger_model)
        else:
            return await self._run_static_game(game, agents, config, game_state, challenger_model)

    async def _run_static_game(self, game, agents, config, game_state, challenger_model: str):
        actions, responses = await self._get_all_actions(game, agents, config, game_state)
        payoffs = game.calculate_payoffs(actions, config, game_state)
        game_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)

        game_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}

        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, challenger_model, list(agents.keys()), actions, payoffs, game_data)

    async def _run_dynamic_game(self, game, agents, config, game_state, challenger_model: str):
        """Dispatcher for dynamic games to their specialized workflows."""
        if config.game_name == 'athey_bagwell':
            return await self._run_athey_bagwell_game(game, agents, config, game_state, challenger_model)
        
        if config.game_name == 'green_porter':
            return await self._run_green_porter_game(game, agents, config, game_state, challenger_model)
        
        self.logger.warning(f"No specialized workflow for dynamic game '{config.game_name}'. Using generic loop.")
        time_horizon = config.constants.get('time_horizon', 50)
        all_rounds_data = []

        for _ in range(time_horizon):
            actions, responses = await self._get_all_actions(game, agents, config, game_state)
            payoffs = game.calculate_payoffs(actions, config, game_state)
            round_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
            
            all_rounds_data.append(round_data)
            game_state = game.update_game_state(game_state, actions, config, payoffs)

        profit_streams = defaultdict(list)
        for round_data in all_rounds_data:
            profits = round_data.get('payoffs', {})
            for p_id, profit in profits.items():
                profit_streams[p_id].append(profit)

        final_npvs = {p_id: calculate_npv(stream, config.constants.get('discount_factor', 0.95)) for p_id, stream in profit_streams.items()}

        game_data = { "constants": config.constants, "rounds": all_rounds_data, "final_npvs": final_npvs }
        
        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, challenger_model, list(agents.keys()), {}, final_npvs, game_data)

    async def _run_green_porter_game(self, game, agents, config, game_state, challenger_model: str):
        """Runs the specialized workflow for Green & Porter, logging both strategic and mechanical rounds."""
        time_horizon = config.constants.get('time_horizon', 50)
        all_rounds_data = []

        while game_state['current_period'] <= time_horizon:
            if game_state['market_state'] == 'Collusive':
                actions, responses = await self._get_all_actions(game, agents, config, game_state)
            else:
                cournot_quantity = config.constants.get('cournot_quantity')
                actions = {pid: {'quantity': cournot_quantity} for pid in agents}
                responses = {}
            
            payoffs = game.calculate_payoffs(actions, config, game_state)
            round_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
            
            if responses:
                round_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}
            
            all_rounds_data.append(round_data)
            game_state = game.update_game_state(game_state, actions, config, payoffs)

        profit_streams = defaultdict(list)
        for round_data in all_rounds_data:
            profits = round_data.get('payoffs', {})
            for p_id, profit in profits.items():
                profit_streams[p_id].append(profit)
        
        final_npvs = {p_id: calculate_npv(stream, config.constants.get('discount_factor', 0.95)) for p_id, stream in profit_streams.items()}
        
        state_history = [r.get('market_state', 'Unknown') for r in all_rounds_data]
        reversion_triggers = sum(1 for i in range(len(state_history) - 1) if state_history[i] == 'Collusive' and state_history[i+1] == 'Reversionary')

        game_data = {
            "constants": config.constants, "rounds": all_rounds_data, "final_npvs": final_npvs,
            "reversion_frequency": reversion_triggers / (time_horizon - 1) if time_horizon > 1 else 0,
            "state_history": state_history
        }

        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, challenger_model, list(agents.keys()), {}, final_npvs, game_data)


    async def _run_athey_bagwell_game(self, game, agents, config, game_state, challenger_model: str):
        """
        Runs the specialized workflow for Athey & Bagwell, correctly modeling
        the sequential odd and even periods as described in the paper.
        """
        time_horizon = config.constants.get('time_horizon', 50)
        all_rounds_data = []
        last_period_reports = {}

        for period in range(1, time_horizon + 1):
            game_state['current_period'] = period
            
            is_odd_period = (period % 2 != 0)

            if is_odd_period:
                reports, responses = await self._get_all_actions(game, agents, config, game_state, stage=1)
                last_period_reports = reports

                low_reporters = [pid for pid, action in reports.items() if action.get('report') == 'low']
                market_shares = {}
                if len(low_reporters) == 1:
                    market_shares = {pid: (1.0 if pid == low_reporters[0] else 0.0) for pid in agents}
                else:
                    share = 1.0 / len(agents)
                    market_shares = {pid: share for pid in agents}
                
                actions_for_payoff = {pid: {'quantity': share} for pid, share in market_shares.items()}
                payoffs = game.calculate_payoffs(actions_for_payoff, config, game_state)
                
                round_data = game.get_game_data_for_logging(reports, payoffs, config, game_state)
                round_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}
                all_rounds_data.append(round_data)

            else:
                low_reporters = [pid for pid, action in last_period_reports.items() if action.get('report') == 'low']
                high_reporters = [pid for pid in agents if pid not in low_reporters]
                
                market_shares = {}
                if high_reporters:
                    share = 1.0 / len(high_reporters)
                    market_shares = {pid: (share if pid in high_reporters else 0.0) for pid in agents}
                else:
                    market_shares = {pid: 1.0 / len(agents) for pid in agents}

                actions = {pid: {'quantity': share} for pid, share in market_shares.items()}
                payoffs = game.calculate_payoffs(actions, config, game_state)
                
                round_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
                all_rounds_data.append(round_data)

        profit_streams = defaultdict(list)
        for round_data in all_rounds_data:
            profits = round_data.get('payoffs', {})
            for p_id, profit in profits.items():
                profit_streams[p_id].append(profit)

        final_npvs = {p_id: calculate_npv(stream, config.constants.get('discount_factor', 0.95)) for p_id, stream in profit_streams.items()}
        hhi_per_round = [sum((s * 100) ** 2 for s in r.get('game_outcomes', {}).get('player_market_shares', {}).values()) for r in all_rounds_data]

        game_data = {
            "constants": config.constants, "rounds": all_rounds_data, "final_npvs": final_npvs,
            "average_hhi": np.mean(hhi_per_round) if hhi_per_round else 0
        }
        
        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, challenger_model, list(agents.keys()), {}, final_npvs, game_data)

    async def _get_all_actions(self, game, agents, config, game_state, stage: int = 1) -> (Dict[str, Any], Dict[str, AgentResponse]):
        """Gets actions and full responses from all agents concurrently."""
        call_id = f"{config.game_name}-{game_state.get('simulation_id', 'N/A')}"

        prompts = {pid: game.generate_player_prompt(pid, game_state, config) for pid in agents}
        
        tasks = {pid: agent.get_response(prompts[pid], call_id, config) for pid, agent in agents.items()}
        responses = await asyncio.gather(*tasks.values())
        response_map = dict(zip(agents.keys(), responses))
        
        actions = {pid: game.parse_llm_response(resp.content, pid, call_id, stage=stage) or {} for pid, resp in response_map.items()}

        return actions, response_map

    def _save_competition_result(self, challenger, config, results):
        """Saves the results of a set of simulations to a JSON file."""
        challenger_dir = self.output_dir / config.game_name / challenger
        challenger_dir.mkdir(parents=True, exist_ok=True)
        filepath = challenger_dir / f"{config.condition_name}_competition_result.json"

        dict_results = [res.__dict__ for res in results]

        output_data = {
            "game_name": config.game_name, "experiment_type": config.experiment_type,
            "condition_name": config.condition_name, "challenger_model": challenger,
            "simulation_results": dict_results
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        self.logger.info(f"Saved results to {filepath}")

def setup_logging(verbose: bool, mock_mode: bool):
    """Configures logging to both console and a timestamped file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"experiment_{timestamp}{'_mock' if mock_mode else ''}.log"

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logging.info(f"Logging initialized. Log file at: {log_file}")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="LLM Game Theory Experiment Runner")
    parser.add_argument('--mock', action='store_true', help="Run in mock mode with random agents.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose DEBUG logging.")
    return parser.parse_args()

async def main():
    """Main entry point for running the experiments."""
    args = parse_arguments()
    setup_logging(args.verbose, args.mock)

    competition = Competition(get_challenger_models(), get_defender_model(), mock_mode=args.mock)
    await competition.run_all_experiments()

if __name__ == "__main__":
    asyncio.run(main())