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
    get_output_dir,
    get_simulation_count
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
        self.output_dir = Path(get_output_dir())
        self.logger = logging.getLogger(self.__class__.__name__)

        try:
            with open("data/master_datasets.json", 'r') as f:
                self.master_datasets = json.load(f)
        except FileNotFoundError:
            self.logger.warning("master_datasets.json not found. Dynamic games may not be reproducible.")
            self.master_datasets = {}

    async def run_all_experiments(self):
        """Runs the full suite of experiments as defined in the config."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING EXPERIMENT SUITE")
        if self.mock_mode: self.logger.info("🎭 RUNNING IN MOCK MODE 🎭")
        self.logger.info("=" * 80)

        all_game_names = ['salop', 'green_porter', 'spulber', 'athey_bagwell']

        for game_name in all_game_names:
            game_configs = get_all_game_configs(game_name)
            for config in game_configs:
                for challenger_model in self.challenger_models:
                    self.logger.info(f"Running: [{game_name}]-[{config.condition_name}] for Challenger: [{get_model_display_name(challenger_model)}]")

                    num_simulations = get_simulation_count(config.experiment_type)
                    tasks = [self.run_single_simulation(challenger_model, config, sim_id) for sim_id in range(num_simulations)]
                    simulation_results = await asyncio.gather(*tasks)

                    self._save_competition_result(challenger_model, config, simulation_results)

    async def run_single_simulation(self, challenger_model: str, config, sim_id: int):
        """Runs a single simulation of a game."""
        game = create_game(config.game_name)
        game_state = game.initialize_game_state(config, sim_id)
        game_state['simulation_id'] = sim_id

        dataset_key = f"{config.game_name}_{config.experiment_type}_{config.condition_name}"
        if dataset_key in self.master_datasets:
            dataset = self.master_datasets[dataset_key]
            sim_specific_data = {}
            for key, value in dataset.items():
                if isinstance(value, list) and len(value) > sim_id:
                    sim_specific_data[key] = value[sim_id]
                elif isinstance(value, dict):
                    player_data = {}
                    for player, data in value.items():
                        player_data[player] = data[sim_id] if isinstance(data, list) and len(data) > sim_id else data
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
        actions, responses, challenger_prompt = await self._get_all_actions(game, agents, config, game_state)
        payoffs = game.calculate_payoffs(actions, config, game_state)
        game_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)

        game_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}
        game_data['initial_prompt_for_challenger'] = challenger_prompt

        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, challenger_model, list(agents.keys()), actions, payoffs, game_data)

    async def _run_dynamic_game(self, game, agents, config, game_state, challenger_model: str):
        time_horizon = config.constants.get('time_horizon', 50)
        all_rounds_data = []

        for _ in range(time_horizon):
            actions, responses, challenger_prompt = await self._get_all_actions(game, agents, config, game_state)
            payoffs = game.calculate_payoffs(actions, config, game_state)
            round_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
            round_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}

            if game_state.get('current_period', 1) == 1:
                round_data['initial_prompt_for_challenger'] = challenger_prompt
            all_rounds_data.append(round_data)
            game_state = game.update_game_state(game_state, actions, config, payoffs)

        profit_streams = defaultdict(list)
        for round_data in all_rounds_data:
            # --- FIXED LOGIC ---
            # Correctly retrieves per-round payoffs from the 'payoffs' key,
            # ensuring profit_streams is populated.
            profits = round_data.get('payoffs', {})
            for p_id, profit in profits.items():
                profit_streams[p_id].append(profit)

        final_npvs = {p_id: calculate_npv(stream, config.constants.get('discount_factor', 0.95)) for p_id, stream in profit_streams.items()}

        game_data = {
            "constants": config.constants,
            "rounds": all_rounds_data,
            "final_npvs": final_npvs,
            "initial_prompt_for_challenger": all_rounds_data[0].get('initial_prompt_for_challenger', '')
        }

        if config.game_name == 'green_porter':
            state_history = [r.get('market_state', 'Unknown') for r in all_rounds_data]
            reversion_triggers = sum(1 for i in range(len(state_history) - 1) if state_history[i] == 'Collusive' and state_history[i+1] == 'Reversionary')
            game_data['reversion_frequency'] = reversion_triggers / (time_horizon - 1) if time_horizon > 1 else 0

        if config.game_name == 'athey_bagwell':
            hhi_per_round = [sum((s * 100) ** 2 for s in r.get('game_outcomes', {}).get('player_market_shares', {}).values()) for r in all_rounds_data]
            game_data['average_hhi'] = np.mean(hhi_per_round) if hhi_per_round else 0

        last_actions = all_rounds_data[-1]['actions'] if all_rounds_data else {}
        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, challenger_model, list(agents.keys()), last_actions, final_npvs, game_data)

    async def _get_all_actions(self, game, agents, config, game_state) -> (Dict[str, Any], Dict[str, AgentResponse], str):
        """Gets actions and full responses from all agents concurrently, and returns the challenger's prompt."""
        call_id = f"{config.game_name}-{game_state.get('simulation_id', 'N/A')}"

        prompts = {pid: game.generate_player_prompt(pid, game_state, config) for pid, agent in agents.items()}
        
        tasks = {pid: agent.get_response(prompts[pid], call_id, config) for pid, agent in agents.items()}
        
        responses = await asyncio.gather(*tasks.values())

        response_map = dict(zip(agents.keys(), responses))
        actions = {pid: game.parse_llm_response(resp.content, pid, call_id) or {} for pid, resp in response_map.items()}

        challenger_prompt = prompts.get('challenger', '')

        return actions, response_map, challenger_prompt

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