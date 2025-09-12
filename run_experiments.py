# run_experiments.py

import asyncio
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

from config.config import (
    get_all_game_configs,
    get_challenger_models,
    get_defender_model,
    get_model_display_name,
    get_output_dir
)
from games import create_game
from agents import create_agent
from metrics import create_game_result

class Competition:
    """Orchestrates a series of game simulations between LLM agents."""

    def __init__(self, challenger_models: List[str], defender_model: str, mock_mode: bool = False):
        self.challenger_models = challenger_models
        self.defender_model = defender_model
        self.mock_mode = mock_mode
        self.output_dir = Path(get_output_dir())
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load pre-generated data for dynamic games
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
        if self.mock_mode:
            self.logger.info("🎭 RUNNING IN MOCK MODE 🎭")
        self.logger.info("=" * 80)

        all_game_names = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
        
        for game_name in all_game_names:
            game_configs = get_all_game_configs(game_name)
            for config in game_configs:
                for challenger_model in self.challenger_models:
                    self.logger.info(f"Running: [{game_name}]-[{config.condition_name}] for Challenger: [{get_model_display_name(challenger_model)}]")
                    
                    simulation_results = []
                    tasks = []
                    for sim_id in range(config.constants.get('num_simulations', 10)): # Fallback to 10 sims
                        tasks.append(self.run_single_simulation(challenger_model, config, sim_id))
                    
                    simulation_results = await asyncio.gather(*tasks)
                    
                    # Save results for this specific competition
                    self._save_competition_result(challenger_model, config, simulation_results)

    async def run_single_simulation(self, challenger_model: str, config, sim_id: int) -> dict:
        """Runs a single simulation of a game."""
        game = create_game(config.game_name)
        game_state = game.initialize_game_state(config, sim_id)
        
        # Inject pre-generated data for reproducibility
        dataset_key = f"{config.game_name}_{config.experiment_type}_{config.condition_name}"
        if dataset_key in self.master_datasets:
            sim_data = {k: v[sim_id] for k, v in self.master_datasets[dataset_key].items()}
            game_state['predefined_sequences'] = sim_data

        agents = {
            'challenger': create_agent(challenger_model, 'challenger', self.mock_mode),
            **{f'defender_{i+1}': create_agent(self.defender_model, f'defender_{i+1}', self.mock_mode)
               for i in range(config.constants['number_of_players'] - 1)}
        }
        
        if hasattr(game, 'update_game_state'): # Dynamic game
            return await self._run_dynamic_game(game, agents, config, game_state)
        else: # Static game
            return await self._run_static_game(game, agents, config, game_state)

    async def _run_static_game(self, game, agents, config, game_state):
        actions = await self._get_all_actions(game, agents, config, game_state)
        payoffs = game.calculate_payoffs(actions, config, game_state)
        game_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, list(agents.keys()), actions, payoffs, game_data)

    async def _run_dynamic_game(self, game, agents, config, game_state):
        time_horizon = config.constants.get('time_horizon', 15)
        all_rounds_data = []

        for _ in range(time_horizon):
            actions = await self._get_all_actions(game, agents, config, game_state)
            payoffs = game.calculate_payoffs(actions, config, game_state)
            round_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
            all_rounds_data.append(round_data)
            game_state = game.update_game_state(game_state, actions, config)
        
        # Final result aggregates data from all rounds
        final_payoffs = {p: sum(r['payoffs'][p] for r in all_rounds_data) for p in agents}
        return create_game_result(game_state['simulation_id'], config.game_name, config.experiment_type, config.condition_name, list(agents.keys()), all_rounds_data[-1]['actions'], final_payoffs, {"rounds": all_rounds_data, **game_state})

    async def _get_all_actions(self, game, agents, config, game_state) -> Dict[str, Any]:
        """Gets actions from all agents concurrently."""
        tasks = []
        for player_id, agent in agents.items():
            prompt = game.generate_player_prompt(player_id, game_state, config)
            tasks.append(agent.get_response(prompt, f"{config.game_name}-{game_state['simulation_id']}"))
        
        responses = await asyncio.gather(*tasks)
        actions = {}
        for i, (player_id, agent) in enumerate(agents.items()):
            resp = responses[i]
            parsed_action = game.parse_llm_response(resp.content, player_id, f"{config.game_name}-{game_state['simulation_id']}")
            actions[player_id] = parsed_action or {} # Use empty dict for failed parses
        return actions

    def _save_competition_result(self, challenger, config, results):
        """Saves the results of a set of simulations to a JSON file."""
        game_dir = self.output_dir / config.game_name
        challenger_dir = game_dir / challenger
        challenger_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{config.condition_name}_competition_result.json"
        filepath = challenger_dir / filename
        
        output_data = {
            "game_name": config.game_name,
            "experiment_type": config.experiment_type,
            "condition_name": config.condition_name,
            "challenger_model": challenger,
            "simulation_results": [res.__dict__ for res in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        self.logger.info(f"Saved results to {filepath}")

def setup_logging(verbose: bool):
    """Configures logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')

def parse_arguments():
    """Parses command-line arguments for the runner."""
    parser = argparse.ArgumentParser(description="LLM Game Theory Experiment Runner")
    parser.add_argument('--mock', action='store_true', help="Run in mock mode with random agents instead of real API calls.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose DEBUG logging.")
    return parser.parse_args()

async def main():
    """Main entry point for running the experiments."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    competition = Competition(challenger_models, defender_model, mock_mode=args.mock)
    await competition.run_all_experiments()

if __name__ == "__main__":
    asyncio.run(main())