# generate_data.py

import json
import numpy as np
import random
from pathlib import Path
import sys
import logging
from typing import Dict, Any

# Add the project root to the Python path to allow for package imports
sys.path.append(str(Path(__file__).resolve().parent))

from config.config import get_all_game_configs, get_experiment_config, GameConfig, get_data_dir

def generate_spulber_data(game_config: GameConfig, num_sims: int) -> Dict[str, Any]:
    """Generates private cost data for all defenders in the Spulber game."""
    constants = game_config.constants
    num_defenders = constants["number_of_players"] - 1
    
    defender_costs = {}
    for i in range(1, num_defenders + 1):
        costs = np.random.normal(
            constants["rival_cost_mean"],
            constants["rival_cost_std"],
            num_sims
        )
        defender_costs[f'defender_{i}'] = np.maximum(0, costs).round(2).tolist()

    # The challenger's cost is fixed in the config, not generated randomly here.
    private_costs = {"challenger": constants.get("your_cost"), **defender_costs}
    return {"player_private_costs": private_costs}

def generate_green_porter_data(game_config: GameConfig, num_sims: int) -> Dict[str, Any]:
    """Generates lists of random demand shocks for the Green & Porter game."""
    constants = game_config.constants
    time_horizon = constants["time_horizon"]
    
    shock_lists = [
        np.random.normal(
            constants.get("demand_shock_mean", 0),
            constants["demand_shock_std"],
            time_horizon
        ).round(2).tolist()
        for _ in range(num_sims)
    ]
    return {"demand_shocks": shock_lists}

def generate_athey_bagwell_data(game_config: GameConfig, num_sims: int) -> Dict[str, Any]:
    """Generates persistent true cost type streams for all players in the Athey & Bagwell game."""
    constants = game_config.constants
    num_players = constants["number_of_players"]
    time_horizon = constants["time_horizon"]
    persistence_prob = constants["persistence_probability"]
    player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

    cost_streams = {}
    for player_id in player_ids:
        sim_streams = []
        for _ in range(num_sims):
            stream = ["low" if random.random() < 0.5 else "high"]
            for _ in range(1, time_horizon):
                next_cost = stream[-1] if random.random() < persistence_prob else ("low" if stream[-1] == "high" else "high")
                stream.append(next_cost)
            sim_streams.append(stream)
        cost_streams[player_id] = sim_streams
        
    return {"player_true_costs": cost_streams}

def main():
    """Generates and saves all master datasets required for the experiments."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    exp_config = get_experiment_config()
    seed = exp_config.get("random_seed")
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_sims = exp_config.get("main_experiment_simulations", 50)
    datasets = {}
    
    game_data_generators = {
        "spulber": generate_spulber_data,
        "green_porter": generate_green_porter_data,
        "athey_bagwell": generate_athey_bagwell_data
    }

    all_games = ["salop", "green_porter", "spulber", "athey_bagwell"]

    for game_name in all_games:
        # Only generate data for games that need it
        if game_name not in game_data_generators:
            continue

        all_configs = get_all_game_configs(game_name)
        for config in all_configs:
            # Create a unique key for each condition that requires pre-generated data
            dataset_key = f"{config.game_name}_{config.experiment_type}_{config.condition_name}"
            logger.info(f"Generating dataset for: {dataset_key}")
            
            generator_func = game_data_generators[game_name]
            # Use the correct simulation count based on experiment type
            sim_count = exp_config.get(f"{config.experiment_type}_simulations", num_sims)
            datasets[dataset_key] = generator_func(config, sim_count)

    output_dir = get_data_dir()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "master_datasets.json"

    with open(output_path, 'w') as f:
        json.dump(datasets, f, indent=2)
        
    logger.info(f"✅ Successfully generated {len(datasets)} master datasets.")
    logger.info(f"   Saved to: {output_path}")

if __name__ == "__main__":
    main()