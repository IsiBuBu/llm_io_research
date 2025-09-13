# data/generate_data.py

import json
import numpy as np
import random
from pathlib import Path
import sys
import logging
# CORRECTED: Import Dict and Any for type hinting
from typing import Dict, Any

# Add the project root to the Python path to allow for package imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import get_game_config, get_output_dir, get_experiment_config, GameConfig

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

    return {"player_private_costs": {"challenger": constants["your_cost"], **defender_costs}}

def generate_green_porter_data(game_config: GameConfig, num_sims: int) -> Dict[str, Any]:
    """Generates lists of random demand shocks for the Green & Porter game."""
    constants = game_config.constants
    time_horizon = constants["time_horizon"]
    
    shock_lists = [
        np.random.normal(0, constants["demand_shock_std"], time_horizon).round(2).tolist()
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

    dataset_definitions = {
        "spulber_structural-variations_more_players": ("spulber", "structural_variations", "more_players", generate_spulber_data),
        "spulber_ablation-studies_wide_cost_range": ("spulber", "ablation_studies", "wide_cost_range", generate_spulber_data),
        "green_porter_structural-variations_long_time_horizon": ("green_porter", "structural_variations", "long_time_horizon", generate_green_porter_data),
        "green_porter_ablation-studies_high_demand_volatility": ("green_porter", "ablation_studies", "high_demand_volatility", generate_green_porter_data),
        "athey_bagwell_structural-variations_long_time_horizon": ("athey_bagwell", "structural_variations", "long_time_horizon", generate_athey_bagwell_data),
        "athey_bagwell_ablation-studies_low_persistence": ("athey_bagwell", "ablation_studies", "low_persistence", generate_athey_bagwell_data),
    }

    for name, (game, exp_type, cond, func) in dataset_definitions.items():
        logger.info(f"Generating dataset: {name}")
        config = get_game_config(game, exp_type, cond)
        datasets[name] = func(config, num_sims)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "master_datasets.json"

    with open(output_path, 'w') as f:
        json.dump(datasets, f, indent=2)
        
    logger.info(f"✅ Successfully generated {len(datasets)} master datasets.")
    logger.info(f"   Saved to: {output_path}")

if __name__ == "__main__":
    main()