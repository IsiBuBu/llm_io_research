#!/usr/bin/env python3
"""
Generate 6 master datasets from config.json using config.py.
This script is corrected based on best practices for reproducibility and clarity.
"""

import json
import numpy as np
import random
from pathlib import Path
from config import get_game_config, get_output_config, get_experiment_config

# --- Data Generation Functions ---

def generate_spulber_data(game_config, num_sims):
    """Generates defender cost data for the Spulber game."""
    num_defenders = game_config.constants["number_of_players"] - 1
    min_cost = game_config.constants.get("min_cost", 0) # Ensure cost is non-negative
    precision = game_config.constants.get("cost_precision", 2)

    defender_costs = {}
    for d in range(1, num_defenders + 1):
        costs = np.random.normal(
            game_config.constants["rival_cost_mean"],
            game_config.constants["rival_cost_std"],
            num_sims
        )
        # Apply minimum cost constraint and rounding
        defender_costs[f'defender_{d}'] = np.maximum(min_cost, costs).round(precision).tolist()

    return {
        "challenger_cost": game_config.constants["your_cost"],
        "defender_costs": defender_costs
    }

def generate_green_porter_data(game_config, num_sims):
    """Generates demand shock lists for the Green & Porter game."""
    time_horizon = game_config.constants["time_horizon"]
    shock_mean = game_config.constants.get("demand_shock_mean", 0)
    shock_precision = game_config.constants.get("shock_precision", 2)

    demand_shock_lists = []
    for _ in range(num_sims):
        shocks = np.random.normal(shock_mean, game_config.constants["demand_shock_std"], time_horizon)
        demand_shock_lists.append(shocks.round(shock_precision).tolist())
        
    return {"demand_shock_lists": demand_shock_lists}

def generate_athey_bagwell_data(game_config, num_sims):
    """Generates player cost type streams for the Athey & Bagwell game."""
    num_players = game_config.constants["number_of_players"]
    time_horizon = game_config.constants["time_horizon"]
    persistence_prob = game_config.constants["persistence_probability"]
    initial_low_prob = game_config.constants.get("initial_low_probability", 0.5)

    player_cost_streams = {}
    # Use consistent naming: player_1 is the challenger
    player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

    for player_id in player_ids:
        sim_streams = []
        for _ in range(num_sims):
            stream = []
            # First period from stationary distribution
            stream.append("low" if random.random() <= initial_low_prob else "high")
            # Subsequent periods based on Markov process
            for _ in range(1, time_horizon):
                prev_cost = stream[-1]
                if random.random() <= persistence_prob:
                    stream.append(prev_cost)
                else:
                    stream.append("high" if prev_cost == "low" else "low")
            sim_streams.append(stream)
        player_cost_streams[player_id] = sim_streams
        
    return {"player_cost_streams": player_cost_streams}

# --- Main Execution ---

def main():
    """Main function to generate and save datasets."""
    experiment_config = get_experiment_config()
    
    # Correctly set random seed from config
    seed = experiment_config.get("random_seed")
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_sims = experiment_config["main_experiment_simulations"]
    datasets = {}

    # 1. Spulber Baseline (5 players)
    spulber_base_config = get_game_config("spulber", "structural_variations", "more_players")
    datasets["spulber_structural_more_players_baseline"] = generate_spulber_data(spulber_base_config, num_sims)

    # 2. Spulber Ablation (5 players, wide cost range)
    spulber_abl_config_base = get_game_config("spulber", "structural_variations", "more_players")
    spulber_abl_params = get_game_config("spulber", "ablation_studies", "wide_cost_range").constants
    spulber_abl_config_base.constants.update(spulber_abl_params)
    datasets["spulber_structural_more_players_ablation_wide_cost_range"] = generate_spulber_data(spulber_abl_config_base, num_sims)

    # 3. Green & Porter Baseline (50 rounds)
    gp_base_config = get_game_config("green_porter", "structural_variations", "long_time_horizon")
    datasets["green_porter_structural_long_time_horizon_baseline"] = generate_green_porter_data(gp_base_config, num_sims)

    # 4. Green & Porter Ablation (50 rounds, high volatility)
    gp_abl_config_base = get_game_config("green_porter", "structural_variations", "long_time_horizon")
    gp_abl_params = get_game_config("green_porter", "ablation_studies", "high_demand_volatility").constants
    gp_abl_config_base.constants.update(gp_abl_params)
    datasets["green_porter_structural_long_time_horizon_ablation_high_demand_volatility"] = generate_green_porter_data(gp_abl_config_base, num_sims)

    # 5. Athey & Bagwell Baseline (50 rounds)
    ab_base_config = get_game_config("athey_bagwell", "structural_variations", "long_time_horizon")
    datasets["athey_bagwell_structural_long_time_horizon_baseline"] = generate_athey_bagwell_data(ab_base_config, num_sims)

    # 6. Athey & Bagwell Ablation (50 rounds, low persistence)
    ab_abl_config_base = get_game_config("athey_bagwell", "structural_variations", "long_time_horizon")
    ab_abl_params = get_game_config("athey_bagwell", "ablation_studies", "low_persistence").constants
    ab_abl_config_base.constants.update(ab_abl_params)
    datasets["athey_bagwell_structural_long_time_horizon_ablation_low_persistence"] = generate_athey_bagwell_data(ab_abl_config_base, num_sims)

    # Save the master datasets to a file
    output_config = get_output_config()
    output_dir = Path(output_config.get("results_dir", "results"))
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "master_datasets.json"
    with open(output_path, 'w') as f:
        json.dump(datasets, f, indent=2)
        
    print(f"✅ Generated 6 master datasets: {list(datasets.keys())}")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    main()