# validate_results.py

import os
import json
import sys
from pathlib import Path

# Ensure the project root is in the Python path to import config functions
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from config.config import (
        load_config,
        get_all_game_configs,
        get_challenger_models,
        get_simulation_count
    )
except ImportError:
    print("ERROR: Could not import from 'config.config'. Make sure this script is in the 'llm_io_research' directory.")
    sys.exit(1)

def validate_all_results():
    """
    Validates the entire directory structure and content of the experimental results
    against the specifications in config.json.
    """
    print("--- Starting Validation of Experimental Results ---")
    
    config = load_config()
    output_dir = Path(config.get('output', {}).get('main_output_dir', 'output'))
    experiments_dir = output_dir / config.get('output', {}).get('experiments_dir', 'experiments')

    if not experiments_dir.exists():
        print(f"❌ ERROR: Experiments directory not found at '{experiments_dir}'")
        return

    all_games = list(config.get('game_configs', {}).keys())
    challenger_models = get_challenger_models()
    
    total_errors = 0
    
    # 1. Check top-level game folders
    for game_name in all_games:
        game_dir = experiments_dir / game_name
        print(f"\n--- Checking Game: {game_name} ---")
        
        if not game_dir.is_dir():
            print(f"❌ ERROR: Missing directory for game '{game_name}'")
            total_errors += 1
            continue
            
        # 2. Check for each model's subfolder
        for model_name in challenger_models:
            model_dir = game_dir / model_name
            if not model_dir.is_dir():
                print(f"  ❌ ERROR: Missing subfolder for model '{model_name}' in '{game_name}'")
                total_errors += 1
                continue
            
            # 3. Check for the correct JSON files within each model folder
            game_configs = get_all_game_configs(game_name)
            for game_config in game_configs:
                expected_filename = f"{game_config.condition_name}_competition_result.json"
                json_path = model_dir / expected_filename
                
                if not json_path.exists():
                    print(f"    ❌ ERROR: Missing result file '{expected_filename}' for model '{model_name}'")
                    total_errors += 1
                    continue
                
                # 4. Validate the content of each JSON file
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    # Validate top-level metadata
                    if data.get('challenger_model') != model_name:
                        print(f"      ❌ ERROR in '{expected_filename}': Incorrect 'challenger_model'. Expected '{model_name}', found '{data.get('challenger_model')}'")
                        total_errors += 1
                    
                    if data.get('experiment_type') != game_config.experiment_type:
                        print(f"      ❌ ERROR in '{expected_filename}': Incorrect 'experiment_type'. Expected '{game_config.experiment_type}', found '{data.get('experiment_type')}'")
                        total_errors += 1
                        
                    if data.get('condition_name') != game_config.condition_name:
                        print(f"      ❌ ERROR in '{expected_filename}': Incorrect 'condition_name'. Expected '{game_config.condition_name}', found '{data.get('condition_name')}'")
                        total_errors += 1

                    # Validate number of simulations
                    expected_sim_count = get_simulation_count(game_config.experiment_type)
                    actual_sim_count = len(data.get('simulation_results', []))
                    if actual_sim_count != expected_sim_count:
                        print(f"      ❌ ERROR in '{expected_filename}': Incorrect number of simulations. Expected {expected_sim_count}, found {actual_sim_count}")
                        total_errors += 1
                        
                    # Validate number of rounds for dynamic games
                    if game_name in ['green_porter', 'athey_bagwell']:
                        expected_rounds = game_config.constants.get('time_horizon', 0)
                        if expected_rounds > 0:
                            first_sim_results = data.get('simulation_results', [{}])[0]
                            rounds_data = first_sim_results.get('game_data', {}).get('rounds', [])
                            if len(rounds_data) != expected_rounds:
                                print(f"      ❌ ERROR in '{expected_filename}': Incorrect number of rounds in simulation. Expected {expected_rounds}, found {len(rounds_data)}")
                                total_errors += 1

                    print(f"    ✅ OK: '{expected_filename}'")

                except json.JSONDecodeError:
                    print(f"    ❌ ERROR: Could not parse JSON in file '{expected_filename}'")
                    total_errors += 1
                except Exception as e:
                    print(f"    ❌ ERROR: An unexpected error occurred while checking '{expected_filename}': {e}")
                    total_errors += 1

    print("\n--- Validation Summary ---")
    if total_errors == 0:
        print("✅ Success! All directories and files are structured and formatted correctly.")
    else:
        print(f"❌ Found {total_errors} error(s). Please review the logs above.")

if __name__ == "__main__":
    validate_all_results()