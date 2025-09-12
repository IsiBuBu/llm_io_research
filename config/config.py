# config/config.py

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache

# --- Data Class for Game Configuration ---

@dataclass
class GameConfig:
    """Represents the fully resolved configuration for a single game condition."""
    game_name: str
    experiment_type: str
    condition_name: str
    constants: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the GameConfig instance to a dictionary."""
        return {
            "game_name": self.game_name,
            "experiment_type": self.experiment_type,
            "condition_name": self.condition_name,
            "constants": self.constants
        }

# --- Core Configuration Loading (Cached) ---

@lru_cache(maxsize=1)
def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """Loads and caches the main JSON configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}'")
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Model Configuration Accessors ---

def get_challenger_models() -> List[str]:
    """Returns the list of challenger model names."""
    return load_config().get('models', {}).get('challenger_models', [])

def get_defender_model() -> str:
    """Returns the defender model name."""
    return load_config().get('models', {}).get('defender_model', '')

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Returns the configuration for a specific model."""
    configs = load_config().get('model_configs', {})
    if model_name not in configs:
        raise ValueError(f"Model '{model_name}' not found in model_configs")
    return configs[model_name]

def get_model_display_name(model_name: str) -> str:
    """Returns the display name for a model, defaulting to the model name."""
    return get_model_config(model_name).get('display_name', model_name)

def is_thinking_enabled(model_name: str) -> bool:
    """Checks if thinking is enabled for a model (i.e., budget is positive)."""
    model_conf = get_model_config(model_name)
    if not model_conf.get('thinking_available', False):
        return False
    thinking_conf = model_conf.get('thinking_config', {})
    return thinking_conf.get('thinking_budget', 0) > 0

# --- Experiment and Game Configuration Accessors ---

def get_experiment_config() -> Dict[str, Any]:
    """Returns the main experiment configuration dictionary."""
    return load_config().get('experiment_config', {})

def get_simulation_count(experiment_type: str) -> int:
    """Gets the number of simulations for a given experiment type."""
    exp_config = get_experiment_config()
    if experiment_type == 'ablation_studies':
        return exp_config.get('ablation_experiment_simulations', 50)
    return exp_config.get('main_experiment_simulations', 50)

def get_all_game_configs(game_name: str) -> List[GameConfig]:
    """Generates all GameConfig instances for a given game."""
    configs = []
    game_data = load_config()['game_configs'][game_name]

    # Add baseline (or structural variations that override it)
    struct_vars = game_data.get('structural_variations', {})
    if not struct_vars:
        configs.append(get_game_config(game_name, 'structural_variations', 'baseline'))
    else:
        for condition in struct_vars:
            configs.append(get_game_config(game_name, 'structural_variations', condition))

    # Add ablation studies
    for condition in game_data.get('ablation_studies', {}):
        configs.append(get_game_config(game_name, 'ablation_studies', condition))

    return configs

def get_game_config(game_name: str, experiment_type: str, condition_name: str) -> GameConfig:
    """
    Constructs a complete GameConfig for a specific game and condition by merging
    baseline, challenger, and experiment-specific parameters.
    """
    all_games_data = load_config().get('game_configs', {})
    if game_name not in all_games_data:
        raise ValueError(f"Game '{game_name}' not found in configuration.")

    game_data = all_games_data[game_name]
    constants = game_data.get('baseline', {}).copy()

    # Merge challenger-specific config if it exists
    constants.update(game_data.get('challenger_config', {}))

    # Merge experiment-specific variations
    if experiment_type in ['structural_variations', 'ablation_studies']:
        variations = game_data.get(experiment_type, {})
        if condition_name in variations:
            # Filter out non-parameter keys like 'description'
            params = {k: v for k, v in variations[condition_name].items() if k != 'description'}
            constants.update(params)
        elif condition_name not in ['baseline', 'few_players', 'more_players', 'short_time_horizon', 'long_time_horizon']:
             raise ValueError(f"Condition '{condition_name}' not found for game '{game_name}' under '{experiment_type}'.")

    return GameConfig(
        game_name=game_name,
        experiment_type=experiment_type,
        condition_name=condition_name,
        constants=constants
    )

# --- Utility Function for Prompt Variables ---

def get_prompt_variables(game_config: GameConfig, player_id: str, **kwargs) -> Dict[str, Any]:
    """
    Consolidates all variables needed to format a game prompt.
    Merges game constants with dynamic data provided via kwargs.
    """
    if not isinstance(game_config, GameConfig):
        raise TypeError(f"Expected GameConfig, but got {type(game_config)}")

    variables = game_config.constants.copy()
    variables['player_id'] = player_id
    variables['number_of_competitors'] = variables.get('number_of_players', 1) - 1

    # Update with any dynamic, run-time variables (e.g., current_round)
    variables.update(kwargs)
    return variables

# --- General Configuration Accessors ---

def get_output_dir() -> str:
    """Returns the path for the results output directory."""
    return load_config().get('output', {}).get('results_dir', 'results')