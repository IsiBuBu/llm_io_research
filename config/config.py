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
def load_config() -> Dict[str, Any]:
    """
    Loads and caches the main JSON configuration file using a path
    relative to this script, making it robust to where it's called from.
    """
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}'")
    with open(config_path, 'r', encoding='utf-8') as f:
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

def get_thinking_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Returns the thinking configuration for a model if it is available."""
    model_conf = get_model_config(model_name)
    if model_conf.get('thinking_available', False):
        return model_conf.get('thinking_config')
    return None

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
    """
    Generates a flexible set of GameConfig instances for a given game based on
    the presence of 'structural_variations' and 'ablation_studies' in config.json.
    """
    configs = []
    game_data = load_config()['game_configs'][game_name]
    baseline_constants = game_data.get('baseline', {}).copy()
    baseline_constants.update(game_data.get('challenger_config', {}))

    # --- Process Structural Variations if they exist ---
    structural_variations = game_data.get('structural_variations', {})
    if structural_variations:
        for struct_name, struct_params in structural_variations.items():
            struct_constants = baseline_constants.copy()
            struct_constants.update(struct_params)
            configs.append(GameConfig(
                game_name=game_name,
                experiment_type='structural_variations',
                condition_name=struct_name,
                constants=struct_constants
            ))

    # --- Process Ablation Studies if they exist ---
    ablation_studies = game_data.get('ablation_studies', {})
    if ablation_studies:
        # Create a 3-player base for all ablations
        ablation_base_constants = baseline_constants.copy()
        if 'few_players' in structural_variations:
            ablation_base_constants.update(structural_variations['few_players'])
        else:
            # If 'few_players' is not defined, enforce a 3-player setup manually
            ablation_base_constants['number_of_players'] = 3
        
        for ablation_name, ablation_data in ablation_studies.items():
            final_ablation_constants = ablation_base_constants.copy()
            ablation_params = {k: v for k, v in ablation_data.items() if k != 'description'}
            final_ablation_constants.update(ablation_params)
            
            # Ensure condition name reflects the 3-player base
            condition_name = f"few_players_{ablation_name}"
            
            configs.append(GameConfig(
                game_name=game_name,
                experiment_type='ablation_studies',
                condition_name=condition_name,
                constants=final_ablation_constants
            ))

    # --- If no variations or ablations, run a single baseline config ---
    if not configs:
        configs.append(GameConfig(
            game_name=game_name,
            experiment_type='baseline',
            condition_name='baseline',
            constants=baseline_constants
        ))

    return configs


def get_game_config(game_name: str, experiment_type: str, condition_name: str) -> GameConfig:
    """Constructs a complete GameConfig by merging baseline and experiment-specific parameters."""
    game_data = load_config()['game_configs'][game_name]
    constants = game_data.get('baseline', {}).copy()
    constants.update(game_data.get('challenger_config', {}))
    
    if experiment_type in game_data and condition_name in game_data[experiment_type]:
        params = {k: v for k, v in game_data[experiment_type][condition_name].items() if k != 'description'}
        constants.update(params)

    return GameConfig(game_name, experiment_type, condition_name, constants)

# --- Utility Function for Prompt Variables ---

def get_prompt_variables(game_config: GameConfig, player_id: str, **kwargs) -> Dict[str, Any]:
    """Consolidates all variables needed to format a game prompt."""
    if not isinstance(game_config, GameConfig):
        raise TypeError(f"Expected GameConfig, but got {type(game_config)}")
    variables = game_config.constants.copy()
    
    if 'cost_types' in variables and isinstance(variables['cost_types'], dict):
        variables['high_cost'] = variables['cost_types'].get('high')
        variables['low_cost'] = variables['cost_types'].get('low')

    variables.update({
        'player_id': player_id,
        'number_of_competitors': variables.get('number_of_players', 1) - 1,
        **kwargs
    })

    if 'price_history' in variables:
        variables['len_price_history'] = len(variables['price_history'])
    return variables

# --- Directory Path Accessors ---

def get_main_output_dir() -> Path:
    """Returns the main output directory Path object."""
    return Path(load_config().get('output', {}).get('main_output_dir', 'output'))

def get_data_dir() -> Path:
    """Returns the path for the data directory."""
    return get_main_output_dir() / load_config().get('output', {}).get('data_dir', 'data')

def get_experiments_dir() -> Path:
    """Returns the path for the experiments (raw results) directory."""
    return get_main_output_dir() / load_config().get('output', {}).get('experiments_dir', 'experiments')

def get_analysis_dir() -> Path:
    """Returns the path for the analysis output directory."""
    return get_main_output_dir() / load_config().get('output', {}).get('analysis_dir', 'analysis')