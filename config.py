"""
Compact configuration system for LLM game theory experiments.
Loads game configs from config.json and provides clean access for prompts and experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass 
class ModelConfig:
    """Configuration for a single model"""
    model_name: str
    thinking_available: bool = False
    display_name: str = ""
    timeout: int = 30

    @property
    def thinking_enabled(self) -> bool:
        # Gemini 2.5 Pro has thinking always on, others can be controlled
        return self.model_name == "gemini-2.5-pro"


@dataclass
class GameConfig:
    """Game configuration with all constants"""
    game_name: str
    experiment_type: str  # 'baseline', 'structural_variations', 'ablation_studies'
    condition_name: str   # 'five_players', 'high_noise', etc.
    constants: Dict[str, Any]


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def load_config_file() -> Dict[str, Any]:
    """Load configuration from config.json"""
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("config.json not found.")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config_cache = json.load(f)
        return _config_cache
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config.json: {e}")


def get_available_models() -> Dict[str, ModelConfig]:
    """Get available models from config"""
    config = load_config_file()
    models = {}
    
    for key, data in config.get('models', {}).items():
        models[key] = ModelConfig(**data)
    
    return models


def get_model_config(model_key: str) -> ModelConfig:
    """Get specific model configuration"""
    models = get_available_models()
    if model_key not in models:
        raise ValueError(f"Unknown model: {model_key}")
    return models[model_key]


def get_game_config(game_name: str, experiment_type: str = 'baseline', condition_name: str = None) -> GameConfig:
    """
    Get complete game configuration with merged constants.
    
    Args:
        game_name: 'salop', 'green_porter', 'spulber', 'athey_bagwell'
        experiment_type: 'baseline', 'structural_variations', 'ablation_studies'
        condition_name: specific condition within experiment_type
    """
    config = load_config_file()
    game_configs = config.get('game_configs', {}).get(game_name, {})
    
    # Start with baseline constants
    constants = game_configs.get('baseline', {}).copy()
    
    # Override with experiment-specific constants
    if experiment_type != 'baseline' and experiment_type in game_configs:
        experiment_section = game_configs[experiment_type]
        
        if condition_name and condition_name in experiment_section:
            condition_constants = experiment_section[condition_name]
            constants.update(condition_constants)
    
    return GameConfig(
        game_name=game_name,
        experiment_type=experiment_type,
        condition_name=condition_name or 'default',
        constants=constants
    )


def get_prompt_variables(game_config: GameConfig, player_id: str = "A", 
                        current_round: int = 1, **dynamic_vars) -> Dict[str, Any]:
    """Get all variables needed for prompt formatting"""
    constants = game_config.constants
    
    # Base variables
    variables = {
        'player_id': player_id,
        'current_round': current_round,
        'number_of_players': constants.get('number_of_players', 3),
    }
    
    # Add all constants
    variables.update(constants)
    
    # Game-specific variable mappings
    if game_config.game_name == 'salop':
        variables.update({
            'market_size': constants.get('market_size', 1000),
            'marginal_cost': constants.get('marginal_cost', 8),
            'fixed_cost': constants.get('fixed_cost', 100),
            'transport_cost': constants.get('transport_cost', 1.5),
            'v': constants.get('v', 30)
        })
    
    elif game_config.game_name == 'green_porter':
        variables.update({
            'base_demand': constants.get('base_demand', 120),
            'marginal_cost': constants.get('marginal_cost', 20),
            'demand_shock_std': constants.get('demand_shock_std', 5),
            'trigger_price': constants.get('trigger_price', 55),
            'punishment_duration': constants.get('punishment_duration', 3),
            'collusive_quantity': constants.get('collusive_quantity', 17),
            'cournot_quantity': constants.get('cournot_quantity', 25),
            'discount_factor': constants.get('discount_factor', 0.95),
            'current_market_state': dynamic_vars.get('current_market_state', 'Collusive'),
            'price_history': dynamic_vars.get('price_history', [])
        })
    
    elif game_config.game_name == 'spulber':
        variables.update({
            'demand_intercept': constants.get('demand_intercept', 100),
            'rival_cost_mean': constants.get('rival_cost_mean', 10),
            'rival_cost_std': constants.get('rival_cost_std', 2.0),
            'your_cost': constants.get('private_values', {}).get('challenger_cost', 8)
        })
    
    elif game_config.game_name == 'athey_bagwell':
        cost_types = constants.get('cost_types', {'low': 15, 'high': 25})
        variables.update({
            'cost_low': cost_types.get('low', 15),
            'cost_high': cost_types.get('high', 25),
            'persistence_probability': constants.get('persistence_probability', 0.7),
            'market_price': constants.get('market_price', 30),
            'market_size': constants.get('market_size', 100),
            'discount_factor': constants.get('discount_factor', 0.95),
            'current_cost_type': dynamic_vars.get('current_cost_type', 'high'),
            'all_reports_history_detailed': dynamic_vars.get('all_reports_history_detailed', [])
        })
    
    # Add any additional dynamic variables
    variables.update(dynamic_vars)
    
    return variables


def get_all_game_configs(game_name: str) -> List[GameConfig]:
    """Get all experiment configurations for a game"""
    config = load_config_file()
    game_configs_data = config.get('game_configs', {}).get(game_name, {})
    
    configs = []
    
    # Baseline
    configs.append(get_game_config(game_name, 'baseline'))
    
    # Structural variations
    if 'structural_variations' in game_configs_data:
        for condition in game_configs_data['structural_variations']:
            configs.append(get_game_config(game_name, 'structural_variations', condition))
    
    # Ablation studies  
    if 'ablation_studies' in game_configs_data:
        for condition in game_configs_data['ablation_studies']:
            configs.append(get_game_config(game_name, 'ablation_studies', condition))
    
    return configs


def get_experiment_config() -> Dict[str, Any]:
    """Get experiment configuration (models, simulation counts)"""
    config = load_config_file()
    return config.get('experiment_config', {})


def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    config = load_config_file()
    return config.get('api', {})


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    config = load_config_file()
    return config.get('logging', {})


def validate_config() -> bool:
    """Basic configuration validation"""
    try:
        config = load_config_file()
        required_sections = ['models', 'api', 'game_configs', 'experiment_config']
        return all(section in config for section in required_sections)
    except Exception:
        return False