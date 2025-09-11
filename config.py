"""
Configuration Management - Updated for Gemini-only experiments with thinking support
Handles game configs, model configs, and prompt variable generation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configuration for a specific game experiment"""
    game_name: str
    experiment_type: str  # 'baseline', 'structural_variations', 'ablation_studies'
    condition_name: str
    constants: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'game_name': self.game_name,
            'experiment_type': self.experiment_type,
            'condition_name': self.condition_name,
            'constants': self.constants
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Provide .get() method for backward compatibility
        Delegates to constants dictionary
        """
        return self.constants.get(key, default)


def load_config_file() -> Dict[str, Any]:
    """Load configuration from config.json"""
    config_path = Path("config.json")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {e}")


def get_game_config(game_name: str, experiment_type: str, 
                   condition_name: Optional[str] = None) -> GameConfig:
    """
    Get game configuration for specific experiment
    
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
            # Specific condition overrides
            condition_constants = experiment_section[condition_name]
            constants.update(condition_constants)
        elif isinstance(experiment_section, dict) and len(experiment_section) == 1:
            # Single condition in experiment_type
            single_condition = list(experiment_section.values())[0]
            constants.update(single_condition)
            condition_name = list(experiment_section.keys())[0]
        else:
            # Use first condition if multiple available
            if experiment_section:
                first_condition = list(experiment_section.keys())[0]
                constants.update(experiment_section[first_condition])
                condition_name = first_condition
    
    # Set default condition name if not specified
    if not condition_name:
        condition_name = 'baseline' if experiment_type == 'baseline' else 'default'
    
    return GameConfig(
        game_name=game_name,
        experiment_type=experiment_type,
        condition_name=condition_name,
        constants=constants
    )


def get_all_game_configs(game_name: str) -> List[GameConfig]:
    """Get all configurations for a specific game"""
    config = load_config_file()
    game_config = config.get('game_configs', {}).get(game_name, {})
    
    configs = []
    
    # Add baseline
    configs.append(get_game_config(game_name, 'baseline'))
    
    # Add structural variations
    structural_variations = game_config.get('structural_variations', {})
    for condition_name in structural_variations:
        configs.append(get_game_config(game_name, 'structural_variations', condition_name))
    
    # Add ablation studies
    ablation_studies = game_config.get('ablation_studies', {})
    for condition_name in ablation_studies:
        configs.append(get_game_config(game_name, 'ablation_studies', condition_name))
    
    return configs


def get_all_experimental_configs() -> List[GameConfig]:
    """Get all experimental configurations across all games"""
    all_configs = []
    games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    
    for game_name in games:
        all_configs.extend(get_all_game_configs(game_name))
    
    return all_configs


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
            'time_horizon': constants.get('time_horizon', 50),
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
            'market_size': constants.get('market_size', 100),
            'demand_intercept': constants.get('demand_intercept', 100),
            'rival_cost_mean': constants.get('rival_cost_mean', 10),
            'rival_cost_std': constants.get('rival_cost_std', 2.0),
            'your_cost': constants.get('private_values', {}).get('challenger_cost', 8)
        })
    
    elif game_config.game_name == 'athey_bagwell':
        cost_types = constants.get('cost_types', {'low': 15, 'high': 25})
        variables.update({
            'time_horizon': constants.get('time_horizon', 50),
            'cost_types': cost_types,  # Keep as dict for template access like {cost_types[high]}
            'persistence_probability': constants.get('persistence_probability', 0.7),
            'market_price': constants.get('market_price', 30),
            'market_size': constants.get('market_size', 100),
            'discount_factor': constants.get('discount_factor', 0.95),
            'your_cost_type': dynamic_vars.get('your_cost_type', 'high'),
            'all_reports_history_detailed': dynamic_vars.get('all_reports_history_detailed', [])
        })
    
    # Add any additional dynamic variables
    variables.update(dynamic_vars)
    
    return variables


# Backward compatibility alias
def generate_prompt_variables(game_config: GameConfig, 
                            dynamic_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backward compatibility wrapper for get_prompt_variables"""
    if dynamic_vars is None:
        dynamic_vars = {}
    return get_prompt_variables(game_config, **dynamic_vars)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    config = load_config_file()
    model_configs = config.get('models', {}).get('model_configs', {})
    
    if model_name not in model_configs:
        raise ValueError(f"Model {model_name} not found in config.json")
    
    return model_configs[model_name]


def get_challenger_models() -> List[str]:
    """Get list of challenger models"""
    config = load_config_file()
    return config.get('models', {}).get('challenger_models', [])


def get_defender_model() -> str:
    """Get defender model name"""
    config = load_config_file()
    return config.get('models', {}).get('defender_model', 'gemini_2_0_flash_lite')


def get_experiment_config() -> Dict[str, Any]:
    """Get experiment configuration"""
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
    """Validate configuration file"""
    logger = logging.getLogger(__name__)
    
    try:
        config = load_config_file()
        
        # Check required sections
        required_sections = ['models', 'api', 'game_configs', 'experiment_config']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Validate models section
        models = config.get('models', {})
        if not models.get('challenger_models'):
            logger.error("No challenger models configured")
            return False
        
        if not models.get('defender_model'):
            logger.error("No defender model configured")
            return False
        
        model_configs = models.get('model_configs', {})
        if not model_configs:
            logger.error("No model configurations found")
            return False
        
        # Validate all challenger models have configs
        for model in models['challenger_models']:
            if model not in model_configs:
                logger.error(f"Missing config for challenger model: {model}")
                return False
        
        # Validate defender model has config
        defender = models['defender_model']
        if defender not in model_configs:
            logger.error(f"Missing config for defender model: {defender}")
            return False
        
        # Validate game configs
        game_configs = config.get('game_configs', {})
        required_games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
        
        for game in required_games:
            if game not in game_configs:
                logger.error(f"Missing game config: {game}")
                return False
            
            game_config = game_configs[game]
            if 'baseline' not in game_config:
                logger.error(f"Missing baseline config for game: {game}")
                return False
        
        # Validate API config
        api_config = config.get('api', {})
        if 'google' not in api_config:
            logger.error("Missing Google API configuration")
            return False
        
        google_config = api_config['google']
        if 'api_key_env' not in google_config:
            logger.error("Missing Google API key environment variable name")
            return False
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def get_simulation_count(experiment_type: str) -> int:
    """Get number of simulations for experiment type"""
    config = load_config_file()
    experiment_config = config.get('experiment_config', {})
    
    if experiment_type in ['baseline', 'structural_variations']:
        return experiment_config.get('main_experiment_simulations', 50)
    elif experiment_type == 'ablation_studies':
        return experiment_config.get('ablation_experiment_simulations', 20)
    else:
        return experiment_config.get('main_experiment_simulations', 50)


def is_thinking_enabled(model_name: str) -> bool:
    """Check if thinking is enabled for a model"""
    try:
        model_config = get_model_config(model_name)
        
        if not model_config.get('thinking_available', False):
            return False
        
        thinking_config = model_config.get('thinking_config', {})
        thinking_budget = thinking_config.get('thinking_budget', 0)
        
        # Thinking is enabled if budget > 0 or -1 (dynamic)
        return thinking_budget != 0
        
    except Exception:
        return False


def get_thinking_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Get thinking configuration for a model"""
    try:
        model_config = get_model_config(model_name)
        
        if not model_config.get('thinking_available', False):
            return None
        
        return model_config.get('thinking_config')
        
    except Exception:
        return None


def get_model_display_name(model_name: str) -> str:
    """Get display name for a model"""
    try:
        model_config = get_model_config(model_name)
        return model_config.get('display_name', model_name)
    except Exception:
        return model_name


def create_experiment_summary() -> Dict[str, Any]:
    """Create summary of experimental configuration"""
    config = load_config_file()
    
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    # Count total configurations
    total_configs = 0
    game_breakdown = {}
    
    for game_name in ['salop', 'green_porter', 'spulber', 'athey_bagwell']:
        game_configs = get_all_game_configs(game_name)
        game_breakdown[game_name] = len(game_configs)
        total_configs += len(game_configs)
    
    # Calculate total competitions
    total_competitions = total_configs * len(challenger_models)
    
    # Estimate simulations
    experiment_config = get_experiment_config()
    estimated_simulations = total_competitions * experiment_config.get('main_experiment_simulations', 50)
    
    return {
        'challenger_models': challenger_models,
        'defender_model': defender_model,
        'total_games': 4,
        'total_configs': total_configs,
        'total_competitions': total_competitions,
        'estimated_simulations': estimated_simulations,
        'game_breakdown': game_breakdown,
        'experiment_config': experiment_config
    }