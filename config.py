"""
Updated Configuration Management - Handles ablation × structural variation matrix
Supports equal sample sizes and proper experimental design with nested model configs
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
        experiment_type: 'baseline', 'structural_variations', 'ablation_studies', 'combined'
        condition_name: specific condition within experiment_type
    """
    config = load_config_file()
    game_configs = config.get('game_configs', {}).get(game_name, {})
    
    # Start with baseline constants
    constants = game_configs.get('baseline', {}).copy()
    
    # Handle different experiment types
    if experiment_type == 'baseline':
        condition_name = 'baseline'
    
    elif experiment_type == 'structural_variations':
        # Apply structural variation only
        structural_variations = game_configs.get('structural_variations', {})
        if condition_name and condition_name in structural_variations:
            constants.update(structural_variations[condition_name])
        else:
            # Use first structural variation if no specific condition
            if structural_variations:
                first_condition = list(structural_variations.keys())[0]
                constants.update(structural_variations[first_condition])
                condition_name = first_condition
    
    elif experiment_type == 'ablation_studies':
        # Apply ablation only (to baseline)
        ablation_studies = game_configs.get('ablation_studies', {})
        if condition_name and condition_name in ablation_studies:
            constants.update(ablation_studies[condition_name])
        else:
            # Use first ablation if no specific condition
            if ablation_studies:
                first_condition = list(ablation_studies.keys())[0]
                constants.update(ablation_studies[first_condition])
                condition_name = first_condition
    
    elif experiment_type == 'combined':
        # Apply both structural variation and ablation
        # condition_name should be in format "structural_ablation"
        if condition_name and '_' in condition_name:
            parts = condition_name.split('_', 1)
            structural_part = parts[0]
            ablation_part = parts[1]
            
            # Apply structural variation first
            structural_variations = game_configs.get('structural_variations', {})
            if structural_part in structural_variations:
                constants.update(structural_variations[structural_part])
            
            # Apply ablation second
            ablation_studies = game_configs.get('ablation_studies', {})
            if ablation_part in ablation_studies:
                constants.update(ablation_studies[ablation_part])
    
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
    """
    Get all configurations for a specific game including ablation × structural variation matrix
    
    Generates:
    1. baseline
    2. structural variations only  
    3. ablation × structural variation combinations
    """
    config = load_config_file()
    game_config = config.get('game_configs', {}).get(game_name, {})
    
    configs = []
    
    # 1. Add baseline
    configs.append(get_game_config(game_name, 'baseline'))
    
    # 2. Add structural variations only
    structural_variations = game_config.get('structural_variations', {})
    for structural_name in structural_variations:
        config_obj = get_game_config(game_name, 'structural_variations', structural_name)
        configs.append(config_obj)
    
    # 3. Add ablation × structural variation combinations
    ablation_studies = game_config.get('ablation_studies', {})
    
    for structural_name in structural_variations:
        for ablation_name in ablation_studies:
            # Create combined condition name: "structural_ablation"
            combined_name = f"{structural_name}_{ablation_name}"
            config_obj = get_game_config(game_name, 'combined', combined_name)
            configs.append(config_obj)
    
    return configs


def get_all_experimental_configs() -> List[GameConfig]:
    """Get all experimental configurations across all games"""
    all_configs = []
    games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    
    for game_name in games:
        all_configs.extend(get_all_game_configs(game_name))
    
    return all_configs


def get_simulation_count(experiment_type: str, condition_type: str = None) -> int:
    """
    Get number of simulations based on experiment type and condition
    
    Args:
        experiment_type: 'baseline', 'structural_variations', 'ablation_studies', 'combined'
        condition_type: Additional classification (deprecated, keeping for compatibility)
    """
    config = load_config_file()
    experiment_config = config.get('experiment_config', {})
    
    if experiment_type == 'baseline':
        return experiment_config.get('main_experiment_simulations', 50)
    elif experiment_type in ['structural_variations', 'ablation_studies', 'combined']:
        return experiment_config.get('ablation_experiment_simulations', 50)
    else:
        return experiment_config.get('main_experiment_simulations', 50)


def get_challenger_models() -> List[str]:
    """Get list of challenger models"""
    config = load_config_file()
    return config.get('models', {}).get('challenger_models', [])


def get_defender_model() -> str:
    """Get defender model"""
    config = load_config_file()
    return config.get('models', {}).get('defender_model', 'gemini-2.0-flash-lite')


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    config = load_config_file()
    # Handle nested model_configs under models section
    model_configs = config.get('models', {}).get('model_configs', {})
    
    if model_name not in model_configs:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    return model_configs[model_name]


def get_experiment_config() -> Dict[str, Any]:
    """Get experiment configuration"""
    config = load_config_file()
    return config.get('experiment_config', {})


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


def validate_config() -> bool:
    """Validate configuration file"""
    logger = logging.getLogger(__name__)
    
    try:
        config = load_config_file()
        
        # Check required sections
        required_sections = ['models', 'experiment_config', 'game_configs']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Check models section structure
        models = config.get('models', {})
        challenger_models = models.get('challenger_models', [])
        defender_model = models.get('defender_model')
        model_configs = models.get('model_configs', {})
        
        if not challenger_models:
            logger.error("No challenger models specified")
            return False
        
        if not defender_model:
            logger.error("No defender model specified")
            return False
        
        if not model_configs:
            logger.error("No model configurations specified")
            return False
        
        # Check model configs exist
        all_models = challenger_models + [defender_model]
        
        for model in all_models:
            if model not in model_configs:
                logger.error(f"Model config missing for: {model}")
                return False
        
        # Check game configs
        game_configs = config.get('game_configs', {})
        required_games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
        
        for game in required_games:
            if game not in game_configs:
                logger.error(f"Game config missing for: {game}")
                return False
            
            game_config = game_configs[game]
            if 'baseline' not in game_config:
                logger.error(f"Baseline config missing for game: {game}")
                return False
        
        logger.info("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def create_experiment_summary() -> Dict[str, Any]:
    """Create summary of experimental configuration"""
    config = load_config_file()
    
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    # Count total configurations using the matrix approach
    total_configs = 0
    game_breakdown = {}
    
    for game_name in ['salop', 'green_porter', 'spulber', 'athey_bagwell']:
        game_configs = get_all_game_configs(game_name)
        game_breakdown[game_name] = len(game_configs)
        total_configs += len(game_configs)
    
    # Calculate total competitions and simulations
    total_competitions = total_configs * len(challenger_models)
    
    # Estimate simulations (assuming equal sample sizes)
    experiment_config = get_experiment_config()
    sims_per_competition = experiment_config.get('main_experiment_simulations', 50)
    estimated_simulations = total_competitions * sims_per_competition
    
    return {
        'challenger_models': challenger_models,
        'defender_model': defender_model,
        'total_configurations': total_configs,
        'total_competitions': total_competitions,
        'estimated_simulations': estimated_simulations,
        'game_breakdown': game_breakdown,
        'experimental_design': 'ablation_structural_matrix',
        'equal_sample_sizes': True
    }


def get_prompt_variables(game_config: GameConfig, player_id: str = "A", 
                        current_round: int = 1, **dynamic_vars) -> Dict[str, Any]:
    """Get all variables needed for prompt formatting"""
    variables = {
        'player_id': player_id,
        'current_round': current_round,
        'game_name': game_config.game_name,
        'condition': game_config.condition_name,
        'experiment_type': game_config.experiment_type
    }
    
    # Add game constants
    variables.update(game_config.constants)
    
    # Add any dynamic variables
    variables.update(dynamic_vars)
    
    return variables


def print_experiment_matrix():
    """Print the complete experimental matrix for review"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 EXPERIMENTAL MATRIX:")
    logger.info("=" * 60)
    
    games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    total_conditions = 0
    
    for game_name in games:
        configs = get_all_game_configs(game_name)
        logger.info(f"\n📊 {game_name.upper()}: {len(configs)} conditions")
        
        for config in configs:
            condition_type = "🔷 " if config.experiment_type == "baseline" else "🔸 "
            logger.info(f"  {condition_type}{config.condition_name}")
        
        total_conditions += len(configs)
    
    challenger_models = get_challenger_models()
    total_competitions = total_conditions * len(challenger_models)
    
    experiment_config = get_experiment_config()
    sims_per_competition = experiment_config.get('main_experiment_simulations', 50)
    total_simulations = total_competitions * sims_per_competition
    
    logger.info(f"\n📈 TOTALS:")
    logger.info(f"  • Conditions: {total_conditions}")
    logger.info(f"  • Challengers: {len(challenger_models)}")
    logger.info(f"  • Competitions: {total_competitions}")
    logger.info(f"  • Simulations: {total_simulations}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Test configuration
    logging.basicConfig(level=logging.INFO)
    
    if validate_config():
        print_experiment_matrix()
        
        # Test config generation
        print("\n🧪 SAMPLE CONFIGS:")
        for game_name in ['salop', 'green_porter']:
            configs = get_all_game_configs(game_name)
            print(f"\n{game_name}:")
            for config in configs[:3]:  # Show first 3
                print(f"  {config.condition_name}: {config.experiment_type}")
    else:
        print("❌ Configuration validation failed")