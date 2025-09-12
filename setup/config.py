#!/usr/bin/env python3
"""
Configuration management for Economics LLM Experiments

Updated to work with the new config.json structure containing:
- models and model_configs sections
- game_configs (instead of games)  
- experiment_config, logging, output, api_config
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class GameConfig:
    """Game configuration data class"""
    constants: Dict[str, Any]
    challenger_config: Optional[Dict[str, Any]] = None

def load_config_file(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return json.load(f)

def get_challenger_models() -> List[str]:
    """Get list of challenger model names"""
    config = load_config_file()
    return config.get('models', {}).get('challenger_models', [])

def get_defender_model() -> str:
    """Get defender model name"""
    config = load_config_file()
    return config.get('models', {}).get('defender_model', '')

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    config = load_config_file()
    model_configs = config.get('model_configs', {})
    
    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' not found in model_configs")
    
    return model_configs[model_name]

def get_model_display_name(model_name: str) -> str:
    """Get display name for a model"""
    try:
        model_config = get_model_config(model_name)
        return model_config.get('display_name', model_name)
    except Exception:
        return model_name

def is_thinking_enabled(model_name: str) -> bool:
    """Check if thinking is enabled for a model"""
    try:
        model_config = get_model_config(model_name)
        
        if not model_config.get('thinking_available', False):
            return False

        thinking_config = model_config.get('thinking_config', {})
        thinking_budget = thinking_config.get('thinking_budget', 0)
        
        # Thinking is enabled if budget > 0
        return thinking_budget > 0
        
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

def get_experiment_config() -> Dict[str, Any]:
    """Get experiment configuration"""
    config = load_config_file()
    return config.get('experiment_config', {})

def get_simulation_count(experiment_type: str) -> int:
    """Get number of simulations for experiment type"""
    config = load_config_file()
    experiment_config = config.get('experiment_config', {})
    
    if experiment_type in ['baseline', 'structural_variations']:
        return experiment_config.get('main_experiment_simulations', 50)
    elif experiment_type == 'ablation_studies':
        return experiment_config.get('ablation_experiment_simulations', 50)
    else:
        return experiment_config.get('main_experiment_simulations', 50)

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
    game_configs = config.get('game_configs', {})
    
    if game_name not in game_configs:
        raise ValueError(f"Game '{game_name}' not found in game_configs")
    
    game_data = game_configs[game_name]
    
    # Get baseline configuration
    baseline = game_data.get('baseline', {})
    
    # Add challenger config if it exists (for games like Spulber)
    challenger_config = game_data.get('challenger_config')
    if challenger_config:
        constants = {**baseline, **challenger_config}
    else:
        constants = baseline.copy()
    
    # Apply experiment-specific modifications
    if experiment_type == 'structural_variations' and condition_name:
        structural_vars = game_data.get('structural_variations', {})
        if condition_name in structural_vars:
            constants.update(structural_vars[condition_name])
    
    elif experiment_type == 'ablation_studies' and condition_name:
        ablations = game_data.get('ablation_studies', {})
        if condition_name in ablations:
            # Remove description field if it exists (it's not a game parameter)
            ablation_params = {k: v for k, v in ablations[condition_name].items() 
                             if k != 'description'}
            constants.update(ablation_params)
    
    return GameConfig(
        constants=constants,
        challenger_config=challenger_config
    )

def get_all_game_names() -> List[str]:
    """Get list of all available game names"""
    config = load_config_file()
    return list(config.get('game_configs', {}).keys())

def get_structural_variations(game_name: str) -> Dict[str, Any]:
    """Get structural variations for a specific game"""
    config = load_config_file()
    game_configs = config.get('game_configs', {})
    
    if game_name not in game_configs:
        raise ValueError(f"Game '{game_name}' not found in game_configs")
    
    return game_configs[game_name].get('structural_variations', {})

def get_ablation_studies(game_name: str) -> Dict[str, Any]:
    """Get ablation studies for a specific game"""
    config = load_config_file()
    game_configs = config.get('game_configs', {})
    
    if game_name not in game_configs:
        raise ValueError(f"Game '{game_name}' not found in game_configs")
    
    return game_configs[game_name].get('ablation_studies', {})

def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    config = load_config_file()
    return config.get('logging', {})

def get_output_config() -> Dict[str, Any]:
    """Get output configuration"""
    config = load_config_file()
    return config.get('output', {})

def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    config = load_config_file()
    return config.get('api_config', {})

def get_baseline_time_horizon(game_name: str) -> Optional[int]:
    """Get baseline time horizon for dynamic games"""
    if game_name not in ['green_porter', 'athey_bagwell']:
        return None
    
    config = load_config_file()
    game_configs = config.get('game_configs', {})
    
    if game_name not in game_configs:
        return None
    
    baseline = game_configs[game_name].get('baseline', {})
    return baseline.get('time_horizon')

def validate_config() -> bool:
    """Validate that config.json has required structure"""
    try:
        config = load_config_file()
        
        # Check required top-level sections
        required_sections = ['models', 'model_configs', 'experiment_config', 'game_configs']
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
        
        # Check models section
        models = config.get('models', {})
        if 'challenger_models' not in models or 'defender_model' not in models:
            print("Missing challenger_models or defender_model in models section")
            return False
        
        # Check that all referenced models have configs
        model_configs = config.get('model_configs', {})
        all_models = models['challenger_models'] + [models['defender_model']]
        
        for model in all_models:
            if model not in model_configs:
                print(f"Model '{model}' referenced but not found in model_configs")
                return False
        
        # Check experiment_config
        exp_config = config.get('experiment_config', {})
        if 'main_experiment_simulations' not in exp_config:
            print("Missing main_experiment_simulations in experiment_config")
            return False
        
        # Check game_configs
        game_configs = config.get('game_configs', {})
        if not game_configs:
            print("No games defined in game_configs")
            return False
        
        # Check each game has baseline
        for game_name, game_data in game_configs.items():
            if 'baseline' not in game_data:
                print(f"Game '{game_name}' missing baseline configuration")
                return False
        
        return True
        
    except Exception as e:
        print(f"Config validation error: {e}")
        return False

def print_config_summary():
    """Print summary of current configuration"""
    try:
        config = load_config_file()
        
        print("="*60)
        print("CONFIG SUMMARY")
        print("="*60)
        
        # Models
        models = config.get('models', {})
        challenger_models = models.get('challenger_models', [])
        defender_model = models.get('defender_model', '')
        
        print(f"Challenger models ({len(challenger_models)}):")
        for model in challenger_models:
            display_name = get_model_display_name(model)
            thinking = "✓" if is_thinking_enabled(model) else "✗"
            print(f"  - {model} ({display_name}) [Thinking: {thinking}]")
        
        print(f"\nDefender model: {defender_model} ({get_model_display_name(defender_model)})")
        
        # Experiment config
        exp_config = get_experiment_config()
        print(f"\nExperiment simulations:")
        print(f"  - Main experiments: {exp_config.get('main_experiment_simulations', 'N/A')}")
        print(f"  - Ablation studies: {exp_config.get('ablation_experiment_simulations', 'N/A')}")
        
        # Games
        game_configs = config.get('game_configs', {})
        print(f"\nGames ({len(game_configs)}):")
        for game_name, game_data in game_configs.items():
            struct_vars = len(game_data.get('structural_variations', {}))
            ablations = len(game_data.get('ablation_studies', {}))
            print(f"  - {game_name}: {struct_vars} structural variations, {ablations} ablations")
        
    except Exception as e:
        print(f"Error printing config summary: {e}")

if __name__ == "__main__":
    # Validate and print config when run directly
    print("Validating config.json...")
    if validate_config():
        print("✓ Configuration is valid")
        print_config_summary()
    else:
        print("✗ Configuration has errors")
        exit(1)