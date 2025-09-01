"""
Minimal configuration system for LLM game theory experiments.
Only includes essential functionality used by runner.py and competition.py
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass 
class ModelConfig:
    """Configuration for a single model"""
    model_name: str
    thinking_available: bool = False
    thinking_enabled: bool = False
    display_name: str = ""
    max_retries: int = 3
    timeout: int = 30

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.model_name

    @property
    def model_key(self) -> str:
        """Alias for model_name for compatibility"""
        return self.model_name


@dataclass
class GameConfig:
    """Configuration for a single game instance"""
    number_of_players: int = 2
    number_of_rounds: int = 1
    num_games: int = 1
    discount_factor: float = 0.95


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment"""
    defender_model_key: str
    challenger_model_keys: List[str]
    game_name: str = "default"
    num_players: int = 2
    num_rounds: int = 1
    num_games: int = 1
    include_thinking: bool = False
    
    def get_defender_model(self) -> ModelConfig:
        """Get defender model configuration"""
        return get_model_config(self.defender_model_key)
    
    def get_challenger_models(self) -> List[ModelConfig]:
        """Get challenger model configurations"""
        return [get_model_config(key) for key in self.challenger_model_keys]


# Global config cache
_config_cache: Optional[Dict[str, Any]] = None


def load_config_file() -> Dict[str, Any]:
    """Load configuration from config.json"""
    global _config_cache
    
    if _config_cache is not None:
        return _config_cache
    
    config_path = Path("config.json")
    if not config_path.exists():
        raise FileNotFoundError("config.json not found. Please create configuration file.")
    
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


def get_game_constants() -> Dict[str, Any]:
    """Get game constants for all games"""
    return load_config_file().get('game_constants', {})


def get_api_config() -> Dict[str, Any]:
    """Get API configuration"""
    return load_config_file().get('api', {})


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    return load_config_file().get('logging', {})


def get_metrics_config() -> Dict[str, Any]:
    """Get metrics configuration"""
    return load_config_file().get('metrics', {'comprehensive_analysis': {'enabled': True}})


def validate_config() -> bool:
    """Basic configuration validation"""
    try:
        # Test loading config
        config = load_config_file()
        
        # Check required sections exist
        required_sections = ['models', 'api', 'game_constants']
        for section in required_sections:
            if section not in config:
                logging.error(f"Missing required config section: {section}")
                return False
        
        # Check we have at least one model
        models = get_available_models()
        if not models:
            logging.error("No models configured")
            return False
    
        # Check game constants exist for expected games
        constants = get_game_constants()
        expected_games = ['salop', 'spulber', 'green_porter', 'athey_bagwell']
        for game in expected_games:
            if game not in constants:
                logging.error(f"Missing game constants for: {game}")
                return False
        
        # Check API config
        api_config = get_api_config()
        if 'gemini_api_key_env' not in api_config:
            logging.error("Missing API key environment variable name in config")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Config validation failed: {e}")
        return False


class GameConstants:
    """Game-specific constants loaded from config"""
    
    def __init__(self, game_config: GameConfig = None):
        self.config = game_config or GameConfig()
        self._constants = get_game_constants()
    
    def get_constants(self, game_name: str) -> Dict[str, Any]:
        """Get constants for a specific game"""
        if game_name not in self._constants:
            raise ValueError(f"No constants defined for game: {game_name}")
        return self._constants[game_name]
    
    # Salop constants
    @property
    def SALOP_BASE_MARKET_SIZE(self) -> int:
        return self._constants['salop']['base_market_size']
    
    @property
    def SALOP_MARGINAL_COST(self) -> float:
        return self._constants['salop']['marginal_cost']
    
    @property
    def SALOP_FIXED_COST(self) -> float:
        return self._constants['salop']['fixed_cost']
    
    @property
    def SALOP_TRANSPORT_COST(self) -> float:
        return self._constants['salop']['transport_cost']
    
    # Spulber constants
    @property
    def SPULBER_BASE_MARKET_SIZE(self) -> int:
        return self._constants['spulber']['base_market_size']
    
    @property
    def SPULBER_MARGINAL_COST(self) -> float:
        return self._constants['spulber']['marginal_cost']
    
    @property
    def SPULBER_MARKET_VALUE(self) -> float:
        return self._constants['spulber']['market_value']
    
    @property
    def SPULBER_RIVAL_COST_MEAN(self) -> float:
        return self._constants['spulber']['rival_cost_mean']
    
    @property
    def SPULBER_RIVAL_COST_STD(self) -> float:
        return self._constants['spulber']['rival_cost_std']
    
    # Green Porter constants
    @property
    def GP_BASE_DEMAND_INTERCEPT(self) -> int:
        return self._constants['green_porter']['base_demand_intercept']
    
    @property
    def GP_MARGINAL_COST(self) -> float:
        return self._constants['green_porter']['marginal_cost']
    
    @property
    def GP_COLLUSIVE_QUANTITY(self) -> float:
        return self._constants['green_porter']['collusive_quantity']
    
    @property
    def GP_DISCOUNT_RATE(self) -> float:
        return self._constants['green_porter']['discount_rate']
    
    # ADDED: Missing Green Porter constants
    @property
    def GP_DEMAND_SHOCK_STD(self) -> float:
        return self._constants['green_porter'].get('demand_shock_std', 5.0)
    
    # Athey Bagwell constants
    @property
    def AB_HIGH_COST(self) -> float:
        return self._constants['athey_bagwell']['high_cost']
    
    @property
    def AB_LOW_COST(self) -> float:
        return self._constants['athey_bagwell']['low_cost']
    
    @property
    def AB_MARKET_PRICE(self) -> float:
        return self._constants['athey_bagwell']['market_price']
    
    @property
    def AB_DISCOUNT_FACTOR(self) -> float:
        return self._constants['athey_bagwell']['discount_factor']
    
    # ADDED: Missing Athey-Bagwell constants  
    @property
    def AB_MARKET_SIZE(self) -> int:
        return self._constants['athey_bagwell'].get('market_size', 1000)
    
    @property
    def AB_COST_PERSISTENCE(self) -> float:
        return self._constants['athey_bagwell'].get('cost_persistence', 0.8)


# Result data classes for compatibility
@dataclass
class PlayerResult:
    """Results for a single player in a game"""
    player_id: str
    profit: float
    actions: List[Dict[str, Any]] = field(default_factory=list)
    win: bool = False
    player_role: str = "unknown"


@dataclass
class GameResult:
    """Results for a complete game"""
    game_name: str
    config: GameConfig
    players: List[PlayerResult]
    total_industry_profit: float = 0.0
    game_id: str = ""