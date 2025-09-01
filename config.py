# config.py - Compact Version for Comprehensive Metrics
"""
Essential configuration system for LLM game theory experiments.
Provides all parameters needed for comprehensive behavioral metrics.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass 
class ModelConfig:
    model_name: str
    thinking_available: bool = False
    thinking_enabled: bool = False
    display_name: str = ""
    max_retries: int = 3
    timeout: int = 30

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.model_name

@dataclass
class GameConfig:
    number_of_players: int = 2
    number_of_rounds: int = 1
    num_games: int = 1
    market_size_override: Optional[int] = None
    demand_intercept_override: Optional[float] = None
    discount_factor: float = 0.95
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameConfig':
        return cls(**data)

@dataclass
class ExperimentConfig:
    defender_model_key: str
    challenger_model_keys: List[str]
    game_name: str = "default"
    num_players: int = 2
    num_rounds: int = 1
    num_games: int = 1
    include_thinking: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(**data)
    
    def get_defender_model(self) -> ModelConfig:
        """Get defender model configuration"""
        return get_model_config(self.defender_model_key)
    
    def get_challenger_models(self) -> List[ModelConfig]:
        """Get challenger model configurations"""
        return [get_model_config(key) for key in self.challenger_model_keys]
    
    @property
    def defender_model(self) -> ModelConfig:
        """Property access to defender model"""
        return self.get_defender_model()
    
    @property
    def challenger_models(self) -> List[ModelConfig]:
        """Property access to challenger models"""
        return self.get_challenger_models()

@dataclass
class PlayerResult:
    player_id: str
    profit: float
    actions: List[Dict[str, Any]] = field(default_factory=list)
    win: bool = False
    player_role: str = "unknown"

@dataclass
class GameResult:
    game_name: str
    config: GameConfig
    players: List[PlayerResult]
    total_industry_profit: float
    experiment_config: Optional[ExperimentConfig] = None
    market_price: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    challenger_model_key: Optional[str] = None

class GameConstants:
    """Economic constants for all games with comprehensive metrics support"""
    
    def __init__(self, config: GameConfig):
        self.config = config
        constants = get_game_constants()
        
        # SALOP - Spatial Competition
        salop = constants.get('salop', {})
        self._SALOP_BASE_MARKET_SIZE = salop.get('base_market_size', 300)
        self._SALOP_MARGINAL_COST = salop.get('marginal_cost', 8.0)
        self._SALOP_FIXED_COST = salop.get('fixed_cost', 100.0)
        self._SALOP_TRANSPORT_COST = salop.get('transport_cost', 0.5)
        
        # SPULBER - Bertrand Competition (CRITICAL for behavioral metrics)
        spulber = constants.get('spulber', {})
        self._SPULBER_BASE_MARKET_SIZE = spulber.get('base_market_size', 1000)
        self._SPULBER_MARGINAL_COST = spulber.get('marginal_cost', 10.0)
        self._SPULBER_DEMAND_SLOPE = spulber.get('demand_slope', 1.0)
        self._SPULBER_MARKET_VALUE = spulber.get('market_value', 100.0)
        self._SPULBER_RIVAL_COST_MEAN = spulber.get('rival_cost_mean', 12.0)
        self._SPULBER_RIVAL_COST_STD = spulber.get('rival_cost_std', 3.0)
        
        # GREEN PORTER - Collusion (CRITICAL for cooperation metrics)
        gp = constants.get('green_porter', {})
        self._GP_BASE_DEMAND_INTERCEPT = gp.get('base_demand_intercept', 100)
        self._GP_MARGINAL_COST = gp.get('marginal_cost', 10.0)
        self._GP_DEMAND_SHOCK_STD = gp.get('demand_shock_std', 5.0)
        self._GP_COLLUSIVE_QUANTITY = gp.get('collusive_quantity', 22.5)
        self._GP_COMPETITIVE_QUANTITY = gp.get('competitive_quantity', 25.0)
        self._GP_DISCOUNT_RATE = gp.get('discount_rate', 0.05)
        
        # ATHEY BAGWELL - Information Collusion (CRITICAL for deception metrics)
        ab = constants.get('athey_bagwell', {})
        self._AB_HIGH_COST = ab.get('high_cost', 15.0)
        self._AB_LOW_COST = ab.get('low_cost', 5.0)
        self._AB_MARKET_PRICE = ab.get('market_price', 50.0)
        self._AB_COST_PERSISTENCE = ab.get('cost_persistence', 0.8)
        self._AB_BASE_MARKET_SIZE = ab.get('base_market_size', 1000)
        self._AB_DISCOUNT_FACTOR = ab.get('discount_factor', 0.95)

    # SALOP Properties
    @property
    def SALOP_MARKET_SIZE(self) -> int:
        override = getattr(self.config, 'market_size_override', None)
        return override if override else self._SALOP_BASE_MARKET_SIZE * self.config.number_of_players
    
    @property
    def SALOP_MARGINAL_COST(self) -> float:
        return self._SALOP_MARGINAL_COST
    
    @property
    def SALOP_FIXED_COST(self) -> float:
        return self._SALOP_FIXED_COST
    
    @property
    def SALOP_TRANSPORT_COST(self) -> float:
        return self._SALOP_TRANSPORT_COST
    
    # SPULBER Properties (CRITICAL for behavioral metrics)
    @property
    def SPULBER_MARKET_SIZE(self) -> int:
        override = getattr(self.config, 'market_size_override', None)
        return override if override else self._SPULBER_BASE_MARKET_SIZE * self.config.number_of_players
    
    @property
    def SPULBER_MARGINAL_COST(self) -> float:
        return self._SPULBER_MARGINAL_COST
    
    @property
    def SPULBER_DEMAND_SLOPE(self) -> float:
        return self._SPULBER_DEMAND_SLOPE
    
    @property
    def SPULBER_MARKET_VALUE(self) -> float:
        return self._SPULBER_MARKET_VALUE
    
    @property
    def SPULBER_RIVAL_COST_MEAN(self) -> float:
        return self._SPULBER_RIVAL_COST_MEAN
    
    @property
    def SPULBER_RIVAL_COST_STD(self) -> float:
        return self._SPULBER_RIVAL_COST_STD
    
    # GREEN PORTER Properties (CRITICAL for cooperation metrics)
    @property
    def GP_MARGINAL_COST(self) -> float:
        return self._GP_MARGINAL_COST
    
    @property
    def GP_DEMAND_SHOCK_STD(self) -> float:
        return self._GP_DEMAND_SHOCK_STD
    
    @property
    def GP_DEMAND_INTERCEPT(self) -> float:
        override = getattr(self.config, 'demand_intercept_override', None)
        return override if override else self._GP_BASE_DEMAND_INTERCEPT * self.config.number_of_players
    
    @property
    def GP_COLLUSIVE_QUANTITY(self) -> float:
        return self._GP_COLLUSIVE_QUANTITY
    
    @property
    def GP_COMPETITIVE_QUANTITY(self) -> float:
        return self._GP_COMPETITIVE_QUANTITY
    
    @property
    def GP_DISCOUNT_RATE(self) -> float:
        return self._GP_DISCOUNT_RATE
    
    # ATHEY BAGWELL Properties (CRITICAL for deception metrics)
    @property
    def AB_HIGH_COST(self) -> float:
        return self._AB_HIGH_COST
    
    @property
    def AB_LOW_COST(self) -> float:
        return self._AB_LOW_COST
    
    @property
    def AB_MARKET_PRICE(self) -> float:
        return self._AB_MARKET_PRICE
    
    @property
    def AB_COST_PERSISTENCE(self) -> float:
        return self._AB_COST_PERSISTENCE
    
    @property
    def AB_BASE_MARKET_SIZE(self) -> int:
        return self._AB_BASE_MARKET_SIZE
    
    @property
    def AB_DISCOUNT_FACTOR(self) -> float:
        return self._AB_DISCOUNT_FACTOR

# Configuration loading
_config_cache = {}

def load_config_file(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    global _config_cache
    
    if _config_cache:
        return _config_cache
    
    try:
        with open(config_path, 'r') as f:
            _config_cache = json.load(f)
    except FileNotFoundError:
        _config_cache = _get_default_config()
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading config: {e}")
        _config_cache = _get_default_config()
    
    return _config_cache

def _get_default_config() -> Dict[str, Any]:
    """Minimal default configuration"""
    return {
        "models": {
            "gemini-2.0-flash-lite": {
                "model_name": "gemini-2.0-flash-lite",
                "display_name": "Gemini 2.0 Flash Lite",
                "thinking_available": False,
                "thinking_enabled": False
            }
        },
        "experiment_presets": {
            "debug_test": {
                "defender_model_key": "gemini-2.0-flash-lite",
                "challenger_model_keys": ["gemini-2.0-flash-lite"],
                "game_name": "debug",
                "num_players": 2,
                "num_rounds": 1,
                "num_games": 1
            }
        },
        "game_constants": {
            "salop": {"base_market_size": 300, "marginal_cost": 8.0, "fixed_cost": 100.0, "transport_cost": 0.5},
            "spulber": {"base_market_size": 1000, "marginal_cost": 10.0, "market_value": 100.0, "rival_cost_mean": 12.0, "rival_cost_std": 3.0},
            "green_porter": {"base_demand_intercept": 100, "marginal_cost": 10.0, "collusive_quantity": 22.5, "discount_rate": 0.05},
            "athey_bagwell": {"high_cost": 15.0, "low_cost": 5.0, "market_price": 50.0, "discount_factor": 0.95}
        },
        "api": {"gemini_api_key_env": "GEMINI_API_KEY", "rate_limit_delay": 1.0},
        "logging": {"level": "INFO"},
        "metrics": {"comprehensive_analysis": {"enabled": True, "behavioral_metrics": True}}
    }

# Core functions
def get_available_models() -> Dict[str, ModelConfig]:
    """Get available models"""
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
    """Get game constants"""
    return load_config_file().get('game_constants', {})

def get_experiment_presets() -> Dict[str, ExperimentConfig]:
    """Get experiment presets"""
    config = load_config_file()
    presets = {}
    for key, data in config.get('experiment_presets', {}).items():
        presets[key] = ExperimentConfig.from_dict(data)
    return presets

def get_game_configs(category: str) -> Dict[str, List[GameConfig]]:
    """Get game configurations by category"""
    config = load_config_file()
    configs_data = config.get('game_configs', {}).get(category, {})
    result = {}
    for game_name, configs in configs_data.items():
        result[game_name] = [GameConfig.from_dict(cfg) for cfg in configs]
    return result

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
        models = get_available_models()
        presets = get_experiment_presets()
        constants = get_game_constants()
        
        # Check basic requirements
        if not models:
            return False
        if not constants:
            return False
            
        # Test GameConstants initialization
        test_config = GameConfig()
        test_constants = GameConstants(test_config)
        
        # Verify critical parameters
        test_constants.SPULBER_MARKET_VALUE
        test_constants.GP_COLLUSIVE_QUANTITY
        test_constants.AB_DISCOUNT_FACTOR
        
        return True
    except Exception:
        return False

# Helper classes
class ExperimentPresets:
    """Pre-configured experiment setups"""
    
    @staticmethod
    def get_preset(preset_key: str) -> ExperimentConfig:
        presets = get_experiment_presets()
        if preset_key not in presets:
            raise ValueError(f"Unknown preset: {preset_key}")
        return presets[preset_key]
    
    @staticmethod
    def debug_test() -> ExperimentConfig:
        return ExperimentPresets.get_preset('debug_test')

class GameConfigs:
    """Pre-configured game setups"""
    
    @staticmethod
    def quick_test_games() -> Dict[str, List[GameConfig]]:
        return get_game_configs('quick_test_games')

# Auto-initialize
try:
    if not Path("config.json").exists():
        with open("config.json", 'w') as f:
            json.dump(_get_default_config(), f, indent=2)
    load_config_file()
except Exception:
    pass