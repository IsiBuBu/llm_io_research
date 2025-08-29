import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for a specific Gemini model"""
    model_name: str
    thinking_available: bool
    thinking_enabled: bool = False
    display_name: str = ""
    api_version: str = "v1"
    max_retries: int = 3
    timeout: int = 30
    
    def __post_init__(self):
        if not self.display_name:
            thinking_suffix = "_thinking" if self.thinking_enabled and self.thinking_available else ""
            self.display_name = f"{self.model_name}{thinking_suffix}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        return cls(**data)

@dataclass
class GameConfig:
    """Configuration for game experiments"""
    number_of_players: int
    number_of_rounds: int = 1
    discount_factor: float = 0.95
    max_retries: int = 3
    num_games: int = 20
    
    # Override constants for specific configurations
    market_size_override: Optional[int] = None
    demand_intercept_override: Optional[float] = None
    market_value_override: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameConfig':
        """Create GameConfig from dictionary"""
        return cls(**data)

@dataclass
class ExperimentConfig:
    """Configuration for experimental setup - defender vs challengers"""
    defender_model_key: str
    challenger_model_keys: List[str]
    game_name: str
    num_players: int = 3
    num_rounds: int = 1
    num_games: int = 10
    include_thinking: bool = True
    
    def get_defender_model(self) -> 'ModelConfig':
        """Get defender model configuration"""
        return get_model_config(self.defender_model_key)
    
    def get_challenger_models(self) -> List['ModelConfig']:
        """Get challenger model configurations"""
        return [get_model_config(key) for key in self.challenger_model_keys]
    
    def get_num_defenders(self, total_players: int) -> int:
        """Number of defenders = total_players - 1 (challenger takes 1 slot)"""
        return max(1, total_players - 1)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create ExperimentConfig from dictionary"""
        return cls(**data)

@dataclass
class PlayerResult:
    """Results for a single player in a game"""
    player_id: str
    profit: float
    actions: List[Dict[str, Any]] = field(default_factory=list)
    win: bool = False

@dataclass
class GameResult:
    """Results for a complete game"""
    game_name: str
    config: GameConfig
    players: List[PlayerResult]
    total_industry_profit: float
    experiment_config: Optional[ExperimentConfig] = None
    market_price: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)

class GameConstants:
    """Economic constants for all games - loaded from config.json"""
    
    def __init__(self, config: GameConfig, constants_dict: Optional[Dict[str, Any]] = None):
        self.config = config
        
        # Load constants from config or use defaults
        if constants_dict is None:
            constants_dict = get_game_constants()
        
        # SALOP game constants
        self._SALOP_BASE_MARKET_SIZE = constants_dict.get('salop', {}).get('base_market_size', 300)
        self._SALOP_MARGINAL_COST = constants_dict.get('salop', {}).get('marginal_cost', 8)
        self._SALOP_FIXED_COST = constants_dict.get('salop', {}).get('fixed_cost', 100)
        self._SALOP_TRANSPORT_COST = constants_dict.get('salop', {}).get('transport_cost', 0.5)
        
        # Spulber game constants
        self._SPULBER_BASE_MARKET_SIZE = constants_dict.get('spulber', {}).get('base_market_size', 1000)
        self._SPULBER_MARGINAL_COST = constants_dict.get('spulber', {}).get('marginal_cost', 10)
        self._SPULBER_DEMAND_SLOPE = constants_dict.get('spulber', {}).get('demand_slope', 1.0)
        
        # Green Porter game constants
        self._GP_BASE_DEMAND_INTERCEPT = constants_dict.get('green_porter', {}).get('base_demand_intercept', 100)
        self._GP_MARGINAL_COST = constants_dict.get('green_porter', {}).get('marginal_cost', 10)
        self._GP_DEMAND_SHOCK_STD = constants_dict.get('green_porter', {}).get('demand_shock_std', 5)
        
        # Athey Bagwell game constants
        self._AB_HIGH_COST = constants_dict.get('athey_bagwell', {}).get('high_cost', 15)
        self._AB_LOW_COST = constants_dict.get('athey_bagwell', {}).get('low_cost', 5)
        self._AB_MARKET_PRICE = constants_dict.get('athey_bagwell', {}).get('market_price', 50)
        self._AB_COST_PERSISTENCE = constants_dict.get('athey_bagwell', {}).get('cost_persistence', 0.8)
        self._AB_BASE_MARKET_SIZE = constants_dict.get('athey_bagwell', {}).get('base_market_size', 1000)

    # Property methods remain the same but now use loaded values
    @property
    def SALOP_MARKET_SIZE(self) -> int:
        if self.config.market_size_override is not None:
            return self.config.market_size_override
        return self._SALOP_BASE_MARKET_SIZE * self.config.number_of_players
    
    @property
    def SALOP_MARGINAL_COST(self) -> float:
        return self._SALOP_MARGINAL_COST
    
    @property
    def SALOP_FIXED_COST(self) -> float:
        return self._SALOP_FIXED_COST
    
    @property 
    def SALOP_TRANSPORT_COST(self) -> float:
        return self._SALOP_TRANSPORT_COST
    
    @property
    def SPULBER_MARKET_SIZE(self) -> int:
        if self.config.market_size_override is not None:
            return self.config.market_size_override
        return self._SPULBER_BASE_MARKET_SIZE * self.config.number_of_players
    
    @property
    def SPULBER_MARGINAL_COST(self) -> float:
        return self._SPULBER_MARGINAL_COST
    
    @property
    def SPULBER_DEMAND_SLOPE(self) -> float:
        return self._SPULBER_DEMAND_SLOPE
    
    @property
    def GP_MARGINAL_COST(self) -> float:
        return self._GP_MARGINAL_COST
    
    @property
    def GP_DEMAND_SHOCK_STD(self) -> float:
        return self._GP_DEMAND_SHOCK_STD
    
    @property
    def GP_DEMAND_INTERCEPT(self) -> float:
        if self.config.demand_intercept_override is not None:
            return self.config.demand_intercept_override
        return self._GP_BASE_DEMAND_INTERCEPT * self.config.number_of_players
    
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
    def AB_MARKET_SIZE(self) -> int:
        if self.config.market_size_override is not None:
            return self.config.market_size_override
        return self._AB_BASE_MARKET_SIZE * self.config.number_of_players

# Global configuration cache
_config_cache = {}
_config_file_path = None

def load_config_file(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file with caching"""
    global _config_cache, _config_file_path
    
    config_path = Path(config_path)
    
    # Check if we need to reload (file changed or not cached)
    if (config_path != _config_file_path or 
        not _config_cache or 
        not config_path.exists()):
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    _config_cache = json.load(f)
                _config_file_path = config_path
                logging.getLogger(__name__).info(f"Loaded configuration from {config_path}")
            else:
                logging.getLogger(__name__).warning(f"Config file {config_path} not found, using defaults")
                _config_cache = _get_default_config()
        except json.JSONDecodeError as e:
            logging.getLogger(__name__).error(f"Invalid JSON in {config_path}: {e}")
            _config_cache = _get_default_config()
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading config {config_path}: {e}")
            _config_cache = _get_default_config()
    
    return _config_cache

def _get_default_config() -> Dict[str, Any]:
    """Get default configuration if config.json doesn't exist"""
    return {
        "models": {
            "gemini-2.0-flash-lite": {
                "model_name": "gemini-2.0-flash-lite",
                "thinking_available": False,
                "thinking_enabled": False,
                "display_name": "Gemini 2.0 Flash Lite",
                "api_version": "v1",
                "max_retries": 3,
                "timeout": 30
            },
            "gemini-2.0-flash": {
                "model_name": "gemini-2.0-flash", 
                "thinking_available": True,
                "thinking_enabled": False,
                "display_name": "Gemini 2.0 Flash",
                "api_version": "v1",
                "max_retries": 3,
                "timeout": 30
            },
            "gemini-2.0-flash-thinking": {
                "model_name": "gemini-2.0-flash",
                "thinking_available": True, 
                "thinking_enabled": True,
                "display_name": "Gemini 2.0 Flash (Thinking)",
                "api_version": "v1",
                "max_retries": 3,
                "timeout": 30
            },
            "gemini-2.5-flash": {
                "model_name": "gemini-2.5-flash",
                "thinking_available": True,
                "thinking_enabled": False,
                "display_name": "Gemini 2.5 Flash",
                "api_version": "v1",
                "max_retries": 3,
                "timeout": 30
            },
            "gemini-2.5-flash-thinking": {
                "model_name": "gemini-2.5-flash",
                "thinking_available": True,
                "thinking_enabled": True, 
                "display_name": "Gemini 2.5 Flash (Thinking)",
                "api_version": "v1",
                "max_retries": 3,
                "timeout": 30
            },
            "gemini-2.5-flash-lite": {
                "model_name": "gemini-2.5-flash-lite",
                "thinking_available": True,
                "thinking_enabled": False,
                "display_name": "Gemini 2.5 Flash Lite", 
                "api_version": "v1",
                "max_retries": 3,
                "timeout": 30
            },
            "gemini-2.5-flash-lite-thinking": {
                "model_name": "gemini-2.5-flash-lite",
                "thinking_available": True,
                "thinking_enabled": True,
                "display_name": "Gemini 2.5 Flash Lite (Thinking)",
                "api_version": "v1", 
                "max_retries": 3,
                "timeout": 30
            },
            "gemini-2.5-pro": {
                "model_name": "gemini-2.5-pro",
                "thinking_available": True,
                "thinking_enabled": True,
                "display_name": "Gemini 2.5 Pro (Thinking Only)",
                "api_version": "v1",
                "max_retries": 3,
                "timeout": 30
            }
        },
        "game_constants": {
            "salop": {
                "base_market_size": 300,
                "marginal_cost": 8,
                "fixed_cost": 100,
                "transport_cost": 0.5
            },
            "spulber": {
                "base_market_size": 1000,
                "marginal_cost": 10,
                "demand_slope": 1.0
            },
            "green_porter": {
                "base_demand_intercept": 100,
                "marginal_cost": 10,
                "demand_shock_std": 5
            },
            "athey_bagwell": {
                "high_cost": 15,
                "low_cost": 5,
                "market_price": 50,
                "cost_persistence": 0.8,
                "base_market_size": 1000
            }
        },
        "experiment_presets": {
            "your_main_setup": {
                "defender_model_key": "gemini-2.0-flash-lite",
                "challenger_model_keys": [
                    "gemini-2.0-flash-lite",
                    "gemini-2.0-flash", 
                    "gemini-2.0-flash-thinking",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-thinking",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-flash-lite-thinking",
                    "gemini-2.5-pro"
                ],
                "game_name": "comprehensive_gemini_comparison",
                "num_games": 20,
                "include_thinking": True
            },
            "thinking_comparison": {
                "defender_model_key": "gemini-2.0-flash-lite",
                "challenger_model_keys": [
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-thinking",
                    "gemini-2.5-flash", 
                    "gemini-2.5-flash-thinking",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-flash-lite-thinking",
                    "gemini-2.5-pro"
                ],
                "game_name": "thinking_capabilities_study",
                "num_games": 15,
                "include_thinking": True
            },
            "debug_test": {
                "defender_model_key": "gemini-2.0-flash-lite",
                "challenger_model_keys": ["gemini-2.5-pro"],
                "game_name": "debug_test",
                "num_games": 3,
                "include_thinking": True
            }
        },
        "game_configs": {
            "static_games": {
                "salop": [
                    {"number_of_players": 3, "number_of_rounds": 1, "num_games": 20},
                    {"number_of_players": 5, "number_of_rounds": 1, "num_games": 20}
                ],
                "spulber": [
                    {"number_of_players": 3, "number_of_rounds": 1, "num_games": 20},
                    {"number_of_players": 5, "number_of_rounds": 1, "num_games": 20}
                ]
            },
            "dynamic_games": {
                "green_porter": [
                    {"number_of_players": 3, "number_of_rounds": 10, "num_games": 15},
                    {"number_of_players": 3, "number_of_rounds": 30, "num_games": 10}
                ],
                "athey_bagwell": [
                    {"number_of_players": 3, "number_of_rounds": 15, "num_games": 15}, 
                    {"number_of_players": 3, "number_of_rounds": 40, "num_games": 10}
                ]
            },
            "quick_test_games": {
                "salop": [{"number_of_players": 3, "number_of_rounds": 1, "num_games": 2}],
                "green_porter": [{"number_of_players": 3, "number_of_rounds": 5, "num_games": 2}]
            }
        },
        "logging": {
            "level": "INFO",
            "file": "llm_experiments.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        },
        "api": {
            "gemini_api_key_env": "GEMINI_API_KEY",
            "rate_limit_delay": 1,
            "default_timeout": 30
        }
    }

def get_available_models() -> Dict[str, ModelConfig]:
    """Get available models from config.json"""
    config = load_config_file()
    models_dict = {}
    
    for model_key, model_data in config.get('models', {}).items():
        try:
            models_dict[model_key] = ModelConfig.from_dict(model_data)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading model {model_key}: {e}")
    
    return models_dict

def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key"""
    available_models = get_available_models()
    if model_key not in available_models:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(available_models.keys())}")
    return available_models[model_key]

def get_game_constants() -> Dict[str, Any]:
    """Get game constants from config.json"""
    config = load_config_file()
    return config.get('game_constants', {})

def get_experiment_presets() -> Dict[str, ExperimentConfig]:
    """Get experiment presets from config.json"""
    config = load_config_file()
    presets_dict = {}
    
    for preset_key, preset_data in config.get('experiment_presets', {}).items():
        try:
            presets_dict[preset_key] = ExperimentConfig.from_dict(preset_data)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error loading preset {preset_key}: {e}")
    
    return presets_dict

def get_game_configs(category: str) -> Dict[str, List[GameConfig]]:
    """Get game configurations by category from config.json"""
    config = load_config_file()
    game_configs_data = config.get('game_configs', {}).get(category, {})
    
    result = {}
    for game_name, configs_list in game_configs_data.items():
        result[game_name] = [GameConfig.from_dict(cfg) for cfg in configs_list]
    
    return result

def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration from config.json"""
    config = load_config_file()
    return config.get('logging', {})

def get_api_config() -> Dict[str, Any]:
    """Get API configuration from config.json"""
    config = load_config_file()
    return config.get('api', {})

# Enhanced experiment presets with JSON loading
class ExperimentPresets:
    """Pre-configured experimental setups loaded from config.json"""
    
    @staticmethod
    def get_preset(preset_key: str) -> ExperimentConfig:
        """Get experiment preset by key"""
        presets = get_experiment_presets()
        if preset_key not in presets:
            raise ValueError(f"Unknown preset: {preset_key}. Available: {list(presets.keys())}")
        return presets[preset_key]
    
    @staticmethod
    def list_available_presets() -> List[str]:
        """List all available experiment presets"""
        return list(get_experiment_presets().keys())
    
    @staticmethod
    def your_main_setup() -> ExperimentConfig:
        """Your main experimental setup - Gemini 2.0 Flash Lite defender"""
        return ExperimentPresets.get_preset('your_main_setup')
    
    @staticmethod
    def thinking_comparison() -> ExperimentConfig:
        """Compare thinking vs non-thinking models"""
        return ExperimentPresets.get_preset('thinking_comparison')
    
    @staticmethod
    def debug_test() -> ExperimentConfig:
        """Minimal setup for debugging"""
        return ExperimentPresets.get_preset('debug_test')
    
    @staticmethod
    def custom_setup(defender_key: str, challenger_keys: List[str], 
                    game_name: str = "custom") -> ExperimentConfig:
        """Create custom experiment configuration"""
        return ExperimentConfig(
            defender_model_key=defender_key,
            challenger_model_keys=challenger_keys,
            game_name=game_name
        )

# Enhanced game configurations with JSON loading  
class GameConfigs:
    """Pre-configured game setups loaded from config.json"""
    
    @staticmethod
    def static_games() -> Dict[str, List[GameConfig]]:
        """Static game configurations testing number of players"""
        return get_game_configs('static_games')
    
    @staticmethod
    def dynamic_games() -> Dict[str, List[GameConfig]]:
        """Dynamic game configurations testing time horizon"""
        return get_game_configs('dynamic_games')
    
    @staticmethod
    def quick_test_games() -> Dict[str, List[GameConfig]]:
        """Minimal configurations for testing setup"""
        return get_game_configs('quick_test_games')
    
    @staticmethod
    def custom_dynamic_config(game_name: str, num_players: int, 
                            num_rounds: int, num_games: int) -> GameConfig:
        """Create custom dynamic game configuration"""
        return GameConfig(
            number_of_players=num_players,
            number_of_rounds=num_rounds,
            num_games=num_games
        )

def create_experiment_config(defender_key: str, challenger_keys: List[str], 
                           game_name: str = "custom") -> ExperimentConfig:
    """Create experiment configuration from model keys"""
    return ExperimentConfig(
        defender_model_key=defender_key,
        challenger_model_keys=challenger_keys,
        game_name=game_name
    )

def validate_config() -> bool:
    """Validate the loaded configuration"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test loading models
        models = get_available_models()
        logger.info(f"Loaded {len(models)} model configurations")
        
        # Test loading presets
        presets = get_experiment_presets()
        logger.info(f"Loaded {len(presets)} experiment presets")
        
        # Test loading game constants
        constants = get_game_constants()
        logger.info(f"Loaded game constants for {len(constants)} games")
        
        # Validate model references in presets
        all_model_keys = set(models.keys())
        for preset_name, preset in presets.items():
            if preset.defender_model_key not in all_model_keys:
                logger.error(f"Preset '{preset_name}' references unknown defender model: {preset.defender_model_key}")
                return False
            
            for challenger_key in preset.challenger_model_keys:
                if challenger_key not in all_model_keys:
                    logger.error(f"Preset '{preset_name}' references unknown challenger model: {challenger_key}")
                    return False
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def reload_config():
    """Force reload configuration from file"""
    global _config_cache, _config_file_path
    _config_cache = {}
    _config_file_path = None
    load_config_file()

def save_default_config(config_path: str = "config.json"):
    """Save default configuration to JSON file"""
    default_config = _get_default_config()
    
    try:
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        logging.getLogger(__name__).info(f"Default configuration saved to {config_path}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving config to {config_path}: {e}")

# Initialize configuration on import
def initialize_config():
    """Initialize configuration system"""
    logger = logging.getLogger(__name__)
    
    # Check if config.json exists, if not create it
    config_path = Path("config.json")
    if not config_path.exists():
        logger.info("config.json not found, creating default configuration")
        save_default_config()
    
    # Load and validate configuration
    load_config_file()
    if not validate_config():
        logger.warning("Configuration validation failed, some features may not work")

# Auto-initialize on import
initialize_config()