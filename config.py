# config.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for a specific Gemini model"""
    model_name: str
    thinking_available: bool
    thinking_enabled: bool = False
    display_name: str = ""
    
    def __post_init__(self):
        if not self.display_name:
            thinking_suffix = "_thinking" if self.thinking_enabled and self.thinking_available else ""
            self.display_name = f"{self.model_name}{thinking_suffix}"

@dataclass
class GameConfig:
    """Configuration for game experiments"""
    number_of_players: int
    number_of_rounds: int = 1
    discount_factor: float = 0.95
    max_retries: int = 3
    num_games: int = 20  # Number of games to run per configuration

@dataclass
class ExperimentConfig:
    """Configuration for experimental setup"""
    challenger_models: List[ModelConfig]
    defender_model: ModelConfig
    num_defenders: int = 2  # Will be adjusted based on number_of_players
    
    def get_num_defenders(self, total_players: int) -> int:
        """Calculate number of defenders based on total players"""
        return max(1, total_players - 1)  # Always leave 1 spot for challenger

@dataclass
class PlayerResult:
    """Results for a single player in a game"""
    player_id: str
    profit: float
    actions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    win: bool = False

@dataclass
class GameResult:
    """Results for a complete game"""
    game_name: str
    config: GameConfig
    experiment_config: ExperimentConfig
    players: List[PlayerResult]
    total_industry_profit: float
    market_price: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)

class GameConstants:
    """Economic constants for all games"""
    
    # Salop (1979) Constants
    SALOP_MARGINAL_COST = 8
    SALOP_FIXED_COST = 100
    SALOP_MARKET_SIZE = 1000
    SALOP_TRANSPORT_COST = 0.5
    
    # Spulber (1995) Constants
    SPULBER_MARGINAL_COST = 8
    SPULBER_MARKET_VALUE = 1000
    SPULBER_RIVAL_COST_MEAN = 10
    SPULBER_RIVAL_COST_STD = 2
    
    # Green & Porter (1984) Constants
    GP_MARGINAL_COST = 20
    GP_DEMAND_INTERCEPT = 100
    GP_DEMAND_SHOCK_STD = 5
    
    # Athey & Bagwell (2008) Constants
    AB_HIGH_COST = 25
    AB_LOW_COST = 15
    AB_MARKET_PRICE = 50
    AB_MARKET_SIZE = 1000
    AB_COST_PERSISTENCE = 0.7

# Available Gemini models with thinking capabilities
AVAILABLE_MODELS = {
    'gemini-2.5-pro': ModelConfig(
        model_name='gemini-2.5-pro',
        thinking_available=True,
        thinking_enabled=True,  # Can't turn off thinking
        display_name='gemini-2.5-pro'
    ),
    'gemini-2.5-flash': ModelConfig(
        model_name='gemini-2.5-flash',
        thinking_available=True,
        thinking_enabled=False,
        display_name='gemini-2.5-flash'
    ),
    'gemini-2.5-flash-thinking': ModelConfig(
        model_name='gemini-2.5-flash',
        thinking_available=True,
        thinking_enabled=True,
        display_name='gemini-2.5-flash-thinking'
    ),
    'gemini-2.5-flash-lite': ModelConfig(
        model_name='gemini-2.5-flash-lite',
        thinking_available=False,
        thinking_enabled=False,
        display_name='gemini-2.5-flash-lite'
    ),
    'gemini-2.0-flash': ModelConfig(
        model_name='gemini-2.0-flash',
        thinking_available=True,
        thinking_enabled=False,
        display_name='gemini-2.0-flash'
    ),
    'gemini-2.0-flash-thinking': ModelConfig(
        model_name='gemini-2.0-flash',
        thinking_available=True,
        thinking_enabled=True,
        display_name='gemini-2.0-flash-thinking'
    )
}

# Pre-configured experiment setups
class ExperimentPresets:
    """Pre-configured experimental setups for different research questions"""
    
    @staticmethod
    def thinking_vs_no_thinking() -> ExperimentConfig:
        """Compare models with and without thinking enabled"""
        return ExperimentConfig(
            challenger_models=[
                AVAILABLE_MODELS['gemini-2.5-flash'],
                AVAILABLE_MODELS['gemini-2.5-flash-thinking'],
                AVAILABLE_MODELS['gemini-2.0-flash'],
                AVAILABLE_MODELS['gemini-2.0-flash-thinking']
            ],
            defender_model=AVAILABLE_MODELS['gemini-2.5-flash']  # Consistent baseline
        )
    
    @staticmethod
    def model_comparison() -> ExperimentConfig:
        """Compare different model architectures"""
        return ExperimentConfig(
            challenger_models=[
                AVAILABLE_MODELS['gemini-2.5-pro'],
                AVAILABLE_MODELS['gemini-2.5-flash'],
                AVAILABLE_MODELS['gemini-2.5-flash-lite'],
                AVAILABLE_MODELS['gemini-2.0-flash']
            ],
            defender_model=AVAILABLE_MODELS['gemini-2.5-flash']
        )
    
    @staticmethod
    def custom_setup(challenger_model_keys: List[str], 
                    defender_model_key: str) -> ExperimentConfig:
        """Create custom experimental setup"""
        challenger_models = [AVAILABLE_MODELS[key] for key in challenger_model_keys]
        defender_model = AVAILABLE_MODELS[defender_model_key]
        
        return ExperimentConfig(
            challenger_models=challenger_models,
            defender_model=defender_model
        )

# Game-specific configurations
class GameConfigs:
    """Pre-configured game setups for thesis experiments"""
    
    @staticmethod
    def static_games() -> Dict[str, List[GameConfig]]:
        """Static game configurations testing number of players"""
        return {
            'salop': [
                GameConfig(number_of_players=3, number_of_rounds=1, num_games=20),
                GameConfig(number_of_players=5, number_of_rounds=1, num_games=20)
            ],
            'spulber': [
                GameConfig(number_of_players=3, number_of_rounds=1, num_games=20),
                GameConfig(number_of_players=5, number_of_rounds=1, num_games=20)
            ]
        }
    
    @staticmethod
    def dynamic_games() -> Dict[str, List[GameConfig]]:
        """Dynamic game configurations testing time horizon"""
        return {
            'green_porter': [
                GameConfig(number_of_players=3, number_of_rounds=10, num_games=15),
                GameConfig(number_of_players=3, number_of_rounds=30, num_games=10)
            ],
            'athey_bagwell': [
                GameConfig(number_of_players=3, number_of_rounds=15, num_games=15),
                GameConfig(number_of_players=3, number_of_rounds=40, num_games=10)
            ]
        }
    
    @staticmethod
    def quick_test_games() -> Dict[str, List[GameConfig]]:
        """Minimal configurations for testing setup"""
        return {
            'salop': [GameConfig(number_of_players=3, number_of_rounds=1, num_games=2)],
            'green_porter': [GameConfig(number_of_players=3, number_of_rounds=5, num_games=2)]
        }
    
    @staticmethod
    def custom_dynamic_config(game_name: str, num_players: int, 
                            num_rounds: int, num_games: int) -> GameConfig:
        """Create custom dynamic game configuration"""
        return GameConfig(
            number_of_players=num_players,
            number_of_rounds=num_rounds,
            num_games=num_games
        )

def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key"""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_key]

def create_experiment_config(challenger_keys: List[str], defender_key: str) -> ExperimentConfig:
    """Create experiment configuration from model keys"""
    return ExperimentPresets.custom_setup(challenger_keys, defender_key)