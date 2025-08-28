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
    
    # Override constants for specific configurations
    market_size_override: Optional[int] = None
    demand_intercept_override: Optional[float] = None
    market_value_override: Optional[int] = None

@dataclass
class ExperimentConfig:
    """Configuration for experimental setup - always 1 challenger vs (n-1) defenders"""
    challenger_model: ModelConfig
    defender_model: ModelConfig
    
    def get_num_defenders(self, total_players: int) -> int:
        """Number of defenders = total_players - 1 (challenger takes 1 slot)"""
        return max(1, total_players - 1)

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
    """Economic constants for all games - configuration-dependent scaling"""
    
    # Base constants (per player or fixed)
    _SALOP_BASE_MARKET_SIZE = 300  # Per player
    _SALOP_MARGINAL_COST = 8       # Fixed per unit
    _SALOP_FIXED_COST = 100        # Fixed per firm
    _SALOP_TRANSPORT_COST = 0.5    # Fixed per unit distance
    
    _SPULBER_MARGINAL_COST = 8     # Fixed per unit
    _SPULBER_MARKET_VALUE = 1000   # Fixed total contract value (doesn't scale)
    _SPULBER_RIVAL_COST_MEAN = 10  # Fixed
    _SPULBER_RIVAL_COST_STD = 2    # Fixed
    
    _GP_BASE_DEMAND_INTERCEPT = 35 # Per player (scales total market)
    _GP_MARGINAL_COST = 20         # Fixed per unit
    _GP_DEMAND_SHOCK_STD = 5       # Fixed
    
    _AB_BASE_MARKET_SIZE = 350     # Per player
    _AB_HIGH_COST = 25             # Fixed per unit
    _AB_LOW_COST = 15              # Fixed per unit
    _AB_MARKET_PRICE = 50          # Fixed per unit
    _AB_COST_PERSISTENCE = 0.7     # Fixed probability
    
    def __init__(self, config: GameConfig):
        """Initialize constants based on game configuration"""
        self.config = config
    
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
    def SALOP_MARKET_SIZE(self) -> int:
        """Market size scales with number of players unless overridden"""
        if self.config.market_size_override is not None:
            return self.config.market_size_override
        return self._SALOP_BASE_MARKET_SIZE * self.config.number_of_players
    
    @property
    def SPULBER_MARGINAL_COST(self) -> float:
        return self._SPULBER_MARGINAL_COST
    
    @property
    def SPULBER_RIVAL_COST_MEAN(self) -> float:
        return self._SPULBER_RIVAL_COST_MEAN
    
    @property
    def SPULBER_RIVAL_COST_STD(self) -> float:
        return self._SPULBER_RIVAL_COST_STD
    
    @property
    def SPULBER_MARKET_VALUE(self) -> int:
        """Market value stays fixed (same contract, more bidders) unless overridden"""
        if self.config.market_value_override is not None:
            return self.config.market_value_override
        return self._SPULBER_MARKET_VALUE
    
    @property
    def GP_MARGINAL_COST(self) -> float:
        return self._GP_MARGINAL_COST
    
    @property
    def GP_DEMAND_SHOCK_STD(self) -> float:
        return self._GP_DEMAND_SHOCK_STD
    
    @property
    def GP_DEMAND_INTERCEPT(self) -> float:
        """Demand intercept scales with players (bigger total market) unless overridden"""
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
        """Market size scales with number of players unless overridden"""
        if self.config.market_size_override is not None:
            return self.config.market_size_override
        return self._AB_BASE_MARKET_SIZE * self.config.number_of_players

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
    ),
    'gemini-2.0-flash-lite': ModelConfig(
        model_name='gemini-2.0-flash-lite',
        thinking_available=False,
        thinking_enabled=False,
        display_name='gemini-2.0-flash-lite'
    )
}

# Pre-configured experiment setups
class ExperimentPresets:
    """Pre-configured experimental setups for different research questions"""
    
    @staticmethod
    def all_models_vs_baseline() -> List[ExperimentConfig]:
        """All available models as challengers vs consistent defender baseline"""
        challenger_keys = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.5-flash-thinking', 
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash',
            'gemini-2.0-flash-thinking',
            'gemini-2.0-flash-lite'
        ]
        baseline_defender = AVAILABLE_MODELS['gemini-2.5-flash']
        
        return [
            ExperimentConfig(
                challenger_model=AVAILABLE_MODELS[key],
                defender_model=baseline_defender
            ) for key in challenger_keys
        ]
    
    @staticmethod
    def thinking_comparison() -> List[ExperimentConfig]:
        """Compare models with and without thinking vs baseline"""
        configs = []
        baseline_defender = AVAILABLE_MODELS['gemini-2.5-flash']
        
        thinking_pairs = [
            ('gemini-2.5-flash', 'gemini-2.5-flash-thinking'),
            ('gemini-2.0-flash', 'gemini-2.0-flash-thinking')
        ]
        
        for no_thinking, with_thinking in thinking_pairs:
            configs.extend([
                ExperimentConfig(
                    challenger_model=AVAILABLE_MODELS[no_thinking],
                    defender_model=baseline_defender
                ),
                ExperimentConfig(
                    challenger_model=AVAILABLE_MODELS[with_thinking], 
                    defender_model=baseline_defender
                )
            ])
        
        return configs
    
    @staticmethod
    def single_challenger(challenger_key: str, defender_key: str = 'gemini-2.5-flash') -> ExperimentConfig:
        """Single challenger vs defender configuration"""
        return ExperimentConfig(
            challenger_model=AVAILABLE_MODELS[challenger_key],
            defender_model=AVAILABLE_MODELS[defender_key]
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