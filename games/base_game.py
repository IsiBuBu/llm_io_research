# games/base_game.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from config import GameConfig

class EconomicGame(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        pass
    
    @abstractmethod 
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig, 
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         round_num: int) -> Dict:
        pass
    
    def initialize_game_state(self, config: GameConfig) -> Dict:
        return {
            'current_round': 1,
            'total_rounds': config.number_of_rounds,
            'number_of_players': config.number_of_players
        }
    
    def validate_action(self, action: Dict[str, Any], player_id: str) -> bool:
        return True
    
    def calculate_game_metrics(self, actions_history: List[Dict[str, Any]], 
                              payoffs_history: List[Dict[str, float]], 
                              config: GameConfig) -> Dict[str, float]:
        return {}

class StaticGame(EconomicGame):
    def __init__(self, name: str):
        super().__init__(name)
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         round_num: int) -> Dict:
        return game_state

class DynamicGame(EconomicGame):
    def __init__(self, name: str, default_rounds: int = 10):
        super().__init__(name)
        self.default_rounds = default_rounds
    
    def calculate_npv(self, payoffs_history: List[Dict[str, float]], 
                     player_id: str, discount_factor: float) -> float:
        npv = 0
        for round_num, payoffs in enumerate(payoffs_history):
            if player_id in payoffs:
                npv += payoffs[player_id] * (discount_factor ** round_num)
        return npv