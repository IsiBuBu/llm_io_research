"""
Base game classes for LLM game theory experiments.
Updated to work with new config system and compact implementation.
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from config import GameConfig


class EconomicGame(ABC):
    """Base class for all economic games"""
    
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def basic_json_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """Basic JSON parsing from LLM response"""
        if not response:
            return None
            
        try:
            # Clean response
            response = response.strip()
            
            # Try direct JSON parse first
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Extract JSON from various formats
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```',     # JSON in any code blocks  
                r'(\{[^{}]*"[^"]*"[^{}]*:[^{}]*\})',  # Simple JSON objects
                r'(\{.*?\})'  # Any braces content
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        action = json.loads(match)
                        if isinstance(action, dict):
                            return action
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception:
            return None

    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt for player using config system"""
        pass
    
    @abstractmethod
    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into action dictionary"""
        pass
    
    @abstractmethod
    def get_default_action(self, player_id: str, game_state: Dict, 
                         game_config: GameConfig) -> Dict[str, Any]:
        """Get default action when parsing fails"""
        pass
    
    @abstractmethod
    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate payoffs for all players"""
        pass
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Update game state after round - default implementation"""
        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get structured data for metrics calculation and logging"""
        return {
            'game_name': game_config.game_name,
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'actions': actions,
            'payoffs': payoffs,
            'constants': game_config.constants,
            'game_state': game_state
        }


class StaticGame(EconomicGame):
    """
    Base class for static (single-round) games like Salop and Spulber.
    No state updates needed between rounds.
    """
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Static games don't change state between rounds"""
        return game_state


class DynamicGame(EconomicGame):
    """
    Base class for dynamic (multi-round) games like Green-Porter and Athey-Bagwell.
    State evolves based on actions and outcomes.
    """
    
    @abstractmethod
    def initialize_game_state(self, game_config: GameConfig, 
                            simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state for new simulation"""
        pass
    
    def get_checkpoint_data(self, game_history: List[Dict], game_config: GameConfig, 
                          checkpoint_round: int) -> Dict[str, Any]:
        """Get data for checkpoint analysis (e.g., round 15 vs 50)"""
        # Filter history up to checkpoint
        checkpoint_history = [h for h in game_history if h.get('round', 1) <= checkpoint_round]
        
        return {
            'checkpoint_round': checkpoint_round,
            'total_rounds': len(checkpoint_history),
            'game_history': checkpoint_history,
            'game_config': game_config
        }


# Utility functions for common operations

def extract_numeric_value(response: str, key: str, default: float = 0.0) -> float:
    """Extract numeric value from response for a given key"""
    patterns = [
        rf'"{key}":\s*([0-9]*\.?[0-9]+)',
        rf'{key}["\']?\s*:\s*([0-9]*\.?[0-9]+)',
        rf'{key}\s*=\s*([0-9]*\.?[0-9]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return default


def validate_action_bounds(action: Dict[str, Any], bounds: Dict[str, tuple]) -> Dict[str, Any]:
    """Validate and clip action values to specified bounds"""
    validated = action.copy()
    
    for key, (min_val, max_val) in bounds.items():
        if key in validated:
            value = validated[key]
            if isinstance(value, (int, float)):
                validated[key] = max(min_val, min(value, max_val))
    
    return validated


def create_player_mapping(num_players: int, challenger_id: str = 'challenger') -> Dict[str, str]:
    """Create mapping between generic player IDs and specific identifiers"""
    player_ids = [challenger_id] + [f'defender_{i}' for i in range(1, num_players)]
    generic_ids = [chr(65 + i) for i in range(num_players)]  # A, B, C, ...
    
    return dict(zip(generic_ids, player_ids))