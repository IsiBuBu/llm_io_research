"""
Random Player Baselines - Implements exact algorithms from t.txt
Non-strategic baselines that choose valid actions uniformly at random
"""

import random
import logging
from typing import Dict, Any, Optional
from config import GameConfig


class RandomPlayer:
    """
    Base random player that implements game-specific random action selection.
    Serves as non-strategic baseline for LLM comparison.
    """
    
    def __init__(self, player_id: str = "random", seed: Optional[int] = None):
        self.player_id = player_id
        self.logger = logging.getLogger(f"{__name__}.RandomPlayer")
        
        # Set seed for reproducible randomness if provided
        if seed is not None:
            random.seed(seed)
    
    def get_action(self, game_name: str, game_state: Dict, game_config: GameConfig, 
                   **player_specific_info) -> Dict[str, Any]:
        """
        Get random action for specified game using algorithms from t.txt
        
        Args:
            game_name: 'salop', 'spulber', 'green_porter', 'athey_bagwell'
            game_state: Current game state
            game_config: Game configuration with constants
            **player_specific_info: Player-specific information (e.g., private costs)
        """
        
        if game_name == 'salop':
            return self._salop_random_action(game_config)
        elif game_name == 'spulber':
            return self._spulber_random_action(game_config, player_specific_info)
        elif game_name == 'green_porter':
            return self._green_porter_random_action(game_config)
        elif game_name == 'athey_bagwell':
            return self._athey_bagwell_random_action(game_config)
        else:
            raise ValueError(f"Unknown game: {game_name}")
    
    def _salop_random_action(self, game_config: GameConfig) -> Dict[str, Any]:
        """
        Salop Random Price Player - Algorithm from t.txt:
        1. Get marginal_cost and v from config
        2. Select uniform random price in [marginal_cost, v]  
        3. Return {"price": <randomly_chosen_price>}
        """
        constants = game_config.constants
        marginal_cost = constants.get('marginal_cost', 8)
        v = constants.get('v', 30)
        
        # Uniform random price in [marginal_cost, v]
        random_price = random.uniform(marginal_cost, v)
        
        return {
            'price': random_price,
            'player_type': 'random',
            'reasoning': f'Random price selection in [{marginal_cost}, {v}]'
        }
    
    def _spulber_random_action(self, game_config: GameConfig, 
                             player_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Spulber Random Bidder - Algorithm from t.txt:
        1. Get private your_cost and demand_intercept
        2. Select uniform random price in [your_cost, demand_intercept]
        3. Return {"price": <randomly_chosen_bid>}
        """
        constants = game_config.constants
        demand_intercept = constants.get('demand_intercept', 100)
        
        # Get player's private cost
        private_values = constants.get('private_values', {})
        if self.player_id == 'challenger':
            your_cost = private_values.get('challenger_cost', 8)
        else:
            your_cost = private_values.get('defender_cost', 10)
        
        # Override with player_info if provided
        your_cost = player_info.get('your_cost', your_cost)
        
        # Uniform random bid in [your_cost, demand_intercept]
        random_price = random.uniform(your_cost, demand_intercept)
        
        return {
            'price': random_price,
            'player_type': 'random',
            'reasoning': f'Random bid selection in [{your_cost}, {demand_intercept}]'
        }
    
    def _green_porter_random_action(self, game_config: GameConfig) -> Dict[str, Any]:
        """
        Green & Porter Random Quantity Player - Algorithm from t.txt:
        1. Get collusive_quantity and cournot_quantity from config
        2. Choose with 50% probability from {collusive_quantity, cournot_quantity}
        3. Return {"quantity": <randomly_chosen_quantity>}
        """
        constants = game_config.constants
        collusive_quantity = constants.get('collusive_quantity', 17)
        cournot_quantity = constants.get('cournot_quantity', 25)
        
        # 50% probability choice between collusive and cournot
        if random.random() < 0.5:
            chosen_quantity = collusive_quantity
            strategy = 'collusive'
        else:
            chosen_quantity = cournot_quantity
            strategy = 'cournot'
        
        return {
            'quantity': chosen_quantity,
            'player_type': 'random',
            'reasoning': f'Random {strategy} quantity selection'
        }
    
    def _athey_bagwell_random_action(self, game_config: GameConfig) -> Dict[str, Any]:
        """
        Athey & Bagwell Random Reporter - Algorithm from t.txt:
        1. Get valid reports {"high", "low"}
        2. Choose with 50% probability from {"high", "low"}
        3. Return {"report": <randomly_chosen_report>}
        """
        # 50% probability choice between "high" and "low"
        if random.random() < 0.5:
            chosen_report = 'high'
        else:
            chosen_report = 'low'
        
        return {
            'report': chosen_report,
            'player_type': 'random',
            'reasoning': f'Random report selection: {chosen_report}'
        }


class SalopRandomPlayer(RandomPlayer):
    """Salop-specific random player"""
    
    def get_action(self, game_state: Dict, game_config: GameConfig) -> Dict[str, Any]:
        return self._salop_random_action(game_config)


class SpulberRandomPlayer(RandomPlayer):
    """Spulber-specific random player"""
    
    def __init__(self, player_id: str = "random", your_cost: float = 10, seed: Optional[int] = None):
        super().__init__(player_id, seed)
        self.your_cost = your_cost
    
    def get_action(self, game_state: Dict, game_config: GameConfig) -> Dict[str, Any]:
        return self._spulber_random_action(game_config, {'your_cost': self.your_cost})


class GreenPorterRandomPlayer(RandomPlayer):
    """Green & Porter specific random player"""
    
    def get_action(self, game_state: Dict, game_config: GameConfig) -> Dict[str, Any]:
        return self._green_porter_random_action(game_config)


class AtheyBagwellRandomPlayer(RandomPlayer):
    """Athey & Bagwell specific random player"""
    
    def get_action(self, game_state: Dict, game_config: GameConfig) -> Dict[str, Any]:
        return self._athey_bagwell_random_action(game_config)


# Factory function for easy instantiation
def create_random_player(game_name: str, player_id: str = "random", 
                        seed: Optional[int] = None, **kwargs) -> RandomPlayer:
    """
    Factory function to create game-specific random players
    
    Args:
        game_name: 'salop', 'spulber', 'green_porter', 'athey_bagwell'
        player_id: Identifier for this player
        seed: Random seed for reproducible behavior
        **kwargs: Game-specific parameters (e.g., your_cost for Spulber)
    """
    
    if game_name == 'salop':
        return SalopRandomPlayer(player_id, seed)
    elif game_name == 'spulber':
        your_cost = kwargs.get('your_cost', 10)
        return SpulberRandomPlayer(player_id, your_cost, seed)
    elif game_name == 'green_porter':
        return GreenPorterRandomPlayer(player_id, seed)
    elif game_name == 'athey_bagwell':
        return AtheyBagwellRandomPlayer(player_id, seed)
    else:
        raise ValueError(f"Unknown game: {game_name}")


# Utility functions for integration with experiment framework
def get_random_action_for_game(game_name: str, player_id: str, game_state: Dict, 
                              game_config: GameConfig, seed: Optional[int] = None,
                              **player_specific_info) -> Dict[str, Any]:
    """
    Convenience function to get random action for any game
    
    Usage:
        action = get_random_action_for_game('salop', 'random_player', {}, game_config)
        action = get_random_action_for_game('spulber', 'random_defender', {}, game_config, your_cost=12)
    """
    player = RandomPlayer(player_id, seed)
    return player.get_action(game_name, game_state, game_config, **player_specific_info)


def validate_random_action(action: Dict[str, Any], game_name: str, 
                          game_config: GameConfig) -> bool:
    """
    Validate that random action is within expected bounds
    
    Returns:
        True if action is valid, False otherwise
    """
    constants = game_config.constants
    
    try:
        if game_name == 'salop':
            price = action.get('price', 0)
            marginal_cost = constants.get('marginal_cost', 8)
            v = constants.get('v', 30)
            return marginal_cost <= price <= v
        
        elif game_name == 'spulber':
            price = action.get('price', 0)
            # Would need player-specific cost info to validate properly
            demand_intercept = constants.get('demand_intercept', 100)
            return 0 <= price <= demand_intercept
        
        elif game_name == 'green_porter':
            quantity = action.get('quantity', 0)
            collusive_quantity = constants.get('collusive_quantity', 17)
            cournot_quantity = constants.get('cournot_quantity', 25)
            return quantity in [collusive_quantity, cournot_quantity]
        
        elif game_name == 'athey_bagwell':
            report = action.get('report', '')
            return report in ['high', 'low']
        
        else:
            return False
            
    except (KeyError, TypeError, ValueError):
        return False