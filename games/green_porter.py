# games/green_porter.py

import numpy as np
from typing import Dict, Any, Optional, List

from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, QuantityParsingMixin

class GreenPorterGame(DynamicGame, QuantityParsingMixin):
    """
    Implements the Green & Porter (1984) dynamic oligopoly game.

    This class manages a multi-round Cournot competition with demand uncertainty.
    It uses a State Transition Algorithm triggered by a price threshold to switch
    between "Collusive" and "Reversionary" (punishment) phases, as specified in t.txt.
    """

    def __init__(self):
        super().__init__("green_porter")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes game state with pre-generated demand shocks for the simulation."""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        demand_shock_std = constants.get('demand_shock_std', 5)
        demand_shock_mean = constants.get('demand_shock_mean', 0)

        # Pre-generate demand shocks for reproducibility
        np.random.seed(simulation_id)
        demand_shocks = np.random.normal(demand_shock_mean, demand_shock_std, time_horizon).tolist()
        
        # Initial state as per t.txt specification
        return {
            'current_period': 1,
            'market_state': 'Collusive',
            'punishment_timer': 0,
            'demand_shocks': demand_shocks,
            'price_history': [],
            'state_history': [],
            'quantity_history': {player_id: [] for player_id in ['challenger'] + [f'defender_{i}' for i in range(1, constants.get('number_of_players', 3))]},
            'profit_history': {player_id: [] for player_id in ['challenger'] + [f'defender_{i}' for i in range(1, constants.get('number_of_players', 3))]}
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a player with current market conditions."""
        variables = get_prompt_variables(
            game_config,
            player_id=player_id,
            current_round=game_state.get('current_period', 1),
            current_market_state=game_state.get('market_state', 'Collusive'),
            price_history=game_state.get('price_history', [])
        )
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parses the LLM's quantity decision using the inherited mixin."""
        return self.parse_quantity_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates player payoffs based on total quantity and the current demand shock."""
        constants = game_config.constants
        demand_intercept = constants.get('base_demand', 120)
        marginal_cost = constants.get('marginal_cost', 20)
        
        current_period = game_state['current_period']
        demand_shock = game_state['demand_shocks'][current_period - 1]

        quantities = {pid: action.get('quantity', constants.get('cournot_quantity', 25)) for pid, action in actions.items()}
        total_quantity = sum(quantities.values())

        # Market Price = base_demand - (Total Industry Quantity) + Demand Shock
        market_price = max(0, demand_intercept - total_quantity + demand_shock)
        
        payoffs = {}
        for player_id, quantity in quantities.items():
            # Profit = (Market Price - marginal_cost) * quantity
            profit = (market_price - marginal_cost) * quantity
            payoffs[player_id] = profit
        
        # Store for state update and logging
        game_state['current_market_price'] = market_price
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig) -> Dict:
        """Updates the game state using the State Transition Algorithm from t.txt."""
        constants = game_config.constants
        trigger_price = constants.get('trigger_price', 55)
        punishment_periods = constants.get('punishment_duration', 3)
        market_price = game_state.get('current_market_price', 0)

        # Update histories
        game_state['price_history'].append(market_price)
        game_state['state_history'].append(game_state['market_state'])
        for pid, action in actions.items():
            game_state['quantity_history'][pid].append(action.get('quantity', constants.get('cournot_quantity', 25)))
        
        # State Transition Algorithm
        if game_state['market_state'] == 'Collusive':
            if market_price < trigger_price:
                game_state['market_state'] = 'Reversionary'
                game_state['punishment_timer'] = punishment_periods
        else: # Reversionary state
            game_state['punishment_timer'] -= 1
            if game_state['punishment_timer'] <= 0:
                game_state['market_state'] = 'Collusive'

        game_state['current_period'] += 1
        return game_state
    
    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers round-specific outcomes for detailed logging."""
        round_data = super().get_game_data_for_logging(actions, payoffs, game_config, game_state)
        round_data.update({
            "period": game_state.get('current_period', 1) -1, # Log the period that just finished
            "market_state": game_state.get('state_history', [])[-1] if game_state.get('state_history') else "Collusive",
            "demand_shock": game_state.get('demand_shocks', [])[game_state.get('current_period', 1) - 2],
            "market_price": game_state.get('current_market_price', 0),
            "player_profits": payoffs,
        })
        return round_data