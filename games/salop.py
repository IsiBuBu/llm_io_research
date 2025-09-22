# games/salop.py

from typing import Dict, Any, Optional, List
from config.config import GameConfig, get_prompt_variables
from games.base_game import StaticGame, PriceParsingMixin

class SalopGame(StaticGame, PriceParsingMixin):
    """
    Implements the Salop (1979) spatial competition game.

    This class defines the logic for a static game where firms compete on price
    in a circular market. It uses the Market Share and Quantity Calculation
    Algorithm as specified in the project documentation to determine payoffs.
    """

    def __init__(self):
        super().__init__("salop")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes the state for a new Salop game simulation."""
        return {
            'game_type': 'static',
            'current_round': 1,
            'simulation_id': simulation_id,
            'constants': game_config.constants
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a player using the game's template and configuration."""
        variables = get_prompt_variables(game_config, player_id=player_id, **game_state)
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        """Parses the LLM's price decision using the inherited mixin."""
        return self.parse_price_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculates player payoffs using the Market Share Algorithm from t.txt.
        """
        constants = game_config.constants
        marginal_cost = constants.get('marginal_cost', 8)
        fixed_cost = constants.get('fixed_cost', 100)
        transport_cost = constants.get('transport_cost', 1.5)
        market_size = constants.get('market_size', 1000)
        v = constants.get('reservation_price', 30)
        num_firms = len(actions)

        prices = {pid: action.get('price', v) for pid, action in actions.items() if isinstance(action, dict)}
        
        player_ids = sorted(prices.keys())
        payoffs = {}
        quantities = {}
        
        for i, player_id in enumerate(player_ids):
            p_i = prices[player_id]
            p_left = prices[player_ids[(i - 1 + num_firms) % num_firms]]
            p_right = prices[player_ids[(i + 1) % num_firms]]

            x_left = (p_left - p_i) / (2 * transport_cost) + (1 / (2 * num_firms))
            x_right = (p_right - p_i) / (2 * transport_cost) + (1 / (2 * num_firms))

            x_max = (v - p_i) / transport_cost

            reach_left = max(0, min(x_left, x_max))
            reach_right = max(0, min(x_right, x_max))
            
            # --- FIX: Constrain total reach to the market's physical limit (circumference 1.0) ---
            total_reach = min(1.0, reach_left + reach_right)
            
            quantity_sold = total_reach * market_size
            quantities[player_id] = quantity_sold
            
            profit = (p_i - marginal_cost) * quantity_sold - fixed_cost
            payoffs[player_id] = profit
            
        # Store quantities for logging
        if game_state is not None:
            game_state['player_quantities'] = quantities
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers all relevant data from the Salop game for detailed logging."""
        return {
            "constants": game_config.constants,
            "player_quantities": game_state.get('player_quantities', {})
        }