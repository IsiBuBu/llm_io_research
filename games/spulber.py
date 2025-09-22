# games/spulber.py

from typing import Dict, Any, Optional

from config.config import GameConfig, get_prompt_variables
from games.base_game import StaticGame, PriceParsingMixin

class SpulberGame(StaticGame, PriceParsingMixin):
    """
    Implements the Spulber (1995) Bertrand competition game with unknown costs.

    This class defines the logic for a static, winner-take-all price auction.
    It uses the Winner Determination Algorithm from t.txt to identify the lowest
    bidder, handle ties by splitting the market, and calculate profits accordingly.
    """

    def __init__(self):
        super().__init__("spulber")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes the state for a new Spulber game simulation."""
        return {
            'game_type': 'static',
            'current_round': 1,
            'simulation_id': simulation_id,
            'constants': game_config.constants,
            'player_private_costs': {} # Explicit field for private costs
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a player using the game's template and configuration."""
        private_costs = game_state.get('player_private_costs', {})
        
        variables = get_prompt_variables(
            game_config,
            player_id=player_id,
            your_cost=private_costs.get(player_id, game_config.constants.get('your_cost')),
            **game_state
        )
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        """Parses the LLM's price/bid decision using the inherited mixin."""
        parsed = self.parse_price_response(response, player_id, call_id)
        if parsed and 'bid' in parsed:
            parsed['price'] = parsed.pop('bid')
        return parsed

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates payoffs using the Winner Determination Algorithm from t.txt."""
        constants = game_config.constants
        demand_intercept = constants.get('demand_intercept', 100)
        
        private_costs = game_state.get('player_private_costs', {})
        prices = {pid: action.get('price', demand_intercept + 1) for pid, action in actions.items() if isinstance(action, dict)}

        if not prices:
            return {pid: 0.0 for pid in actions}

        min_price = min(prices.values())
        winners = [pid for pid, price in prices.items() if price == min_price]
        
        payoffs = {}
        quantity_demanded = max(0, demand_intercept - min_price)

        for player_id in actions:
            if player_id in winners:
                market_share = 1.0 / len(winners)
                quantity_sold = quantity_demanded * market_share
                
                player_cost = private_costs.get(player_id)
                
                # --- FIX: Ensure player_cost is a number, not a list ---
                if isinstance(player_cost, list):
                    # This handles the case where the full list was passed instead of a single value
                    sim_id = game_state.get('simulation_id', 0)
                    if sim_id < len(player_cost):
                        player_cost = player_cost[sim_id]
                    else:
                        # Fallback if the list is unexpectedly short
                        player_cost = constants.get('your_cost')
                elif player_cost is None:
                     # Fallback for the challenger if no pre-generated cost is found
                    player_cost = constants.get('your_cost')

                profit = (min_price - player_cost) * quantity_sold
                payoffs[player_id] = profit
            else:
                payoffs[player_id] = 0.0
                
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers all relevant data from the Spulber game for detailed logging."""
        prices = {pid: action.get('price') for pid, action in actions.items() if isinstance(action, dict) and 'price' in action}
        min_price = min(prices.values()) if prices else 0
        winners = [pid for pid, price in prices.items() if price == min_price]

        private_costs = game_state.get('player_private_costs', {})

        return {
            "constants": game_config.constants,
            "winner_ids": winners,
            "player_private_costs": private_costs
        }