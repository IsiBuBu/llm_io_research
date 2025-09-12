"""
Spulber Bertrand Competition Game - Updated implementation with robust response parsing
Implements Winner Determination Algorithm from t.txt for comprehensive metrics analysis
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import StaticGame, PriceParsingMixin, extract_numeric_value, validate_action_bounds
from config import GameConfig, get_prompt_variables


class SpulberGame(StaticGame, PriceParsingMixin):
    """
    Spulber Bertrand Competition - winner-take-all price auction with incomplete information
    Implements Winner Determination Algorithm from t.txt
    """
    
    def __init__(self):
        super().__init__("spulber")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state for static Spulber game"""
        return {
            'game_type': 'static',
            'current_round': 1,
            'simulation_id': simulation_id,
            'constants': game_config.constants
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt using template and config"""
        
        # Get all template variables from config
        variables = get_prompt_variables(game_config, player_id=player_id)
        
        # Format template with variables
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse bidding decision from LLM response using inherited mixin"""
        
        # Use the PriceParsingMixin method (handles both 'price' and 'bid' fields)
        result = self.parse_price_response(response, player_id, call_id)
        
        if result:
            # Normalize field names - convert 'bid' to 'price' for consistency
            if 'bid' in result and 'price' not in result:
                result['price'] = result.pop('bid')
            
            self.logger.debug(f"[{call_id}] Successfully parsed bid/price: {result.get('price', 'N/A')} for {player_id}")
            return result
        
        self.logger.warning(f"[{call_id}] Could not parse bid/price from {player_id}")
        return None

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Spulber Bertrand payoffs using Winner Determination Algorithm from t.txt
        
        Winner Determination Algorithm from t.txt:
        1. Find minimum bid: p_min = min(p_1, p_2, ..., p_N)
        2. Identify all players (K) who bid p_min
        3. Tie-Breaking Rule: market_share for each winner = 1/K, others = 0
        4. For each winner: quantity_sold = (demand_intercept - p_min) × market_share
        5. For each winner: profit = (p_min - cost) × quantity_sold
        6. All other players receive zero profit
        """
        
        # Extract game constants
        constants = game_config.constants
        demand_intercept = constants.get('demand_intercept', 100)
        demand_slope = constants.get('demand_slope', 1)
        
        # Extract prices and costs from actions
        prices = {}
        costs = {}
        
        for player_id, action in actions.items():
            if isinstance(action, dict):
                # Extract price/bid
                if 'price' in action:
                    prices[player_id] = action['price']
                elif 'bid' in action:
                    prices[player_id] = action['bid']
                else:
                    self.logger.warning(f"No price/bid found for {player_id}: {action}")
                    prices[player_id] = demand_intercept  # High fallback price (likely loses)
                
                # Extract cost (if available in action, otherwise from constants)
                if 'cost' in action:
                    costs[player_id] = action['cost']
                else:
                    # Use default cost from constants or player-specific cost
                    costs[player_id] = constants.get(f'{player_id}_cost', constants.get('marginal_cost', 10))
            else:
                self.logger.warning(f"Invalid action format for {player_id}: {action}")
                prices[player_id] = demand_intercept
                costs[player_id] = constants.get('marginal_cost', 10)
        
        # Step 1: Find minimum price
        min_price = min(prices.values())
        
        # Step 2: Identify winners (all players who bid the minimum price)
        winners = [player_id for player_id, price in prices.items() if price == min_price]
        
        # Step 3: Calculate market shares (equal split among winners)
        market_shares = {}
        for player_id in prices.keys():
            if player_id in winners:
                market_shares[player_id] = 1.0 / len(winners)  # Equal share among winners
            else:
                market_shares[player_id] = 0.0  # Losers get nothing
        
        # Step 4 & 5: Calculate quantities and profits
        payoffs = {}
        
        # Calculate total quantity demanded at winning price
        total_quantity = max(0, demand_intercept - demand_slope * min_price)
        
        for player_id in prices.keys():
            if player_id in winners:
                # Winner gets share of market
                quantity_sold = total_quantity * market_shares[player_id]
                player_cost = costs[player_id]
                profit = (min_price - player_cost) * quantity_sold
                payoffs[player_id] = profit
                
                self.logger.debug(f"Winner {player_id}: price={min_price:.2f}, cost={player_cost:.2f}, "
                               f"quantity={quantity_sold:.1f}, profit={profit:.2f}")
            else:
                # Loser gets zero profit
                payoffs[player_id] = 0.0
                self.logger.debug(f"Loser {player_id}: price={prices[player_id]:.2f}, profit=0.0")
        
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get comprehensive structured data for metrics calculation and logging
        Implements t.txt specification for Spulber metrics data collection
        """
        
        # Extract basic data
        constants = game_config.constants
        prices = {}
        costs = {}
        
        for player_id, action in actions.items():
            if isinstance(action, dict):
                # Extract price/bid
                prices[player_id] = action.get('price', action.get('bid', 0))
                # Extract cost
                costs[player_id] = action.get('cost', constants.get(f'{player_id}_cost', 
                                                                  constants.get('marginal_cost', 10)))
            else:
                prices[player_id] = 0
                costs[player_id] = constants.get('marginal_cost', 10)
        
        # Calculate market shares (from payoff calculation logic)
        min_price = min(prices.values()) if prices else 0
        winners = [player_id for player_id, price in prices.items() if price == min_price]
        
        market_shares = {}
        for player_id in prices.keys():
            market_shares[player_id] = 1.0 / len(winners) if player_id in winners else 0.0
        
        # Calculate additional metrics for t.txt specification
        max_profit = max(payoffs.values()) if payoffs else 0
        win_status = {}
        profitable_status = {}
        market_capture_status = {}
        rational_bid_status = {}
        
        for player_id, profit in payoffs.items():
            # Win Status: 1 if player had highest profit, 0 otherwise (t.txt spec)
            win_status[player_id] = 1 if profit == max_profit else 0
            
            # Profitable Status: 1 if profit >= 0, 0 otherwise (MAgIC metrics)
            profitable_status[player_id] = 1 if profit >= 0 else 0
            
            # Market Capture Status: 1/K if won market, 0 otherwise (t.txt Market Capture Rate)
            if player_id in winners and len(winners) > 0:
                market_capture_status[player_id] = 1.0 / len(winners)
            else:
                market_capture_status[player_id] = 0
            
            # Rational Bid Status: 1 if bid >= own cost, 0 otherwise (MAgIC Rationality)
            player_price = prices.get(player_id, 0)
            player_cost = costs.get(player_id, 0)
            rational_bid_status[player_id] = 1 if player_price >= player_cost else 0
        
        return {
            # Core identifiers
            'game_name': 'spulber',
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'constants': game_config.constants,
            
            # Required data for t.txt metrics calculation
            'actions': actions,
            'payoffs': payoffs,
            'prices': prices,
            'costs': costs,
            'market_shares': market_shares,
            
            # t.txt specific metrics data
            'win_status': win_status,
            'profitable_status': profitable_status,
            'market_capture_status': market_capture_status,
            'rational_bid_status': rational_bid_status,
            
            # Winner determination results
            'min_price': min_price,
            'winners': winners,
            'number_of_winners': len(winners),
            
            # Additional metadata
            'game_state': game_state
        }