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
        5. Profit = (p_min - private_cost) × quantity_sold
        """
        
        constants = game_config.constants
        
        # Extract constants from t.txt specification
        demand_intercept = constants.get('demand_intercept', 100)
        private_values = constants.get('private_values', {})
        
        # Extract prices and costs from actions
        player_ids = list(actions.keys())
        prices = {}
        costs = {}
        
        for player_id, action in actions.items():
            price = action.get('price', 0)
            prices[player_id] = max(0.0, price)  # Ensure non-negative bids
            
            # Get player's private cost from config
            if player_id == 'challenger':
                costs[player_id] = private_values.get('challenger_cost', 8)
            else:
                costs[player_id] = private_values.get('defender_cost', 10)
        
        # Step 1: Find minimum bid
        min_price = min(prices.values()) if prices else 0
        
        # Step 2: Identify all players who bid p_min
        winners = [player_id for player_id, price in prices.items() if price == min_price]
        K = len(winners)  # Number of winners
        
        # Step 3: Tie-Breaking Rule - market_share for each winner = 1/K
        market_share_per_winner = 1.0 / K if K > 0 else 0
        
        # Calculate payoffs for each player
        payoffs = {}
        quantities = {}
        market_shares = {}
        
        for player_id in player_ids:
            if player_id in winners:
                # Winner gets market share
                market_share = market_share_per_winner
                
                # Step 4: Calculate quantity sold (from t.txt)
                quantity_sold = (demand_intercept - min_price) * market_share
                quantity_sold = max(0, quantity_sold)  # Non-negative quantity
                
                # Step 5: Calculate profit (from t.txt)  
                profit = (min_price - costs[player_id]) * quantity_sold
            else:
                # Loser gets nothing
                market_share = 0
                quantity_sold = 0
                profit = 0
            
            payoffs[player_id] = profit
            quantities[player_id] = quantity_sold
            market_shares[player_id] = market_share
            
            self.logger.debug(f"Player {player_id}: bid={prices[player_id]:.2f}, "
                            f"cost={costs[player_id]:.2f}, won={player_id in winners}, profit={profit:.2f}")
        
        # Store additional data in game_state for metrics calculation
        if game_state is not None:
            game_state.update({
                'prices': prices,
                'costs': costs,
                'quantities': quantities,
                'market_shares': market_shares,
                'min_price': min_price,
                'winners': winners
            })
        
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for t.txt metrics calculation only"""
        
        # Get calculated data from game_state
        prices = game_state.get('prices', {}) if game_state else {}
        costs = game_state.get('costs', {}) if game_state else {}
        market_shares = game_state.get('market_shares', {}) if game_state else {}
        min_price = game_state.get('min_price', 0) if game_state else 0
        winners = game_state.get('winners', []) if game_state else []
        
        # Calculate win status for Win Rate metric (from t.txt)
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