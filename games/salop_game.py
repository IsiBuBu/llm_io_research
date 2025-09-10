"""
Salop Spatial Competition Game - Updated implementation with robust response parsing
Implements Market Share and Quantity Calculation Algorithm from t.txt
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import StaticGame, PriceParsingMixin, extract_numeric_value, validate_action_bounds
from config import GameConfig, get_prompt_variables


class SalopGame(StaticGame, PriceParsingMixin):
    """
    Salop Spatial Competition - firms compete on prices in circular market
    Implements Market Share and Quantity Calculation Algorithm from t.txt
    """
    
    def __init__(self):
        super().__init__("salop")

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
        """Parse pricing decision from LLM response using inherited mixin"""
        
        # Use the PriceParsingMixin method
        result = self.parse_price_response(response, player_id, call_id)
        
        if result:
            # Validate price bounds if specified in config
            constants = getattr(self, '_current_config', {}).get('constants', {})
            max_price = constants.get('max_price', 100.0)  # Default reasonable upper bound
            min_price = constants.get('min_price', 0.0)    # Default lower bound
            
            if 'price' in result:
                result['price'] = max(min_price, min(max_price, result['price']))
            
            self.logger.debug(f"[{call_id}] Successfully parsed price: {result.get('price', 'N/A')} for {player_id}")
            return result
        
        self.logger.warning(f"[{call_id}] Could not parse price from {player_id}")
        return None

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Salop spatial competition payoffs using Market Share Algorithm from t.txt
        
        Algorithm from t.txt:
        1. Calculate Competitive Boundaries (x_right, x_left for each firm)
        2. Calculate Monopoly Boundary (x_max for unconstrained market reach)
        3. Determine Final Market Reach (min of competitive and monopoly boundaries)
        4. Calculate Market Share and Quantities
        5. Calculate Profits = (Price - MC) * Quantity - Fixed Cost
        """
        
        constants = game_config.constants
        
        # Extract constants
        marginal_cost = constants.get('marginal_cost', 8)
        fixed_cost = constants.get('fixed_cost', 100)
        transport_cost = constants.get('transport_cost', 2)
        market_size = constants.get('market_size', 100)
        v = constants.get('v', 50)  # Consumer reservation price
        number_of_players = constants.get('number_of_players', 3)
        
        # Extract prices from actions
        prices = {}
        player_ids = list(actions.keys())
        
        for player_id, action in actions.items():
            price = action.get('price', marginal_cost + 1)  # Fallback price
            prices[player_id] = price
        
        # Calculate payoffs for each player
        payoffs = {}
        
        for i, player_id in enumerate(player_ids):
            p_i = prices[player_id]
            
            # Initialize market boundaries
            total_market_share = 0.0
            
            # For circular market, each firm competes with adjacent firms
            # In simplified model, assume uniform competition
            if number_of_players == 1:
                # Monopoly case
                if p_i <= v:
                    market_share = market_size
                else:
                    market_share = 0
            else:
                # Competition case - calculate against each other player
                competitive_shares = []
                
                for j, other_player_id in enumerate(player_ids):
                    if i == j:
                        continue
                        
                    p_j = prices[other_player_id]
                    
                    # Calculate competitive boundary between firms i and j
                    # Customer indifferent at distance x where: p_i + t*x = p_j + t*(1/n - x)
                    # Solving: x = (p_j - p_i)/(2*t) + 1/(2*n)
                    
                    distance_between_firms = 1.0 / number_of_players  # Equal spacing on circle
                    
                    if abs(p_i - p_j) < 1e-6:  # Essentially same price
                        competitive_boundary = distance_between_firms / 2
                    else:
                        competitive_boundary = (p_j - p_i) / (2 * transport_cost) + distance_between_firms / 2
                    
                    # Clamp boundary to valid range
                    competitive_boundary = max(0, min(distance_between_firms, competitive_boundary))
                    competitive_shares.append(competitive_boundary)
                
                # Calculate monopoly boundary (max reach before losing to reservation price)
                # Customer at distance x pays p_i + t*x, must be <= v
                monopoly_boundary = (v - p_i) / transport_cost if p_i < v else 0
                monopoly_boundary = max(0, monopoly_boundary)
                
                # Market share is minimum of competitive constraints
                if competitive_shares:
                    avg_competitive_share = sum(competitive_shares) / len(competitive_shares)
                    # Each firm gets market share in both directions (circular market)
                    market_share_fraction = min(avg_competitive_share * 2, monopoly_boundary / (1.0 / number_of_players))
                    market_share_fraction = min(market_share_fraction, 1.0)  # Can't exceed full market
                else:
                    market_share_fraction = min(monopoly_boundary / (1.0 / number_of_players), 1.0)
                
                market_share = market_share_fraction * market_size / number_of_players
            
            # Calculate quantity (market share determines quantity sold)
            quantity = max(0, market_share)
            
            # Calculate profit
            if quantity > 0:
                revenue = p_i * quantity
                variable_cost = marginal_cost * quantity
                profit = revenue - variable_cost - fixed_cost
            else:
                profit = -fixed_cost  # Still pay fixed cost even with zero sales
            
            payoffs[player_id] = profit
            
            # Log detailed calculation for debugging
            self.logger.debug(f"Player {player_id}: price={p_i:.2f}, quantity={quantity:.2f}, profit={profit:.2f}")
        
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get structured data for metrics calculation and logging"""
        
        # Extract key metrics for analysis
        prices = [action.get('price', 0) for action in actions.values()]
        
        base_data = super().get_game_data_for_logging(actions, payoffs, game_config, game_state)
        
        # Add Salop-specific metrics
        salop_metrics = {
            'average_price': sum(prices) / len(prices) if prices else 0,
            'price_variance': sum((p - sum(prices)/len(prices))**2 for p in prices) / len(prices) if len(prices) > 1 else 0,
            'min_price': min(prices) if prices else 0,
            'max_price': max(prices) if prices else 0,
            'marginal_cost': game_config.constants.get('marginal_cost', 8),
            'transport_cost': game_config.constants.get('transport_cost', 2),
            'market_size': game_config.constants.get('market_size', 100),
            'price_dispersion': (max(prices) - min(prices)) if len(prices) > 1 else 0,
            'markup_rates': [(prices[i] - game_config.constants.get('marginal_cost', 8)) / prices[i] 
                           if prices[i] > 0 else 0 for i in range(len(prices))]
        }
        
        base_data.update(salop_metrics)
        return base_data