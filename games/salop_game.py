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

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state for static Salop game"""
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
        5. Calculate Profits (Revenue - Costs)
        """
        
        # Store current config for use in parsing
        self._current_config = game_config
        
        # Extract game constants
        constants = game_config.constants
        marginal_cost = constants.get('marginal_cost', 0.5)
        transport_cost = constants.get('transport_cost', 1.0)
        market_size = constants.get('market_size', 100)
        num_firms = constants.get('num_firms', len(actions))
        
        # Extract prices from actions
        prices = {}
        for player_id, action in actions.items():
            if isinstance(action, dict) and 'price' in action:
                prices[player_id] = action['price']
            else:
                self.logger.warning(f"Invalid action format for {player_id}: {action}")
                prices[player_id] = marginal_cost + 1.0  # Default fallback price
        
        # Convert to list for algorithm (maintaining order)
        player_ids = sorted(prices.keys())  # Ensure consistent ordering
        price_list = [prices[pid] for pid in player_ids]
        
        # Calculate market shares using Salop algorithm
        market_shares = self._calculate_salop_market_shares(price_list, transport_cost, num_firms)
        
        # Calculate quantities and payoffs
        payoffs = {}
        for i, player_id in enumerate(player_ids):
            quantity = market_shares[i] * market_size
            price = price_list[i]
            profit = (price - marginal_cost) * quantity
            payoffs[player_id] = profit
            
            self.logger.debug(f"Player {player_id}: price={price:.2f}, market_share={market_shares[i]:.3f}, "
                           f"quantity={quantity:.1f}, profit={profit:.2f}")
        
        return payoffs
    
    def _calculate_salop_market_shares(self, prices: List[float], transport_cost: float, 
                                     num_firms: int) -> List[float]:
        """
        Calculate market shares in Salop circular city model
        
        Algorithm from t.txt:
        For each firm i, calculate market boundaries with adjacent firms i-1 and i+1
        Market share = distance covered on circle / total circumference
        """
        
        if num_firms <= 0:
            return []
        
        if num_firms == 1:
            return [1.0]  # Monopoly
            
        market_shares = []
        
        for i in range(num_firms):
            # Get adjacent firm indices (circular)
            left_neighbor = (i - 1) % num_firms
            right_neighbor = (i + 1) % num_firms
            
            # Calculate competitive boundaries
            # Left boundary: point where consumers are indifferent between firm i and left neighbor
            x_left = self._calculate_boundary(prices[i], prices[left_neighbor], transport_cost, num_firms)
            
            # Right boundary: point where consumers are indifferent between firm i and right neighbor  
            x_right = self._calculate_boundary(prices[i], prices[right_neighbor], transport_cost, num_firms)
            
            # Market share is the distance covered (as fraction of circle circumference)
            market_share = (x_left + x_right) / num_firms
            
            # Ensure non-negative market share
            market_share = max(0.0, market_share)
            
            market_shares.append(market_share)
        
        # Normalize to ensure sum equals 1 (handle rounding errors)
        total_share = sum(market_shares)
        if total_share > 0:
            market_shares = [share / total_share for share in market_shares]
        else:
            # If all shares are 0, distribute equally
            market_shares = [1.0 / num_firms] * num_firms
            
        return market_shares
    
    def _calculate_boundary(self, price_i: float, price_j: float, transport_cost: float, 
                          num_firms: int) -> float:
        """
        Calculate competitive boundary between two adjacent firms
        
        Returns distance from firm i where consumer is indifferent between firms i and j
        Derived from setting total costs equal: price_i + t*x = price_j + t*(1/n - x)
        """
        
        # Distance between adjacent firms on circle
        firm_distance = 1.0 / num_firms
        
        # Solve for indifference point: price_i + t*x = price_j + t*(firm_distance - x)
        # Rearranging: x = (price_j - price_i + t*firm_distance) / (2*t)
        
        if transport_cost <= 0:
            # If no transport cost, split market equally
            return firm_distance / 2.0
        
        boundary = (price_j - price_i + transport_cost * firm_distance) / (2.0 * transport_cost)
        
        # Ensure boundary is within valid range [0, firm_distance]
        boundary = max(0.0, min(firm_distance, boundary))
        
        return boundary