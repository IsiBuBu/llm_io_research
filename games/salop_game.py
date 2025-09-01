"""
Salop Spatial Competition Game - Minimal implementation for comprehensive_metrics.py
Focuses only on core Salop economics and generating proper data for metrics analysis
"""

import json
import re
from typing import Dict, Any, Optional, List
from games.base_game import StaticGame
from config import GameConfig, GameConstants


class SalopGame(StaticGame):
    """
    Salop Spatial Competition - firms compete on prices in circular market
    Generates data needed by comprehensive_metrics.py for behavioral analysis
    """
    
    def __init__(self):
        super().__init__("Salop Spatial Competition")

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             config: GameConfig, constants: GameConstants) -> str:
        """Generate strategic prompt for Salop pricing decision"""
        
        neighbors = self._get_neighbors(player_id, config.number_of_players)
        equilibrium_price = self._calculate_equilibrium_price(config, constants)
        
        prompt = f"""**SALOP SPATIAL COMPETITION**

You are Firm {player_id} in a circular market with {config.number_of_players} firms positioned around a circle.

**MARKET STRUCTURE:**
- Total consumers: {constants.SALOP_BASE_MARKET_SIZE:,} distributed uniformly around circle
- Your neighbors: {neighbors[0]} (left) and {neighbors[1]} (right)  
- Competition: You compete primarily with immediate neighbors
- Consumer cost: Your Price + (distance × ${constants.SALOP_TRANSPORT_COST:.2f})

**YOUR ECONOMICS:**
- Marginal cost: ${constants.SALOP_MARGINAL_COST:.2f} per unit
- Fixed cost: ${constants.SALOP_FIXED_COST:.2f} per period
- Profit = (Price - ${constants.SALOP_MARGINAL_COST:.2f}) × Quantity - ${constants.SALOP_FIXED_COST:.2f}

**STRATEGIC SITUATION:**
- Lower prices expand your market territory but reduce margins
- Higher prices increase margins but risk losing customers to neighbors
- Market boundaries determined by where consumers switch to neighbors
- Theoretical equilibrium price: ${equilibrium_price:.2f}

**PRICING GUIDANCE:**
- Minimum: ${constants.SALOP_MARGINAL_COST:.2f} (marginal cost)
- Competitive range: ${constants.SALOP_MARGINAL_COST + 1:.2f} - ${constants.SALOP_MARGINAL_COST + 5:.2f}
- Maximum reasonable: ${constants.SALOP_MARGINAL_COST + 8:.2f}

**REQUIRED FORMAT:**
{{"price": <your_price>, "reasoning": "<explanation of your spatial pricing strategy>"}}

Choose your price to maximize profit in this circular spatial competition."""

        return prompt

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse pricing decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.basic_json_parse(response)
        if json_action and 'price' in json_action:
            price = self._validate_price(json_action['price'])
            if price is not None:
                json_action['price'] = price
                json_action['raw_response'] = response
                return json_action
        
        # Try regex patterns for price extraction
        price_patterns = [
            r'"price":\s*(\d+\.?\d*)',
            r'price["\']?:\s*(\d+\.?\d*)',
            r'price:\s*\$?(\d+\.?\d*)',
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d+)'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    price = self._validate_price(float(matches[0]))
                    if price is not None:
                        return {
                            'price': price,
                            'reasoning': response[:200],
                            'parsing_method': 'regex',
                            'raw_response': response
                        }
                except ValueError:
                    continue
        
        self.logger.warning(f"[{call_id}] Could not parse price from {player_id} response")
        return None

    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Get default pricing action when parsing fails"""
        constants = GameConstants()
        default_price = self._calculate_equilibrium_price(config, constants)
        
        return {
            'price': default_price,
            'reasoning': f'Default equilibrium pricing due to parsing failure',
            'parsing_success': False,
            'player_id': player_id,
            'round': game_state.get('current_round', 1)
        }

    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate Salop spatial competition payoffs"""
        
        call_id = game_state.get('call_id', 'unknown') if game_state else 'unknown'
        constants = GameConstants()
        
        # Extract prices
        prices = {}
        for agent_id, action in actions.items():
            price = action.get('price', constants.SALOP_MARGINAL_COST + 3)
            prices[agent_id] = float(price)
        
        profits = {}
        player_list = list(prices.keys())
        n = len(player_list)
        
        if n == 1:
            # Single player gets full market
            agent_id = player_list[0]
            price = prices[agent_id]
            quantity = constants.SALOP_BASE_MARKET_SIZE
            profit = (price - constants.SALOP_MARGINAL_COST) * quantity - constants.SALOP_FIXED_COST
            profits[agent_id] = max(0, profit)
            
        else:
            # Multi-player spatial competition
            for i, agent_id in enumerate(player_list):
                price = prices[agent_id]
                
                # Get neighbors (circular topology)
                left_neighbor = player_list[(i - 1) % n]
                right_neighbor = player_list[(i + 1) % n]
                left_price = prices[left_neighbor]
                right_price = prices[right_neighbor]
                
                # Calculate market share using spatial competition
                market_share = self._calculate_market_share(
                    price, left_price, right_price, constants, n
                )
                
                # Calculate profit
                quantity = market_share * constants.SALOP_BASE_MARKET_SIZE
                revenue = price * quantity
                total_cost = constants.SALOP_MARGINAL_COST * quantity + constants.SALOP_FIXED_COST
                profit = revenue - total_cost
                
                profits[agent_id] = max(0, profit)
                
                self.logger.debug(f"[{call_id}] {agent_id}: price=${price:.2f}, "
                                f"share={market_share:.3f}, profit=${profit:.2f}")
        
        # Log market summary for analysis
        total_profit = sum(profits.values())
        avg_price = sum(prices.values()) / len(prices)
        
        self.logger.info(f"[{call_id}] Salop market: Avg price=${avg_price:.2f}, "
                       f"Total profit=${total_profit:.2f}")
        
        return profits

    def _validate_price(self, price: float) -> Optional[float]:
        """Validate price is in reasonable range"""
        try:
            price = float(price)
            if 5.0 <= price <= 25.0:  # Reasonable bounds
                return round(price, 2)
            else:
                return None
        except (ValueError, TypeError):
            return None

    def _calculate_market_share(self, price: float, left_price: float, right_price: float, 
                               constants: GameConstants, num_firms: int) -> float:
        """Calculate market share using Salop spatial competition model"""
        
        try:
            # Distance between adjacent firms on circle
            firm_distance = 1.0 / num_firms
            transport_cost = constants.SALOP_TRANSPORT_COST
            
            # Calculate indifference points with neighbors
            
            # Left boundary (indifference with left neighbor)
            if abs(price - left_price) < 0.01:  # Equal prices
                left_boundary = firm_distance / 2
            else:
                # Consumer indifference: price + t*d = left_price + t*(firm_distance - d)
                # Solve for d: d = (firm_distance * t + left_price - price) / (2*t)
                left_boundary = (firm_distance * transport_cost + left_price - price) / (2 * transport_cost)
                left_boundary = max(0, min(firm_distance, left_boundary))
            
            # Right boundary (indifference with right neighbor)  
            if abs(price - right_price) < 0.01:  # Equal prices
                right_boundary = firm_distance / 2
            else:
                right_boundary = (firm_distance * transport_cost + right_price - price) / (2 * transport_cost)
                right_boundary = max(0, min(firm_distance, right_boundary))
            
            # Total market share = left segment + right segment
            market_share = (left_boundary + right_boundary) / firm_distance
            
            # Bound to reasonable range
            return max(0.01, min(0.99, market_share))
            
        except Exception as e:
            self.logger.warning(f"Error calculating market share: {e}")
            return 1.0 / num_firms  # Equal share fallback

    def _get_neighbors(self, player_id: str, num_players: int) -> List[str]:
        """Get left and right neighbors on the circle"""
        
        # Create standardized player list
        if num_players == 2:
            player_list = ["defender", "challenger"]
        else:
            player_list = ["challenger"] + [f"defender_{i}" for i in range(num_players - 1)]
        
        try:
            current_index = player_list.index(player_id)
        except ValueError:
            return ["neighbor_left", "neighbor_right"]
        
        # Get circular neighbors
        left_neighbor = player_list[(current_index - 1) % num_players]
        right_neighbor = player_list[(current_index + 1) % num_players]
        
        return [left_neighbor, right_neighbor]

    def _calculate_equilibrium_price(self, config: GameConfig, constants: GameConstants) -> float:
        """Calculate theoretical Salop equilibrium price"""
        # Standard Salop equilibrium: P* = MC + (t/n)
        equilibrium_price = (constants.SALOP_MARGINAL_COST + 
                           constants.SALOP_TRANSPORT_COST / config.number_of_players)
        return round(equilibrium_price, 2)