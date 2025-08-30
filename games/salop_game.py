import numpy as np
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict

from games.base_game import StaticGame
from config import GameConfig, GameConstants, PlayerResult

class SalopGame(StaticGame):
    """
    Enhanced Salop Spatial Competition Game with JSON configuration and comprehensive debugging.
    
    This implementation models spatial competition where firms are positioned on a circle
    and compete on prices, with consumers incurring transportation costs to reach firms.
    Based on Steven Salop's classic 1979 model of monopolistic competition.
    """
    
    def __init__(self):
        super().__init__("Salop Spatial Competition")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Game-specific tracking
        self.price_history = []
        self.market_share_evolution = {}
        self.spatial_equilibrium_analysis = []
        
        self.logger.info("Salop spatial competition game initialized with enhanced debugging")
    
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        """
        Create enhanced strategic prompt for Salop spatial competition with comprehensive context.
        """
        try:
            constants = GameConstants(config)
            neighbors = self._get_neighbors(player_id, config.number_of_players)
            call_id = game_state.get('call_id', 'unknown')
            
            # Calculate spatial competition statistics
            spatial_stats = self._calculate_spatial_statistics(game_state, config)
            competitor_analysis = self._analyze_competitor_positioning(player_id, game_state, config)
            
            self.logger.debug(f"[{call_id}] Creating Salop prompt for {player_id}")
            self.logger.debug(f"[{call_id}] Neighbors: {neighbors}, Spatial stats: {spatial_stats}")
            
            # Enhanced strategic context with spatial competition theory
            prompt = f"""**STRATEGIC CONTEXT: Salop Spatial Competition Model**

You are Firm {player_id} competing in a classic circular market with {config.number_of_players} firms positioned around a circle. This is a sophisticated spatial competition scenario where your success depends on optimal pricing relative to your geographic position and competitor locations.

**SPATIAL MARKET STRUCTURE:**
- **Market Geography**: All firms are located on a circle with consumers evenly distributed
- **Your Position**: You are strategically positioned between Firm {neighbors[0]} (left neighbor) and Firm {neighbors[1]} (right neighbor)
- **Total Market**: {constants.SALOP_MARKET_SIZE:,} consumers distributed uniformly around the circle
- **Consumer Behavior**: Each consumer buys exactly one unit from the firm offering lowest total cost (price + transportation cost)

**ECONOMIC FUNDAMENTALS:**
- **Your Marginal Cost**: ${constants.SALOP_MARGINAL_COST:.2f} per unit (constant across all units)
- **Your Fixed Cost**: ${constants.SALOP_FIXED_COST:.2f} per period (must be covered regardless of sales)
- **Transportation Cost**: ${constants.SALOP_TRANSPORT_COST:.2f} per unit of distance (paid by consumers)
- **Profit Formula**: (Your Price - ${constants.SALOP_MARGINAL_COST:.2f}) × Units Sold - ${constants.SALOP_FIXED_COST:.2f}

**SPATIAL COMPETITION DYNAMICS:**
The Salop model creates intense **localized competition** between neighboring firms:

1. **Market Boundaries**: Your market share is determined by the midpoint between you and each neighbor where consumers are indifferent between firms
2. **Price Competition**: If you price too high, nearby customers switch to competitors; too low sacrifices profit margins
3. **Geographic Advantage**: Your circular position gives you some market power, but neighbors constrain your pricing flexibility
4. **Consumer Indifference**: A consumer located distance 'd' from you pays: Your Price + (d × ${constants.SALOP_TRANSPORT_COST:.2f})

**STRATEGIC TRADE-OFFS:**
- **Market Share vs. Margins**: Lower prices expand your market territory but reduce per-unit profits
- **Neighbor Competition**: Your optimal price depends critically on your neighbors' pricing strategies  
- **Cost Coverage**: You must generate enough revenue to cover your fixed cost of ${constants.SALOP_FIXED_COST:.2f}
- **Spatial Protection**: Transportation costs provide some insulation from distant competitors

**MARKET ANALYSIS:**
- **Market Coverage**: Each firm naturally serves approximately {100/config.number_of_players:.1f}% of the market in symmetric equilibrium
- **Competitive Intensity**: With {config.number_of_players} firms, competition is {"intense" if config.number_of_players >= 5 else "moderate" if config.number_of_players >= 3 else "limited"}
- **Expected Equilibrium Range**: Theoretical prices typically range from ${constants.SALOP_MARGINAL_COST + 2:.2f} to ${constants.SALOP_MARGINAL_COST + 8:.2f}

**CURRENT COMPETITIVE LANDSCAPE:**
{competitor_analysis['market_context']}

**STRATEGIC CONSIDERATIONS:**
Consider these critical factors in your pricing decision:

1. **Neighbor Price Sensitivity**: 
   - If Left Neighbor (Firm {neighbors[0]}) prices high, you can capture more left-side market share with competitive pricing
   - If Right Neighbor (Firm {neighbors[1]}) prices low, you need aggressive pricing to defend right-side territory

2. **Market Boundary Optimization**:
   - Your optimal boundaries depend on achieving the profit-maximizing balance between market share and margins
   - Consider how your price affects the indifference points with both neighbors

3. **Fixed Cost Recovery**:
   - You need minimum revenue of ${constants.SALOP_FIXED_COST:.2f} to break even
   - This requires selling at least {constants.SALOP_FIXED_COST / (constants.SALOP_MARGINAL_COST + 1):.0f} units at reasonable margins

4. **Competitive Response Anticipation**:
   - Your neighbors will simultaneously choose their prices, considering your likely response
   - Think strategically about what price signals cooperation vs. aggressive competition

**PRICING STRATEGY GUIDANCE:**
- **Conservative Strategy**: Price around ${constants.SALOP_MARGINAL_COST + 4:.2f} (moderate markup, steady market share)
- **Aggressive Strategy**: Price around ${constants.SALOP_MARGINAL_COST + 2:.2f} (expand market share, pressure neighbors)
- **Premium Strategy**: Price around ${constants.SALOP_MARGINAL_COST + 6:.2f} (high margins, risk losing market share)

**SPATIAL EQUILIBRIUM THEORY:**
In symmetric Salop equilibrium:
- All firms price at: P* = MC + (t/n) where t = transport cost, n = number of firms
- Your theoretical equilibrium price: ${self._calculate_theoretical_equilibrium_price(config):.2f}
- Market share per firm: {constants.SALOP_MARKET_SIZE // config.number_of_players:,} customers
- Profit per firm: ${self._calculate_theoretical_equilibrium_profit(config):.2f}

**DECISION FRAMEWORK:**
Think through these steps:
1. **Assess Market Position**: How does your circular position affect your competitive advantages?
2. **Neighbor Analysis**: What pricing strategies are your immediate neighbors likely to employ?
3. **Market Share Goals**: Do you prioritize expanding territory or maximizing profit per customer?
4. **Risk Assessment**: How much competitive response risk are you willing to accept?

**OUTPUT REQUIREMENTS:**
Provide your decision in the following JSON format with detailed spatial reasoning:

{{"price": <price between {constants.SALOP_MARGINAL_COST:.2f} and {constants.SALOP_MARGINAL_COST + 10:.2f}>, "reasoning": "<detailed explanation of your spatial competition strategy, neighbor analysis, and pricing rationale>"}}

**PRICING GUIDELINES:**
- **Minimum Viable Price**: ${constants.SALOP_MARGINAL_COST:.2f} (marginal cost - zero margin)
- **Maximum Reasonable Price**: ${constants.SALOP_MARGINAL_COST + 10:.2f} (risk of losing all customers to neighbors)
- **Typical Competitive Range**: ${constants.SALOP_MARGINAL_COST + 2:.2f} - ${constants.SALOP_MARGINAL_COST + 6:.2f}

**IMPACT OF YOUR DECISION:**
Your price directly determines:
1. **Market Boundaries**: Where customers switch between you and neighbors
2. **Market Share**: Fraction of circular market you capture  
3. **Profit Margins**: Revenue per customer minus costs
4. **Competitive Dynamics**: Whether you trigger price wars or maintain market stability

Choose your price considering both immediate profit maximization and the strategic equilibrium of the circular market."""

            # Log prompt statistics
            self.logger.debug(f"[{call_id}] Generated Salop prompt: {len(prompt)} chars for player {player_id}")
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error creating Salop prompt for {player_id}: {e}")
            # Return simplified fallback prompt
            return f"""You are Firm {player_id} in a circular market with {config.number_of_players} firms. 
            Your neighbors are Firms {self._get_neighbors(player_id, config.number_of_players)}.
            Your marginal cost is ${GameConstants(config).SALOP_MARGINAL_COST}, transportation cost is ${GameConstants(config).SALOP_TRANSPORT_COST} per unit distance.
            Choose your price to maximize profit. Format: {{"price": <number>, "reasoning": "<explanation>"}}"""
    
    def parse_action(self, response: str, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Enhanced action parsing for Salop pricing decisions.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            self.logger.debug(f"[{call_id}] Parsing Salop pricing action for {player_id}")
            
            # Try JSON parsing first
            action = self._parse_json_response(response, player_id, call_id)
            if action and 'price' in action:
                price = self._validate_price(action['price'], player_id, call_id, config)
                action['price'] = price
                action['parsing_method'] = 'json'
                return self._enhance_salop_action(action, player_id, game_state, config)
            
            # Try pattern-based parsing
            action = self._parse_salop_patterns(response, player_id, call_id, config)
            if action:
                return self._enhance_salop_action(action, player_id, game_state, config)
            
            # Use base class intelligent parsing with Salop enhancement
            action = self._parse_intelligent_response(response, player_id, call_id)
            if action:
                action = self._enhance_intelligent_parsing(action, response.lower())
                if action and 'price' in action:
                    return self._enhance_salop_action(action, player_id, game_state, config)
            
            # Fallback to default
            self.logger.warning(f"[{call_id}] All parsing methods failed for {player_id}, using Salop default")
            return self.get_default_action(player_id, game_state, config)
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error parsing Salop action for {player_id}: {e}")
            return self.get_default_action(player_id, game_state, config)
    
    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Provide intelligent default pricing action when parsing fails"""
        call_id = game_state.get('call_id', 'unknown')
        
        # Use theoretical equilibrium price as reasonable default
        default_price = self._calculate_theoretical_equilibrium_price(config)
        
        self.logger.warning(f"[{call_id}] Using Salop equilibrium default price for {player_id}: ${default_price:.2f}")
        
        return {
            'price': default_price,
            'reasoning': f'Default Salop equilibrium pricing due to parsing failure. Using theoretical symmetric equilibrium price.',
            'raw_response': 'PARSING_FAILED',
            'parsing_method': 'salop_default',
            'player_id': player_id,
            'round': game_state.get('current_round', 1)
        }
    
    def get_required_action_fields(self) -> List[str]:
        """Required fields for Salop pricing actions"""
        return ['price', 'reasoning']
    
    def validate_action(self, action: Dict[str, Any], player_id: str, config: GameConfig) -> bool:
        """Enhanced validation for Salop pricing actions"""
        try:
            # Check required fields
            if 'price' not in action:
                self.logger.warning(f"Missing 'price' field in {player_id} action")
                return False
            
            # Validate price range
            constants = GameConstants(config)
            price = float(action['price'])
            
            if price < constants.SALOP_MARGINAL_COST:
                self.logger.warning(f"{player_id} price ${price:.2f} below marginal cost ${constants.SALOP_MARGINAL_COST}")
                return False
            
            if price > constants.SALOP_MARGINAL_COST + 20:  # Reasonable upper bound
                self.logger.warning(f"{player_id} price ${price:.2f} unreasonably high")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid price format for {player_id}: {e}")
            return False
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig, 
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Enhanced payoff calculation using Salop spatial competition model with comprehensive logging.
        """
        call_id = game_state.get('call_id', 'unknown') if game_state else 'unknown'
        
        try:
            constants = GameConstants(config)
            
            # Extract and validate prices
            prices = {}
            for player_id, action in actions.items():
                price = self._safe_get_numeric(action, 'price', constants.SALOP_MARGINAL_COST + 4)
                prices[player_id] = price
                self.logger.debug(f"[{call_id}] {player_id} price: ${price:.2f}")
            
            profits = {}
            market_shares = {}
            n = config.number_of_players
            
            # Create mapping from player IDs to positions (0-indexed)
            player_list = list(actions.keys())
            
            # Ensure we have enough players for spatial competition
            if len(player_list) < 2:
                self.logger.warning(f"[{call_id}] Not enough players ({len(player_list)}) for spatial competition")
                # Return basic profits for single player case
                return {player_id: max(0, (price - constants.SALOP_MARGINAL_COST) * constants.SALOP_MARKET_SIZE - constants.SALOP_FIXED_COST) 
                       for player_id, price in prices.items()}
            
            player_positions = {player_id: i for i, player_id in enumerate(player_list)}
            
            # Calculate profits for each firm using Salop model
            for player_id, price in prices.items():
                player_pos = player_positions[player_id]
                
                # Get neighbor positions with circular topology
                left_neighbor_pos = (player_pos - 1) % len(player_list)
                right_neighbor_pos = (player_pos + 1) % len(player_list)
                
                # Get neighbor player IDs - use bounds checking
                if left_neighbor_pos >= len(player_list) or right_neighbor_pos >= len(player_list):
                    self.logger.error(f"[{call_id}] Index out of bounds: pos={player_pos}, left={left_neighbor_pos}, right={right_neighbor_pos}, players={len(player_list)}")
                    # Use the player's own price as neighbor prices for safety
                    left_neighbor_id = player_id
                    right_neighbor_id = player_id
                else:
                    left_neighbor_id = player_list[left_neighbor_pos]
                    right_neighbor_id = player_list[right_neighbor_pos]
                
                left_price = prices.get(left_neighbor_id, price)
                right_price = prices.get(right_neighbor_id, price)
                
                # Calculate market boundaries using Salop indifference conditions
                left_boundary, right_boundary = self._calculate_market_boundaries(
                    price, left_price, right_price, constants, n
                )
                
                # Total market share is sum of left and right boundaries
                market_share = left_boundary + right_boundary
                market_shares[player_id] = market_share
                
                # Quantity sold = market share × total market size
                quantity_sold = market_share * constants.SALOP_MARKET_SIZE
                
                # Profit calculation with fixed costs
                profit = ((price - constants.SALOP_MARGINAL_COST) * quantity_sold - 
                         constants.SALOP_FIXED_COST)
                
                profits[player_id] = max(0, profit)  # Firms can't have negative profits (exit option)
                
                self.logger.debug(f"[{call_id}] {player_id}: price=${price:.2f}, share={market_share:.3f}, qty={quantity_sold:.1f}, profit=${profit:.2f}")
            
            # Store market analysis for debugging
            if game_state:
                market_analysis = {
                    'prices': prices.copy(),
                    'market_shares': market_shares.copy(),
                    'total_industry_profit': sum(profits.values()),
                    'price_dispersion': np.std(list(prices.values())),
                    'herfindahl_index': sum(share**2 for share in market_shares.values())
                }
                game_state['salop_market_analysis'] = market_analysis
            
            # Log market outcome
            avg_price = np.mean(list(prices.values()))
            total_profit = sum(profits.values())
            self.logger.info(f"[{call_id}] Salop market: Avg price=${avg_price:.2f}, Total profit=${total_profit:.2f}, Price std=${np.std(list(prices.values())):.2f}")
            
            return profits
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error calculating Salop payoffs: {e}")
            # Return safe default profits
            constants = GameConstants(config)
            default_profit = (constants.SALOP_MARGINAL_COST + 4 - constants.SALOP_MARGINAL_COST) * (constants.SALOP_MARKET_SIZE / config.number_of_players) - constants.SALOP_FIXED_COST
            return {player_id: max(0, default_profit) for player_id in actions.keys()}
    
    # Helper methods for enhanced Salop functionality
    
    def _parse_salop_patterns(self, response: str, player_id: str, call_id: str, config: GameConfig) -> Optional[Dict[str, Any]]:
        """Parse Salop-specific pricing patterns"""
        try:
            response_lower = response.lower()
            
            # Salop-specific price patterns
            price_patterns = [
                r'price["\']?\s*[:=]\s*\$?(\d+(?:\.\d+)?)',
                r'charge\s+\$?(\d+(?:\.\d+)?)',
                r'set.*price.*\$?(\d+(?:\.\d+)?)',
                r'my price is \$?(\d+(?:\.\d+)?)',
                r'i will price at \$?(\d+(?:\.\d+)?)',
                r'\$(\d+(?:\.\d+)?)\s*per\s*unit'
            ]
            
            for pattern in price_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    price = float(match.group(1))
                    validated_price = self._validate_price(price, player_id, call_id, config)
                    
                    # Extract reasoning
                    reasoning = self._extract_reasoning_around_match(response, match)
                    
                    return {
                        'price': validated_price,
                        'reasoning': reasoning,
                        'raw_response': response[:300],
                        'parsing_method': 'salop_pattern'
                    }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"[{call_id}] Salop pattern parsing failed for {player_id}: {e}")
            return None
    
    def _enhance_intelligent_parsing(self, parsed_action: Dict[str, Any], response_lower: str) -> Optional[Dict[str, Any]]:
        """Enhance intelligent parsing with Salop-specific price extraction"""
        try:
            decision_text = parsed_action.get('decision_text', '')
            
            # Look for price indicators in decision text
            price_indicators = [
                r'\$?(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*dollars?',
                r'price.*?(\d+(?:\.\d+)?)'
            ]
            
            for pattern in price_indicators:
                match = re.search(pattern, decision_text)
                if match:
                    try:
                        price = float(match.group(1))
                        if 5 <= price <= 25:  # Reasonable price range for Salop
                            parsed_action['price'] = price
                            return parsed_action
                    except ValueError:
                        continue
            
            return None
            
        except Exception:
            return None
    
    def _enhance_salop_action(self, action: Dict[str, Any], player_id: str, 
                            game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Enhance parsed action with Salop-specific analysis"""
        try:
            constants = GameConstants(config)
            price = action['price']
            
            # Add Salop-specific metadata
            action.update({
                'markup': price - constants.SALOP_MARGINAL_COST,
                'markup_percentage': (price - constants.SALOP_MARGINAL_COST) / constants.SALOP_MARGINAL_COST * 100,
                'theoretical_equilibrium_price': self._calculate_theoretical_equilibrium_price(config),
                'pricing_strategy': self._classify_pricing_strategy(price, config),
                'neighbors': self._get_neighbors(player_id, config.number_of_players),
                'expected_market_share': 1 / config.number_of_players,  # Symmetric approximation
                'break_even_quantity': constants.SALOP_FIXED_COST / (price - constants.SALOP_MARGINAL_COST) if price > constants.SALOP_MARGINAL_COST else float('inf')
            })
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error enhancing Salop action: {e}")
            return action
    
    def _validate_price(self, price: float, player_id: str, call_id: str, config: GameConfig) -> float:
        """Validate and constrain price values for Salop model"""
        try:
            constants = GameConstants(config)
            price = float(price)
            
            # Apply Salop-specific bounds
            min_price = constants.SALOP_MARGINAL_COST
            max_price = constants.SALOP_MARGINAL_COST + 15  # Reasonable upper bound
            
            if price < min_price:
                self.logger.warning(f"[{call_id}] {player_id} price ${price:.2f} below MC, setting to ${min_price:.2f}")
                return min_price
            elif price > max_price:
                self.logger.warning(f"[{call_id}] {player_id} price ${price:.2f} too high, setting to ${max_price:.2f}")
                return max_price
            
            return price
            
        except (ValueError, TypeError):
            constants = GameConstants(config)
            default_price = constants.SALOP_MARGINAL_COST + 4
            self.logger.warning(f"[{call_id}] Invalid price for {player_id}, using ${default_price:.2f}")
            return default_price
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        """Safely extract numeric value from action"""
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _get_neighbors(self, player_id: str, total_players: int) -> Tuple[str, str]:
        """Get circular neighbors for spatial competition"""
        try:
            # For now, return generic neighbor names since we don't have the full player list here
            # The actual neighbor calculation is done in calculate_payoffs where we have all players
            return "neighbor_left", "neighbor_right"
        except:
            return "neighbor_left", "neighbor_right"
    
    def _get_neighbor_id(self, neighbor_num: int) -> str:
        """Convert neighbor number to consistent ID format"""
        return f"player_{neighbor_num}" if neighbor_num <= 10 else str(neighbor_num)
    
    def _calculate_market_boundaries(self, price: float, left_price: float, right_price: float,
                                   constants: GameConstants, n: int) -> Tuple[float, float]:
        """Calculate market boundaries using Salop indifference conditions"""
        try:
            # Base market share per firm in symmetric case
            base_share = 1 / n
            
            # Left boundary calculation
            if left_price == price:
                left_boundary = base_share / 2
            else:
                # Indifference point where consumers switch
                left_distance = (left_price - price) / (2 * constants.SALOP_TRANSPORT_COST)
                left_boundary = max(0, min(base_share, base_share/2 + left_distance))
            
            # Right boundary calculation  
            if right_price == price:
                right_boundary = base_share / 2
            else:
                right_distance = (right_price - price) / (2 * constants.SALOP_TRANSPORT_COST)
                right_boundary = max(0, min(base_share, base_share/2 + right_distance))
            
            return left_boundary, right_boundary
            
        except Exception:
            # Fallback to symmetric allocation
            return 1/(2*n), 1/(2*n)
    
    def _calculate_theoretical_equilibrium_price(self, config: GameConfig) -> float:
        """Calculate theoretical Salop equilibrium price"""
        constants = GameConstants(config)
        # Standard Salop formula: P* = MC + t/n where t is transport cost per unit distance
        return constants.SALOP_MARGINAL_COST + (constants.SALOP_TRANSPORT_COST / config.number_of_players)
    
    def _calculate_theoretical_equilibrium_profit(self, config: GameConfig) -> float:
        """Calculate theoretical Salop equilibrium profit per firm"""
        constants = GameConstants(config)
        equilibrium_price = self._calculate_theoretical_equilibrium_price(config)
        equilibrium_quantity = constants.SALOP_MARKET_SIZE / config.number_of_players
        return (equilibrium_price - constants.SALOP_MARGINAL_COST) * equilibrium_quantity - constants.SALOP_FIXED_COST
    
    def _classify_pricing_strategy(self, price: float, config: GameConfig) -> str:
        """Classify pricing strategy relative to theoretical equilibrium"""
        constants = GameConstants(config)
        equilibrium_price = self._calculate_theoretical_equilibrium_price(config)
        
        if price < equilibrium_price * 0.9:
            return "Aggressive (Below Equilibrium)"
        elif price > equilibrium_price * 1.1:
            return "Premium (Above Equilibrium)"
        else:
            return "Equilibrium (Near Theoretical)"
    
    def _calculate_spatial_statistics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Calculate spatial market statistics"""
        try:
            salop_analysis = game_state.get('salop_market_analysis', {})
            
            if not salop_analysis:
                return {
                    'market_structure': f"Circular market with {config.number_of_players} firms",
                    'competition_intensity': 'Standard',
                    'theoretical_equilibrium': self._calculate_theoretical_equilibrium_price(config)
                }
            
            prices = salop_analysis.get('prices', {})
            
            return {
                'average_market_price': np.mean(list(prices.values())) if prices else 0,
                'price_dispersion': np.std(list(prices.values())) if len(prices) > 1 else 0,
                'market_structure': f"Circular market with {config.number_of_players} firms",
                'herfindahl_index': salop_analysis.get('herfindahl_index', 1/config.number_of_players),
                'theoretical_equilibrium': self._calculate_theoretical_equilibrium_price(config)
            }
            
        except Exception:
            return {'error': 'Unable to calculate spatial statistics'}
    
    def _analyze_competitor_positioning(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, str]:
        """Analyze competitor positioning and market context"""
        try:
            neighbors = self._get_neighbors(player_id, config.number_of_players)
            constants = GameConstants(config)
            
            market_context = f"""**Your Competitive Position Analysis:**
- **Direct Competitors**: Your immediate neighbors (Firms {neighbors[0]} and {neighbors[1]}) pose the greatest competitive threat
- **Market Density**: With {config.number_of_players} firms on the circle, each firm naturally serves ~{constants.SALOP_MARKET_SIZE // config.number_of_players:,} customers in equilibrium
- **Spatial Protection**: Transportation costs of ${constants.SALOP_TRANSPORT_COST:.2f} per distance unit provide some insulation from distant competitors
- **Competition Intensity**: {"High - many nearby competitors" if config.number_of_players >= 5 else "Moderate - balanced competition" if config.number_of_players >= 3 else "Low - few competitors"}"""
            
            return {
                'neighbors': neighbors,
                'market_context': market_context
            }
            
        except Exception:
            return {
                'neighbors': ('unknown', 'unknown'),
                'market_context': 'Standard circular market competition'
            }
    
    def _extract_reasoning_around_match(self, response: str, match) -> str:
        """Extract reasoning text around a regex match"""
        try:
            match_pos = match.start()
            
            # Look for reasoning before the match
            before_text = response[:match_pos].strip()
            reasoning_start = max(
                before_text.rfind('.') + 1,
                before_text.rfind('!') + 1,  
                before_text.rfind('?') + 1,
                0
            )
            
            # Look for reasoning after the match
            after_text = response[match.end():].strip()
            reasoning_end = after_text.find('.')
            if reasoning_end == -1:
                reasoning_end = min(100, len(after_text))
            
            reasoning = (before_text[reasoning_start:] + " " + match.group(0) + " " + after_text[:reasoning_end]).strip()
            
            return reasoning if reasoning else "Price decision based on competitive analysis"
            
        except Exception:
            return "Salop spatial competition pricing"
    
    def _calculate_strategic_metrics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Calculate Salop-specific strategic metrics"""
        try:
            salop_analysis = game_state.get('salop_market_analysis', {})
            
            if not salop_analysis:
                return {'status': 'no_analysis_data'}
            
            prices = salop_analysis.get('prices', {})
            market_shares = salop_analysis.get('market_shares', {})
            
            if not prices:
                return {'status': 'no_price_data'}
            
            constants = GameConstants(config)
            theoretical_price = self._calculate_theoretical_equilibrium_price(config)
            theoretical_profit = self._calculate_theoretical_equilibrium_profit(config)
            
            # Calculate spatial competition metrics
            metrics = {
                'price_competition': {
                    'average_price': np.mean(list(prices.values())),
                    'price_dispersion': np.std(list(prices.values())),
                    'min_price': min(prices.values()),
                    'max_price': max(prices.values()),
                    'deviation_from_theory': abs(np.mean(list(prices.values())) - theoretical_price)
                },
                'market_structure': {
                    'herfindahl_index': sum(share**2 for share in market_shares.values()),
                    'market_concentration': max(market_shares.values()) if market_shares else 0,
                    'market_symmetry': 1 - np.std(list(market_shares.values())) if len(market_shares) > 1 else 1
                },
                'spatial_equilibrium': {
                    'theoretical_price': theoretical_price,
                    'theoretical_profit': theoretical_profit,
                    'actual_total_profit': salop_analysis.get('total_industry_profit', 0),
                    'efficiency_ratio': salop_analysis.get('total_industry_profit', 0) / (theoretical_profit * config.number_of_players) if theoretical_profit > 0 else 0
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating Salop strategic metrics: {e}")
            return {'error': str(e)}