import numpy as np
import json
import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import asdict

from games.base_game import DynamicGame
from config import GameConfig, GameConstants, PlayerResult

class GreenPorterGame(DynamicGame):
    """
    Enhanced Green Porter Dynamic Oligopoly Game with JSON configuration and comprehensive debugging.
    
    This implementation models a dynamic oligopoly where firms compete on quantities over multiple periods,
    with demand uncertainty that creates strategic challenges in maintaining collusive equilibria.
    """
    
    def __init__(self):
        super().__init__("Green Porter Dynamic Oligopoly", default_rounds=25)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Game-specific tracking
        self.demand_shocks = []  # Track demand shocks for analysis
        self.price_volatility = []  # Track market price volatility
        self.quantity_history = {}  # Detailed quantity tracking per player
        
        self.logger.info("Green Porter game initialized with enhanced debugging")
    
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        """
        Create enhanced strategic prompt for Green Porter oligopoly with comprehensive context.
        """
        try:
            constants = GameConstants(config)
            history = game_state.get('price_history', [])
            player_history = game_state.get('player_histories', {}).get(player_id, {})
            current_round = game_state.get('current_round', 1)
            call_id = game_state.get('call_id', 'unknown')
            
            # Calculate market statistics for enhanced context
            market_stats = self._calculate_market_statistics(game_state, config)
            player_stats = self._calculate_player_statistics(player_id, game_state)
            
            self.logger.debug(f"[{call_id}] Creating prompt for {player_id}, round {current_round}")
            self.logger.debug(f"[{call_id}] Market stats: {market_stats}")
            
            # Enhanced strategic context with thinking support
            prompt = f"""**STRATEGIC CONTEXT: Green Porter Dynamic Oligopoly Game**

You are Firm {player_id} in a {config.number_of_players}-firm oligopoly producing a homogeneous product over {config.number_of_rounds} periods. This market presents one of the most challenging strategic environments in industrial organization - maintaining cooperation under demand uncertainty.

**MARKET DYNAMICS & ECONOMICS:**
- **Market Price Formula**: P = ${constants.GP_DEMAND_INTERCEPT} - (Total Industry Quantity) + Demand Shock
- **Demand Uncertainty**: Each period, demand shock ~ Normal(0, ${constants.GP_DEMAND_SHOCK_STD})
- **Your Marginal Cost**: ${constants.GP_MARGINAL_COST} per unit (constant across periods)
- **Discount Factor**: {config.discount_factor} (future profits worth {int(config.discount_factor*100)}% of current profits)
- **Profit Formula**: (Market Price - ${constants.GP_MARGINAL_COST}) × Your Quantity

**THE STRATEGIC DILEMMA:**
This market embodies the classic "Green Porter problem": when market prices fall, is it due to:
1. **Competitor Defection**: Rivals increasing output to gain market share
2. **Demand Shock**: Random negative demand shock reducing market price
3. **Both**: Combination making detection of cheating extremely difficult

This uncertainty creates a fundamental tension between:
- **Cooperative Strategy**: Restrict output to maintain high prices and joint profits
- **Competitive Pressure**: Temptation to increase output for individual gain
- **Punishment Dilemma**: Whether to trigger price wars in response to low prices

**CURRENT MARKET INFORMATION:**
- **Period**: {current_round} of {config.number_of_rounds}
- **Recent Market Prices**: {history[-5:] if len(history) >= 5 else history}
- **Your Recent Quantities**: {player_history.get('quantities', [])[-5:]}
- **Your Recent Profits**: {[round(p, 2) for p in player_history.get('profits', [])[-5:]]}

**MARKET STATISTICS:**
- **Average Market Price**: ${market_stats['avg_price']:.2f} (last {min(5, len(history))} periods)
- **Price Volatility**: ${market_stats['price_std']:.2f} (standard deviation)
- **Market Trend**: {market_stats['price_trend']} 
- **Total Industry Output**: {market_stats['total_quantity']:.1f} units (last period)

**YOUR PERFORMANCE:**
- **Your Market Share**: {player_stats['market_share']:.1%}
- **Average Quantity**: {player_stats['avg_quantity']:.1f} units
- **Cumulative Profit**: ${player_stats['total_profit']:.2f}
- **Profit Trend**: {player_stats['profit_trend']}

**STRATEGIC CONSIDERATIONS:**
Consider these critical factors in your decision:

1. **Market Discipline**: Higher quantities increase current profits but may signal aggressive competition, potentially triggering industry-wide competitive responses.

2. **Information Asymmetry**: Low market prices could indicate competitor cheating OR negative demand shocks. Your response should account for this uncertainty.

3. **Reputation Effects**: Your quantity choices build your reputation as either a "cooperative" or "aggressive" player, affecting future competitor behavior.

4. **End Game Effects**: With {config.number_of_rounds - current_round + 1} periods remaining, consider how time horizon affects cooperation incentives.

5. **Trigger Strategies**: Industry participants often use "grim trigger" or "tit-for-tat" strategies where detected cheating leads to permanent or temporary punishment phases.

**TYPICAL STRATEGIC APPROACHES:**
- **Collusive Quantity**: ~{self._suggest_collusive_quantity(config):.1f} units (maximize joint profits)
- **Cournot Quantity**: ~{self._suggest_cournot_quantity(config):.1f} units (Nash equilibrium)
- **Competitive Quantity**: ~{self._suggest_competitive_quantity(config):.1f} units (marginal cost pricing)

**DECISION FRAMEWORK:**
Think through these steps:
1. **Interpret Recent Market Performance**: What do recent prices suggest about competitor behavior?
2. **Assess Cooperation Level**: Is the market currently in a cooperative or competitive phase?
3. **Consider Strategic Response**: Should you reward cooperation, punish defection, or test the market?
4. **Plan Future Implications**: How will your choice affect future market dynamics?

**OUTPUT REQUIREMENTS:**
Provide your decision in the following JSON format with detailed reasoning:

{{"quantity": <number between 0 and 100>, "reasoning": "<detailed explanation of your strategic thinking, market interpretation, and decision rationale>"}}

**QUANTITY GUIDELINES:**
- Minimum: 0 units (market exit)
- Maximum: 100 units (aggressive expansion)
- Typical Range: 15-40 units depending on strategy
- Consider: Higher quantities = higher current profit but may trigger competitive responses

Your quantity decision will directly impact:
1. This period's market price and your profit
2. Competitors' interpretations of market conditions
3. Future market dynamics and cooperation levels
4. Your long-term strategic reputation

Choose wisely, balancing immediate gains against long-term market health and competitive relationships."""

            # Log prompt statistics
            self.logger.debug(f"[{call_id}] Generated prompt for {player_id}: {len(prompt)} chars")
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error creating prompt for {player_id}: {e}")
            # Return simplified fallback prompt
            return f"""You are Firm {player_id} in round {game_state.get('current_round', 1)} of {config.number_of_rounds}. 
            Choose your quantity (0-50 units). Previous market prices: {game_state.get('price_history', [])[-3:]}.
            Format: {{"quantity": <number>, "reasoning": "<explanation>"}}"""
    
    def parse_action(self, response: str, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Enhanced action parsing with comprehensive error handling and debugging.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            self.logger.debug(f"[{call_id}] Parsing action for {player_id}")
            self.logger.debug(f"[{call_id}] Raw response: {response[:200]}...")
            
            # Try JSON parsing first
            action = self._parse_json_response(response, player_id, call_id)
            
            if action:
                quantity = action.get('quantity', 25.0)
                reasoning = action.get('reasoning', 'No reasoning provided')
                
                # Validate quantity
                validated_quantity = self._validate_quantity(quantity, player_id, call_id)
                
                parsed_action = {
                    'quantity': validated_quantity,
                    'reasoning': reasoning,
                    'raw_response': response[:500],  # Store for debugging
                    'parsing_method': 'json',
                    'player_id': player_id,
                    'round': game_state.get('current_round', 1)
                }
                
                self.logger.debug(f"[{call_id}] Successfully parsed JSON action for {player_id}: quantity={validated_quantity}")
                return parsed_action
            
            # Fallback to pattern matching
            action = self._parse_pattern_response(response, player_id, call_id)
            if action:
                return action
            
            # Final fallback to default
            self.logger.warning(f"[{call_id}] Could not parse response for {player_id}, using default")
            return self.get_default_action(player_id, game_state, config)
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error parsing action for {player_id}: {e}")
            return self.get_default_action(player_id, game_state, config)
    
    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Provide default action when parsing fails"""
        call_id = game_state.get('call_id', 'unknown')
        
        # Use Cournot quantity as reasonable default
        default_quantity = self._suggest_cournot_quantity(config)
        
        self.logger.warning(f"[{call_id}] Using default action for {player_id}: quantity={default_quantity}")
        
        return {
            'quantity': default_quantity,
            'reasoning': f'Default action due to parsing failure. Using Cournot-style quantity.',
            'raw_response': 'PARSING_FAILED',
            'parsing_method': 'default',
            'player_id': player_id,
            'round': game_state.get('current_round', 1)
        }
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Tuple[Dict[str, float], float]:
        """
        Enhanced payoff calculation with comprehensive logging and analysis.
        """
        call_id = game_state.get('call_id', 'unknown') if game_state else 'unknown'
        
        try:
            constants = GameConstants(config)
            
            # Extract quantities with validation
            quantities = {}
            for player_id, action in actions.items():
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                quantities[player_id] = quantity
                
                self.logger.debug(f"[{call_id}] {player_id} quantity: {quantity}")
            
            # Calculate market dynamics
            total_quantity = sum(quantities.values())
            demand_shock = np.random.normal(0, constants.GP_DEMAND_SHOCK_STD)
            market_price = max(0, constants.GP_DEMAND_INTERCEPT - total_quantity + demand_shock)
            
            # Store demand shock for analysis
            self.demand_shocks.append(demand_shock)
            
            # Calculate individual profits
            profits = {}
            for player_id, quantity in quantities.items():
                profit = max(0, (market_price - constants.GP_MARGINAL_COST) * quantity)
                profits[player_id] = profit
                
                self.logger.debug(f"[{call_id}] {player_id} profit: ${profit:.2f}")
            
            # Log market outcome
            self.logger.info(f"[{call_id}] Market outcome: Price=${market_price:.2f}, Total Q={total_quantity:.1f}, Shock={demand_shock:.2f}")
            
            # Update quantity history
            for player_id, quantity in quantities.items():
                if player_id not in self.quantity_history:
                    self.quantity_history[player_id] = []
                self.quantity_history[player_id].append(quantity)
            
            return profits, market_price
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error calculating payoffs: {e}")
            # Return safe default payoffs
            default_profits = {player_id: 0.0 for player_id in actions.keys()}
            return default_profits, 0.0
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], config: GameConfig) -> Dict:
        """
        Enhanced game state update with comprehensive tracking and analysis.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            # Initialize state components if needed
            if 'price_history' not in game_state:
                game_state['price_history'] = []
            if 'player_histories' not in game_state:
                game_state['player_histories'] = {}
            if 'market_analysis' not in game_state:
                game_state['market_analysis'] = {}
            
            # Calculate payoffs and market price
            profits, market_price = self.calculate_payoffs(actions, config, game_state)
            
            # Update price history
            game_state['price_history'].append(market_price)
            
            # Update player histories with detailed tracking
            for player_id, action in actions.items():
                if player_id not in game_state['player_histories']:
                    game_state['player_histories'][player_id] = {
                        'quantities': [],
                        'profits': [],
                        'reasoning': [],
                        'cumulative_profit': 0.0
                    }
                
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                profit = profits[player_id]
                reasoning = action.get('reasoning', 'No reasoning provided')
                
                # Update histories
                player_hist = game_state['player_histories'][player_id]
                player_hist['quantities'].append(quantity)
                player_hist['profits'].append(profit)
                player_hist['reasoning'].append(reasoning)
                player_hist['cumulative_profit'] += profit
                
                self.logger.debug(f"[{call_id}] Updated {player_id}: Q={quantity}, π=${profit:.2f}, Total=${player_hist['cumulative_profit']:.2f}")
            
            # Update market analysis
            game_state['market_analysis'] = self._analyze_market_dynamics(game_state, config)
            
            # Calculate price volatility
            if len(game_state['price_history']) >= 2:
                recent_prices = game_state['price_history'][-10:]  # Last 10 periods
                price_volatility = np.std(recent_prices) if len(recent_prices) > 1 else 0
                self.price_volatility.append(price_volatility)
            
            self.logger.debug(f"[{call_id}] Game state updated successfully")
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error updating game state: {e}")
            return game_state  # Return unchanged state on error
    
    def calculate_final_results(self, game_state: Dict, config: GameConfig) -> List[PlayerResult]:
        """
        Calculate final results with enhanced analysis and comprehensive metrics.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            self.logger.info(f"[{call_id}] Calculating final results for Green Porter game")
            
            player_results = []
            player_histories = game_state.get('player_histories', {})
            
            if not player_histories:
                self.logger.warning(f"[{call_id}] No player histories found")
                return []
            
            # Calculate discounted profits and comprehensive metrics
            for player_id, history in player_histories.items():
                profits = history.get('profits', [])
                quantities = history.get('quantities', [])
                
                # Calculate NPV with proper discounting
                discounted_profit = 0.0
                for round_num, profit in enumerate(profits):
                    discounted_profit += profit * (config.discount_factor ** round_num)
                
                # Enhanced player metrics
                player_metrics = {
                    'total_profit': discounted_profit,
                    'average_quantity': np.mean(quantities) if quantities else 0,
                    'quantity_volatility': np.std(quantities) if len(quantities) > 1 else 0,
                    'market_share': np.mean(quantities) / sum(np.mean(h.get('quantities', [0])) 
                                                            for h in player_histories.values()) if quantities else 0,
                    'profit_per_unit': discounted_profit / sum(quantities) if sum(quantities) > 0 else 0,
                    'strategic_consistency': self._calculate_strategic_consistency(quantities)
                }
                
                player_result = PlayerResult(
                    player_id=player_id,
                    profit=discounted_profit,
                    actions=[{
                        'quantity': q,
                        'reasoning': r,
                        'profit': p,
                        'round': i + 1
                    } for i, (q, r, p) in enumerate(zip(
                        quantities,
                        history.get('reasoning', [''] * len(quantities)),
                        profits
                    ))],
                    win=False  # Will be set after comparison
                )
                
                # Add enhanced metrics as additional attributes
                player_result.additional_metrics = player_metrics
                player_results.append(player_result)
                
                self.logger.debug(f"[{call_id}] {player_id} final profit: ${discounted_profit:.2f}")
            
            # Determine winners (highest profit)
            if player_results:
                max_profit = max(pr.profit for pr in player_results)
                for pr in player_results:
                    pr.win = (abs(pr.profit - max_profit) < 0.01)
                
                # Log final standings
                sorted_results = sorted(player_results, key=lambda x: x.profit, reverse=True)
                self.logger.info(f"[{call_id}] Final standings:")
                for i, pr in enumerate(sorted_results):
                    winner_status = "WINNER" if pr.win else f"#{i+1}"
                    self.logger.info(f"[{call_id}]   {winner_status}: {pr.player_id} - ${pr.profit:.2f}")
            
            return player_results
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error calculating final results: {e}")
            return []
    
    # Helper methods for enhanced functionality
    
    def _parse_json_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse JSON-formatted response"""
        try:
            # Look for JSON object in the response
            json_match = re.search(r'\{[^{}]*"quantity"[^{}]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            
            # Try parsing entire response as JSON
            return json.loads(response)
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            self.logger.debug(f"[{call_id}] JSON parsing failed for {player_id}: {e}")
            return None
    
    def _parse_pattern_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse response using pattern matching"""
        try:
            # Look for quantity patterns
            quantity_patterns = [
                r'quantity["\']?\s*[:=]\s*(\d+(?:\.\d+)?)',
                r'produce\s+(\d+(?:\.\d+)?)\s*units',
                r'choose\s+(\d+(?:\.\d+)?)',
                r'quantity\s+of\s+(\d+(?:\.\d+)?)'
            ]
            
            for pattern in quantity_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    quantity = float(match.group(1))
                    
                    # Extract reasoning (first sentence or paragraph)
                    reasoning_match = re.search(r'[.!?]\s*([^.!?]+)', response)
                    reasoning = reasoning_match.group(1) if reasoning_match else "Pattern-matched quantity"
                    
                    return {
                        'quantity': quantity,
                        'reasoning': reasoning,
                        'raw_response': response[:200],
                        'parsing_method': 'pattern'
                    }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"[{call_id}] Pattern parsing failed for {player_id}: {e}")
            return None
    
    def _validate_quantity(self, quantity: float, player_id: str, call_id: str) -> float:
        """Validate and constrain quantity values"""
        try:
            quantity = float(quantity)
            
            # Apply reasonable bounds
            if quantity < 0:
                self.logger.warning(f"[{call_id}] {player_id} quantity {quantity} below 0, setting to 0")
                return 0.0
            elif quantity > 100:
                self.logger.warning(f"[{call_id}] {player_id} quantity {quantity} above 100, setting to 100")
                return 100.0
            
            return quantity
            
        except (ValueError, TypeError):
            self.logger.warning(f"[{call_id}] Invalid quantity for {player_id}: {quantity}, using default 25")
            return 25.0
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        """Safely extract numeric value from action"""
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_market_statistics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Calculate comprehensive market statistics"""
        price_history = game_state.get('price_history', [])
        
        if not price_history:
            return {
                'avg_price': 0.0,
                'price_std': 0.0,
                'price_trend': 'Unknown',
                'total_quantity': 0.0
            }
        
        recent_prices = price_history[-5:]  # Last 5 periods
        avg_price = np.mean(recent_prices)
        price_std = np.std(recent_prices) if len(recent_prices) > 1 else 0.0
        
        # Determine trend
        if len(recent_prices) >= 3:
            if recent_prices[-1] > recent_prices[-2] > recent_prices[-3]:
                trend = "Rising"
            elif recent_prices[-1] < recent_prices[-2] < recent_prices[-3]:
                trend = "Falling"
            else:
                trend = "Stable/Mixed"
        else:
            trend = "Insufficient data"
        
        # Calculate total quantity from last round
        total_quantity = 0.0
        player_histories = game_state.get('player_histories', {})
        for player_id, history in player_histories.items():
            quantities = history.get('quantities', [])
            if quantities:
                total_quantity += quantities[-1]
        
        return {
            'avg_price': avg_price,
            'price_std': price_std,
            'price_trend': trend,
            'total_quantity': total_quantity
        }
    
    def _calculate_player_statistics(self, player_id: str, game_state: Dict) -> Dict[str, Any]:
        """Calculate player-specific statistics"""
        player_histories = game_state.get('player_histories', {})
        player_history = player_histories.get(player_id, {})
        
        quantities = player_history.get('quantities', [])
        profits = player_history.get('profits', [])
        
        if not quantities:
            return {
                'market_share': 0.0,
                'avg_quantity': 0.0,
                'total_profit': 0.0,
                'profit_trend': 'No data'
            }
        
        # Calculate market share
        total_market_quantity = 0.0
        for pid, hist in player_histories.items():
            if hist.get('quantities'):
                total_market_quantity += hist['quantities'][-1]
        
        market_share = quantities[-1] / total_market_quantity if total_market_quantity > 0 else 0.0
        
        # Profit trend
        if len(profits) >= 3:
            if profits[-1] > profits[-2] > profits[-3]:
                profit_trend = "Improving"
            elif profits[-1] < profits[-2] < profits[-3]:
                profit_trend = "Declining"
            else:
                profit_trend = "Mixed"
        else:
            profit_trend = "Insufficient data"
        
        return {
            'market_share': market_share,
            'avg_quantity': np.mean(quantities),
            'total_profit': sum(profits),
            'profit_trend': profit_trend
        }
    
    def _analyze_market_dynamics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Analyze market dynamics for strategic insights"""
        try:
            price_history = game_state.get('price_history', [])
            
            if len(price_history) < 3:
                return {'status': 'insufficient_data'}
            
            recent_prices = price_history[-5:]
            price_volatility = np.std(recent_prices)
            
            # Detect market regime
            constants = GameConstants(config)
            theoretical_collusive_price = constants.GP_DEMAND_INTERCEPT - (constants.GP_DEMAND_INTERCEPT - constants.GP_MARGINAL_COST) / 4
            theoretical_competitive_price = constants.GP_MARGINAL_COST
            
            avg_recent_price = np.mean(recent_prices)
            
            if avg_recent_price >= theoretical_collusive_price * 0.9:
                market_regime = "Highly Cooperative"
            elif avg_recent_price >= (theoretical_collusive_price + theoretical_competitive_price) / 2:
                market_regime = "Moderately Cooperative"
            else:
                market_regime = "Competitive"
            
            return {
                'market_regime': market_regime,
                'price_volatility': price_volatility,
                'avg_recent_price': avg_recent_price,
                'theoretical_collusive_price': theoretical_collusive_price,
                'cooperation_index': min(1.0, avg_recent_price / theoretical_collusive_price)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market dynamics: {e}")
            return {'status': 'analysis_failed'}
    
    def _calculate_strategic_consistency(self, quantities: List[float]) -> float:
        """Calculate how consistent a player's strategy has been"""
        if len(quantities) < 3:
            return 1.0  # Perfect consistency with insufficient data
        
        # Calculate coefficient of variation (std dev / mean)
        mean_q = np.mean(quantities)
        std_q = np.std(quantities)
        
        if mean_q == 0:
            return 0.0
        
        cv = std_q / mean_q
        consistency = max(0.0, 1.0 - cv)  # Convert to consistency score (0-1)
        
        return consistency
    
    def _suggest_collusive_quantity(self, config: GameConfig) -> float:
        """Suggest quantity for collusive strategy"""
        constants = GameConstants(config)
        # Monopoly quantity divided by number of firms
        collusive_total = (constants.GP_DEMAND_INTERCEPT - constants.GP_MARGINAL_COST) / 2
        return collusive_total / config.number_of_players
    
    def _suggest_cournot_quantity(self, config: GameConfig) -> float:
        """Suggest quantity for Cournot Nash equilibrium"""
        constants = GameConstants(config)
        # Standard Cournot quantity
        return (constants.GP_DEMAND_INTERCEPT - constants.GP_MARGINAL_COST) / (config.number_of_players + 1)
    
    def _suggest_competitive_quantity(self, config: GameConfig) -> float:
        """Suggest quantity for competitive strategy"""
        constants = GameConstants(config)
        # Higher than Cournot, closer to competitive
        cournot_q = self._suggest_cournot_quantity(config)
        return cournot_q * 1.5  # 50% higher than Cournot
    
    def get_game_summary(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Generate comprehensive game summary for analysis"""
        try:
            market_analysis = game_state.get('market_analysis', {})
            price_history = game_state.get('price_history', [])
            player_histories = game_state.get('player_histories', {})
            
            summary = {
                'game_name': self.name,
                'total_rounds': len(price_history),
                'final_market_price': price_history[-1] if price_history else 0,
                'average_market_price': np.mean(price_history) if price_history else 0,
                'price_volatility': np.std(price_history) if len(price_history) > 1 else 0,
                'market_regime': market_analysis.get('market_regime', 'Unknown'),
                'cooperation_index': market_analysis.get('cooperation_index', 0),
                'demand_shocks_std': np.std(self.demand_shocks) if self.demand_shocks else 0,
                'player_count': len(player_histories),
                'total_industry_profit': sum(
                    sum(hist.get('profits', [])) 
                    for hist in player_histories.values()
                )
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating game summary: {e}")
            return {'error': str(e)}