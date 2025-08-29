import numpy as np
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict

from games.base_game import StaticGame
from config import GameConfig, GameConstants, PlayerResult

class SpulberGame(StaticGame):
    """
    Enhanced Spulber Bertrand Competition Game with JSON configuration and comprehensive debugging.
    
    This implementation models a winner-take-all auction/Bertrand competition where the lowest-bidding
    firm captures the entire market. Based on Daniel Spulber's models of competitive bidding under
    incomplete information about rival costs.
    """
    
    def __init__(self):
        super().__init__("Spulber Bertrand Competition")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Game-specific tracking
        self.bid_history = []
        self.winner_analysis = []
        self.profit_margins_history = []
        self.competitive_intensity_metrics = []
        
        self.logger.info("Spulber Bertrand competition game initialized with enhanced debugging")
    
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        """
        Create enhanced strategic prompt for Spulber Bertrand competition with comprehensive context.
        """
        try:
            constants = GameConstants(config)
            call_id = game_state.get('call_id', 'unknown')
            
            # Calculate competitive analysis
            competitive_analysis = self._analyze_competitive_environment(game_state, config)
            strategic_guidance = self._generate_strategic_guidance(config)
            
            self.logger.debug(f"[{call_id}] Creating Spulber prompt for {player_id}")
            self.logger.debug(f"[{call_id}] Market value: ${constants.SPULBER_MARKET_VALUE:,}")
            
            # Enhanced strategic context with auction theory
            prompt = f"""**STRATEGIC CONTEXT: Spulber Bertrand Winner-Take-All Competition**

You are Firm {player_id} competing in a high-stakes winner-take-all market where only the LOWEST bidder wins the entire contract worth ${constants.SPULBER_MARKET_VALUE:,}. This is a classic Bertrand competition under incomplete information - you know your own costs but must strategically bid against {config.number_of_players - 1} rivals with unknown cost structures.

**AUCTION MECHANISM & RULES:**
- **Winner-Take-All**: The firm submitting the LOWEST bid wins the entire market revenue of ${constants.SPULBER_MARKET_VALUE:,}
- **All-or-Nothing**: Second-place and below earn ZERO profit (no participation prizes)
- **Single-Round**: All firms submit sealed bids simultaneously - no second chances
- **Information Asymmetry**: You know your costs but competitors' costs are private information

**YOUR COST STRUCTURE:**
- **Your Marginal Cost**: ${constants.SPULBER_MARGINAL_COST:.2f} per unit of production
- **Your Cost Advantage**: Your cost of ${constants.SPULBER_MARGINAL_COST:.2f} is {"BELOW" if constants.SPULBER_MARGINAL_COST < constants.SPULBER_RIVAL_COST_MEAN else "ABOVE" if constants.SPULBER_MARGINAL_COST > constants.SPULBER_RIVAL_COST_MEAN else "EQUAL TO"} the expected rival cost of ${constants.SPULBER_RIVAL_COST_MEAN:.2f}
- **Profit If You Win**: ${constants.SPULBER_MARKET_VALUE:,} - Your Bid = Your Profit
- **Profit If You Lose**: $0 (complete loss)

**COMPETITOR INTELLIGENCE:**
- **Rival Cost Distribution**: Each competitor's marginal cost is independently drawn from Normal(μ=${constants.SPULBER_RIVAL_COST_MEAN:.2f}, σ=${constants.SPULBER_RIVAL_COST_STD:.2f})
- **Rival Cost Range**: Approximately 95% of competitors have costs between ${constants.SPULBER_RIVAL_COST_MEAN - 2*constants.SPULBER_RIVAL_COST_STD:.2f} and ${constants.SPULBER_RIVAL_COST_MEAN + 2*constants.SPULBER_RIVAL_COST_STD:.2f}
- **Number of Rivals**: {config.number_of_players - 1} competing firms (increasing your competition)
- **Strategic Implication**: {"You have a significant cost advantage" if constants.SPULBER_MARGINAL_COST < constants.SPULBER_RIVAL_COST_MEAN - constants.SPULBER_RIVAL_COST_STD else "You have a moderate cost advantage" if constants.SPULBER_MARGINAL_COST < constants.SPULBER_RIVAL_COST_MEAN else "You face cost disadvantage pressure"}

**THE BERTRAND PARADOX & STRATEGIC DYNAMICS:**
This auction embodies the classic Bertrand competition dilemma:
1. **Aggressive Bidding**: Lower bids increase your winning probability but destroy profit margins
2. **Conservative Bidding**: Higher bids preserve margins but risk losing to aggressive competitors  
3. **Information Uncertainty**: You can't observe rival costs, making optimal bidding incredibly challenging
4. **Winner's Curse Risk**: Winning might indicate you bid too low (left money on the table)

**STRATEGIC TRADE-OFF ANALYSIS:**
{strategic_guidance['trade_off_analysis']}

**COMPETITIVE PROBABILITY ASSESSMENT:**
Based on your cost advantage and the competitive landscape:
{competitive_analysis['probability_assessment']}

**BIDDING STRATEGY FRAMEWORK:**
Consider these strategic approaches:

1. **Aggressive Market Capture**:
   - Bid Range: ${constants.SPULBER_MARGINAL_COST + 1:.2f} - ${constants.SPULBER_MARGINAL_COST + 3:.2f}
   - Philosophy: Maximize winning probability, accept lower margins
   - Risk: Potential winner's curse if you're too aggressive
   - Suitable When: You have strong cost advantage and risk tolerance

2. **Balanced Risk-Reward**:
   - Bid Range: ${constants.SPULBER_MARGINAL_COST + 3:.2f} - ${constants.SPULBER_MARGINAL_COST + 6:.2f}
   - Philosophy: Balance winning probability with reasonable profit margins
   - Risk: Moderate chance of losing to very aggressive bidders
   - Suitable When: You want steady expected returns

3. **High-Margin Conservative**:
   - Bid Range: ${constants.SPULBER_MARGINAL_COST + 6:.2f} - ${constants.SPULBER_MARGINAL_COST + 10:.2f}
   - Philosophy: Win only if competitors are inefficient, but win big
   - Risk: Lower winning probability, but excellent margins if you win
   - Suitable When: You prefer high-reward, lower-probability outcomes

**AUCTION THEORY INSIGHTS:**
- **Optimal Bid Theory**: In symmetric auctions, optimal bid = marginal cost + markup based on rival distribution
- **Your Theoretical Advantage**: Expected markup opportunity of ~${max(0, constants.SPULBER_RIVAL_COST_MEAN - constants.SPULBER_MARGINAL_COST):.2f} over average rival
- **Competition Intensity**: With {config.number_of_players - 1} rivals, expect {"intense" if config.number_of_players >= 5 else "moderate" if config.number_of_players >= 3 else "limited"} price pressure

**ECONOMIC DECISION FACTORS:**
Consider these critical elements:

1. **Win Probability Estimation**:
   - How likely are you to have the lowest bid given your cost advantage?
   - What's the probability that all rivals bid above your target price?

2. **Expected Value Calculation**:
   - Expected Profit = (Probability of Winning) × (Market Value - Your Bid)
   - Optimize your bid to maximize this expected value

3. **Risk Tolerance Assessment**:
   - Are you willing to accept lower win probability for higher margins?
   - How much profit do you need to make this venture worthwhile?

4. **Competitive Response Anticipation**:
   - Assume rivals are also optimizing their expected profits
   - Consider that other efficient firms may bid aggressively too

**CURRENT MARKET INTELLIGENCE:**
- **Market Value**: ${constants.SPULBER_MARKET_VALUE:,} (fixed prize)
- **Your Break-Even**: Any bid above ${constants.SPULBER_MARGINAL_COST:.2f} generates profit if you win
- **Competitive Pressure**: {config.number_of_players - 1} rivals creates {"high competitive intensity" if config.number_of_players >= 4 else "moderate pressure"}
- **Expected Rival Range**: Most competitors likely to bid between ${max(constants.SPULBER_MARGINAL_COST, constants.SPULBER_RIVAL_COST_MEAN - constants.SPULBER_RIVAL_COST_STD):.2f} and ${constants.SPULBER_RIVAL_COST_MEAN + 2*constants.SPULBER_RIVAL_COST_STD:.2f}

**DECISION FRAMEWORK:**
Think through these steps:
1. **Cost Advantage Assessment**: How significant is your ${constants.SPULBER_MARGINAL_COST:.2f} cost vs. expected rival costs of ~${constants.SPULBER_RIVAL_COST_MEAN:.2f}?
2. **Risk Preference**: Do you prioritize winning probability or profit margins?
3. **Expected Value Optimization**: What bid maximizes (Win Probability) × (Profit if Win)?
4. **Competitive Intelligence**: How aggressively are efficient rivals likely to bid?

**OUTPUT REQUIREMENTS:**
Provide your decision in the following JSON format with detailed competitive analysis:

{{"bid": <bid amount between {constants.SPULBER_MARGINAL_COST:.2f} and {constants.SPULBER_MARKET_VALUE:.0f}>, "reasoning": "<detailed explanation of your bidding strategy, win probability assessment, expected value calculation, and competitive reasoning>"}}

**BIDDING GUIDELINES:**
- **Minimum Rational Bid**: ${constants.SPULBER_MARGINAL_COST:.2f} (your marginal cost - zero profit)
- **Maximum Possible Bid**: ${constants.SPULBER_MARKET_VALUE:.0f} (entire market value - zero profit if you win)
- **Typical Competitive Range**: ${constants.SPULBER_MARGINAL_COST + 1:.2f} - ${constants.SPULBER_MARGINAL_COST + 8:.2f}
- **Strategic Sweet Spot**: Consider bids that balance win probability with meaningful profit margins

**IMPACT OF YOUR DECISION:**
Your bid directly determines:
1. **Win Probability**: Lower bids increase chances of beating all rivals
2. **Profit Margin**: Higher bids increase profit if you win, but reduce win probability
3. **Expected Return**: The mathematical product of win probability and profit margin
4. **Market Dynamics**: Your aggressiveness influences overall competitive intensity

Choose your bid wisely - in winner-take-all markets, there are no second chances, but the rewards for optimal strategy are substantial."""

            # Log prompt statistics
            self.logger.debug(f"[{call_id}] Generated Spulber prompt: {len(prompt)} chars for player {player_id}")
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error creating Spulber prompt for {player_id}: {e}")
            # Return simplified fallback prompt
            return f"""You are bidding in a winner-take-all auction against {config.number_of_players - 1} rivals. 
            The lowest bid wins ${GameConstants(config).SPULBER_MARKET_VALUE:,}. Your cost is ${GameConstants(config).SPULBER_MARGINAL_COST}. 
            Rival costs are normally distributed around ${GameConstants(config).SPULBER_RIVAL_COST_MEAN}. 
            Choose your bid. Format: {{"bid": <number>, "reasoning": "<explanation>"}}"""
    
    def parse_action(self, response: str, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Enhanced action parsing for Spulber bidding decisions.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            self.logger.debug(f"[{call_id}] Parsing Spulber bidding action for {player_id}")
            
            # Try JSON parsing first
            action = self._parse_json_response(response, player_id, call_id)
            if action and 'bid' in action:
                bid = self._validate_bid(action['bid'], player_id, call_id, config)
                action['bid'] = bid
                action['parsing_method'] = 'json'
                return self._enhance_spulber_action(action, player_id, game_state, config)
            
            # Try pattern-based parsing
            action = self._parse_spulber_patterns(response, player_id, call_id, config)
            if action:
                return self._enhance_spulber_action(action, player_id, game_state, config)
            
            # Use base class intelligent parsing with Spulber enhancement
            action = self._parse_intelligent_response(response, player_id, call_id)
            if action:
                action = self._enhance_intelligent_parsing(action, response.lower())
                if action and 'bid' in action:
                    return self._enhance_spulber_action(action, player_id, game_state, config)
            
            # Fallback to default
            self.logger.warning(f"[{call_id}] All parsing methods failed for {player_id}, using Spulber default")
            return self.get_default_action(player_id, game_state, config)
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error parsing Spulber action for {player_id}: {e}")
            return self.get_default_action(player_id, game_state, config)
    
    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Provide intelligent default bidding action when parsing fails"""
        call_id = game_state.get('call_id', 'unknown')
        
        # Use expected value maximizing bid as reasonable default
        default_bid = self._calculate_expected_value_optimal_bid(config)
        
        self.logger.warning(f"[{call_id}] Using Spulber EV-optimal default bid for {player_id}: ${default_bid:.2f}")
        
        return {
            'bid': default_bid,
            'reasoning': f'Default expected-value-maximizing bid due to parsing failure. Balancing win probability with profit margins.',
            'raw_response': 'PARSING_FAILED',
            'parsing_method': 'spulber_default',
            'player_id': player_id,
            'round': game_state.get('current_round', 1)
        }
    
    def get_required_action_fields(self) -> List[str]:
        """Required fields for Spulber bidding actions"""
        return ['bid', 'reasoning']
    
    def validate_action(self, action: Dict[str, Any], player_id: str, config: GameConfig) -> bool:
        """Enhanced validation for Spulber bidding actions"""
        try:
            # Check required fields
            if 'bid' not in action:
                self.logger.warning(f"Missing 'bid' field in {player_id} action")
                return False
            
            # Validate bid range
            constants = GameConstants(config)
            bid = float(action['bid'])
            
            if bid < constants.SPULBER_MARGINAL_COST:
                self.logger.warning(f"{player_id} bid ${bid:.2f} below marginal cost ${constants.SPULBER_MARGINAL_COST}")
                return False
            
            if bid > constants.SPULBER_MARKET_VALUE:
                self.logger.warning(f"{player_id} bid ${bid:.2f} above market value ${constants.SPULBER_MARKET_VALUE}")
                return False
            
            return True
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid bid format for {player_id}: {e}")
            return False
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig, 
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Enhanced payoff calculation using Spulber winner-take-all auction model with comprehensive logging.
        """
        call_id = game_state.get('call_id', 'unknown') if game_state else 'unknown'
        
        try:
            constants = GameConstants(config)
            
            # Extract and validate bids
            bids = {}
            for player_id, action in actions.items():
                bid = self._safe_get_numeric(action, 'bid', constants.SPULBER_MARGINAL_COST + 5)
                bids[player_id] = bid
                self.logger.debug(f"[{call_id}] {player_id} bid: ${bid:.2f}")
            
            # Determine winner (lowest bid wins)
            min_bid = min(bids.values())
            winners = [player_id for player_id, bid in bids.items() if abs(bid - min_bid) < 0.01]
            
            # Handle ties (split the market equally among winners)
            profits = {}
            market_share_per_winner = 1.0 / len(winners)
            
            for player_id, bid in bids.items():
                if player_id in winners:
                    # Winner gets market value minus their bid, divided by number of winners
                    profit = (constants.SPULBER_MARKET_VALUE - bid) * market_share_per_winner
                    profits[player_id] = max(0, profit)
                    
                    win_status = f"WINNER ({'TIED' if len(winners) > 1 else 'SOLE'})"
                    self.logger.debug(f"[{call_id}] {player_id}: {win_status}, bid=${bid:.2f}, profit=${profit:.2f}")
                else:
                    # Losers get zero profit
                    profits[player_id] = 0.0
                    self.logger.debug(f"[{call_id}] {player_id}: LOSER, bid=${bid:.2f}, profit=$0.00")
            
            # Store auction analysis for debugging
            if game_state:
                auction_analysis = {
                    'bids': bids.copy(),
                    'winner_count': len(winners),
                    'winners': winners,
                    'winning_bid': min_bid,
                    'total_profit_generated': sum(profits.values()),
                    'bid_spread': max(bids.values()) - min(bids.values()),
                    'competitive_intensity': self._calculate_competitive_intensity(bids),
                    'market_efficiency': min_bid / constants.SPULBER_MARKET_VALUE
                }
                game_state['spulber_auction_analysis'] = auction_analysis
            
            # Log auction outcome
            avg_bid = np.mean(list(bids.values()))
            total_profit = sum(profits.values())
            self.logger.info(f"[{call_id}] Spulber auction: Winner(s): {winners}, Winning bid: ${min_bid:.2f}, Avg bid: ${avg_bid:.2f}, Total profit: ${total_profit:.2f}")
            
            return profits
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error calculating Spulber payoffs: {e}")
            # Return safe default profits (no winner)
            return {player_id: 0.0 for player_id in actions.keys()}
    
    # Helper methods for enhanced Spulber functionality
    
    def _parse_spulber_patterns(self, response: str, player_id: str, call_id: str, config: GameConfig) -> Optional[Dict[str, Any]]:
        """Parse Spulber-specific bidding patterns"""
        try:
            response_lower = response.lower()
            
            # Spulber-specific bid patterns
            bid_patterns = [
                r'bid["\']?\s*[:=]\s*\$?(\d+(?:\.\d+)?)',
                r'i bid \$?(\d+(?:\.\d+)?)',
                r'my bid is \$?(\d+(?:\.\d+)?)',
                r'i will bid \$?(\d+(?:\.\d+)?)',
                r'bidding \$?(\d+(?:\.\d+)?)',
                r'offer \$?(\d+(?:\.\d+)?)',
                r'\$(\d+(?:\.\d+)?)\s*(?:bid|offer)'
            ]
            
            for pattern in bid_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    bid = float(match.group(1))
                    validated_bid = self._validate_bid(bid, player_id, call_id, config)
                    
                    # Extract reasoning
                    reasoning = self._extract_reasoning_around_match(response, match)
                    
                    return {
                        'bid': validated_bid,
                        'reasoning': reasoning,
                        'raw_response': response[:300],
                        'parsing_method': 'spulber_pattern'
                    }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"[{call_id}] Spulber pattern parsing failed for {player_id}: {e}")
            return None
    
    def _enhance_intelligent_parsing(self, parsed_action: Dict[str, Any], response_lower: str) -> Optional[Dict[str, Any]]:
        """Enhance intelligent parsing with Spulber-specific bid extraction"""
        try:
            decision_text = parsed_action.get('decision_text', '')
            
            # Look for bid indicators in decision text
            bid_indicators = [
                r'\$?(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*dollars?',
                r'bid.*?(\d+(?:\.\d+)?)',
                r'offer.*?(\d+(?:\.\d+)?)'
            ]
            
            for pattern in bid_indicators:
                match = re.search(pattern, decision_text)
                if match:
                    try:
                        bid = float(match.group(1))
                        if 5 <= bid <= 1000:  # Reasonable bid range for Spulber
                            parsed_action['bid'] = bid
                            return parsed_action
                    except ValueError:
                        continue
            
            return None
            
        except Exception:
            return None
    
    def _enhance_spulber_action(self, action: Dict[str, Any], player_id: str, 
                              game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Enhance parsed action with Spulber-specific analysis"""
        try:
            constants = GameConstants(config)
            bid = action['bid']
            
            # Calculate competitive metrics
            profit_if_win = constants.SPULBER_MARKET_VALUE - bid
            markup = bid - constants.SPULBER_MARGINAL_COST
            margin_percentage = profit_if_win / constants.SPULBER_MARKET_VALUE * 100
            
            # Estimate win probability (simplified model)
            win_probability = self._estimate_win_probability(bid, config)
            expected_profit = win_probability * profit_if_win
            
            # Add Spulber-specific metadata
            action.update({
                'profit_if_win': profit_if_win,
                'markup_over_cost': markup,
                'margin_percentage': margin_percentage,
                'estimated_win_probability': win_probability,
                'expected_profit': expected_profit,
                'bidding_strategy': self._classify_bidding_strategy(bid, config),
                'risk_assessment': self._assess_bid_risk(bid, config),
                'competitive_position': 'Aggressive' if bid < constants.SPULBER_RIVAL_COST_MEAN else 'Conservative',
                'theoretical_optimality': abs(bid - self._calculate_expected_value_optimal_bid(config))
            })
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error enhancing Spulber action: {e}")
            return action
    
    def _validate_bid(self, bid: float, player_id: str, call_id: str, config: GameConfig) -> float:
        """Validate and constrain bid values for Spulber model"""
        try:
            constants = GameConstants(config)
            bid = float(bid)
            
            # Apply Spulber-specific bounds
            min_bid = constants.SPULBER_MARGINAL_COST
            max_bid = constants.SPULBER_MARKET_VALUE
            
            if bid < min_bid:
                self.logger.warning(f"[{call_id}] {player_id} bid ${bid:.2f} below marginal cost, setting to ${min_bid:.2f}")
                return min_bid
            elif bid > max_bid:
                self.logger.warning(f"[{call_id}] {player_id} bid ${bid:.2f} above market value, setting to ${max_bid:.2f}")
                return max_bid
            
            return bid
            
        except (ValueError, TypeError):
            constants = GameConstants(config)
            default_bid = constants.SPULBER_MARGINAL_COST + 5
            self.logger.warning(f"[{call_id}] Invalid bid for {player_id}, using ${default_bid:.2f}")
            return default_bid
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        """Safely extract numeric value from action"""
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_expected_value_optimal_bid(self, config: GameConfig) -> float:
        """Calculate expected value maximizing bid using simplified auction theory"""
        try:
            constants = GameConstants(config)
            
            # Simplified optimal bid calculation
            # In symmetric auctions: optimal_bid ≈ marginal_cost + (expected_rival_cost - marginal_cost) * (n-1)/n
            n = config.number_of_players
            cost_advantage = max(0, constants.SPULBER_RIVAL_COST_MEAN - constants.SPULBER_MARGINAL_COST)
            
            # Apply competitive adjustment
            competitive_multiplier = (n - 1) / n if n > 1 else 0.5
            optimal_markup = cost_advantage * competitive_multiplier
            
            optimal_bid = constants.SPULBER_MARGINAL_COST + optimal_markup
            
            # Ensure reasonable bounds
            max_reasonable = min(constants.SPULBER_MARKET_VALUE * 0.8, constants.SPULBER_MARGINAL_COST + 20)
            
            return min(optimal_bid, max_reasonable)
            
        except Exception:
            constants = GameConstants(config)
            return constants.SPULBER_MARGINAL_COST + 5
    
    def _estimate_win_probability(self, bid: float, config: GameConfig) -> float:
        """Estimate probability of winning with given bid"""
        try:
            constants = GameConstants(config)
            
            # Probability that a single rival bids above our bid
            # Assuming rival bids near their cost with some markup
            rival_cost_mean = constants.SPULBER_RIVAL_COST_MEAN
            rival_cost_std = constants.SPULBER_RIVAL_COST_STD
            
            # Simplified: assume rivals bid at cost + small markup
            rival_markup_mean = 2.0  # Average markup assumption
            rival_markup_std = 1.0
            
            rival_bid_mean = rival_cost_mean + rival_markup_mean
            rival_bid_std = np.sqrt(rival_cost_std**2 + rival_markup_std**2)
            
            # Probability single rival bids above our bid (normal CDF)
            from scipy.stats import norm
            prob_single_rival_above = 1 - norm.cdf(bid, rival_bid_mean, rival_bid_std)
            
            # Probability all rivals bid above our bid
            n_rivals = config.number_of_players - 1
            prob_win = prob_single_rival_above ** n_rivals
            
            return max(0.01, min(0.99, prob_win))  # Bound between 1% and 99%
            
        except Exception:
            # Fallback: simple heuristic
            constants = GameConstants(config)
            if bid <= constants.SPULBER_MARGINAL_COST + 1:
                return 0.8  # Very aggressive
            elif bid <= constants.SPULBER_MARGINAL_COST + 5:
                return 0.5  # Moderate
            else:
                return 0.2  # Conservative
    
    def _classify_bidding_strategy(self, bid: float, config: GameConfig) -> str:
        """Classify bidding strategy relative to cost and market conditions"""
        constants = GameConstants(config)
        
        markup = bid - constants.SPULBER_MARGINAL_COST
        
        if markup <= 2:
            return "Highly Aggressive (Low Markup)"
        elif markup <= 5:
            return "Moderately Aggressive"
        elif markup <= 10:
            return "Balanced Risk-Reward"
        else:
            return "Conservative (High Markup)"
    
    def _assess_bid_risk(self, bid: float, config: GameConfig) -> str:
        """Assess risk level of bidding strategy"""
        constants = GameConstants(config)
        
        # Risk based on markup and competitive pressure
        markup = bid - constants.SPULBER_MARGINAL_COST
        competitive_pressure = config.number_of_players - 1
        
        if markup <= 2 and competitive_pressure >= 3:
            return "High Risk (Very Aggressive in Competitive Market)"
        elif markup <= 5:
            return "Moderate Risk (Balanced Approach)"
        elif markup >= 10:
            return "Low Risk (Conservative, Lower Win Probability)"
        else:
            return "Medium Risk (Standard Competitive Bid)"
    
    def _calculate_competitive_intensity(self, bids: Dict[str, float]) -> float:
        """Calculate competitive intensity metric"""
        try:
            bid_values = list(bids.values())
            if len(bid_values) < 2:
                return 0.0
            
            # Competitive intensity = 1 - (coefficient of variation)
            # Lower variation = higher competitive intensity
            mean_bid = np.mean(bid_values)
            std_bid = np.std(bid_values)
            
            if mean_bid == 0:
                return 0.0
            
            cv = std_bid / mean_bid
            intensity = max(0.0, 1.0 - cv)
            
            return intensity
            
        except Exception:
            return 0.5  # Default moderate intensity
    
    def _analyze_competitive_environment(self, game_state: Dict, config: GameConfig) -> Dict[str, str]:
        """Analyze competitive environment and provide strategic insights"""
        try:
            constants = GameConstants(config)
            n_rivals = config.number_of_players - 1
            
            # Cost advantage analysis
            cost_advantage = constants.SPULBER_RIVAL_COST_MEAN - constants.SPULBER_MARGINAL_COST
            
            if cost_advantage > constants.SPULBER_RIVAL_COST_STD:
                advantage_level = "significant cost advantage"
            elif cost_advantage > 0:
                advantage_level = "moderate cost advantage" 
            else:
                advantage_level = "cost disadvantage"
            
            # Competition pressure analysis
            if n_rivals >= 4:
                competition_level = "intense competition"
            elif n_rivals >= 2:
                competition_level = "moderate competition"
            else:
                competition_level = "limited competition"
            
            probability_assessment = f"""**Win Probability Analysis:**
- **Your Cost Position**: You have a {advantage_level} with costs of ${constants.SPULBER_MARGINAL_COST:.2f} vs. expected rival costs of ${constants.SPULBER_RIVAL_COST_MEAN:.2f}
- **Competition Level**: Facing {n_rivals} rivals creates {competition_level}
- **Strategic Implication**: {"Bid aggressively to capitalize on cost advantage" if cost_advantage > 0 else "Bid carefully due to cost disadvantage"}
- **Uncertainty Factor**: Rival cost uncertainty (σ=${constants.SPULBER_RIVAL_COST_STD:.2f}) means some rivals may be very efficient"""
            
            return {
                'advantage_level': advantage_level,
                'competition_level': competition_level,
                'probability_assessment': probability_assessment
            }
            
        except Exception:
            return {
                'advantage_level': 'uncertain',
                'competition_level': 'standard',
                'probability_assessment': 'Standard competitive environment with uncertain rival costs.'
            }
    
    def _generate_strategic_guidance(self, config: GameConfig) -> Dict[str, str]:
        """Generate strategic guidance based on market conditions"""
        try:
            constants = GameConstants(config)
            
            trade_off_analysis = f"""**The Core Strategic Trade-Off:**
Every dollar increase in your bid has two opposing effects:
- **Negative**: Reduces your profit if you win (${constants.SPULBER_MARKET_VALUE:,} - Higher Bid = Lower Profit)
- **Positive**: Increases probability of winning (fewer rivals likely to bid below you)

**Mathematical Optimization**: Your optimal bid maximizes Expected Profit = P(Win) × (${constants.SPULBER_MARKET_VALUE:,} - Bid)

**Key Insight**: The optimal balance depends on:
1. How much cost advantage you have over rivals
2. How many competitors you're facing
3. Your risk tolerance for winner-take-all outcomes"""
            
            return {
                'trade_off_analysis': trade_off_analysis
            }
            
        except Exception:
            return {
                'trade_off_analysis': 'Balance win probability against profit margins to maximize expected returns.'
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
                reasoning_end = min(150, len(after_text))
            
            reasoning = (before_text[reasoning_start:] + " " + match.group(0) + " " + after_text[:reasoning_end]).strip()
            
            return reasoning if reasoning else "Bid decision based on competitive analysis"
            
        except Exception:
            return "Spulber Bertrand competition bidding"
    
    def _calculate_strategic_metrics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Calculate Spulber-specific strategic metrics"""
        try:
            auction_analysis = game_state.get('spulber_auction_analysis', {})
            
            if not auction_analysis:
                return {'status': 'no_analysis_data'}
            
            bids = auction_analysis.get('bids', {})
            winners = auction_analysis.get('winners', [])
            
            if not bids:
                return {'status': 'no_bid_data'}
            
            constants = GameConstants(config)
            
            # Calculate auction efficiency metrics
            winning_bid = auction_analysis.get('winning_bid', 0)
            theoretical_efficient_bid = constants.SPULBER_MARGINAL_COST + 1  # Near-marginal cost
            
            metrics = {
                'auction_efficiency': {
                    'winning_bid': winning_bid,
                    'bid_spread': max(bids.values()) - min(bids.values()),
                    'average_bid': np.mean(list(bids.values())),
                    'competitive_intensity': auction_analysis.get('competitive_intensity', 0),
                    'market_efficiency_ratio': winning_bid / constants.SPULBER_MARKET_VALUE
                },
                'strategic_outcomes': {
                    'winner_count': len(winners),
                    'total_surplus_captured': sum(max(0, constants.SPULBER_MARKET_VALUE - bid) for bid in bids.values() if any(p_id in winners for p_id in bids.keys())),
                    'allocative_efficiency': min(bids.values()) / theoretical_efficient_bid if theoretical_efficient_bid > 0 else 1,
                    'revenue_efficiency': winning_bid / constants.SPULBER_MARKET_VALUE
                },
                'competitive_dynamics': {
                    'bid_dispersion': np.std(list(bids.values())),
                    'aggressive_bidders': sum(1 for bid in bids.values() if bid <= constants.SPULBER_MARGINAL_COST + 3),
                    'conservative_bidders': sum(1 for bid in bids.values() if bid >= constants.SPULBER_MARGINAL_COST + 8),
                    'market_power_exercise': max(bids.values()) - constants.SPULBER_MARGINAL_COST
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating Spulber strategic metrics: {e}")
            return {'error': str(e)}