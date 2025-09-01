"""
Spulber Bertrand Competition Game - Minimal implementation for comprehensive_metrics.py
Focuses only on core Bertrand auction economics and generating proper data for metrics analysis
"""

import json
import re
from typing import Dict, Any, Optional, List
from games.base_game import StaticGame
from config import GameConfig, GameConstants


class SpulberGame(StaticGame):
    """
    Spulber Bertrand Competition - winner-take-all auction under asymmetric cost information
    Generates data needed by comprehensive_metrics.py for behavioral analysis
    """
    
    def __init__(self):
        super().__init__("Spulber Bertrand Competition")

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             config: GameConfig, constants: GameConstants) -> str:
        """Generate strategic prompt for Spulber bidding decision"""
        
        cost_advantage = self._analyze_cost_position(constants)
        winning_probability = self._estimate_winning_probability(constants, config)
        bid_range = self._calculate_bid_guidance(constants)
        
        prompt = f"""**SPULBER BERTRAND COMPETITION - WINNER-TAKE-ALL AUCTION**

You are Firm {player_id} competing in a sealed-bid auction against {config.number_of_players - 1} rivals.

**AUCTION RULES:**
- **Winner-Take-All**: LOWEST bidder wins entire market value of ${constants.SPULBER_MARKET_VALUE:,}
- **All-or-Nothing**: Only winner gets profit, losers get $0
- **Sealed Bids**: All firms bid simultaneously without knowing rival bids
- **Single Round**: One bid, final decision

**YOUR ECONOMICS:**
- **Your Cost**: ${constants.SPULBER_MARGINAL_COST:.2f} per unit
- **Your Position**: {cost_advantage}
- **If You Win**: Profit = ${constants.SPULBER_MARKET_VALUE:,} - Your Bid
- **If You Lose**: Profit = $0

**RIVAL INFORMATION:**
- **Rival Costs**: Each rival's cost ~ Normal(μ=${constants.SPULBER_RIVAL_COST_MEAN:.2f}, σ=${constants.SPULBER_RIVAL_COST_STD:.2f})
- **Rival Cost Range**: Approximately ${constants.SPULBER_RIVAL_COST_MEAN - 2*constants.SPULBER_RIVAL_COST_STD:.2f} to ${constants.SPULBER_RIVAL_COST_MEAN + 2*constants.SPULBER_RIVAL_COST_STD:.2f}
- **Your Winning Probability**: ~{winning_probability:.0%} with optimal bidding

**BERTRAND COMPETITION THEORY:**
- Under perfect information: price wars drive profits to zero
- Under asymmetric information: positive profits possible
- Key trade-off: aggressive bidding (higher win chance, lower margins) vs conservative bidding (lower win chance, higher margins)

**STRATEGIC CONSIDERATIONS:**
1. **Cost Advantage Strategy**: {"Bid aggressively to exploit your cost advantage" if constants.SPULBER_MARGINAL_COST < constants.SPULBER_RIVAL_COST_MEAN else "Bid conservatively due to cost disadvantage"}
2. **Competition Intensity**: {config.number_of_players - 1} rivals means {"high" if config.number_of_players > 3 else "moderate"} competition
3. **Winner's Curse Risk**: Winning might mean you bid too low (left money on table)
4. **Information Uncertainty**: You can't see rival costs, must use probability

**BIDDING GUIDANCE:**
- **Minimum Viable**: ${constants.SPULBER_MARGINAL_COST:.2f} (your cost)
- **Suggested Range**: {bid_range}
- **Maximum**: ${constants.SPULBER_MARKET_VALUE:,} (no profit)

**REQUIRED FORMAT:**
{{"bid": <your_bid>, "reasoning": "<detailed explanation of your bidding strategy considering cost uncertainty>"}}

Choose your bid to maximize expected profit in this Bertrand competition under asymmetric information."""

        return prompt

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse bidding decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.basic_json_parse(response)
        if json_action and 'bid' in json_action:
            bid = self._validate_bid(json_action['bid'])
            if bid is not None:
                json_action['bid'] = bid
                json_action['raw_response'] = response
                return json_action
        
        # Try regex patterns for bid extraction
        bid_patterns = [
            r'"bid":\s*(\d+\.?\d*)',
            r'bid["\']?:\s*(\d+\.?\d*)',
            r'bid:\s*\$?(\d+\.?\d*)',
            r'I bid\s*\$?(\d+\.?\d*)',
            r'my bid:\s*\$?(\d+\.?\d*)',
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d+)'
        ]
        
        for pattern in bid_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    bid = self._validate_bid(float(matches[0]))
                    if bid is not None:
                        return {
                            'bid': bid,
                            'reasoning': response[:200],
                            'parsing_method': 'regex',
                            'raw_response': response
                        }
                except ValueError:
                    continue
        
        self.logger.warning(f"[{call_id}] Could not parse bid from {player_id} response")
        return None

    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Get default bidding action when parsing fails"""
        constants = GameConstants()
        # Conservative default: cost + moderate markup
        default_bid = constants.SPULBER_MARGINAL_COST + 15.0
        
        return {
            'bid': default_bid,
            'reasoning': f'Default cost-plus bidding due to parsing failure',
            'parsing_success': False,
            'player_id': player_id,
            'round': game_state.get('current_round', 1)
        }

    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate Spulber Bertrand competition payoffs - winner-take-all"""
        
        call_id = game_state.get('call_id', 'unknown') if game_state else 'unknown'
        constants = GameConstants()
        
        # Extract bids
        bids = {}
        for agent_id, action in actions.items():
            bid = action.get('bid', constants.SPULBER_MARGINAL_COST + 10)
            bids[agent_id] = float(bid)
        
        # Find winner (lowest bidder)
        min_bid = min(bids.values())
        winners = [agent_id for agent_id, bid in bids.items() if abs(bid - min_bid) < 0.01]
        
        # Calculate profits (winner-take-all)
        profits = {}
        
        if len(winners) == 1:
            # Single winner
            winner = winners[0]
            winning_bid = bids[winner]
            
            for agent_id in bids.keys():
                if agent_id == winner:
                    # Winner gets market value minus bid
                    profit = constants.SPULBER_MARKET_VALUE - winning_bid
                    profits[agent_id] = max(0, profit)
                else:
                    # Losers get zero
                    profits[agent_id] = 0.0
                    
        else:
            # Tie - split market value among winners
            shared_revenue = constants.SPULBER_MARKET_VALUE / len(winners)
            
            for agent_id in bids.keys():
                if agent_id in winners:
                    profit = shared_revenue - bids[agent_id]
                    profits[agent_id] = max(0, profit)
                else:
                    profits[agent_id] = 0.0
        
        # Log auction results for analysis
        winner_info = winners[0] if len(winners) == 1 else f"{len(winners)} tied winners"
        
        self.logger.info(f"[{call_id}] Spulber auction: Winner={winner_info}, "
                       f"Winning bid=${min_bid:.2f}, Market value=${constants.SPULBER_MARKET_VALUE:,}")
        
        for agent_id in bids.keys():
            bid = bids[agent_id]
            profit = profits[agent_id]
            status = "🏆 WINNER" if agent_id in winners else "❌ LOSER"
            self.logger.info(f"[{call_id}]   {status}: {agent_id} - Bid=${bid:.2f}, Profit=${profit:.2f}")
        
        return profits

    def _validate_bid(self, bid: float) -> Optional[float]:
        """Validate bid is in reasonable range"""
        try:
            bid = float(bid)
            constants = GameConstants()
            
            # Bounds: above 0, below market value
            min_bid = 1.0
            max_bid = constants.SPULBER_MARKET_VALUE
            
            if min_bid <= bid <= max_bid:
                return round(bid, 2)
            else:
                return None
                
        except (ValueError, TypeError):
            return None

    def _analyze_cost_position(self, constants: GameConstants) -> str:
        """Analyze cost position relative to rivals"""
        my_cost = constants.SPULBER_MARGINAL_COST
        rival_mean = constants.SPULBER_RIVAL_COST_MEAN
        rival_std = constants.SPULBER_RIVAL_COST_STD
        
        # Calculate relative position
        cost_diff = rival_mean - my_cost
        z_score = cost_diff / rival_std
        
        if z_score > 1.0:
            return f"SIGNIFICANT cost advantage (${cost_diff:.2f} below rivals)"
        elif z_score > 0.5:
            return f"MODERATE cost advantage (${cost_diff:.2f} below rivals)"
        elif z_score > -0.5:
            return f"SIMILAR costs to rivals (${abs(cost_diff):.2f} difference)"
        else:
            return f"COST DISADVANTAGE (${abs(cost_diff):.2f} above rivals)"

    def _estimate_winning_probability(self, constants: GameConstants, config: GameConfig) -> float:
        """Estimate winning probability with optimal bidding"""
        my_cost = constants.SPULBER_MARGINAL_COST
        rival_mean = constants.SPULBER_RIVAL_COST_MEAN
        rival_std = constants.SPULBER_RIVAL_COST_STD
        num_rivals = config.number_of_players - 1
        
        # Simplified probability based on cost advantage
        z_score = (rival_mean - my_cost) / rival_std
        base_prob = 0.5 + 0.2 * z_score  # Higher for cost advantage
        
        # Adjust for number of competitors
        adjusted_prob = base_prob / (1 + 0.1 * num_rivals)
        
        return max(0.1, min(0.9, adjusted_prob))

    def _calculate_bid_guidance(self, constants: GameConstants) -> str:
        """Calculate suggested bidding range"""
        my_cost = constants.SPULBER_MARGINAL_COST
        market_value = constants.SPULBER_MARKET_VALUE
        rival_mean = constants.SPULBER_RIVAL_COST_MEAN
        
        # Conservative: bid near rival mean
        conservative = min(rival_mean + 5, market_value - 10)
        
        # Aggressive: bid closer to own cost
        aggressive = my_cost + 5
        
        return f"${aggressive:.2f} (aggressive) to ${conservative:.2f} (conservative)"