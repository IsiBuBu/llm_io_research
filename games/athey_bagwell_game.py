"""
Athey-Bagwell Information Collusion Game - Minimal implementation for comprehensive_metrics.py
Focuses only on core information collusion economics and generating proper data for metrics analysis
"""

import json
import re
import numpy as np
from typing import Dict, Any, Optional, List
from games.base_game import DynamicGame
from config import GameConfig, GameConstants


class AtheyBagwellGame(DynamicGame):
    """
    Athey-Bagwell Information Collusion - cartel with private cost information
    Generates data needed by comprehensive_metrics.py for behavioral analysis
    """
    
    def __init__(self):
        super().__init__("Athey Bagwell Information Collusion")

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             config: GameConfig, constants: GameConstants) -> str:
        """Generate strategic prompt for Athey-Bagwell cost reporting decision"""
        
        current_cost = game_state.get('current_costs', {}).get(player_id, 'high')
        current_round = game_state.get('current_round', 1)
        
        prompt = f"""**ATHEY-BAGWELL INFORMATION COLLUSION**

You are Firm {player_id} in a {config.number_of_players}-firm cartel, Round {current_round}/{config.number_of_rounds}.

**CARTEL STRUCTURE:**
- **Market Price**: Fixed at ${constants.AB_MARKET_PRICE} per unit
- **Total Market**: {constants.AB_MARKET_SIZE} units allocated among cartel members
- **Market Share Allocation**: Based on reported costs (lower costs get larger shares)
- **Your True Cost**: {"High" if current_cost == 'high' else "Low"} cost type this period

**COST INFORMATION:**
- **High Cost**: ${constants.AB_HIGH_COST} per unit
- **Low Cost**: ${constants.AB_LOW_COST} per unit
- **Cost Persistence**: {int(constants.AB_COST_PERSISTENCE*100)}% chance your cost stays same next period
- **Your Profit**: (${constants.AB_MARKET_PRICE} - True Cost) × Market Share

**STRATEGIC DILEMMA:**
- **Truth-telling**: Report actual cost type for efficient cartel allocation
- **Strategic Misreporting**: Report "low" to get larger market share regardless of true cost
- **Reputation Effects**: Your reporting history affects future trust and allocations
- **Long-term vs Short-term**: Discount factor {constants.AB_DISCOUNT_FACTOR} weights future profits

**INFORMATION ASYMMETRY:**
- Other firms cannot observe your true costs
- They only see your cost reports
- Market shares depend on relative reported costs
- Cartel stability requires some level of truthful reporting

**REPORTING GUIDANCE:**
- Report "high" if you have high costs and value cartel stability
- Report "low" if you want maximum market share this period
- Consider long-term cartel sustainability in your decision

**REQUIRED FORMAT:**
{{"report": "<high or low>", "reasoning": "<explanation of your reporting strategy>"}}

Should you report your cost as "high" or "low" this period?"""
        
        return prompt

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse cost report from LLM response"""
        try:
            # Try JSON parsing first
            action = self._parse_json_response(response)
            if action and 'report' in action and action['report'] in ['high', 'low']:
                return action
            
            # Pattern-based fallback
            response_lower = response.lower()
            
            # Look for report statements
            if 'report' in response_lower:
                if 'high' in response_lower and 'low' not in response_lower:
                    return {"report": "high", "reasoning": "Pattern-parsed report"}
                elif 'low' in response_lower and 'high' not in response_lower:
                    return {"report": "low", "reasoning": "Pattern-parsed report"}
            
            # Simple keyword detection
            if 'low' in response_lower:
                return {"report": "low", "reasoning": "Keyword-parsed report"}
            else:
                return {"report": "high", "reasoning": "Default high cost report"}
                
        except Exception as e:
            self.logger.warning(f"Parsing error for {player_id}: {e}")
            return None

    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Default action when parsing fails - report true cost"""
        current_cost = game_state.get('current_costs', {}).get(player_id, 'high')
        return {
            "report": current_cost,
            "reasoning": "Default truthful reporting"
        }

    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate payoffs based on cost reports and true costs"""
        constants = GameConstants()
        reports = {pid: action.get('report', 'high') for pid, action in actions.items()}
        true_costs = game_state.get('current_costs', {}) if game_state else {}
        
        # Allocate market shares based on reports
        market_shares = self._allocate_market_shares(reports, constants.AB_MARKET_SIZE)
        
        # Calculate profits
        profits = {}
        for player_id in reports.keys():
            market_share = market_shares.get(player_id, 0)
            true_cost_value = constants.AB_HIGH_COST if true_costs.get(player_id, 'high') == 'high' else constants.AB_LOW_COST
            profit = (constants.AB_MARKET_PRICE - true_cost_value) * market_share
            profits[player_id] = max(0, profit)
        
        return profits

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """Update game state with cost evolution"""
        # Call parent method for basic round history
        game_state = super().update_game_state(game_state, actions, config)
        
        # Initialize or evolve costs for each player
        constants = GameConstants()
        
        if 'current_costs' not in game_state:
            game_state['current_costs'] = {}
        
        for player_id in actions.keys():
            if game_state['current_round'] == 1:
                # Initialize random cost types
                game_state['current_costs'][player_id] = np.random.choice(['high', 'low'])
            else:
                # Evolve costs with persistence
                current_cost = game_state['current_costs'][player_id]
                if np.random.random() > constants.AB_COST_PERSISTENCE:
                    # Switch cost type
                    game_state['current_costs'][player_id] = 'low' if current_cost == 'high' else 'high'
        
        return game_state

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response"""
        try:
            json_patterns = [
                r'\{[^{}]*"report"[^{}]*\}',  # Simple JSON with report
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```'     # JSON in any code blocks
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    action = json.loads(match)
                    if isinstance(action, dict):
                        return action
        except:
            pass
        return None

    def _allocate_market_shares(self, reports: Dict[str, str], total_market: int) -> Dict[str, float]:
        """Allocate market shares based on reported costs"""
        low_reporters = [pid for pid, report in reports.items() if report == 'low']
        high_reporters = [pid for pid, report in reports.items() if report == 'high']
        
        shares = {}
        
        if low_reporters:
            # Low cost reporters get larger shares
            low_share = total_market * 0.7 / len(low_reporters)
            high_share = total_market * 0.3 / len(high_reporters) if high_reporters else 0
            
            for pid in low_reporters:
                shares[pid] = low_share
            for pid in high_reporters:
                shares[pid] = high_share
        else:
            # All high reporters get equal shares
            share_per_firm = total_market / len(high_reporters)
            for pid in high_reporters:
                shares[pid] = share_per_firm
        
        return shares

    def _analyze_cost_position(self, constants: GameConstants) -> str:
        """Analyze relative cost position"""
        return "Standard cartel member"

    def _estimate_winning_probability(self, constants: GameConstants, config: GameConfig) -> float:
        """Estimate probability of getting large market share"""
        return 0.5  # Simplified estimate

    def _calculate_bid_guidance(self, constants: GameConstants) -> Dict[str, float]:
        """Calculate guidance for cost reporting"""
        return {
            'min': 0.3,  # Conservative reporting
            'max': 0.7   # Aggressive reporting
        }