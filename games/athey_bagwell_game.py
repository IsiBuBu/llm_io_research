# games/athey_bagwell_game.py
import numpy as np
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from .base_game import DynamicGame
from config import GameConfig, GameConstants


class AtheyBagwellGame(DynamicGame):
    """
    Athey-Bagwell Information Collusion Game Implementation
    
    Models a cartel where firms have private cost information and must decide
    whether to truthfully report costs for market share allocation.
    """
    
    def __init__(self):
        super().__init__("Athey Bagwell Information Collusion", default_rounds=20)
        
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        """Create game-specific prompt for Athey-Bagwell reporting decisions"""
        constants = GameConstants(config)
        player_history = game_state.get('player_histories', {}).get(player_id, {})
        current_cost = game_state.get('current_costs', {}).get(player_id, 'high')
        current_round = game_state.get('current_round', 1)
        
        return f"""**Context:** You participate in a {config.number_of_players}-firm cartel operating over {config.number_of_rounds} periods. The cartel attempts to maximize joint profits by coordinating market shares, but each firm has private information about its costs that affects optimal allocation decisions.

**Information Structure:** 
- Each period, your marginal cost is either "high" (${constants.AB_HIGH_COST}) or "low" (${constants.AB_LOW_COST})
- Cost types are persistent ({int(constants.AB_COST_PERSISTENCE*100)}% probability your cost stays the same next period)
- Other firms cannot observe your true costs - they only observe your cost reports
- Market shares are allocated based on reported costs: lower reported costs receive larger shares

**Economic Information:**
- Market price: Fixed at ${constants.AB_MARKET_PRICE} per unit
- Total market size: {constants.AB_MARKET_SIZE} units per period
- Your profit: (${constants.AB_MARKET_PRICE} - True Cost) × Allocated Market Share
- Discount factor: {config.discount_factor} (future profits worth {int(config.discount_factor*100)}% of current value)

**Strategic Dilemma:** You face a fundamental tension between truthfulness and profit maximization. Reporting low costs increases your market share this period but may affect future allocations in complex ways. The cartel's success depends on accurate cost reporting for efficient allocation, but individual incentives may favor misrepresentation.

**Cartel Rules:** Market shares are allocated using a mechanism that aims to maximize total cartel profits based on reported costs. The exact allocation formula considers both current reports and historical patterns to maintain stability.

**Current Game State:**
- Period: {current_round} of {config.number_of_rounds}
- Your true cost this period: {current_cost}
- Your previous cost reports: {player_history.get('reports', [])[-3:]}
- Previous market shares you received: {player_history.get('market_shares', [])[-3:]} units
- Your previous profits: {[round(p, 2) for p in player_history.get('profits', [])[-3:]]}

**Strategic Considerations:** Consider how your current report affects not only this period's allocation but also the cartel's long-term stability and your future market share. Other firms are making similar calculations about their own reporting strategies.

**Your Task:** Decide whether to report your cost as "high" or "low" for this period.

**Output Format:** {{"report": "<high or low>", "reasoning": "<brief explanation of your information management strategy>"}}"""
    
    def parse_action(self, response: str, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Enhanced action parsing for Athey-Bagwell reporting decisions.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            self.logger.debug(f"[{call_id}] Parsing Athey-Bagwell report action for {player_id}")
            
            # Try JSON parsing first
            action = self._parse_json_response(response, player_id, call_id)
            if action and 'report' in action:
                report = self._validate_report(action['report'], player_id, call_id)
                action['report'] = report
                action['parsing_method'] = 'json'
                return self._enhance_athey_action(action, player_id, game_state, config)
            
            # Try pattern-based parsing
            action = self._parse_athey_patterns(response, player_id, call_id)
            if action:
                return self._enhance_athey_action(action, player_id, game_state, config)
            
            # Use base class intelligent parsing with Athey enhancement
            action = self._parse_intelligent_response(response, player_id, call_id)
            if action:
                action = self._enhance_intelligent_parsing(action, response.lower())
                if action and 'report' in action:
                    return self._enhance_athey_action(action, player_id, game_state, config)
            
            # Fallback to default
            self.logger.warning(f"[{call_id}] All parsing methods failed for {player_id}, using Athey default")
            return self.get_default_action(player_id, game_state, config)
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error parsing Athey action for {player_id}: {e}")
            return self.get_default_action(player_id, game_state, config)
    
    def _parse_athey_patterns(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse Athey-Bagwell specific patterns from response"""
        try:
            response_lower = response.lower()
            
            # Look for explicit report statements
            report_patterns = [
                r'report[:\s]+["\']?(high|low)["\']?',
                r'i\s+report[:\s]+["\']?(high|low)["\']?',
                r'my\s+report[:\s]+is[:\s]+["\']?(high|low)["\']?',
                r'cost[:\s]+report[:\s]+["\']?(high|low)["\']?',
                r'["\']?(high|low)["\']?\s+cost\s+report',
                r'announce[:\s]+["\']?(high|low)["\']?',
                r'declare[:\s]+["\']?(high|low)["\']?'
            ]
            
            for pattern in report_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    report = match.group(1)
                    reasoning = self._extract_reasoning_from_pattern(response, match)
                    
                    return {
                        'report': report,
                        'reasoning': reasoning,
                        'raw_response': response[:500],
                        'parsing_method': 'pattern'
                    }
            
            return None
            
        except Exception as e:
            self.logger.debug(f"[{call_id}] Pattern parsing error for {player_id}: {e}")
            return None
    
    def _enhance_intelligent_parsing(self, parsed_action: Dict[str, Any], response_lower: str) -> Optional[Dict[str, Any]]:
        """Enhance intelligent parsing with Athey-Bagwell specific logic"""
        try:
            decision_text = parsed_action.get('decision_text', '')
            
            # Look for cost reporting language in decision text
            if 'high' in decision_text or 'expensive' in decision_text or 'costly' in decision_text:
                parsed_action['report'] = 'high'
                return parsed_action
            elif 'low' in decision_text or 'cheap' in decision_text or 'inexpensive' in decision_text:
                parsed_action['report'] = 'low'
                return parsed_action
            
            # Look for strategic language patterns
            if any(word in response_lower for word in ['truth', 'honest', 'accurate']):
                parsed_action['report'] = 'high'  # Assume truthful high cost reporting
                return parsed_action
            elif any(word in response_lower for word in ['misreport', 'lie', 'deceive', 'strategic']):
                parsed_action['report'] = 'low'   # Strategic low cost claim
                return parsed_action
            
            return None
            
        except Exception:
            return None
    
    def _validate_report(self, report: str, player_id: str, call_id: str) -> str:
        """Validate and normalize cost report"""
        try:
            report = str(report).lower().strip().strip('"\'')
            
            if report in ['high', 'h', 'hi', 'expensive', 'costly']:
                return 'high'
            elif report in ['low', 'l', 'lo', 'cheap', 'inexpensive']:
                return 'low'
            else:
                self.logger.warning(f"[{call_id}] Invalid report '{report}' for {player_id}, defaulting to 'high'")
                return 'high'
                
        except (ValueError, TypeError):
            self.logger.warning(f"[{call_id}] Could not parse report for {player_id}, defaulting to 'high'")
            return 'high'
    
    def _enhance_athey_action(self, action: Dict[str, Any], player_id: str, 
                             game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Enhance parsed action with Athey-Bagwell specific metadata"""
        try:
            call_id = game_state.get('call_id', 'unknown')
            current_cost = game_state.get('current_costs', {}).get(player_id, 'high')
            report = action.get('report', 'high')
            
            # Add Athey-specific metadata
            action.update({
                'player_id': player_id,
                'round': game_state.get('current_round', 1),
                'true_cost': current_cost,
                'is_truthful': report == current_cost,
                'strategic_classification': self._classify_report_strategy(report, current_cost),
                'expected_market_share': self._estimate_market_share(report, config),
                'call_id': call_id
            })
            
            self.logger.debug(f"[{call_id}] Enhanced Athey action for {player_id}: report={report}, truthful={action['is_truthful']}")
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error enhancing Athey action: {e}")
            return action
    
    def _classify_report_strategy(self, report: str, true_cost: str) -> str:
        """Classify the strategic nature of the cost report"""
        if report == true_cost:
            return 'truthful'
        elif report == 'low' and true_cost == 'high':
            return 'strategic_underreport'
        else:  # report == 'high' and true_cost == 'low'
            return 'strategic_overreport'
    
    def _estimate_market_share(self, report: str, config: GameConfig) -> float:
        """Estimate expected market share based on report"""
        constants = GameConstants(config)
        n = config.number_of_players
        
        if report == 'low':
            # Assuming half the players report low (equilibrium approximation)
            expected_low_reporters = n / 2
            return (constants.AB_MARKET_SIZE * 1.5) / (expected_low_reporters * 1.5 + (n - expected_low_reporters))
        else:
            # High cost report gets standard allocation
            expected_high_reporters = n / 2
            expected_low_reporters = n - expected_high_reporters
            return constants.AB_MARKET_SIZE / (expected_low_reporters * 1.5 + expected_high_reporters)
    
    def _extract_reasoning_from_pattern(self, response: str, match) -> str:
        """Extract reasoning from response based on pattern match"""
        try:
            # Take everything before the matched pattern as reasoning
            reasoning_text = response[:match.start()].strip()
            
            # If too long, take the last part
            if len(reasoning_text) > 200:
                reasoning_text = "..." + reasoning_text[-200:]
            
            return reasoning_text if reasoning_text else "Pattern-based parsing"
            
        except Exception:
            return "Pattern-based parsing"
    
    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Provide intelligent default reporting action when parsing fails"""
        call_id = game_state.get('call_id', 'unknown')
        current_cost = game_state.get('current_costs', {}).get(player_id, 'high')
        
        # Use truthful reporting as economically reasonable default
        self.logger.warning(f"[{call_id}] Using truthful default report for {player_id}: {current_cost}")
        
        return {
            'report': current_cost,
            'reasoning': 'Default truthful reporting due to parsing failure. Maintains cartel stability.',
            'parsing_method': 'default',
            'parsing_success': False,
            'raw_response': 'PARSING_FAILED',
            'player_id': player_id,
            'round': game_state.get('current_round', 1),
            'true_cost': current_cost,
            'is_truthful': True,
            'strategic_classification': 'truthful',
            'expected_market_share': self._estimate_market_share(current_cost, config)
        }
    
    def validate_action(self, action: Dict[str, Any], player_id: str, config: GameConfig) -> bool:
        """Validate that the action contains required Athey-Bagwell elements"""
        try:
            report = action.get('report')
            return report in ['high', 'low']
        except:
            return False
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate payoffs for all players based on cost reports and true costs
        """
        constants = GameConstants(config)
        reports = {pid: action.get('report', 'high') for pid, action in actions.items()}
        true_costs = game_state.get('current_costs', {}) if game_state else {}
        
        # Allocate market shares based on reports
        market_shares = self._allocate_market_shares(reports, constants.AB_MARKET_SIZE)
        
        profits = {}
        for player_id in reports.keys():
            market_share = market_shares.get(player_id, 0)
            true_cost_type = true_costs.get(player_id, 'high')
            
            true_cost = (constants.AB_HIGH_COST if true_cost_type == 'high' 
                        else constants.AB_LOW_COST)
            
            profit = (constants.AB_MARKET_PRICE - true_cost) * market_share
            profits[player_id] = max(0, profit)
        
        return profits
    
    def _allocate_market_shares(self, reports: Dict[str, str], total_market: int) -> Dict[str, float]:
        """
        Allocate market shares based on reported costs using weighted allocation
        Low-cost reporters receive 1.5x weight advantage
        """
        low_reporters = [pid for pid, report in reports.items() if report == 'low']
        high_reporters = [pid for pid, report in reports.items() if report == 'high']
        
        n_low = len(low_reporters)
        n_high = len(high_reporters)
        
        shares = {}
        
        if n_low > 0 and n_high > 0:
            # Mixed reporting: low-cost gets 1.5x weight
            total_weights = n_low * 1.5 + n_high * 1.0
            low_share = (total_market * 1.5) / total_weights
            high_share = total_market / total_weights
            
            for pid in low_reporters:
                shares[pid] = low_share
            for pid in high_reporters:
                shares[pid] = high_share
                
        elif n_low > 0:  # Only low reporters
            share_per_firm = total_market / n_low
            for pid in low_reporters:
                shares[pid] = share_per_firm
                
        else:  # Only high reporters (or no valid reports)
            share_per_firm = total_market / max(n_high, 1)
            for pid in high_reporters:
                shares[pid] = share_per_firm
        
        return shares
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         round_num: int) -> Dict:
        """
        Update game state with cost evolution and comprehensive historical tracking
        """
        # Create temporary config for constants
        temp_config = GameConfig(number_of_players=len(actions))
        constants = GameConstants(temp_config)
        
        # Initialize game state components
        if 'player_histories' not in game_state:
            game_state['player_histories'] = {}
        if 'current_costs' not in game_state:
            game_state['current_costs'] = {}
         
        game_state['current_round'] = round_num
        
        # Initialize or evolve costs for each player
        for player_id in actions.keys():
            if player_id not in game_state['player_histories']:
                game_state['player_histories'][player_id] = {
                    'reports': [], 'market_shares': [], 'profits': []
                }
            
            # Cost evolution with persistence parameter
            if round_num == 1:
                # Initialize random cost types
                game_state['current_costs'][player_id] = np.random.choice(['high', 'low'])
            else:
                # Evolve costs with persistence
                if np.random.random() < constants.AB_COST_PERSISTENCE:
                    pass  # Keep same cost type
                else:
                    # Switch cost type
                    current_cost = game_state['current_costs'][player_id]
                    game_state['current_costs'][player_id] = 'low' if current_cost == 'high' else 'high'
        
        # Calculate market outcomes
        reports = {pid: action.get('report', 'high') for pid, action in actions.items()}
        market_shares = self._allocate_market_shares(reports, constants.AB_MARKET_SIZE)
        profits = self.calculate_payoffs(actions, temp_config, game_state)
        
        # Update player histories
        for player_id, action in actions.items():
            report = action.get('report', 'high')
            market_share = market_shares.get(player_id, 0)
            profit = profits.get(player_id, 0)
            
            game_state['player_histories'][player_id]['reports'].append(report)
            game_state['player_histories'][player_id]['market_shares'].append(market_share)
            game_state['player_histories'][player_id]['profits'].append(profit)
        
        return game_state
    
    def _summarize_action(self, action: Dict[str, Any]) -> str:
        """Create brief summary of reporting action"""
        report = action.get('report', 'unknown')
        is_truthful = action.get('is_truthful')
        truthfulness_indicator = "✓" if is_truthful else "✗" if is_truthful is False else "?"
        return f"Report={report} {truthfulness_indicator}"
    
    def _classify_strategy(self, action: Dict[str, Any], game_state: Dict, player_id: str) -> str:
        """Classify the strategic nature of a reporting action"""
        return action.get('strategic_classification', 'unknown_strategy')
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        """Safely extract numeric value from action dictionary"""
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default