"""
Athey & Bagwell Information Collusion Game - Compact implementation with full config integration
Implements Market Allocation Algorithm from t.txt for comprehensive metrics analysis
"""

import json
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import DynamicGame, extract_numeric_value
from config import GameConfig, get_prompt_variables


class AtheyBagwellGame(DynamicGame):
    """
    Athey & Bagwell Information Collusion - cartel with private cost information
    Implements Market Allocation Algorithm and cost persistence from t.txt
    """
    
    def __init__(self):
        super().__init__("athey_bagwell")
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template from markdown file"""
        prompt_path = Path("prompts/athey_bagwell.md")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract template between ``` blocks
        template_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if not template_match:
            raise ValueError("No template found in athey_bagwell.md")
        
        return template_match.group(1)

    def initialize_game_state(self, game_config: GameConfig, 
                            simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state with cost persistence for all players"""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        persistence_probability = constants.get('persistence_probability', 0.7)
        num_players = constants.get('number_of_players', 3)
        
        # Generate cost sequences for all players using Markov process
        np.random.seed(simulation_id)  # Reproducible costs per simulation
        
        cost_sequences = {}
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]
        
        for player_id in player_ids:
            # Start with stationary distribution (50/50)
            costs = ['high' if np.random.random() < 0.5 else 'low']
            
            # Generate sequence using persistence probability
            for t in range(1, time_horizon):
                if np.random.random() < persistence_probability:
                    # Keep same cost
                    costs.append(costs[-1])
                else:
                    # Flip cost
                    costs.append('low' if costs[-1] == 'high' else 'high')
            
            cost_sequences[player_id] = costs
        
        return {
            'current_round': 1,
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids},
            'true_cost_history': {pid: [] for pid in player_ids},
            'profit_history': {pid: [] for pid in player_ids},
            'market_share_history': {pid: [] for pid in player_ids},
            'total_rounds': time_horizon
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt using template and config"""
        
        current_round = game_state.get('current_round', 1)
        cost_sequences = game_state.get('cost_sequences', {})
        report_history = game_state.get('report_history', {})
        
        # Get current cost for this player
        current_cost_type = 'high'
        if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_round:
            current_cost_type = cost_sequences[player_id][current_round - 1]
        
        # Format detailed report history
        all_reports_history_detailed = self._format_report_history(report_history, current_round)
        
        # Get template variables from config
        variables = get_prompt_variables(
            game_config, 
            player_id=player_id,
            current_round=current_round,
            current_cost_type=current_cost_type,
            all_reports_history_detailed=all_reports_history_detailed
        )
        
        # Format template with variables
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def _format_report_history(self, report_history: Dict[str, List], current_round: int) -> str:
        """Format detailed report history for prompt"""
        if current_round <= 1:
            return "No previous reports."
        
        history_lines = []
        for round_num in range(1, current_round):
            round_reports = []
            for player_id, reports in report_history.items():
                if len(reports) >= round_num:
                    round_reports.append(f"{player_id}: {reports[round_num - 1]}")
            
            if round_reports:
                history_lines.append(f"Round {round_num}: {', '.join(round_reports)}")
        
        return '; '.join(history_lines) if history_lines else "No previous reports."

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse cost report decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.basic_json_parse(response)
        if json_action and 'report' in json_action:
            report = json_action['report'].lower().strip()
            if report in ['high', 'low']:
                return {'report': report, 'raw_response': response}
        
        # Try direct text extraction
        response_lower = response.lower()
        
        # Look for explicit report statements
        if '"high"' in response_lower or "'high'" in response_lower or 'report "high"' in response_lower:
            return {'report': 'high', 'parsing_method': 'text', 'raw_response': response}
        elif '"low"' in response_lower or "'low'" in response_lower or 'report "low"' in response_lower:
            return {'report': 'low', 'parsing_method': 'text', 'raw_response': response}
        
        # Look for decision keywords
        if 'high cost' in response_lower or 'report high' in response_lower:
            return {'report': 'high', 'parsing_method': 'keyword', 'raw_response': response}
        elif 'low cost' in response_lower or 'report low' in response_lower:
            return {'report': 'low', 'parsing_method': 'keyword', 'raw_response': response}
        
        self.logger.warning(f"[{call_id}] Could not parse report from {player_id}")
        return None

    def get_default_action(self, player_id: str, game_state: Dict, 
                         game_config: GameConfig) -> Dict[str, Any]:
        """Default report action when parsing fails"""
        # Default to truthful reporting
        current_round = game_state.get('current_round', 1)
        cost_sequences = game_state.get('cost_sequences', {})
        
        current_cost = 'high'
        if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_round:
            current_cost = cost_sequences[player_id][current_round - 1]
        
        return {
            'report': current_cost,
            'reasoning': 'Default truthful reporting due to parsing failure',
            'parsing_success': False,
            'player_id': player_id
        }

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Athey-Bagwell payoffs using Market Allocation Algorithm from t.txt
        
        Algorithm from t.txt:
        1. Count players who reported "low" (N_low)
        2. If N_low = 1: "low" reporter gets 100% market share, others get 0
        3. If N_low > 1: each "low" reporter gets 1/N_low share, others get 0  
        4. If N_low = 0: each player gets 1/number_of_players share
        """
        constants = game_config.constants
        cost_types = constants.get('cost_types', {'low': 15, 'high': 25})
        market_price = constants.get('market_price', 30)
        market_size = constants.get('market_size', 100)
        num_players = constants.get('number_of_players', 3)
        
        current_round = game_state.get('current_round', 1) if game_state else 1
        cost_sequences = game_state.get('cost_sequences', {}) if game_state else {}
        
        # Extract reports and get true costs
        players = list(actions.keys())
        reports = {}
        true_costs = {}
        
        for player_id, action in actions.items():
            report = action.get('report', 'high')
            reports[player_id] = report
            
            # Get true cost for this player and round
            if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_round:
                true_cost_type = cost_sequences[player_id][current_round - 1]
                true_costs[player_id] = cost_types[true_cost_type]
            else:
                true_costs[player_id] = cost_types['high']  # Default
        
        # Step 1: Count players who reported "low"
        low_reporters = [pid for pid, report in reports.items() if report == 'low']
        N_low = len(low_reporters)
        
        # Step 2-4: Market Allocation Algorithm
        payoffs = {}
        market_shares = {}
        
        for player_id in players:
            if N_low == 1 and player_id in low_reporters:
                # Single "low" reporter gets everything
                market_share = 1.0
            elif N_low > 1 and player_id in low_reporters:
                # Multiple "low" reporters split evenly
                market_share = 1.0 / N_low
            elif N_low == 0:
                # All reported "high", split evenly
                market_share = 1.0 / num_players
            else:
                # Reported "high" when others reported "low"
                market_share = 0.0
            
            market_shares[player_id] = market_share
            
            # Calculate profit: (market_price - true_cost) × market_share × market_size
            true_cost = true_costs[player_id]
            profit = (market_price - true_cost) * market_share * market_size
            payoffs[player_id] = profit
        
        # Store additional data in game_state for logging
        if game_state is not None:
            game_state.update({
                'current_reports': reports,
                'current_true_costs': true_costs,
                'current_market_shares': market_shares,
                'current_profits': payoffs,
                'N_low': N_low,
                'low_reporters': low_reporters
            })
        
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Update game state with current round results"""
        
        current_round = game_state.get('current_round', 1)
        current_reports = game_state.get('current_reports', {})
        current_true_costs = game_state.get('current_true_costs', {})
        current_profits = game_state.get('current_profits', {})
        current_market_shares = game_state.get('current_market_shares', {})
        
        # Update histories
        report_history = game_state.get('report_history', {})
        true_cost_history = game_state.get('true_cost_history', {})
        profit_history = game_state.get('profit_history', {})
        market_share_history = game_state.get('market_share_history', {})
        
        for player_id in current_reports:
            if player_id not in report_history:
                report_history[player_id] = []
                true_cost_history[player_id] = []
                profit_history[player_id] = []
                market_share_history[player_id] = []
            
            report_history[player_id].append(current_reports[player_id])
            profit_history[player_id].append(current_profits.get(player_id, 0))
            market_share_history[player_id].append(current_market_shares.get(player_id, 0))
            
            # Add true cost type (not value)
            cost_sequences = game_state.get('cost_sequences', {})
            if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_round:
                true_cost_type = cost_sequences[player_id][current_round - 1]
                true_cost_history[player_id].append(true_cost_type)
            else:
                true_cost_history[player_id].append('high')
        
        # Update state
        game_state.update({
            'current_round': current_round + 1,
            'report_history': report_history,
            'true_cost_history': true_cost_history,
            'profit_history': profit_history,
            'market_share_history': market_share_history
        })
        
        return game_state

    def calculate_npv(self, profit_stream: List[float], discount_factor: float) -> float:
        """Calculate Net Present Value from profit stream"""
        npv = 0.0
        for t, profit in enumerate(profit_stream):
            npv += (discount_factor ** t) * profit
        return npv

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for metrics calculation as specified in t.txt"""
        
        constants = game_config.constants
        discount_factor = constants.get('discount_factor', 0.95)
        
        # Get game state data
        current_round = game_state.get('current_round', 1) if game_state else 1
        report_history = game_state.get('report_history', {}) if game_state else {}
        true_cost_history = game_state.get('true_cost_history', {}) if game_state else {}
        profit_history = game_state.get('profit_history', {}) if game_state else {}
        market_share_history = game_state.get('market_share_history', {}) if game_state else {}
        
        # Calculate NPVs and status indicators for MAgIC metrics
        npvs = {}
        deception_status = {}
        reasoning_status = {}
        cooperation_status = {}
        
        for player_id in payoffs:
            # Calculate NPV if we have profit history
            if player_id in profit_history and profit_history[player_id]:
                npv = self.calculate_npv(profit_history[player_id], discount_factor)
                npvs[player_id] = npv
            else:
                npvs[player_id] = payoffs[player_id]  # Single period fallback
            
            # Deception status (reported "low" when true cost was "high")
            if player_id in report_history and player_id in true_cost_history:
                reports = report_history[player_id]
                true_costs = true_cost_history[player_id]
                
                deceptions = 0
                opportunities = 0
                for i in range(len(reports)):
                    if i < len(true_costs) and true_costs[i] == 'high':
                        opportunities += 1
                        if reports[i] == 'low':
                            deceptions += 1
                
                deception_status[player_id] = {
                    'deceptions': deceptions,
                    'opportunities': opportunities,
                    'deception_rate': deceptions / opportunities if opportunities > 0 else 0
                }
            else:
                deception_status[player_id] = {'deceptions': 0, 'opportunities': 0, 'deception_rate': 0}
            
            # Reasoning status (high-profit actions)
            if player_id in profit_history and profit_history[player_id]:
                profits = profit_history[player_id]
                avg_profit = sum(profits) / len(profits)
                high_profit_actions = sum(1 for p in profits if p > avg_profit)
                reasoning_status[player_id] = {
                    'high_profit_actions': high_profit_actions,
                    'total_actions': len(profits),
                    'high_profit_rate': high_profit_actions / len(profits)
                }
            else:
                reasoning_status[player_id] = {'high_profit_actions': 0, 'total_actions': 1, 'high_profit_rate': 0}
            
            # Cooperation status (valid reports)
            if player_id in report_history:
                reports = report_history[player_id]
                valid_reports = sum(1 for r in reports if r in ['high', 'low'])
                cooperation_status[player_id] = {
                    'valid_reports': valid_reports,
                    'total_reports': len(reports),
                    'adherence_rate': valid_reports / len(reports) if reports else 1
                }
            else:
                cooperation_status[player_id] = {'valid_reports': 0, 'total_reports': 0, 'adherence_rate': 1}
        
        # Calculate win status based on NPV
        max_npv = max(npvs.values()) if npvs else 0
        win_status = {pid: (1 if npvs[pid] == max_npv else 0) for pid in npvs}
        
        return {
            'game_name': 'athey_bagwell',
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'constants': game_config.constants,
            
            # Core data for metrics (from t.txt requirements)
            'actions': actions,
            'payoffs': payoffs,
            'npvs': npvs,
            'report_history': report_history,
            'true_cost_history': true_cost_history,
            'profit_history': profit_history,
            'market_share_history': market_share_history,
            
            # Status indicators for MAgIC metrics
            'win_status': win_status,
            'deception_status': deception_status,
            'reasoning_status': reasoning_status,
            'cooperation_status': cooperation_status,
            
            # Current state info
            'current_round': current_round,
            'total_rounds': game_state.get('total_rounds', 50) if game_state else 50,
            
            # Additional metadata
            'game_state': game_state
        }