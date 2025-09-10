"""
Athey & Bagwell Information Collusion Game - Updated implementation with t.txt algorithms only
Implements Market Allocation Algorithm and cost persistence from t.txt specification
"""

import json
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import DynamicGame, ReportParsingMixin, extract_numeric_value
from config import GameConfig, get_prompt_variables


class AtheyBagwellGame(DynamicGame, ReportParsingMixin):
    """
    Athey & Bagwell Information Collusion - cartel with private cost information
    Implements Market Allocation Algorithm and cost persistence from t.txt
    """
    
    def __init__(self):
        super().__init__("athey_bagwell")

    def initialize_game_state(self, game_config: GameConfig, 
                            simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state with cost persistence for all players (from t.txt)"""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        persistence_probability = constants.get('persistence_probability', 0.7)
        num_players = constants.get('number_of_players', 3)
        
        # Generate cost sequences for all players using Markov process (reproducible per simulation)
        np.random.seed(simulation_id)
        
        cost_sequences = {}
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]
        
        for player_id in player_ids:
            # Start with stationary distribution (50/50)
            costs = ['high' if np.random.random() < 0.5 else 'low']
            
            # Generate sequence using persistence probability (Markov process)
            for t in range(1, time_horizon):
                if np.random.random() < persistence_probability:
                    # Keep same cost type (persistence)
                    costs.append(costs[t-1])
                else:
                    # Switch cost type
                    costs.append('low' if costs[t-1] == 'high' else 'high')
            
            cost_sequences[player_id] = costs
        
        return {
            'current_period': 1,               # t.txt: Set current_period = 1
            'cost_sequences': cost_sequences,  # Pre-generated cost lists (t.txt)
            'report_history': {},
            'true_cost_history': {},
            'profit_history': {},
            'market_share_history': {},
            'total_periods': time_horizon
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt with private cost and public report history (from t.txt)"""
        
        # Get current period and cost sequences
        current_period = game_state.get('current_period', 1)
        cost_sequences = game_state.get('cost_sequences', {})
        report_history = game_state.get('report_history', {})
        
        # Get current true cost for this player
        current_cost_type = 'high'  # Default
        if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_period:
            current_cost_type = cost_sequences[player_id][current_period - 1]
        
        # Format detailed history with player-specific reports
        all_reports_history_detailed = self._format_detailed_history(report_history, current_period)
        
        # Get template variables from config
        variables = get_prompt_variables(
            game_config, 
            player_id=player_id,
            current_round=current_period,
            current_cost_type=current_cost_type,
            all_reports_history_detailed=all_reports_history_detailed
        )
        
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def _format_detailed_history(self, report_history: Dict, current_period: int) -> str:
        """Format detailed history with specific player reports"""
        if current_period <= 1:
            return "No previous reports."
        
        history_lines = []
        for period_num in range(1, current_period):
            period_reports = []
            for player_id, reports in report_history.items():
                if len(reports) >= period_num:
                    period_reports.append(f"{player_id}: {reports[period_num - 1]}")
            
            if period_reports:
                history_lines.append(f"Period {period_num}: {', '.join(period_reports)}")
        
        return '; '.join(history_lines) if history_lines else "No previous reports."

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse cost report decision from LLM response using inherited mixin"""
        
        # Use the ReportParsingMixin method
        result = self.parse_report_response(response, player_id, call_id)
        
        if result:
            self.logger.debug(f"[{call_id}] Successfully parsed report: {result.get('report', 'N/A')} for {player_id}")
            return result
        
        self.logger.warning(f"[{call_id}] Could not parse report from {player_id}")
        return None

    def get_default_action(self, player_id: str, game_state: Dict, 
                         game_config: GameConfig) -> Dict[str, Any]:
        """Default report action when parsing fails - truthful reporting"""
        
        # Default to truthful reporting (report true cost)
        current_period = game_state.get('current_period', 1)
        cost_sequences = game_state.get('cost_sequences', {})
        
        current_cost = 'high'
        if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_period:
            current_cost = cost_sequences[player_id][current_period - 1]
        
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
        
        Market Allocation Algorithm from t.txt:
        1. Count players who reported "low" (N_low)
        2. If N_low = 1: "low" reporter gets market_share = 1.0, others get 0
        3. If N_low > 1: each "low" reporter gets market_share = 1/N_low, others get 0
        4. If N_low = 0: each player gets market_share = 1/number_of_players
        5. Profit = (market_price - true_cost) × market_share × market_size
        """
        
        constants = game_config.constants
        
        # Extract constants from t.txt specification
        cost_types = constants.get('cost_types', {'low': 15, 'high': 25})
        market_price = constants.get('market_price', 30)
        market_size = constants.get('market_size', 100)
        num_players = constants.get('number_of_players', 3)
        
        # Get game state data
        current_period = game_state.get('current_period', 1) if game_state else 1
        cost_sequences = game_state.get('cost_sequences', {}) if game_state else {}
        
        # Extract reports and get true costs
        players = list(actions.keys())
        reports = {}
        true_costs = {}
        true_cost_types = {}
        
        for player_id, action in actions.items():
            report = action.get('report', 'high')
            reports[player_id] = report
            
            # Get true cost for this player and period
            if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_period:
                true_cost_type = cost_sequences[player_id][current_period - 1]
                true_cost_types[player_id] = true_cost_type
                true_costs[player_id] = cost_types[true_cost_type]
            else:
                true_cost_types[player_id] = 'high'  # Default
                true_costs[player_id] = cost_types['high']
        
        # Step 1: Count players who reported "low" (from t.txt)
        low_reporters = [pid for pid, report in reports.items() if report == 'low']
        N_low = len(low_reporters)
        
        # Steps 2-4: Market Allocation Algorithm (exactly from t.txt)
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
            
            # Step 5: Calculate profit (from t.txt)
            profit = (market_price - true_costs[player_id]) * market_share * market_size
            payoffs[player_id] = profit
            
            self.logger.debug(f"Player {player_id}: report={reports[player_id]}, "
                            f"true_cost={true_cost_types[player_id]}, market_share={market_share:.3f}, profit={profit:.2f}")
        
        # Store data needed for metrics calculation
        if game_state is not None:
            game_state.update({
                'last_reports': reports.copy(),
                'last_true_cost_types': true_cost_types.copy(),
                'last_market_shares': market_shares.copy()
            })
        
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Update game state after each period"""
        
        current_period = game_state.get('current_period', 1)
        
        # Update report history
        for player_id, action in actions.items():
            if player_id not in game_state['report_history']:
                game_state['report_history'][player_id] = []
            
            report = action.get('report', 'high')
            game_state['report_history'][player_id].append(report)
        
        # Update true cost history
        cost_sequences = game_state.get('cost_sequences', {})
        for player_id in actions.keys():
            if player_id not in game_state['true_cost_history']:
                game_state['true_cost_history'][player_id] = []
            
            if player_id in cost_sequences and len(cost_sequences[player_id]) >= current_period:
                true_cost_type = cost_sequences[player_id][current_period - 1]
                game_state['true_cost_history'][player_id].append(true_cost_type)
        
        # Update market share history
        last_market_shares = game_state.get('last_market_shares', {})
        for player_id in actions.keys():
            if player_id not in game_state['market_share_history']:
                game_state['market_share_history'][player_id] = []
            
            market_share = last_market_shares.get(player_id, 0)
            game_state['market_share_history'][player_id].append(market_share)
        
        # Advance to next period
        game_state['current_period'] = current_period + 1
        
        return game_state

    def calculate_npv(self, profit_stream: List[float], discount_factor: float) -> float:
        """Calculate Net Present Value as specified in t.txt"""
        npv = 0.0
        for t, profit in enumerate(profit_stream):
            npv += (discount_factor ** t) * profit
        return npv

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for t.txt metrics calculation only"""
        
        if not game_state:
            return {}
        
        constants = game_config.constants
        discount_factor = constants.get('discount_factor', 0.95)
        
        # Get game histories
        current_period = game_state.get('current_period', 1)
        report_history = game_state.get('report_history', {})
        true_cost_history = game_state.get('true_cost_history', {})
        profit_history = game_state.get('profit_history', {})
        market_share_history = game_state.get('market_share_history', {})
        
        # Add current period's profits to history
        for player_id, profit in payoffs.items():
            if player_id not in profit_history:
                profit_history[player_id] = []
            profit_history[player_id].append(profit)
        
        # Calculate NPVs for win status determination
        npvs = {}
        strategic_inertia = {}
        deceptive_reports = {}
        appropriate_reports = {}
        profitable_periods = {}
        
        for player_id in payoffs.keys():
            # Calculate NPV from profit stream
            player_profits = profit_history.get(player_id, [])
            if player_profits:
                npvs[player_id] = self.calculate_npv(player_profits, discount_factor)
            else:
                npvs[player_id] = payoffs[player_id]
            
            # Calculate Strategic Inertia (t.txt metric)
            player_reports = report_history.get(player_id, [])
            if len(player_reports) > 1:
                repeats = sum(1 for i in range(1, len(player_reports)) 
                            if player_reports[i] == player_reports[i-1])
                strategic_inertia[player_id] = repeats / (len(player_reports) - 1)
            else:
                strategic_inertia[player_id] = 0
            
            # Count deceptive reports (report "low" when true cost is "high")
            player_true_costs = true_cost_history.get(player_id, [])
            deceptive_count = 0
            for i, report in enumerate(player_reports):
                if i < len(player_true_costs):
                    if report == 'low' and player_true_costs[i] == 'high':
                        deceptive_count += 1
            deceptive_reports[player_id] = deceptive_count
            
            # Count appropriate reports (truthful reporting)
            appropriate_count = 0
            for i, report in enumerate(player_reports):
                if i < len(player_true_costs):
                    if report == player_true_costs[i]:
                        appropriate_count += 1
            appropriate_reports[player_id] = appropriate_count
            
            # Count profitable periods (positive profit)
            profitable_count = sum(1 for profit in player_profits if profit > 0)
            profitable_periods[player_id] = profitable_count
        
        # Calculate Herfindahl-Hirschman Index (HHI) (t.txt metric)
        hhi_values = []
        for period_idx in range(len(market_share_history.get(list(payoffs.keys())[0], []))):
            period_hhi = 0
            for player_id in payoffs.keys():
                if player_id in market_share_history and period_idx < len(market_share_history[player_id]):
                    market_share_pct = market_share_history[player_id][period_idx] * 100
                    period_hhi += market_share_pct ** 2
            hhi_values.append(period_hhi)
        
        average_hhi = sum(hhi_values) / len(hhi_values) if hhi_values else 0
        
        # Calculate win status based on NPV
        max_npv = max(npvs.values()) if npvs else 0
        win_status = {pid: (1 if npvs[pid] == max_npv else 0) for pid in npvs}
        
        return {
            # Core identifiers
            'game_name': 'athey_bagwell',
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'constants': game_config.constants,
            
            # Required data for t.txt metrics calculation
            'actions': actions,
            'payoffs': payoffs,
            'npvs': npvs,
            
            # Game history data (required for t.txt metrics)
            'report_history': report_history,
            'true_cost_history': true_cost_history,
            'profit_history': profit_history,
            'market_share_history': market_share_history,
            
            # t.txt specific metrics data
            'win_status': win_status,
            'strategic_inertia': strategic_inertia,
            'deceptive_reports': deceptive_reports,
            'appropriate_reports': appropriate_reports,
            'profitable_periods': profitable_periods,
            'average_hhi': average_hhi,
            
            # Current state information
            'current_period': current_period,
            'total_periods': game_state.get('total_periods', 50),
            
            # Additional metadata
            'game_state': game_state
        }