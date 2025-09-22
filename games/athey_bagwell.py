# games/athey_bagwell.py

import numpy as np
from typing import Dict, Any, Optional

from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, ReportParsingMixin, QuantityParsingMixin

class AtheyBagwellGame(DynamicGame, ReportParsingMixin, QuantityParsingMixin):
    """
    Implements the Athey & Bagwell (2008) game of collusion with persistent cost shocks,
    using a two-stage, odd-even scheme to manage incentives.
    """

    def __init__(self):
        super().__init__("athey_bagwell")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes the game state for the two-stage, odd-even scheme."""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        num_players = constants.get('number_of_players', 3)
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

        # Pre-generate the persistent cost streams for all players
        np.random.seed(simulation_id)
        cost_sequences = {}
        for player_id in player_ids:
            costs = ['high' if np.random.rand() < 0.5 else 'low']
            persistence = constants.get('persistence_probability', 0.7)
            for _ in range(1, time_horizon):
                next_cost = costs[-1] if np.random.rand() < persistence else ('low' if costs[-1] == 'high' else 'high')
                costs.append(next_cost)
            cost_sequences[player_id] = costs
            
        return {
            'current_period': 1,
            'period_type': 'Odd',  # Can be 'Odd' or 'Even'
            'stage': 1,            # Can be 1 (Reporting) or 2 (Allocation)
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids},
            'last_period_reports': None # Stores reports from the previous odd period
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """
        Generates a prompt for a player ONLY during Stage 1 of an Odd period,
        which is the only time a strategic decision is made.
        """
        if game_state['period_type'] != 'Odd' or game_state['stage'] != 1:
            raise ValueError("Prompts should only be generated for Stage 1 of Odd periods.")

        current_period = game_state['current_period']
        true_cost = game_state['cost_sequences'][player_id][current_period - 1]
        
        # Format history for the prompt
        report_history = game_state.get('report_history', {})
        your_history = report_history.get(player_id, [])
        other_history = {pid: reports for pid, reports in report_history.items() if pid != player_id}
        your_history_str = ", ".join(your_history) or "N/A"
        other_history_lines = []
        # History is only up to the *previous* period
        for i in range(current_period - 1):
            line = f"Period {i+1}: " + ", ".join([f"{pid}: {reports[i]}" for pid, reports in other_history.items() if i < len(reports)])
            other_history_lines.append(line)
        other_history_str = "; ".join(other_history_lines) or "No other player reports yet."

        variables = get_prompt_variables(
            game_config, player_id=player_id, current_round=current_period,
            your_cost_type=true_cost,
            your_reports_history_detailed=your_history_str,
            all_other_reports_history_detailed=other_history_str
        )
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int) -> Optional[Dict[str, Any]]:
        """Parses the LLM's response based on the current game stage."""
        if stage == 1: # Reporting stage
            return self.parse_report_response(response, player_id, call_id)
        elif stage == 2: # Allocation stage (if it were implemented with an LLM)
             # As discussed, Stage 2 is mechanical, but we include the parser for completeness
            return self.parse_quantity_response(response, player_id, call_id)
        return None

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates payoffs based on market allocation rules."""
        constants = game_config.constants
        costs = constants.get('cost_types', {'low': 15, 'high': 25})
        market_price = constants.get('market_price', 30)
        market_size = constants.get('market_size', 100)
        
        # Actions in this context are the quantity/market share allocations
        market_shares = {pid: action.get('quantity', 0) for pid, action in actions.items()}

        payoffs = {}
        current_period = game_state['current_period']
        for pid in actions:
            true_cost = costs[game_state['cost_sequences'][pid][current_period - 1]]
            profit = (market_price - true_cost) * market_shares.get(pid, 0) * market_size
            payoffs[pid] = profit
            
        game_state['last_market_shares'] = market_shares
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig, payoffs: Dict[str, float]) -> Dict:
        """Updates the game state based on the odd-even scheme."""
        if game_state['period_type'] == 'Odd':
            if game_state['stage'] == 1:
                # Stage 1 (Reporting) is over. Record the reports.
                reports = actions
                game_state['last_period_reports'] = reports
                for pid, action in reports.items():
                    game_state['report_history'][pid].append(action.get('report', 'high'))
                # Transition to Stage 2 (Allocation) within the same period
                game_state['stage'] = 2
            elif game_state['stage'] == 2:
                # Stage 2 (Allocation) is over. Transition to the Even period.
                game_state['period_type'] = 'Even'
                game_state['stage'] = 1 # Reset stage for the next odd period
        
        elif game_state['period_type'] == 'Even':
            # Even period is over. Transition to the next Odd period.
            game_state['period_type'] = 'Odd'
            game_state['current_period'] += 1 # Only advance the period counter after the full cycle

        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers round-specific outcomes for detailed logging."""
        period = game_state.get('current_period', 1)
        return {
            "period": period,
            "period_type": game_state.get('period_type'),
            "stage": game_state.get('stage'),
            "actions": actions, # These will be reports or quantities depending on the stage
            "payoffs": payoffs,
            "player_true_costs": {pid: seq[period-1] for pid, seq in game_state.get('cost_sequences', {}).items()},
            "game_outcomes": {
                "player_market_shares": game_state.get('last_market_shares', {}),
                "player_profits": payoffs
            }
        }