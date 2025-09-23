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
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids}
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """
        Generates a prompt for a player ONLY during an Odd period,
        which is the only time a strategic decision is made.
        """
        current_period = game_state['current_period']
        if current_period % 2 == 0:
             raise ValueError("Prompts should not be generated for Even periods.")

        true_cost = game_state['cost_sequences'][player_id][current_period - 1]
        
        # Format history for the prompt
        report_history = game_state.get('report_history', {})
        your_history = report_history.get(player_id, [])
        other_history = {pid: reports for pid, reports in report_history.items() if pid != player_id}
        your_history_str = ", ".join(your_history) or "N/A"
        other_history_lines = []
        
        # History is only up to the *previous* period
        for i in range(len(next(iter(other_history.values()), []))):
            line = f"Period {i*2+1}: " + ", ".join([f"{pid}: {reports[i]}" for pid, reports in other_history.items()])
            other_history_lines.append(line)
        other_history_str = "; ".join(other_history_lines) or "No other player reports yet."

        variables = get_prompt_variables(
            game_config, player_id=player_id, current_round=current_period,
            your_cost_type=true_cost,
            your_reports_history_detailed=your_history_str,
            all_other_reports_history_detailed=other_history_str
        )
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        """Parses the LLM's report decision."""
        return self.parse_report_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates payoffs based on market allocation rules."""
        constants = game_config.constants
        costs = constants.get('cost_types', {'low': 15, 'high': 25})
        market_price = constants.get('market_price', 30)
        market_size = constants.get('market_size', 100)
        
        market_shares = {pid: action.get('quantity', 0) for pid, action in actions.items()}

        payoffs = {}
        current_period = game_state['current_period']
        for pid in actions:
            true_cost_type = game_state['cost_sequences'][pid][current_period - 1]
            true_cost = costs[true_cost_type]
            profit = (market_price - true_cost) * market_shares.get(pid, 0) * market_size
            payoffs[pid] = profit
            
        game_state['last_market_shares'] = market_shares
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig, payoffs: Dict[str, float]) -> Dict:
        """Updates the game state by recording the most recent reports."""
        is_report_action = any('report' in action for action in actions.values())
        if is_report_action:
            for pid, action in actions.items():
                 game_state['report_history'][pid].append(action.get('report', 'high'))
        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers round-specific outcomes for detailed logging."""
        period = game_state.get('current_period', 1)
        period_type = "Odd" if period % 2 != 0 else "Even"

        return {
            "period": period,
            "period_type": period_type,
            "actions": actions, 
            "payoffs": payoffs,
            "player_true_costs": {pid: seq[period-1] for pid, seq in game_state.get('cost_sequences', {}).items() if period-1 < len(seq)},
            "game_outcomes": {
                "player_market_shares": game_state.get('last_market_shares', {}),
                "player_profits": payoffs
            }
        }
