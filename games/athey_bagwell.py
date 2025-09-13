# games/athey_bagwell.py

import numpy as np
from typing import Dict, Any, Optional

from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, ReportParsingMixin

class AtheyBagwellGame(DynamicGame, ReportParsingMixin):
    """
    Implements the Athey & Bagwell (2008) game of collusion with persistent cost shocks.
    """

    def __init__(self):
        super().__init__("athey_bagwell")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes game state with pre-generated, persistent cost streams for each player."""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        persistence = constants.get('persistence_probability', 0.7)
        num_players = constants.get('number_of_players', 3)
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

        np.random.seed(simulation_id)
        cost_sequences = {}
        for player_id in player_ids:
            costs = ['high' if np.random.rand() < 0.5 else 'low']
            for _ in range(1, time_horizon):
                next_cost = costs[-1] if np.random.rand() < persistence else ('low' if costs[-1] == 'high' else 'high')
                costs.append(next_cost)
            cost_sequences[player_id] = costs
            
        return {
            'current_period': 1,
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids},
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a player with their private cost and separated public report history."""
        current_period = game_state['current_period']
        true_cost = game_state['cost_sequences'][player_id][current_period - 1]
        
        report_history = game_state.get('report_history', {})
        your_history = report_history.get(player_id, [])
        other_history = {pid: reports for pid, reports in report_history.items() if pid != player_id}

        your_history_str = ", ".join(your_history) or "N/A"
        
        other_history_lines = []
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

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parses the LLM's 'high' or 'low' report using the inherited mixin."""
        return self.parse_report_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates payoffs using the Market Allocation Algorithm from t.txt."""
        constants = game_config.constants
        costs = constants.get('cost_types', {'low': 15, 'high': 25})
        market_price = constants.get('market_price', 30)
        market_size = constants.get('market_size', 100)
        
        reports = {pid: action.get('report', 'high') for pid, action in actions.items()}
        low_reporters = [pid for pid, report in reports.items() if report == 'low']
        
        market_shares = {}
        if len(low_reporters) == 1:
            for pid in actions: market_shares[pid] = 1.0 if pid == low_reporters[0] else 0.0
        elif len(low_reporters) > 1:
            share = 1.0 / len(low_reporters)
            for pid in actions: market_shares[pid] = share if pid in low_reporters else 0.0
        else:
            for pid in actions: market_shares[pid] = 1.0 / len(actions)

        payoffs = {}
        current_period = game_state['current_period']
        for pid in actions:
            true_cost = costs[game_state['cost_sequences'][pid][current_period - 1]]
            payoffs[pid] = (market_price - true_cost) * market_shares[pid] * market_size
            
        game_state['last_market_shares'] = market_shares
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig) -> Dict:
        """Updates report histories and advances the game to the next period."""
        for pid, action in actions.items():
            game_state['report_history'][pid].append(action.get('report', 'high'))
        game_state['current_period'] += 1
        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers round-specific outcomes, including actions, for detailed logging."""
        period = game_state.get('current_period', 1)
        # CORRECTED: Added 'actions': actions to the returned dictionary.
        return {
            "period": period,
            "actions": actions, # This was the missing key
            "player_true_costs": {pid: seq[period-1] for pid, seq in game_state.get('cost_sequences', {}).items()},
            "game_outcomes": {
                "player_market_shares": game_state.get('last_market_shares', {}),
                "player_profits": payoffs
            }
        }