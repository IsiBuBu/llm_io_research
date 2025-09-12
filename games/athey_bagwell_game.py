# games/athey_bagwell.py

import numpy as np
from typing import Dict, Any, Optional

from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, ReportParsingMixin

class AtheyBagwellGame(DynamicGame, ReportParsingMixin):
    """
    Implements the Athey & Bagwell (2008) game of collusion with persistent cost shocks.

    This class manages a multi-round game where players have private, persistent
    cost information ("high" or "low") and must make public reports. Payoffs
    are determined by the Market Allocation Algorithm based on these reports.
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

        # Generate persistent cost sequences for all players for the entire game
        np.random.seed(simulation_id)
        cost_sequences = {}
        for player_id in player_ids:
            costs = ['high' if np.random.rand() < 0.5 else 'low']
            for _ in range(1, time_horizon):
                if np.random.rand() < persistence:
                    costs.append(costs[-1])  # Cost persists
                else:
                    costs.append('low' if costs[-1] == 'high' else 'high') # Cost flips
            cost_sequences[player_id] = costs
            
        return {
            'current_period': 1,
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids},
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a player with their private cost and public report history."""
        current_period = game_state['current_period']
        true_cost = game_state['cost_sequences'][player_id][current_period - 1]
        
        # Format history for the prompt
        history_str = "; ".join([f"Period {t+1}: " + ", ".join([f"{pid}: {reports[t]}" for pid, reports in game_state['report_history'].items()]) for t in range(current_period - 1)])
        if not history_str:
            history_str = "No previous reports."

        variables = get_prompt_variables(
            game_config,
            player_id=player_id,
            current_round=current_period,
            current_cost_type=true_cost,
            all_reports_history_detailed=history_str
        )
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parses the LLM's 'high' or 'low' report using the inherited mixin."""
        return self.parse_report_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates payoffs using the Market Allocation Algorithm from t.txt."""
        constants = game_config.constants
        costs = {'low': constants.get('low_cost_value', 15), 'high': constants.get('high_cost_value', 25)}
        market_price = constants.get('market_price', 30)
        market_size = constants.get('market_size', 100)
        num_players = len(actions)
        
        reports = {pid: action.get('report', 'high') for pid, action in actions.items()}
        low_reporters = [pid for pid, report in reports.items() if report == 'low']
        
        # Market Allocation Algorithm
        market_shares = {}
        if len(low_reporters) == 1:
            winner_id = low_reporters[0]
            for pid in actions:
                market_shares[pid] = 1.0 if pid == winner_id else 0.0
        elif len(low_reporters) > 1:
            share = 1.0 / len(low_reporters)
            for pid in actions:
                market_shares[pid] = share if pid in low_reporters else 0.0
        else: # N_low = 0
            share = 1.0 / num_players
            for pid in actions:
                market_shares[pid] = share

        payoffs = {}
        current_period = game_state['current_period']
        for player_id in actions:
            true_cost_type = game_state['cost_sequences'][player_id][current_period - 1]
            true_cost = costs[true_cost_type]
            # Profit = (Price - True Cost) * Market Share * Market Size
            profit = (market_price - true_cost) * market_shares[player_id] * market_size
            payoffs[player_id] = profit
            
        game_state['last_market_shares'] = market_shares # Store for logging
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig) -> Dict:
        """Updates report histories and advances the game to the next period."""
        for pid, action in actions.items():
            game_state['report_history'][pid].append(action.get('report', 'high'))
        
        game_state['current_period'] += 1
        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers round-specific outcomes for detailed logging."""
        current_period = game_state.get('current_period', 1) -1
        return {
            "period": current_period,
            "player_true_costs": {pid: seq[current_period-1] for pid, seq in game_state.get('cost_sequences', {}).items()},
            "game_outcomes": {
                "player_market_shares": game_state.get('last_market_shares', {}),
                "player_profits": payoffs
            }
        }