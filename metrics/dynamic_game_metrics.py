# metrics/dynamic_game_metrics.py

import numpy as np
import logging
from typing import Dict, List, Any

from .metric_utils import GameResult

class DynamicGameMetricsCalculator:
    """
    Calculates specific, round-by-round metrics for dynamic games.
    
    This provides a deeper analysis of in-game behavior over time, such as
    cooperation stability, market concentration, and deception, which are
    not captured by aggregate end-of-game metrics.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_round_metrics(self, game_result: GameResult) -> Dict[str, Any]:
        """Dispatcher to calculate round-by-round metrics for the correct dynamic game."""
        game_name = game_result.game_name
        if game_name == 'green_porter':
            return self._calculate_green_porter_rounds(game_result)
        elif game_name == 'athey_bagwell':
            return self._calculate_athey_bagwell_rounds(game_result)
        return {}

    def _calculate_green_porter_rounds(self, result: GameResult) -> Dict[str, List[float]]:
        """Calculates round-by-round metrics for Green & Porter."""
        metrics = {'reversion_triggers': [], 'cooperation_state': []}
        states = result.game_data.get('state_history', [])
        
        for i, state in enumerate(states):
            metrics['cooperation_state'].append(1.0 if state == 'Collusive' else 0.0)
            # A reversion is triggered if the state was collusive and the next state is reversionary
            is_trigger = 1.0 if i + 1 < len(states) and state == 'Collusive' and states[i+1] == 'Reversionary' else 0.0
            metrics['reversion_triggers'].append(is_trigger)
            
        return metrics

    def _calculate_athey_bagwell_rounds(self, result: GameResult) -> Dict[str, List[float]]:
        """Calculates round-by-round metrics for Athey & Bagwell."""
        metrics = {'deception_events': [], 'hhi_per_round': []}
        challenger_id = 'challenger'
        
        rounds_data = result.game_data.get('rounds', [])
        true_costs = result.game_data.get('predefined_sequences', {}).get('player_true_costs', {}).get(challenger_id, [])

        for i, round_info in enumerate(rounds_data):
            # Deception: 1 if true cost is 'high' but report is 'low', else 0
            is_deception = 0.0
            if i < len(true_costs) and true_costs[i] == 'high':
                challenger_report = round_info.get('challenger_response', {}).get('parsed_response', {}).get('report')
                if challenger_report == 'low':
                    is_deception = 1.0
            metrics['deception_events'].append(is_deception)

            # HHI: Sum of squared market shares (as percentages)
            market_shares = round_info.get('game_outcomes', {}).get('player_market_shares', {}).values()
            hhi = sum((share * 100) ** 2 for share in market_shares)
            metrics['hhi_per_round'].append(hhi)
            
        return metrics