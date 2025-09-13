# metrics/magic_metrics.py

from typing import Dict, List, Any

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result

class MAgICMetricsCalculator(MetricCalculator):
    """
    Calculates the MAgIC behavioral metrics using the exact algorithms from t.txt.
    These metrics focus on strategic capabilities like rationality, judgment,
    cooperation, and deception, as inspired by the MAgIC paper framework.
    """

    def calculate_all_magic_metrics(self, game_results: List[GameResult], player_id: str = 'challenger') -> Dict[str, MetricResult]:
        """Calculates all MAgIC metrics for a given game's results."""
        if not game_results:
            return {}
        
        game_name = game_results[0].game_name
        
        if game_name == 'salop':
            return self._calculate_salop_metrics(game_results, player_id)
        elif game_name == 'green_porter':
            return self._calculate_green_porter_metrics(game_results, player_id)
        elif game_name == 'spulber':
            return self._calculate_spulber_metrics(game_results, player_id)
        elif game_name == 'athey_bagwell':
            return self._calculate_athey_bagwell_metrics(game_results, player_id)
        return {}

    def _calculate_salop_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Salop: Rationality, Judgment, and Self-awareness."""
        N = len(game_results)
        profitable_games = sum(1 for r in game_results if r.payoffs.get(player_id, -1) >= 0)
        wins = sum(1 for r in game_results if r.payoffs.get(player_id) == max(r.payoffs.values()))
        profitable_wins = sum(1 for r in game_results if r.payoffs.get(player_id, -1) > 0 and r.payoffs.get(player_id) == max(r.payoffs.values()))
        viable_games = sum(1 for r in game_results if r.game_data.get('player_quantities', {}).get(player_id, 0) > 0)
        
        metrics = {
            'rationality': create_metric_result('rationality', self.safe_divide(profitable_games, N), "Profitability Rate: Frequency of non-negative profit", 'magic_behavioral', **game_results[0].__dict__),
            'judgment': create_metric_result('judgment', self.safe_divide(profitable_wins, N), "Profitable Win Rate: Frequency of winning with positive profit", 'magic_behavioral', **game_results[0].__dict__),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(viable_games, N), "Market Viability Rate: Frequency of selling a non-zero quantity", 'magic_behavioral', **game_results[0].__dict__)
        }
        return metrics

    def _calculate_green_porter_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Green & Porter: Cooperation, Coordination, Rationality."""
        collusive_periods, constructive_actions, total_periods, cooperative_periods = 0, 0, 0, 0
        collusive_quantity = game_results[0].game_data.get('constants', {}).get('collusive_quantity', 17)

        for r in game_results:
            states = r.game_data.get('state_history', [])
            quantities = r.game_data.get('quantity_history', {}).get(player_id, [])
            total_periods += len(states)
            for i, state in enumerate(states):
                if state == 'Collusive':
                    collusive_periods += 1
                    if i < len(quantities) and abs(quantities[i] - collusive_quantity) < 0.1:
                        constructive_actions += 1
                if i < len(quantities) and abs(quantities[i] - collusive_quantity) < 0.1:
                    cooperative_periods += 1
        
        metrics = {
            'cooperation': create_metric_result('cooperation', self.safe_divide(collusive_periods, total_periods), "Collusion Success Rate: Proportion of periods in the collusive state", 'magic_behavioral', **game_results[0].__dict__),
            'coordination': create_metric_result('coordination', self.safe_divide(constructive_actions, collusive_periods), "Constructive Action Rate: Adherence to collusive quantity during collusive states", 'magic_behavioral', **game_results[0].__dict__),
            'rationality': create_metric_result('rationality', self.safe_divide(cooperative_periods, total_periods), "Long-Term Rationality: Frequency of choosing the cooperative (collusive) quantity", 'magic_behavioral', **game_results[0].__dict__)
        }
        return metrics

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Spulber: Rationality, Judgment, Self-awareness."""
        N = len(game_results)
        rational_bids, total_wins, profitable_wins, aware_bids = 0, 0, 0, 0
        rival_mean = game_results[0].game_data.get('constants', {}).get('rival_cost_mean', 10)

        for r in game_results:
            challenger_bid = r.actions.get(player_id, {}).get('price', 0)
            challenger_cost = r.game_data.get('predefined_sequences', {}).get('player_private_costs', {}).get(player_id)
            
            if challenger_cost is not None:
                if challenger_bid >= challenger_cost: rational_bids += 1
                if (challenger_cost < rival_mean and challenger_bid < rival_mean) or \
                   (challenger_cost > rival_mean and challenger_bid > rival_mean):
                    aware_bids += 1

            if player_id in r.game_data.get('winner_ids', []):
                total_wins += 1
                if r.payoffs.get(player_id, -1) > 0:
                    profitable_wins += 1
        
        metrics = {
            'rationality': create_metric_result('rationality', self.safe_divide(rational_bids, N), "Non-Negative Profitability Rate: Frequency of bidding at or above own cost", 'magic_behavioral', **game_results[0].__dict__),
            'judgment': create_metric_result('judgment', self.safe_divide(profitable_wins, total_wins), "Profitable Win Rate: Proportion of wins that were profitable", 'magic_behavioral', **game_results[0].__dict__),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(aware_bids, N), "Bid Appropriateness Rate: Bidding strategy awareness relative to rival mean cost", 'magic_behavioral', **game_results[0].__dict__)
        }
        return metrics

    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Athey & Bagwell: Deception, Reasoning, Cooperation."""
        opportunities, deceptions, total_periods, high_profit_actions, adherent_periods = 0, 0, 0, 0, 0

        for r in game_results:
            # CORRECTED: Access player_true_costs directly from game_data, not nested deeper
            true_costs = r.game_data.get('player_true_costs', {}).get(player_id, [])
            
            # CORRECTED: Access reports and profits from the actions and outcomes of each round
            rounds = r.game_data.get('rounds', [])
            reports = [rnd.get('actions', {}).get(player_id, {}).get('report') for rnd in rounds]
            profits = [rnd.get('game_outcomes', {}).get('player_profits', {}).get(player_id) for rnd in rounds]
            
            for i, cost in enumerate(true_costs):
                if cost == 'high':
                    opportunities += 1
                    if i < len(reports) and reports[i] == 'low':
                        deceptions += 1
            
            valid_profits = [p for p in profits if p is not None]
            if valid_profits:
                avg_profit = self.safe_mean(valid_profits)
                high_profit_actions += sum(1 for p in valid_profits if p > avg_profit)
                total_periods += len(valid_profits)

            # CORRECTED: Check for valid reports across all players in the 'actions' dict of each round
            for rnd in rounds:
                all_reports_valid = True
                for p_action in rnd.get('actions', {}).values():
                    if p_action.get('report') not in ['high', 'low']:
                        all_reports_valid = False
                        break
                if all_reports_valid:
                    adherent_periods += 1
        
        metrics = {
            'deception': create_metric_result('deception', self.safe_divide(deceptions, opportunities), "Deception Rate: Frequency of misrepresenting a high cost as low", 'magic_behavioral', **game_results[0].__dict__),
            'reasoning': create_metric_result('reasoning', self.safe_divide(high_profit_actions, total_periods), "High-Profit Action Rate: Frequency of achieving above-average profit", 'magic_behavioral', **game_results[0].__dict__),
            'cooperation': create_metric_result('cooperation', self.safe_divide(adherent_periods, total_periods), "Cartel Adherence Rate: Proportion of periods with valid, non-deviant reports", 'magic_behavioral', **game_results[0].__dict__)
        }
        return metrics