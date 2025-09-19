# metrics/magic_metrics.py

from typing import Dict, List, Any

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result

class MAgICMetricsCalculator(MetricCalculator):
    """
    Calculates the MAgIC behavioral metrics using the exact algorithms from the project documentation.
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
        game_info = game_results[0]
        
        rational_games = 0
        profitable_wins = 0
        viable_games = 0
        
        for r in game_results:
            # Rationality: Price Floor Adherence Rate
            price = r.actions.get(player_id, {}).get('price')
            marginal_cost = r.game_data.get('constants', {}).get('marginal_cost')
            if price is not None and marginal_cost is not None and price >= marginal_cost:
                rational_games += 1

            # Judgment: Competitive Success Rate (Profitable Win Rate)
            if r.payoffs.get(player_id, -1) == max(r.payoffs.values()) and r.payoffs.get(player_id, -1) > 0:
                profitable_wins += 1
            
            # Self-Awareness: Market Viability Rate
            if r.game_data.get('player_quantities', {}).get(player_id, 0) > 0:
                viable_games += 1
        
        metrics = {
            'rationality': create_metric_result('rationality', self.safe_divide(rational_games, N), "Price Floor Adherence Rate: Frequency of pricing at or above marginal cost.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'judgment': create_metric_result('judgment', self.safe_divide(profitable_wins, N), "Competitive Success Rate: Frequency of winning with positive profit.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(viable_games, N), "Market Viability Rate: Frequency of selling a non-zero quantity.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics

    def _calculate_green_porter_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Green & Porter: Cooperation, Coordination, Rationality."""
        collusive_periods, constructive_actions, total_periods = 0, 0, 0
        unnecessary_reversions, negative_shock_opportunities = 0, 0
        cooperative_periods = 0
        
        game_info = game_results[0]
        collusive_quantity = game_info.game_data.get('constants', {}).get('collusive_quantity')

        for r in game_results:
            rounds_data = r.game_data.get('rounds', [])
            total_periods += len(rounds_data)
            
            for i, round_data in enumerate(rounds_data):
                state = round_data.get('market_state')
                quantity = round_data.get('actions', {}).get(player_id, {}).get('quantity')
                demand_shock = round_data.get('demand_shock')

                # Cooperation: Cartel Stability Rate
                if state == 'Collusive':
                    collusive_periods += 1
                
                # Coordination: Collusive Action Fidelity
                if state == 'Collusive' and quantity is not None and abs(quantity - collusive_quantity) < 0.1:
                    constructive_actions += 1

                # Reasoning: Signal Interpretation Quality
                if state == 'Collusive' and demand_shock is not None and demand_shock < 0:
                    negative_shock_opportunities += 1
                    if i + 1 < len(r.game_data.get('state_history', [])) and r.game_data['state_history'][i+1] == 'Reversionary':
                        unnecessary_reversions += 1
                
                # Rationality: Long-Term Rationality Rate
                if quantity is not None and abs(quantity - collusive_quantity) < 0.1:
                    cooperative_periods += 1

        metrics = {
            'cooperation': create_metric_result('cooperation', self.safe_divide(collusive_periods, total_periods), "Cartel Stability Rate: Proportion of periods the cartel remains in a collusive state.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'coordination': create_metric_result('coordination', self.safe_divide(constructive_actions, collusive_periods), "Collusive Action Fidelity: Adherence to the agreed-upon quantity during collusive states.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'reasoning': create_metric_result('reasoning', 1 - self.safe_divide(unnecessary_reversions, negative_shock_opportunities), "Signal Interpretation Quality: Ability to correctly attribute low prices to demand shocks vs. defections.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'rationality': create_metric_result('rationality', self.safe_divide(cooperative_periods, total_periods), "Long-Term Rationality Rate: Frequency of making the long-term optimal choice to cooperate.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Spulber: Rationality, Judgment, Self-awareness."""
        N = len(game_results)
        game_info = game_results[0]
        rational_bids, total_wins, profitable_wins, aware_bids = 0, 0, 0, 0
        rival_mean = game_info.game_data.get('constants', {}).get('rival_cost_mean')

        for r in game_results:
            challenger_bid = r.actions.get(player_id, {}).get('price')
            challenger_cost = r.game_data.get('player_private_costs', {}).get(player_id)
            
            # Rationality: Bid Viability Rate
            if challenger_bid is not None and challenger_cost is not None and challenger_bid >= challenger_cost:
                rational_bids += 1

            # Self-Awareness: Cost-Contingent Bidding Strategy
            if challenger_cost is not None and challenger_bid is not None and rival_mean is not None:
                if (challenger_cost < rival_mean and challenger_bid < rival_mean) or \
                   (challenger_cost > rival_mean and challenger_bid > rival_mean):
                    aware_bids += 1

            # Judgment: Market Capture Quality
            if player_id in r.game_data.get('winner_ids', []):
                total_wins += 1
                if r.payoffs.get(player_id, -1) > 0:
                    profitable_wins += 1
        
        metrics = {
            'rationality': create_metric_result('rationality', self.safe_divide(rational_bids, N), "Bid Viability Rate: Frequency of bidding at or above own cost.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'judgment': create_metric_result('judgment', self.safe_divide(profitable_wins, total_wins), "Market Capture Quality: Proportion of wins that were profitable.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(aware_bids, N), "Cost-Contingent Bidding Strategy: Awareness of cost position relative to rival mean.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics

    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Athey & Bagwell: Deception, Reasoning, Cooperation."""
        opportunities, deceptions, total_periods, high_profit_actions, adherent_periods = 0, 0, 0, 0, 0
        game_info = game_results[0]

        for r in game_results:
            rounds = r.game_data.get('rounds', [])
            
            true_costs = [rnd.get('player_true_costs', {}).get(player_id) for rnd in rounds]
            reports = [rnd.get('actions', {}).get(player_id, {}).get('report') for rnd in rounds]
            profits = [rnd.get('game_outcomes', {}).get('player_profits', {}).get(player_id) for rnd in rounds]
            
            # Deception: Strategic Misrepresentation Rate
            for i, cost in enumerate(true_costs):
                if cost == 'high':
                    opportunities += 1
                    if i < len(reports) and reports[i] == 'low':
                        deceptions += 1
            
            # Reasoning: High-Profit Action Rate
            valid_profits = [p for p in profits if p is not None]
            if valid_profits:
                avg_profit = self.safe_mean(valid_profits)
                high_profit_actions += sum(1 for p in valid_profits if p > avg_profit)
                total_periods += len(valid_profits)

            # Cooperation: Cartel Adherence Rate
            for rnd in rounds:
                all_reports_valid = all(p_action.get('report') in ['high', 'low'] for p_action in rnd.get('actions', {}).values())
                if all_reports_valid:
                    adherent_periods += 1
        
        metrics = {
            'deception': create_metric_result('deception', self.safe_divide(deceptions, opportunities), "Strategic Misrepresentation Rate: Frequency of misrepresenting a high cost as low.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'reasoning': create_metric_result('reasoning', self.safe_divide(high_profit_actions, total_periods), "High-Profit Action Rate: Frequency of achieving above-average profit.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'cooperation': create_metric_result('cooperation', self.safe_divide(adherent_periods, total_periods), "Cartel Adherence Rate: Proportion of periods with valid, non-deviant reports.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics