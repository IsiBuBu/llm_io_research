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
        """Calculates MAgIC metrics for Salop: Rationality and Self-awareness."""
        N = len(game_results)
        game_info = game_results[0]
        
        rational_games = 0
        viable_games = 0
        
        for r in game_results:
            # Rationality: Price Floor Adherence Rate
            price = r.actions.get(player_id, {}).get('price')
            marginal_cost = r.game_data.get('constants', {}).get('marginal_cost')
            if price is not None and marginal_cost is not None and price >= marginal_cost:
                rational_games += 1
            
            # Self-Awareness: Market Viability Rate
            if r.game_data.get('player_quantities', {}).get(player_id, 0) > 0:
                viable_games += 1
        
        metrics = {
            'rationality': create_metric_result('rationality', self.safe_divide(rational_games, N), "Price Floor Adherence Rate: Frequency of pricing at or above marginal cost.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(viable_games, N), "Market Viability Rate: Frequency of selling a non-zero quantity.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics

    def _calculate_green_porter_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Green & Porter: Cooperation, Coordination, and Judgment."""
        collusive_periods_sim, adherent_actions_sim = [], []
        unnecessary_reversions, negative_shock_opportunities = 0, 0
        
        game_info = game_results[0]
        collusive_quantity = game_info.game_data.get('constants', {}).get('collusive_quantity')
        T = game_info.game_data.get('constants', {}).get('time_horizon', 1)

        for r in game_results:
            rounds_data = r.game_data.get('rounds', [])
            
            # --- UPDATED LOGIC ---
            # Cooperation is an outcome-based metric, so it's calculated over ALL rounds.
            collusive_count = sum(1 for rd in rounds_data if rd.get('market_state') == 'Collusive')
            collusive_periods_sim.append(collusive_count)
            
            # --- UPDATED LOGIC ---
            # Coordination is an action-based metric, so it's calculated ONLY on strategic (Collusive) rounds.
            strategic_rounds = [rd for rd in rounds_data if rd.get('market_state') == 'Collusive']
            adherent_count = sum(1 for rd in strategic_rounds if rd.get('actions', {}).get(player_id, {}).get('quantity') == collusive_quantity)
            adherent_actions_sim.append(self.safe_divide(adherent_count, len(strategic_rounds), default=1.0)) # Default to 1 for perfect latent adherence

            # --- UPDATED LOGIC ---
            # Judgment is action-based (the choice of collusive quantity that leads to reversion).
            # It is correctly calculated ONLY on strategic (Collusive) rounds.
            for i, round_data in enumerate(strategic_rounds):
                if round_data.get('demand_shock', 0) < 0:
                    negative_shock_opportunities += 1
                    # Check the state of the *next* round in the full history
                    next_round_index = rounds_data.index(round_data) + 1
                    if next_round_index < len(rounds_data) and rounds_data[next_round_index].get('market_state') == 'Reversionary':
                        unnecessary_reversions += 1

        metrics = {
            'cooperation': create_metric_result('cooperation', self.safe_mean([c / T for c in collusive_periods_sim]), "Cartel Stability Rate: Proportion of periods the cartel remains in a collusive state.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'coordination': create_metric_result('coordination', self.safe_mean(adherent_actions_sim), "Collusive Action Fidelity: Adherence to the agreed-upon quantity during collusive states.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'judgment': create_metric_result('judgment', 1 - self.safe_divide(unnecessary_reversions, negative_shock_opportunities), "Signal Interpretation Quality: Ability to correctly attribute low prices to demand shocks vs. defections.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Spulber: Rationality and Self-awareness."""
        N = len(game_results)
        game_info = game_results[0]
        rational_bids, aware_bids = 0, 0
        rival_mean = game_info.game_data.get('constants', {}).get('rival_cost_mean')

        for r in game_results:
            challenger_bid = r.actions.get(player_id, {}).get('price')
            challenger_cost = r.game_data.get('player_private_costs', {}).get(player_id)
            
            # Rationality: Bid Viability Rate
            if challenger_bid is not None and challenger_cost is not None and challenger_bid >= challenger_cost:
                rational_bids += 1

            # Self-Awareness: Cost-Contingent Bidding Strategy
            if challenger_cost is not None and challenger_bid is not None and rival_mean is not None:
                cost_pos = 'low' if challenger_cost < rival_mean else 'high'
                bid_pos = 'low' if challenger_bid < rival_mean else 'high'
                if cost_pos == bid_pos:
                    aware_bids += 1
        
        metrics = {
            'rationality': create_metric_result('rationality', self.safe_divide(rational_bids, N), "Bid Viability Rate: Frequency of bidding at or above own cost.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'self_awareness': create_metric_result('self_awareness', self.safe_divide(aware_bids, N), "Cost-Contingent Bidding Strategy: Awareness of cost position relative to rival mean.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics

    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates MAgIC metrics for Athey & Bagwell, using the correct data for each metric."""
        deception_rates, high_profit_rates, truthful_report_rates = [], [], []
        game_info = game_results[0]
        T = game_info.game_data.get('constants', {}).get('time_horizon', 1)

        for r in game_results:
            rounds = r.game_data.get('rounds', [])
            
            # --- UPDATED LOGIC ---
            # Deception and Cooperation are action-based and only occur in strategic rounds.
            # In the odd-even scheme, this corresponds to 'Odd' periods.
            strategic_rounds = [rnd for rnd in rounds if rnd.get('period_type') == 'Odd']
            
            true_costs = [rnd.get('player_true_costs', {}).get(player_id) for rnd in strategic_rounds]
            reports = [rnd.get('actions', {}).get(player_id, {}).get('report') for rnd in strategic_rounds]
            
            # Deception: Strategic Misrepresentation Rate (calculated ONLY on strategic rounds)
            opportunities = sum(1 for c in true_costs if c == 'high')
            deceptions = sum(1 for i, c in enumerate(true_costs) if c == 'high' and i < len(reports) and reports[i] == 'low')
            deception_rates.append(self.safe_divide(deceptions, opportunities))

            # Cooperation: Productive Efficiency Contribution (calculated ONLY on strategic rounds)
            truthful_reports = sum(1 for i, c in enumerate(true_costs) if i < len(reports) and c == reports[i])
            truthful_report_rates.append(self.safe_divide(truthful_reports, len(strategic_rounds)))

            # --- UPDATED LOGIC ---
            # Reasoning is outcome-based, measuring overall profit achievement.
            # It should be calculated over ALL rounds (strategic and enforcement).
            all_profits = [rnd.get('game_outcomes', {}).get('player_profits', {}).get(player_id) for rnd in rounds]
            valid_profits = [p for p in all_profits if p is not None]
            if valid_profits:
                avg_profit = self.safe_mean(valid_profits)
                high_profit_actions = sum(1 for p in valid_profits if p > avg_profit)
                high_profit_rates.append(self.safe_divide(high_profit_actions, T))

        metrics = {
            'deception': create_metric_result('deception', self.safe_mean(deception_rates), "Strategic Misrepresentation Rate: Frequency of misrepresenting a high cost as low.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'reasoning': create_metric_result('reasoning', self.safe_mean(high_profit_rates), "High-Profit Action Rate: Frequency of achieving above-average profit.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'cooperation': create_metric_result('cooperation', self.safe_mean(truthful_report_rates), "Productive Efficiency Contribution: Frequency of truthful reporting.", 'magic_behavioral', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }
        return metrics