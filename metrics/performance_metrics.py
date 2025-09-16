# metrics/performance_metrics.py

from typing import Dict, List, Any

from .metric_utils import MetricCalculator, MetricResult, GameResult, create_metric_result

class PerformanceMetricsCalculator(MetricCalculator):
    """
    Calculates standard performance metrics for all games using the exact
    algorithms specified in the t.txt documentation. This includes universal
    metrics like win rate and profit, as well as game-specific measures.
    """

    def calculate_all_performance_metrics(self, game_results: List[GameResult], player_id: str = 'challenger') -> Dict[str, MetricResult]:
        """
        Calculates all applicable performance metrics for a given list of game results.
        This method acts as a dispatcher to the appropriate game-specific function.
        """
        if not game_results:
            return {}
        
        game_name = game_results[0].game_name
        metrics = self._calculate_universal_metrics(game_results, player_id)

        if game_name == 'salop':
            metrics.update(self._calculate_salop_metrics(game_results, player_id))
        elif game_name == 'spulber':
            metrics.update(self._calculate_spulber_metrics(game_results, player_id))
        elif game_name in ['green_porter', 'athey_bagwell']:
            metrics.update(self._calculate_dynamic_game_metrics(game_results, player_id))
        
        return metrics

    def _calculate_universal_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """
        Calculates performance metrics that are common across all games.
        This corrected version correctly handles both static (profit) and dynamic (NPV) games.
        """
        game_info = game_results[0]
        N = len(game_results)
        
        is_dynamic = game_info.game_name in ['green_porter', 'athey_bagwell']
        
        # The 'payoffs' object correctly holds single-round profit for static games
        # and the final NPV for dynamic games.
        all_player_outcomes = [r.payoffs for r in game_results]
        challenger_outcomes = [r.payoffs.get(player_id, 0.0) for r in game_results]

        # --- Win Rate ---
        wins = sum(1 for outcome in all_player_outcomes if outcome and challenger_outcomes[all_player_outcomes.index(outcome)] == max(outcome.values()))
        win_rate = self.safe_divide(wins, N)
        
        # --- Average Profit / NPV ---
        avg_outcome = self.safe_mean(challenger_outcomes)
        
        # --- Profit Volatility ---
        outcome_volatility = self.safe_std(challenger_outcomes)

        profit_metric_name = "Average NPV" if is_dynamic else "Average Profit"
        
        return {
            'win_rate': create_metric_result('win_rate', win_rate, "Frequency of achieving the highest profit/NPV", 'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'average_profit': create_metric_result('average_profit', avg_outcome, f"Mean of the challenger's {profit_metric_name}", 'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name),
            'profit_volatility': create_metric_result('profit_volatility', outcome_volatility, "Standard deviation of the challenger's profit/NPV stream", 'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }

    def _calculate_salop_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates Salop-specific metrics like Market Share Captured."""
        market_shares = []
        game_info = game_results[0]
        for r in game_results:
            quantity = r.game_data.get('player_quantities', {}).get(player_id, 0)
            market_size = r.game_data.get('constants', {}).get('market_size', 1)
            market_shares.append(self.safe_divide(quantity, market_size))
            
        avg_market_share = self.safe_mean(market_shares)
        return {
            'market_share': create_metric_result('market_share', avg_market_share, "Average portion of the market served by the challenger", 'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates Spulber-specific metrics like Market Capture Rate."""
        market_captures = []
        game_info = game_results[0]
        for r in game_results:
            winners = r.game_data.get('winner_ids', [])
            if player_id in winners:
                market_captures.append(1.0 / len(winners)) # Account for ties
            else:
                market_captures.append(0.0)
        
        market_capture_rate = self.safe_mean(market_captures)
        return {
            'market_capture_rate': create_metric_result('market_capture_rate', market_capture_rate, "Frequency of winning the market by setting the lowest price", 'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name)
        }

    def _calculate_dynamic_game_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, Any]:
        """Calculates metrics specific to dynamic games like Reversion Frequency or HHI."""
        game_name = game_results[0].game_name
        game_info = game_results[0]
        metrics = {}
        
        if game_name == 'green_porter':
            # The reversion frequency is calculated post-simulation in run_experiments.py
            reversion_freqs = [r.game_data.get('reversion_frequency', 0) for r in game_results]
            avg_reversion_freq = self.safe_mean(reversion_freqs)
            metrics['reversion_frequency'] = create_metric_result('reversion_frequency', avg_reversion_freq, "Rate at which the cartel enters a punishment phase", 'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name)

        if game_name == 'athey_bagwell':
            # The average HHI is calculated post-simulation in run_experiments.py
            hhi_values = [r.game_data.get('average_hhi', 0) for r in game_results]
            avg_hhi = self.safe_mean(hhi_values)
            metrics['hhi'] = create_metric_result('hhi', avg_hhi, "Average market concentration (Herfindahl-Hirschman Index)", 'performance', game_info.game_name, game_info.experiment_type, game_info.condition_name)

        return metrics