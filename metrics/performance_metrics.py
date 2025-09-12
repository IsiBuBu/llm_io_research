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
        """Calculates performance metrics that are common across all games."""
        game_info = game_results[0]
        N = len(game_results)
        
        # For static games, profit is direct. For dynamic, it's the NPV.
        is_dynamic = game_info.game_name in ['green_porter', 'athey_bagwell']
        
        all_player_profits = []
        challenger_profits = []

        for r in game_results:
            if is_dynamic:
                # Extract NPVs calculated during the simulation
                challenger_profit = r.game_data.get('npvs', {}).get(player_id, 0.0)
                all_profits = r.game_data.get('npvs', {})
            else:
                challenger_profit = r.payoffs.get(player_id, 0.0)
                all_profits = r.payoffs
            
            challenger_profits.append(challenger_profit)
            all_player_profits.append(all_profits)

        # --- Win Rate ---
        wins = sum(1 for profits in all_player_profits if profits and challenger_profits[all_player_profits.index(profits)] == max(profits.values()))
        win_rate = self.safe_divide(wins, N)
        
        # --- Average Profit / NPV ---
        avg_profit = self.safe_mean(challenger_profits)
        
        # --- Profit Volatility ---
        profit_volatility = self.safe_std(challenger_profits)

        profit_metric_name = "Average NPV" if is_dynamic else "Average Profit"
        
        return {
            'win_rate': create_metric_result('win_rate', win_rate, "Frequency of achieving the highest profit/NPV", 'performance', **game_info.__dict__),
            'average_profit': create_metric_result('average_profit', avg_profit, f"Mean of the challenger's {profit_metric_name}", 'performance', **game_info.__dict__),
            'profit_volatility': create_metric_result('profit_volatility', profit_volatility, "Standard deviation of the challenger's profit/NPV stream", 'performance', **game_info.__dict__)
        }

    def _calculate_salop_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates Salop-specific metrics like Market Share Captured."""
        market_shares = []
        for r in game_results:
            # This assumes quantity_sold and market_size are logged in game_data
            quantity = r.game_data.get('player_quantities', {}).get(player_id, 0)
            market_size = r.game_data.get('constants', {}).get('market_size', 1)
            market_shares.append(self.safe_divide(quantity, market_size))
            
        avg_market_share = self.safe_mean(market_shares)
        return {
            'market_share': create_metric_result('market_share', avg_market_share, "Average portion of the market served by the challenger", 'performance', **game_results[0].__dict__)
        }

    def _calculate_spulber_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, MetricResult]:
        """Calculates Spulber-specific metrics like Market Capture Rate."""
        market_captures = []
        for r in game_results:
            winners = r.game_data.get('winner_ids', [])
            if player_id in winners:
                market_captures.append(1.0 / len(winners)) # Account for ties
            else:
                market_captures.append(0.0)
        
        market_capture_rate = self.safe_mean(market_captures)
        return {
            'market_capture_rate': create_metric_result('market_capture_rate', market_capture_rate, "Frequency of winning the market by setting the lowest price", 'performance', **game_results[0].__dict__)
        }

    def _calculate_dynamic_game_metrics(self, game_results: List[GameResult], player_id: str) -> Dict[str, Any]:
        """Calculates metrics specific to dynamic games like Reversion Frequency or HHI."""
        game_name = game_results[0].game_name
        metrics = {}
        
        if game_name == 'green_porter':
            reversion_freqs = [r.game_data.get('reversion_frequency', 0) for r in game_results]
            avg_reversion_freq = self.safe_mean(reversion_freqs)
            metrics['reversion_frequency'] = create_metric_result('reversion_frequency', avg_reversion_freq, "Rate at which the cartel enters a punishment phase", 'performance', **game_results[0].__dict__)

        if game_name == 'athey_bagwell':
            hhi_values = [r.game_data.get('average_hhi', 0) for r in game_results]
            avg_hhi = self.safe_mean(hhi_values)
            metrics['hhi'] = create_metric_result('hhi', avg_hhi, "Average market concentration (Herfindahl-Hirschman Index)", 'performance', **game_results[0].__dict__)

        return metrics