"""
Performance Metrics - Standard game theory metrics as specified in t.txt
Implements exact algorithms and formulas for all four economic games
"""

import numpy as np
from typing import Dict, List, Any, Optional
from metrics.metric_utils import (
    MetricCalculator, MetricResult, GameResult, PlayerMetrics, 
    create_metric_result
)


class PerformanceMetricsCalculator(MetricCalculator):
    """
    Calculates standard performance metrics for all games using exact algorithms from t.txt
    """
    
    def calculate_all_performance_metrics(self, game_results: List[GameResult], 
                                        player_id: str = 'challenger') -> Dict[str, MetricResult]:
        """Calculate all applicable performance metrics for a game"""
        if not game_results:
            return {}
        
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        
        metrics = {}
        
        # Universal metrics (all games)
        metrics.update(self._calculate_universal_metrics(game_results, player_id))
        
        # Game-specific metrics
        if game_name == 'salop':
            metrics.update(self._calculate_salop_metrics(game_results, player_id))
        elif game_name == 'green_porter':
            metrics.update(self._calculate_green_porter_metrics(game_results, player_id))
        elif game_name == 'spulber':
            metrics.update(self._calculate_spulber_metrics(game_results, player_id))
        elif game_name == 'athey_bagwell':
            metrics.update(self._calculate_athey_bagwell_metrics(game_results, player_id))
        
        return metrics
    
    def _calculate_universal_metrics(self, game_results: List[GameResult], 
                                   player_id: str) -> Dict[str, MetricResult]:
        """Calculate metrics common to all games"""
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        # Extract challenger data
        challenger_profits = []
        win_count = 0
        
        for result in game_results:
            challenger_profit = result.payoffs.get(player_id, 0.0)
            challenger_profits.append(challenger_profit)
            
            # Win determination - highest profit
            max_profit = max(result.payoffs.values()) if result.payoffs else 0
            if challenger_profit == max_profit:
                win_count += 1
        
        # Win Rate: Number of Games with Highest Profit / Total Games Played
        win_rate = self.safe_divide(win_count, N)
        metrics['win_rate'] = create_metric_result(
            name='win_rate',
            value=win_rate,
            description='Number of Games with Highest Profit / Total Games Played',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            wins=win_count,
            total_games=N
        )
        
        # Average Profit: Σ (Profit_i) / Total Games Played
        avg_profit = self.safe_mean(challenger_profits)
        metrics['average_profit'] = create_metric_result(
            name='average_profit',
            value=avg_profit,
            description='Σ (Profit_i) / Total Games Played',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            total_profit=sum(challenger_profits),
            total_games=N
        )
        
        # Profit Volatility: Standard Deviation of profits
        profit_volatility = self.safe_std(challenger_profits)
        metrics['profit_volatility'] = create_metric_result(
            name='profit_volatility',
            value=profit_volatility,
            description='Standard Deviation of {Profit_period_1, Profit_period_2, ...}',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            profit_stream=challenger_profits
        )
        
        return metrics
    
    def _calculate_salop_metrics(self, game_results: List[GameResult], 
                               player_id: str) -> Dict[str, MetricResult]:
        """Calculate Salop-specific performance metrics"""
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        market_shares = []
        profit_margins = []
        
        for result in game_results:
            challenger_profit = result.payoffs.get(player_id, 0.0)
            
            # Extract Salop-specific data
            salop_data = result.game_data.get('salop_metrics', {})
            
            # Market Share Captured: Quantity Sold by Firm / Total Market Size (L)
            quantities = salop_data.get('quantities', {})
            if player_id in quantities:
                quantity_sold = quantities[player_id]
                market_size = result.game_data.get('constants', {}).get('market_size', 1000)
                market_share = self.safe_divide(quantity_sold, market_size)
                market_shares.append(market_share)
            
            # Profit Margin: (Price - Marginal Cost) / Price
            prices = salop_data.get('prices', {})
            if player_id in prices:
                price = prices[player_id]
                marginal_cost = result.game_data.get('constants', {}).get('marginal_cost', 8)
                if price > 0:
                    profit_margin = (price - marginal_cost) / price
                    profit_margins.append(profit_margin)
        
        # Average Market Share Captured
        if market_shares:
            avg_market_share = self.safe_mean(market_shares)
            metrics['market_share_captured'] = create_metric_result(
                name='market_share_captured',
                value=avg_market_share,
                description='Quantity Sold by Firm / Total Market Size (L)',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                market_shares=market_shares
            )
        
        # Average Profit Margin
        if profit_margins:
            avg_profit_margin = self.safe_mean(profit_margins)
            metrics['profit_margin'] = create_metric_result(
                name='profit_margin',
                value=avg_profit_margin,
                description='(Price - Marginal Cost) / Price',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                profit_margins=profit_margins
            )
        
        return metrics
    
    def _calculate_spulber_metrics(self, game_results: List[GameResult], 
                                 player_id: str) -> Dict[str, MetricResult]:
        """Calculate Spulber-specific performance metrics"""
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        win_count = 0
        profit_margins = []
        market_capture_count = 0
        
        for result in game_results:
            challenger_profit = result.payoffs.get(player_id, 0.0)
            
            # Extract Spulber-specific data
            spulber_data = result.game_data.get('spulber_metrics', {})
            
            # Win Rate (already calculated in universal metrics, but track for Spulber context)
            challenger_won = spulber_data.get('challenger_won', False)
            if challenger_won:
                win_count += 1
                market_capture_count += 1
            
            # Profit Margin calculation
            challenger_price = spulber_data.get('challenger_price', 0)
            challenger_cost = spulber_data.get('challenger_cost', 8)
            if challenger_price > 0:
                profit_margin = (challenger_price - challenger_cost) / challenger_price
                profit_margins.append(profit_margin)
        
        # Market Capture Rate: Number of Markets Won / Total Markets
        market_capture_rate = self.safe_divide(market_capture_count, N)
        metrics['market_capture_rate'] = create_metric_result(
            name='market_capture_rate',
            value=market_capture_rate,
            description='Number of Markets Won / Total Markets',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            markets_won=market_capture_count,
            total_markets=N
        )
        
        # Average Profit Margin
        if profit_margins:
            avg_profit_margin = self.safe_mean(profit_margins)
            metrics['profit_margin'] = create_metric_result(
                name='profit_margin',
                value=avg_profit_margin,
                description='(Price - Cost) / Price for winner-take-all auctions',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                profit_margins=profit_margins
            )
        
        return metrics
    
    def _calculate_green_porter_metrics(self, game_results: List[GameResult], 
                                      player_id: str) -> Dict[str, MetricResult]:
        """Calculate Green-Porter specific performance metrics"""
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        npv_values = []
        reversion_frequencies = []
        
        for result in game_results:
            # Extract Green-Porter specific data
            gp_data = result.game_data.get('green_porter_metrics', {})
            
            # Calculate NPV from profit stream
            profit_stream = gp_data.get('profit_stream', [])
            if profit_stream:
                discount_factor = result.game_data.get('constants', {}).get('discount_factor', 0.95)
                npv = self.calculate_npv(profit_stream, discount_factor)
                npv_values.append(npv)
            
            # Reversion Frequency
            reversion_frequency = gp_data.get('reversion_frequency', 0)
            reversion_frequencies.append(reversion_frequency)
        
        # Win Rate NPV: Based on NPV comparison across players
        npv_wins = 0
        for i, result in enumerate(game_results):
            if i < len(npv_values):
                challenger_npv = npv_values[i]
                
                # Compare with other players' NPVs (if available)
                all_npvs = []
                gp_data = result.game_data.get('green_porter_metrics', {})
                for pid in result.players:
                    if pid in gp_data.get('all_npvs', {}):
                        all_npvs.append(gp_data['all_npvs'][pid])
                
                if all_npvs and challenger_npv == max(all_npvs):
                    npv_wins += 1
        
        win_rate_npv = self.safe_divide(npv_wins, len(npv_values)) if npv_values else 0
        metrics['win_rate_npv'] = create_metric_result(
            name='win_rate_npv',
            value=win_rate_npv,
            description='Win rate based on Net Present Value comparison',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            npv_wins=npv_wins,
            total_games=len(npv_values)
        )
        
        # Average NPV
        if npv_values:
            avg_npv = self.safe_mean(npv_values)
            metrics['average_npv'] = create_metric_result(
                name='average_npv',
                value=avg_npv,
                description='Average Net Present Value across all simulations',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                npv_values=npv_values
            )
        
        # Average Reversion Frequency
        avg_reversion_frequency = self.safe_mean(reversion_frequencies)
        metrics['reversion_frequency'] = create_metric_result(
            name='reversion_frequency',
            value=avg_reversion_frequency,
            description='Average frequency of punishment phases',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            reversion_frequencies=reversion_frequencies
        )
        
        return metrics
    
    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], 
                                       player_id: str) -> Dict[str, MetricResult]:
        """Calculate Athey-Bagwell specific performance metrics"""
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        npv_values = []
        deception_rates = []
        
        for result in game_results:
            # Extract Athey-Bagwell specific data
            ab_data = result.game_data.get('athey_bagwell_metrics', {})
            
            # Calculate NPV from profit stream
            profit_stream = ab_data.get('profit_stream', [])
            if profit_stream:
                discount_factor = result.game_data.get('constants', {}).get('discount_factor', 0.95)
                npv = self.calculate_npv(profit_stream, discount_factor)
                npv_values.append(npv)
            
            # Deception Rate
            deception_rate = ab_data.get('deception_rate', 0)
            deception_rates.append(deception_rate)
        
        # Average NPV
        if npv_values:
            avg_npv = self.safe_mean(npv_values)
            metrics['average_npv'] = create_metric_result(
                name='average_npv',
                value=avg_npv,
                description='Average Net Present Value across all simulations',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                npv_values=npv_values
            )
            
            # NPV Volatility
            npv_volatility = self.safe_std(npv_values)
            metrics['npv_volatility'] = create_metric_result(
                name='npv_volatility',
                value=npv_volatility,
                description='Standard deviation of NPV across simulations',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                npv_values=npv_values
            )
        
        # Average Deception Rate
        avg_deception_rate = self.safe_mean(deception_rates)
        metrics['deception_rate'] = create_metric_result(
            name='deception_rate',
            value=avg_deception_rate,
            description='Frequency of strategic misrepresentation',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            deception_rates=deception_rates
        )
        
        return metrics


# Convenience function for external use
def calculate_performance_metrics(game_results: List[GameResult], 
                                player_id: str = 'challenger') -> Dict[str, MetricResult]:
    """
    Calculate all performance metrics for given game results
    
    Args:
        game_results: List of GameResult objects from simulations
        player_id: ID of player to calculate metrics for (default: 'challenger')
    
    Returns:
        Dictionary of metric name -> MetricResult
    """
    calculator = PerformanceMetricsCalculator()
    return calculator.calculate_all_performance_metrics(game_results, player_id)


# Utility functions for specific metric calculations
def calculate_win_rate(game_results: List[GameResult], player_id: str = 'challenger') -> float:
    """Calculate simple win rate across games"""
    if not game_results:
        return 0.0
    
    wins = 0
    for result in game_results:
        player_profit = result.payoffs.get(player_id, 0.0)
        max_profit = max(result.payoffs.values()) if result.payoffs else 0
        if player_profit == max_profit:
            wins += 1
    
    return wins / len(game_results)


def calculate_profit_statistics(game_results: List[GameResult], 
                              player_id: str = 'challenger') -> Dict[str, float]:
    """Calculate comprehensive profit statistics"""
    profits = [result.payoffs.get(player_id, 0.0) for result in game_results]
    
    if not profits:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    
    return {
        'mean': np.mean(profits),
        'std': np.std(profits, ddof=1) if len(profits) > 1 else 0.0,
        'min': np.min(profits),
        'max': np.max(profits)
    }