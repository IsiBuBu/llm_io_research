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
            challenger_payoff = result.payoffs.get(player_id, 0.0)
            challenger_profits.append(challenger_payoff)
            
            # Check if challenger won (highest profit)
            max_profit = max(result.payoffs.values()) if result.payoffs else 0
            if challenger_payoff == max_profit:
                win_count += 1
        
        # Win Rate - Formula: Win Rate = (1/N) * Σ I(ChallengerProfit_i = max(AllProfits_i))
        win_rate = self.safe_divide(win_count, N)
        metrics['win_rate'] = create_metric_result(
            name='win_rate',
            value=win_rate,
            description='Frequency with which challenger achieves highest profit',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            total_games=N,
            wins=win_count
        )
        
        # Average Profit - Formula: Profit = (1/N) * Σ ChallengerProfit_i
        avg_profit = self.safe_mean(challenger_profits)
        metrics['average_profit'] = create_metric_result(
            name='average_profit',
            value=avg_profit,
            description='Mean of challenger single-period profits over all games',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            profit_stream=challenger_profits
        )
        
        # Profit Volatility - Formula: Volatility = sqrt((1/(N-1)) * Σ(ChallengerProfit_i - Profit)²)
        profit_volatility = self.safe_std(challenger_profits)
        metrics['profit_volatility'] = create_metric_result(
            name='profit_volatility',
            value=profit_volatility,
            description='Sample standard deviation of challenger profit stream',
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
        
        # Market Share Captured - Algorithm from t.txt
        for result in game_results:
            game_data = result.game_data
            challenger_market_share = 0.0
            
            # Get market share from game data
            if 'market_shares' in game_data and player_id in game_data['market_shares']:
                challenger_market_share = game_data['market_shares'][player_id]
            elif 'quantities' in game_data and 'constants' in game_data:
                # Calculate from quantities if market_shares not available
                quantities = game_data.get('quantities', {})
                constants = game_data.get('constants', {})
                market_size = constants.get('market_size', 1000)
                
                if player_id in quantities and market_size > 0:
                    challenger_quantity = quantities[player_id]
                    challenger_market_share = challenger_quantity / market_size
            
            market_shares.append(challenger_market_share)
        
        # Formula: Market Share = (1/N) * Σ Share_i
        avg_market_share = self.safe_mean(market_shares)
        metrics['market_share_captured'] = create_metric_result(
            name='market_share_captured',
            value=avg_market_share,
            description='Average portion of total market served by challenger',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            market_shares=market_shares
        )
        
        return metrics
    
    def _calculate_green_porter_metrics(self, game_results: List[GameResult], 
                                      player_id: str) -> Dict[str, MetricResult]:
        """Calculate Green & Porter specific performance metrics"""
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        # Get discount factor from first result
        constants = game_results[0].game_data.get('constants', {})
        discount_factor = constants.get('discount_factor', 0.95)
        
        # Calculate NPV-based metrics
        challenger_npvs = []
        npv_win_count = 0
        strategic_inertia_rates = []
        reversion_frequencies = []
        
        for result in game_results:
            game_data = result.game_data
            
            # Calculate NPV if profit history available
            if 'profit_history' in game_data and player_id in game_data['profit_history']:
                profit_stream = game_data['profit_history'][player_id]
                challenger_npv = self.calculate_npv(profit_stream, discount_factor)
                challenger_npvs.append(challenger_npv)
                
                # Check NPV-based win
                if 'npvs' in game_data:
                    all_npvs = game_data['npvs']
                    max_npv = max(all_npvs.values()) if all_npvs else 0
                    if challenger_npv == max_npv:
                        npv_win_count += 1
            else:
                # Fallback to single-period payoff
                challenger_npvs.append(result.payoffs.get(player_id, 0.0))
            
            # Strategic Inertia - Algorithm from t.txt
            if 'quantity_history' in game_data and player_id in game_data['quantity_history']:
                quantities = game_data['quantity_history'][player_id]
                if len(quantities) > 1:
                    repeated_choices = 0
                    for t in range(1, len(quantities)):
                        if abs(quantities[t] - quantities[t-1]) < 0.01:  # Nearly equal
                            repeated_choices += 1
                    inertia_rate = repeated_choices / (len(quantities) - 1)
                    strategic_inertia_rates.append(inertia_rate)
            
            # Reversion Frequency - Algorithm from t.txt
            if 'state_history' in game_data:
                state_history = game_data['state_history']
                if len(state_history) > 1:
                    reversions = 0
                    for t in range(1, len(state_history)):
                        if (state_history[t] == 'Price War' and 
                            state_history[t-1] == 'Collusive'):
                            reversions += 1
                    reversion_freq = reversions / (len(state_history) - 1)
                    reversion_frequencies.append(reversion_freq)
        
        # Win Rate (NPV-based) - Formula: Win Rate = (1/N) * Σ I(ChallengerNPV_i = max(AllNPVs_i))
        npv_win_rate = self.safe_divide(npv_win_count, N)
        metrics['win_rate_npv'] = create_metric_result(
            name='win_rate_npv',
            value=npv_win_rate,
            description='Frequency with which challenger achieves highest NPV',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            total_simulations=N,
            npv_wins=npv_win_count
        )
        
        # Average Profit (NPV) - Formula: NPV = (1/N) * Σ(Σ δ^(t-1) × Profit_i,t)
        avg_npv = self.safe_mean(challenger_npvs)
        metrics['average_profit_npv'] = create_metric_result(
            name='average_profit_npv',
            value=avg_npv,
            description='Average Net Present Value of challenger profit stream',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            npv_stream=challenger_npvs,
            discount_factor=discount_factor
        )
        
        # Strategic Inertia - Formula: Inertia = (1/N) * Σ((1/(T-1)) * Σ I(q_i,t = q_i,t-1))
        if strategic_inertia_rates:
            avg_strategic_inertia = self.safe_mean(strategic_inertia_rates)
            metrics['strategic_inertia'] = create_metric_result(
                name='strategic_inertia',
                value=avg_strategic_inertia,
                description='Frequency with which challenger repeats quantity choice',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                inertia_rates=strategic_inertia_rates
            )
        
        # Reversion Frequency - Formula: Reversion Freq. = (1/N) * Σ((1/(T-1)) * Σ I(State_t = Reversionary ∧ State_t-1 = Collusive))
        if reversion_frequencies:
            avg_reversion_frequency = self.safe_mean(reversion_frequencies)
            metrics['reversion_frequency'] = create_metric_result(
                name='reversion_frequency',
                value=avg_reversion_frequency,
                description='Rate at which cartel enters punishment price war phase',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                reversion_rates=reversion_frequencies
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
        market_capture_scores = []
        
        # Market Capture Rate - Algorithm from t.txt
        for result in game_results:
            game_data = result.game_data
            capture_score = 0.0
            
            # Get market capture data
            if 'market_capture' in game_data and player_id in game_data['market_capture']:
                capture_score = game_data['market_capture'][player_id]
            elif 'win_status' in game_data and 'num_winners' in game_data:
                # Calculate from win status and tie information
                if player_id in game_data['win_status'] and game_data['win_status'][player_id] > 0:
                    K = game_data.get('num_winners', 1)
                    capture_score = 1.0 / K
            
            market_capture_scores.append(capture_score)
        
        # Formula: Market Capture Rate = (1/N) * Σ (1/K_i) * I(ChallengerPrice_i = p_min,i)
        avg_market_capture = self.safe_mean(market_capture_scores)
        metrics['market_capture_rate'] = create_metric_result(
            name='market_capture_rate',
            value=avg_market_capture,
            description='Frequency with which challenger wins entire market by setting lowest price',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            capture_scores=market_capture_scores
        )
        
        return metrics
    
    def _calculate_athey_bagwell_metrics(self, game_results: List[GameResult], 
                                       player_id: str) -> Dict[str, MetricResult]:
        """Calculate Athey & Bagwell specific performance metrics"""
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        # Get discount factor from first result
        constants = game_results[0].game_data.get('constants', {})
        discount_factor = constants.get('discount_factor', 0.95)
        
        # Calculate NPV-based metrics (same as Green-Porter)
        challenger_npvs = []
        npv_win_count = 0
        strategic_inertia_rates = []
        hhi_values = []
        
        for result in game_results:
            game_data = result.game_data
            
            # Calculate NPV
            if 'profit_history' in game_data and player_id in game_data['profit_history']:
                profit_stream = game_data['profit_history'][player_id]
                challenger_npv = self.calculate_npv(profit_stream, discount_factor)
                challenger_npvs.append(challenger_npv)
                
                # Check NPV-based win
                if 'npvs' in game_data:
                    all_npvs = game_data['npvs']
                    max_npv = max(all_npvs.values()) if all_npvs else 0
                    if challenger_npv == max_npv:
                        npv_win_count += 1
            else:
                challenger_npvs.append(result.payoffs.get(player_id, 0.0))
            
            # Strategic Inertia (for reports) - Algorithm from t.txt
            if 'report_history' in game_data and player_id in game_data['report_history']:
                reports = game_data['report_history'][player_id]
                if len(reports) > 1:
                    repeated_reports = 0
                    for t in range(1, len(reports)):
                        if reports[t] == reports[t-1]:
                            repeated_reports += 1
                    inertia_rate = repeated_reports / (len(reports) - 1)
                    strategic_inertia_rates.append(inertia_rate)
            
            # Herfindahl-Hirschman Index (HHI) - Algorithm from t.txt
            if 'market_share_history' in game_data:
                market_share_history = game_data['market_share_history']
                
                # Calculate HHI for each period, then average
                period_hhis = []
                max_periods = max(len(shares) for shares in market_share_history.values()) if market_share_history else 0
                
                for t in range(max_periods):
                    period_hhi = 0.0
                    for player_shares in market_share_history.values():
                        if t < len(player_shares):
                            share_percent = player_shares[t] * 100  # Convert to percentage
                            period_hhi += share_percent ** 2
                    period_hhis.append(period_hhi)
                
                if period_hhis:
                    simulation_hhi = self.safe_mean(period_hhis)
                    hhi_values.append(simulation_hhi)
        
        # Win Rate (NPV-based)
        npv_win_rate = self.safe_divide(npv_win_count, N)
        metrics['win_rate_npv'] = create_metric_result(
            name='win_rate_npv',
            value=npv_win_rate,
            description='Frequency with which challenger achieves highest NPV',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            total_simulations=N,
            npv_wins=npv_win_count
        )
        
        # Average Profit (NPV)
        avg_npv = self.safe_mean(challenger_npvs)
        metrics['average_profit_npv'] = create_metric_result(
            name='average_profit_npv',
            value=avg_npv,
            description='Average Net Present Value of challenger profit stream',
            metric_type='performance',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            npv_stream=challenger_npvs,
            discount_factor=discount_factor
        )
        
        # Strategic Inertia (for cost reports)
        if strategic_inertia_rates:
            avg_strategic_inertia = self.safe_mean(strategic_inertia_rates)
            metrics['strategic_inertia'] = create_metric_result(
                name='strategic_inertia',
                value=avg_strategic_inertia,
                description='Frequency with which challenger repeats cost report',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                inertia_rates=strategic_inertia_rates
            )
        
        # Herfindahl-Hirschman Index (HHI)
        if hhi_values:
            avg_hhi = self.safe_mean(hhi_values)
            metrics['herfindahl_hirschman_index'] = create_metric_result(
                name='herfindahl_hirschman_index',
                value=avg_hhi,
                description='Average market concentration per period',
                metric_type='performance',
                game_name=game_name,
                experiment_type=experiment_type,
                condition_name=condition_name,
                hhi_values=hhi_values
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