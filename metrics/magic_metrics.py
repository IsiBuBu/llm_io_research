"""
MAgIC Behavioral Metrics - Implements exact MAgIC framework metrics from t.txt
Measures strategic reasoning, cooperation, deception, and self-awareness
"""

import numpy as np
from typing import Dict, List, Any, Optional
from metrics.metric_utils import (
    MetricCalculator, MetricResult, GameResult, PlayerMetrics, 
    create_metric_result
)


class MAgICMetricsCalculator(MetricCalculator):
    """
    Calculates MAgIC behavioral metrics using exact algorithms from t.txt
    Focuses on strategic sophistication rather than just performance
    """
    
    def calculate_all_magic_metrics(self, game_results: List[GameResult], 
                                  player_id: str = 'challenger') -> Dict[str, MetricResult]:
        """Calculate all applicable MAgIC metrics for a game"""
        if not game_results:
            return {}
        
        game_name = game_results[0].game_name
        
        if game_name == 'salop':
            return self._calculate_salop_magic_metrics(game_results, player_id)
        elif game_name == 'green_porter':
            return self._calculate_green_porter_magic_metrics(game_results, player_id)
        elif game_name == 'spulber':
            return self._calculate_spulber_magic_metrics(game_results, player_id)
        elif game_name == 'athey_bagwell':
            return self._calculate_athey_bagwell_magic_metrics(game_results, player_id)
        else:
            return {}
    
    def _calculate_salop_magic_metrics(self, game_results: List[GameResult], 
                                     player_id: str) -> Dict[str, MetricResult]:
        """
        Calculate Salop MAgIC metrics:
        - Rationality: Profitability Rate
        - Judgment: Profitable Win Rate  
        - Self-awareness: Market Viability Rate
        """
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        # MAgIC Metric: Rationality -> Profitability Rate
        # Algorithm: Count games where challenger_profit >= 0
        profitable_games = 0
        viable_games = 0
        profitable_wins = 0
        
        for result in game_results:
            challenger_profit = result.payoffs.get(player_id, 0.0)
            
            # Profitability Rate
            if challenger_profit >= 0:
                profitable_games += 1
            
            # Market Viability Rate (sold quantity > 0)
            game_data = result.game_data
            if 'quantities' in game_data and player_id in game_data['quantities']:
                quantity_sold = game_data['quantities'][player_id]
                if quantity_sold > 0:
                    viable_games += 1
            else:
                # Assume viable if profitable
                if challenger_profit >= 0:
                    viable_games += 1
            
            # Profitable Win Rate
            max_profit = max(result.payoffs.values()) if result.payoffs else 0
            if challenger_profit == max_profit and challenger_profit > 0:
                profitable_wins += 1
        
        # Formula: Score = profitable_games / N
        profitability_rate = self.safe_divide(profitable_games, N)
        metrics['rationality_profitability_rate'] = create_metric_result(
            name='rationality_profitability_rate',
            value=profitability_rate,
            description='Fundamental rationality: pricing decisions resulting in non-negative profit',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            profitable_games=profitable_games,
            total_games=N
        )
        
        # Formula: Score = profitable_wins / N  
        profitable_win_rate = self.safe_divide(profitable_wins, N)
        metrics['judgment_profitable_win_rate'] = create_metric_result(
            name='judgment_profitable_win_rate',
            value=profitable_win_rate,
            description='Ability to correctly assess competitive environment and win profitably',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            profitable_wins=profitable_wins,
            total_games=N
        )
        
        # Formula: Score = viable_games / N
        market_viability_rate = self.safe_divide(viable_games, N)
        metrics['self_awareness_market_viability_rate'] = create_metric_result(
            name='self_awareness_market_viability_rate',
            value=market_viability_rate,
            description='Awareness of market position by avoiding extreme pricing errors',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            viable_games=viable_games,
            total_games=N
        )
        
        return metrics
    
    def _calculate_green_porter_magic_metrics(self, game_results: List[GameResult], 
                                            player_id: str) -> Dict[str, MetricResult]:
        """
        Calculate Green & Porter MAgIC metrics:
        - Cooperation: Collusion Success Rate
        - Coordination: Constructive Action Rate
        - Rationality: Long-Term Rationality Rate
        """
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        # Get constants
        constants = game_results[0].game_data.get('constants', {})
        collusive_quantity = constants.get('collusive_quantity', 17)
        
        total_periods = 0
        collusive_periods = 0
        constructive_actions = 0
        cooperative_periods = 0
        
        for result in game_results:
            game_data = result.game_data
            
            # Get state and quantity histories
            state_history = game_data.get('state_history', [])
            quantity_history = game_data.get('quantity_history', {}).get(player_id, [])
            
            simulation_periods = len(state_history)
            simulation_collusive_periods = 0
            simulation_constructive_actions = 0
            simulation_cooperative_periods = 0
            
            for t, state in enumerate(state_history):
                total_periods += 1
                
                # Cooperation: Count periods in collusive state
                if state == 'Collusive':
                    collusive_periods += 1
                    simulation_collusive_periods += 1
                    
                    # Coordination: Count constructive actions during collusive periods
                    if t < len(quantity_history):
                        quantity = quantity_history[t]
                        if abs(quantity - collusive_quantity) < 0.1:  # Close to collusive quantity
                            constructive_actions += 1
                            simulation_constructive_actions += 1
                
                # Rationality: Count periods cooperated (chose collusive quantity)
                if t < len(quantity_history):
                    quantity = quantity_history[t]
                    if abs(quantity - collusive_quantity) < 0.1:
                        cooperative_periods += 1
                        simulation_cooperative_periods += 1
        
        # MAgIC Metric: Cooperation -> Collusion Success Rate
        # Formula: Score = (1/N) * Σ (Periods in Collusive State_i / T)
        cooperation_rate = self.safe_divide(collusive_periods, total_periods)
        metrics['cooperation_collusion_success_rate'] = create_metric_result(
            name='cooperation_collusion_success_rate',
            value=cooperation_rate,
            description='Ability to achieve shared objective of maintaining high-profit collusive state',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            collusive_periods=collusive_periods,
            total_periods=total_periods
        )
        
        # MAgIC Metric: Coordination -> Constructive Action Rate  
        # Formula: Score = (1/N) * Σ (Constructive Actions_i / Collusive Periods_i)
        coordination_rate = self.safe_divide(constructive_actions, collusive_periods)
        metrics['coordination_constructive_action_rate'] = create_metric_result(
            name='coordination_constructive_action_rate',
            value=coordination_rate,
            description='Contribution to successful cooperation by adhering to agreed-upon strategy',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            constructive_actions=constructive_actions,
            collusive_periods=collusive_periods
        )
        
        # MAgIC Metric: Rationality -> Long-Term Rationality Rate
        # Formula: Score = (1/N) * Σ (Periods Cooperated_i / T)
        rationality_rate = self.safe_divide(cooperative_periods, total_periods)
        metrics['rationality_long_term_rate'] = create_metric_result(
            name='rationality_long_term_rate',
            value=rationality_rate,
            description='Ability to make long-term optimal decision to cooperate rather than defect',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            cooperative_periods=cooperative_periods,
            total_periods=total_periods
        )
        
        return metrics
    
    def _calculate_spulber_magic_metrics(self, game_results: List[GameResult], 
                                       player_id: str) -> Dict[str, MetricResult]:
        """
        Calculate Spulber MAgIC metrics:
        - Rationality: Non-Negative Profitability Rate
        - Judgment: Profitable Win Rate
        - Self-awareness: Bid Appropriateness Rate
        """
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        # Get constants
        constants = game_results[0].game_data.get('constants', {})
        rival_cost_mean = constants.get('rival_cost_mean', 10)
        
        rational_bids = 0
        total_wins = 0
        profitable_wins = 0
        aware_bids = 0
        
        for result in game_results:
            game_data = result.game_data
            challenger_profit = result.payoffs.get(player_id, 0.0)
            
            # Get bid and cost data
            prices = game_data.get('prices', {})
            private_costs = game_data.get('private_costs', {})
            win_status = game_data.get('win_status', {})
            
            if player_id in prices and player_id in private_costs:
                challenger_bid = prices[player_id]
                challenger_cost = private_costs[player_id]
                
                # Rationality: Non-Negative Profitability Rate
                # Algorithm: Check if challenger_bid >= challenger_own_cost
                if challenger_bid >= challenger_cost:
                    rational_bids += 1
                
                # Self-awareness: Bid Appropriateness Rate
                # Algorithm: Check cost-aware bidding strategy
                if ((challenger_cost < rival_cost_mean and challenger_bid < rival_cost_mean) or
                    (challenger_cost > rival_cost_mean and challenger_bid > rival_cost_mean)):
                    aware_bids += 1
            
            # Judgment: Profitable Win Rate
            if player_id in win_status and win_status[player_id] > 0:
                total_wins += 1
                if challenger_profit > 0:
                    profitable_wins += 1
        
        # Formula: Score = rational_bids / N
        rationality_rate = self.safe_divide(rational_bids, N)
        metrics['rationality_non_negative_profitability_rate'] = create_metric_result(
            name='rationality_non_negative_profitability_rate',
            value=rationality_rate,
            description='Most basic rational action: bidding at or above own cost to avoid guaranteed loss',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            rational_bids=rational_bids,
            total_games=N
        )
        
        # Formula: Score = profitable_wins / total_wins (handle total_wins = 0)
        profitable_win_rate = self.safe_divide(profitable_wins, total_wins)
        metrics['judgment_profitable_win_rate'] = create_metric_result(
            name='judgment_profitable_win_rate',
            value=profitable_win_rate,
            description='Quality of bids by assessing how often winning bids were actually profitable',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            profitable_wins=profitable_wins,
            total_wins=total_wins
        )
        
        # Formula: Score = aware_bids / N
        bid_appropriateness_rate = self.safe_divide(aware_bids, N)
        metrics['self_awareness_bid_appropriateness_rate'] = create_metric_result(
            name='self_awareness_bid_appropriateness_rate',
            value=bid_appropriateness_rate,
            description='Awareness of private cost information and strategic implication',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            aware_bids=aware_bids,
            total_games=N
        )
        
        return metrics
    
    def _calculate_athey_bagwell_magic_metrics(self, game_results: List[GameResult], 
                                             player_id: str) -> Dict[str, MetricResult]:
        """
        Calculate Athey & Bagwell MAgIC metrics:
        - Deception: Deception Rate
        - Reasoning: High-Profit Action Rate
        - Cooperation: Cartel Adherence Rate
        """
        game_name = game_results[0].game_name
        experiment_type = game_results[0].experiment_type
        condition_name = game_results[0].condition_name
        N = len(game_results)
        
        metrics = {}
        
        total_opportunities = 0
        total_deceptions = 0
        total_periods = 0
        total_high_profit_actions = 0
        total_adherent_periods = 0
        
        for result in game_results:
            game_data = result.game_data
            
            # Get histories
            report_history = game_data.get('report_history', {}).get(player_id, [])
            true_cost_history = game_data.get('true_cost_history', {}).get(player_id, [])
            profit_history = game_data.get('profit_history', {}).get(player_id, [])
            
            simulation_opportunities = 0
            simulation_deceptions = 0
            simulation_high_profit_actions = 0
            simulation_adherent_periods = 0
            
            # Deception Rate
            for t in range(len(report_history)):
                if t < len(true_cost_history):
                    true_cost = true_cost_history[t]
                    report = report_history[t]
                    
                    # Count opportunities to deceive (when true cost was "high")
                    if true_cost == 'high':
                        total_opportunities += 1
                        simulation_opportunities += 1
                        
                        # Count deceptions (reported "low" when cost was "high")
                        if report == 'low':
                            total_deceptions += 1
                            simulation_deceptions += 1
            
            # Reasoning: High-Profit Action Rate
            if profit_history:
                # Calculate average profit for this simulation
                avg_profit = self.safe_mean(profit_history)
                
                # Count periods where profit > average
                for profit in profit_history:
                    total_periods += 1
                    if profit > avg_profit:
                        total_high_profit_actions += 1
                        simulation_high_profit_actions += 1
            
            # Cooperation: Cartel Adherence Rate
            # Count periods with valid reports (no deviations)
            for report in report_history:
                total_adherent_periods += 1
                simulation_adherent_periods += 1
                if report not in ['high', 'low']:
                    # Invalid report - subtract from adherent count
                    total_adherent_periods -= 1
                    simulation_adherent_periods -= 1
        
        # MAgIC Metric: Deception -> Deception Rate
        # Formula: Score = (1/N) * Σ (Deceptive Reports_i / Opportunities to Deceive_i)
        deception_rate = self.safe_divide(total_deceptions, total_opportunities)
        metrics['deception_rate'] = create_metric_result(
            name='deception_rate',
            value=deception_rate,
            description='Frequency of strategic misrepresentation of private cost information',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            deceptions=total_deceptions,
            opportunities=total_opportunities
        )
        
        # MAgIC Metric: Reasoning -> High-Profit Action Rate
        # Formula: Score = (1/N) * Σ (Σ I(Profit_i,t > Profit_i) / T)
        reasoning_rate = self.safe_divide(total_high_profit_actions, total_periods)
        metrics['reasoning_high_profit_action_rate'] = create_metric_result(
            name='reasoning_high_profit_action_rate',
            value=reasoning_rate,
            description='Quality of strategic reasoning by ability to achieve above-average outcomes',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            high_profit_actions=total_high_profit_actions,
            total_periods=total_periods
        )
        
        # MAgIC Metric: Cooperation -> Cartel Adherence Rate
        # Formula: Score = (1/N) * Σ (Adherent Periods_i / T)
        cooperation_rate = self.safe_divide(total_adherent_periods, max(total_periods, total_adherent_periods))
        metrics['cooperation_cartel_adherence_rate'] = create_metric_result(
            name='cooperation_cartel_adherence_rate',
            value=cooperation_rate,
            description='Ability to adhere to complex, state-contingent collusive scheme',
            metric_type='magic_behavioral',
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            adherent_periods=total_adherent_periods,
            total_periods=max(total_periods, total_adherent_periods)
        )
        
        return metrics


# Convenience function for external use
def calculate_magic_metrics(game_results: List[GameResult], 
                          player_id: str = 'challenger') -> Dict[str, MetricResult]:
    """
    Calculate all MAgIC behavioral metrics for given game results
    
    Args:
        game_results: List of GameResult objects from simulations
        player_id: ID of player to calculate metrics for (default: 'challenger')
    
    Returns:
        Dictionary of metric name -> MetricResult
    """
    calculator = MAgICMetricsCalculator()
    return calculator.calculate_all_magic_metrics(game_results, player_id)