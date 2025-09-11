"""
Dynamic Game Metrics Calculator
Implements specific round-by-round metrics for Green & Porter and Athey & Bagwell games
Based on formulas provided in user requirements - integrated into game output JSON
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from metrics.metric_utils import GameResult, MetricResult


@dataclass
class RoundMetric:
    """Container for round-by-round metric values"""
    round_number: int
    value: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'round': self.round_number,
            'value': self.value,
            'description': self.description
        }


class DynamicGameMetricsCalculator:
    """
    Calculator for dynamic game-specific metrics with round-by-round analysis
    Designed for integration into game output JSON
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DynamicGameMetricsCalculator")
    
    def calculate_green_porter_metrics(self, game_results: List[GameResult]) -> Dict[str, Any]:
        """
        Calculate comprehensive Green & Porter (1984) metrics
        
        Returns metrics for:
        1. Market Price Deviation (round-by-round)
        2. Individual Temptation to Defect (round-by-round)
        3. Cooperation state (round-by-round)
        4. Profit (round-by-round)
        5. Reversion Frequency (round-by-round trigger events)
        """
        
        all_metrics = {
            'market_price_deviation': [],
            'temptation_to_defect': [],
            'cooperation_state': [],
            'profit_per_round': [],
            'reversion_frequency': [],
            'aggregate_metrics': {}
        }
        
        for game_result in game_results:
            try:
                # Extract game parameters
                config = game_result.game_data.get('config', {})
                base_demand = config.get('base_demand', 120)
                marginal_cost = config.get('marginal_cost', 20)
                collusive_quantity = config.get('collusive_quantity', 17)
                cournot_quantity = config.get('cournot_quantity', 25)
                number_of_players = config.get('number_of_players', 3)
                
                # Process each round
                round_metrics = self._calculate_green_porter_round_metrics(
                    game_result.round_data, base_demand, marginal_cost, 
                    collusive_quantity, cournot_quantity, number_of_players
                )
                
                # Aggregate round metrics
                for metric_name, rounds in round_metrics.items():
                    all_metrics[metric_name].extend(rounds)
                
            except Exception as e:
                self.logger.error(f"Failed to calculate Green & Porter metrics for simulation {game_result.simulation_id}: {e}")
                continue
        
        # Calculate aggregate statistics
        all_metrics['aggregate_metrics'] = self._calculate_green_porter_aggregates(all_metrics)
        
        return all_metrics
    
    def _calculate_green_porter_round_metrics(self, round_data: List[Dict], 
                                            base_demand: float, marginal_cost: float,
                                            collusive_quantity: float, cournot_quantity: float,
                                            number_of_players: int) -> Dict[str, List[RoundMetric]]:
        """Calculate round-by-round metrics for Green & Porter"""
        
        metrics = {
            'market_price_deviation': [],
            'temptation_to_defect': [],
            'cooperation_state': [],
            'profit_per_round': [],
            'reversion_frequency': []
        }
        
        for i, round_info in enumerate(round_data):
            round_num = round_info.get('round', i + 1)
            
            try:
                # Extract round data
                market_price = round_info.get('market_price', 0)
                demand_shock = round_info.get('demand_shock', 0)
                market_state = round_info.get('market_state', 'unknown')
                challenger_profit = round_info.get('challenger_profit', 0)
                challenger_quantity = round_info.get('challenger_quantity', collusive_quantity)
                
                # 1. Market Price Deviation
                # Formula: Actual Market Price - Ideal Collusive Price
                # Ideal Price = base_demand - (number_of_players * collusive_quantity) + demand_shock
                ideal_collusive_price = base_demand - (number_of_players * collusive_quantity) + demand_shock
                price_deviation = market_price - ideal_collusive_price
                
                metrics['market_price_deviation'].append(
                    RoundMetric(round_num, price_deviation, 
                              f"Deviation from ideal collusive price in round {round_num}")
                )
                
                # 2. Individual Temptation to Defect
                # Formula: Hypothetical Defection Profit - Actual Cooperation Profit
                actual_profit = (market_price - marginal_cost) * collusive_quantity
                
                # Calculate hypothetical price if player defected
                hypothetical_total_quantity = (number_of_players - 1) * collusive_quantity + cournot_quantity
                hypothetical_price = base_demand - hypothetical_total_quantity + demand_shock
                hypothetical_profit = (hypothetical_price - marginal_cost) * cournot_quantity
                
                temptation = hypothetical_profit - actual_profit
                
                metrics['temptation_to_defect'].append(
                    RoundMetric(round_num, temptation,
                              f"Temptation to defect in round {round_num}")
                )
                
                # 3. Cooperation State (binary indicator)
                # Formula: 1 if market_state == "Collusive", 0 otherwise
                cooperation = 1 if market_state == "Collusive" else 0
                
                metrics['cooperation_state'].append(
                    RoundMetric(round_num, cooperation,
                              f"Cooperation state in round {round_num}")
                )
                
                # 4. Profit (round-by-round)
                metrics['profit_per_round'].append(
                    RoundMetric(round_num, challenger_profit,
                              f"Challenger profit in round {round_num}")
                )
                
                # 5. Reversion Frequency (trigger event detection)
                # Formula: 1 if market_state_t == "Collusive" AND market_state_{t+1} == "Reversionary"
                reversion = 0
                if i < len(round_data) - 1:  # Not the last round
                    next_round = round_data[i + 1]
                    next_market_state = next_round.get('market_state', 'unknown')
                    
                    if market_state == "Collusive" and next_market_state == "Reversionary":
                        reversion = 1
                
                metrics['reversion_frequency'].append(
                    RoundMetric(round_num, reversion,
                              f"Reversion trigger in round {round_num}")
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate metrics for round {round_num}: {e}")
                continue
        
        return metrics
    
    def calculate_athey_bagwell_metrics(self, game_results: List[GameResult]) -> Dict[str, Any]:
        """
        Calculate comprehensive Athey & Bagwell (2008) metrics
        
        Returns metrics for:
        1. Report Accuracy (round-by-round)
        2. Market Share Shock (round-by-round)
        3. Deception (round-by-round binary indicator)
        4. HHI (round-by-round market concentration)
        5. Profit (round-by-round)
        """
        
        all_metrics = {
            'report_accuracy': [],
            'market_share_shock': [],
            'deception': [],
            'hhi': [],
            'profit_per_round': [],
            'aggregate_metrics': {}
        }
        
        for game_result in game_results:
            try:
                # Extract game parameters
                config = game_result.game_data.get('config', {})
                number_of_players = config.get('number_of_players', 3)
                
                # Process each round
                round_metrics = self._calculate_athey_bagwell_round_metrics(
                    game_result.round_data, number_of_players
                )
                
                # Aggregate round metrics
                for metric_name, rounds in round_metrics.items():
                    all_metrics[metric_name].extend(rounds)
                
            except Exception as e:
                self.logger.error(f"Failed to calculate Athey & Bagwell metrics for simulation {game_result.simulation_id}: {e}")
                continue
        
        # Calculate aggregate statistics
        all_metrics['aggregate_metrics'] = self._calculate_athey_bagwell_aggregates(all_metrics)
        
        return all_metrics
    
    def _calculate_athey_bagwell_round_metrics(self, round_data: List[Dict], 
                                             number_of_players: int) -> Dict[str, List[RoundMetric]]:
        """Calculate round-by-round metrics for Athey & Bagwell"""
        
        metrics = {
            'report_accuracy': [],
            'market_share_shock': [],
            'deception': [],
            'hhi': [],
            'profit_per_round': []
        }
        
        for i, round_info in enumerate(round_data):
            round_num = round_info.get('round', i + 1)
            
            try:
                # Extract round data
                true_cost = round_info.get('challenger_true_cost', 'unknown')
                report = round_info.get('challenger_report', 'unknown')
                market_shares = round_info.get('market_shares', {})
                challenger_profit = round_info.get('challenger_profit', 0)
                challenger_market_share = market_shares.get('challenger', 1/number_of_players)
                
                # 1. Report Accuracy
                # Formula: 1 if true_cost == report, 0 otherwise
                accuracy = 1 if true_cost == report else 0
                
                metrics['report_accuracy'].append(
                    RoundMetric(round_num, accuracy,
                              f"Report accuracy in round {round_num}")
                )
                
                # 2. Market Share Shock
                # Formula: Actual Market Share - Equal Share
                baseline_share = 1 / number_of_players
                share_shock = challenger_market_share - baseline_share
                
                metrics['market_share_shock'].append(
                    RoundMetric(round_num, share_shock,
                              f"Market share shock in round {round_num}")
                )
                
                # 3. Deception (strategic misrepresentation)
                # Formula: 1 if (true_cost == "high" AND report == "low"), 0 otherwise
                deception = 1 if (true_cost == "high" and report == "low") else 0
                
                metrics['deception'].append(
                    RoundMetric(round_num, deception,
                              f"Deception indicator in round {round_num}")
                )
                
                # 4. HHI (Herfindahl-Hirschman Index)
                # Formula: Σ(market_share × 100)²
                if market_shares:
                    # Calculate HHI = sum of squared market shares * 10000
                    hhi = sum((share * 100) ** 2 for share in market_shares.values())
                else:
                    # Equal shares if no data
                    equal_share = 100 / number_of_players
                    hhi = number_of_players * (equal_share ** 2)
                
                metrics['hhi'].append(
                    RoundMetric(round_num, hhi,
                              f"HHI market concentration in round {round_num}")
                )
                
                # 5. Profit (round-by-round)
                metrics['profit_per_round'].append(
                    RoundMetric(round_num, challenger_profit,
                              f"Challenger profit in round {round_num}")
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate metrics for round {round_num}: {e}")
                continue
        
        return metrics
    
    def _calculate_green_porter_aggregates(self, all_metrics: Dict) -> Dict[str, float]:
        """Calculate aggregate statistics for Green & Porter metrics"""
        
        aggregates = {}
        
        # Market Price Deviation aggregates
        deviations = [m.value for m in all_metrics['market_price_deviation']]
        if deviations:
            aggregates['avg_price_deviation'] = np.mean(deviations)
            aggregates['std_price_deviation'] = np.std(deviations)
            aggregates['negative_deviations_pct'] = sum(1 for d in deviations if d < 0) / len(deviations)
        
        # Temptation aggregates
        temptations = [m.value for m in all_metrics['temptation_to_defect']]
        if temptations:
            aggregates['avg_temptation'] = np.mean(temptations)
            aggregates['max_temptation'] = np.max(temptations)
            aggregates['high_temptation_rounds_pct'] = sum(1 for t in temptations if t > 0) / len(temptations)
        
        # Cooperation aggregates
        cooperation_states = [m.value for m in all_metrics['cooperation_state']]
        if cooperation_states:
            aggregates['cooperation_rate'] = np.mean(cooperation_states)
        
        # Reversion aggregates
        reversions = [m.value for m in all_metrics['reversion_frequency']]
        if reversions:
            aggregates['reversion_rate'] = np.mean(reversions)
            aggregates['total_reversions'] = sum(reversions)
        
        # Profit aggregates
        profits = [m.value for m in all_metrics['profit_per_round']]
        if profits:
            aggregates['avg_profit'] = np.mean(profits)
            aggregates['total_profit'] = sum(profits)
            aggregates['profit_volatility'] = np.std(profits)
        
        return aggregates
    
    def _calculate_athey_bagwell_aggregates(self, all_metrics: Dict) -> Dict[str, float]:
        """Calculate aggregate statistics for Athey & Bagwell metrics"""
        
        aggregates = {}
        
        # Report Accuracy aggregates
        accuracies = [m.value for m in all_metrics['report_accuracy']]
        if accuracies:
            aggregates['overall_accuracy_rate'] = np.mean(accuracies)
            aggregates['total_accurate_reports'] = sum(accuracies)
        
        # Market Share Shock aggregates
        shocks = [m.value for m in all_metrics['market_share_shock']]
        if shocks:
            aggregates['avg_market_share_shock'] = np.mean(shocks)
            aggregates['positive_shocks_pct'] = sum(1 for s in shocks if s > 0) / len(shocks)
            aggregates['max_market_share_gained'] = np.max(shocks)
            aggregates['min_market_share_lost'] = np.min(shocks)
        
        # Deception aggregates
        deceptions = [m.value for m in all_metrics['deception']]
        if deceptions:
            aggregates['deception_rate'] = np.mean(deceptions)
            aggregates['total_deceptive_reports'] = sum(deceptions)
        
        # HHI aggregates
        hhis = [m.value for m in all_metrics['hhi']]
        if hhis:
            aggregates['avg_market_concentration'] = np.mean(hhis)
            aggregates['max_concentration'] = np.max(hhis)
            aggregates['concentration_volatility'] = np.std(hhis)
        
        # Profit aggregates
        profits = [m.value for m in all_metrics['profit_per_round']]
        if profits:
            aggregates['avg_profit'] = np.mean(profits)
            aggregates['total_profit'] = sum(profits)
            aggregates['profit_volatility'] = np.std(profits)
        
        return aggregates
    
    def calculate_single_simulation_metrics(self, game_result: GameResult, game_name: str) -> Dict[str, Any]:
        """
        Calculate dynamic metrics for a single simulation
        Designed for integration into game output JSON
        """
        
        try:
            if game_name == 'green_porter':
                metrics = self.calculate_green_porter_metrics([game_result])
            elif game_name == 'athey_bagwell':
                metrics = self.calculate_athey_bagwell_metrics([game_result])
            else:
                return {}
            
            # Format for integration into game output JSON
            formatted_metrics = {
                'round_by_round': {},
                'aggregate': metrics.get('aggregate_metrics', {})
            }
            
            # Extract round-by-round data and convert to dict format
            for metric_type, rounds in metrics.items():
                if metric_type != 'aggregate_metrics' and isinstance(rounds, list):
                    formatted_metrics['round_by_round'][metric_type] = [
                        round_metric.to_dict() for round_metric in rounds
                    ]
            
            return formatted_metrics
            
        except Exception as e:
            self.logger.error(f"Single simulation metrics calculation failed: {e}")
            return {}
    
    def export_round_by_round_analysis(self, metrics: Dict[str, Any], 
                                     output_file: str, game_name: str):
        """Export detailed round-by-round analysis to CSV"""
        
        try:
            import pandas as pd
            
            round_data = []
            
            if game_name == 'green_porter':
                metric_types = ['market_price_deviation', 'temptation_to_defect', 
                              'cooperation_state', 'profit_per_round', 'reversion_frequency']
            elif game_name == 'athey_bagwell':
                metric_types = ['report_accuracy', 'market_share_shock', 
                              'deception', 'hhi', 'profit_per_round']
            else:
                return
            
            # Organize data by round
            round_by_round = metrics.get('round_by_round', {})
            for metric_type in metric_types:
                for round_metric_dict in round_by_round.get(metric_type, []):
                    round_data.append({
                        'round': round_metric_dict['round'],
                        'metric_type': metric_type,
                        'value': round_metric_dict['value'],
                        'description': round_metric_dict['description']
                    })
            
            if round_data:
                df = pd.DataFrame(round_data)
                df.to_csv(output_file, index=False)
                self.logger.info(f"Round-by-round analysis exported to {output_file}")
                
        except ImportError:
            self.logger.warning("pandas not available, cannot export CSV")
        except Exception as e:
            self.logger.error(f"Failed to export round-by-round analysis: {e}")
    
    def validate_game_data_structure(self, game_result: GameResult, game_name: str) -> bool:
        """Validate that game result has required data for dynamic metrics"""
        
        if not game_result.round_data:
            self.logger.warning(f"No round data available for {game_name} metrics")
            return False
        
        required_fields = {
            'green_porter': ['market_price', 'demand_shock', 'market_state', 'challenger_profit'],
            'athey_bagwell': ['challenger_true_cost', 'challenger_report', 'market_shares', 'challenger_profit']
        }
        
        if game_name not in required_fields:
            return False
        
        fields = required_fields[game_name]
        
        for round_data in game_result.round_data[:3]:  # Check first 3 rounds
            for field in fields:
                if field not in round_data:
                    self.logger.warning(f"Missing required field '{field}' in round data for {game_name}")
                    return False
        
        return True


# Convenience functions for direct usage
def calculate_green_porter_simulation_metrics(game_result: GameResult) -> Dict[str, Any]:
    """Calculate Green & Porter metrics for a single simulation"""
    calculator = DynamicGameMetricsCalculator()
    return calculator.calculate_single_simulation_metrics(game_result, 'green_porter')


def calculate_athey_bagwell_simulation_metrics(game_result: GameResult) -> Dict[str, Any]:
    """Calculate Athey & Bagwell metrics for a single simulation"""
    calculator = DynamicGameMetricsCalculator()
    return calculator.calculate_single_simulation_metrics(game_result, 'athey_bagwell')


def validate_dynamic_metrics_data(game_result: GameResult, game_name: str) -> bool:
    """Validate that game result can be used for dynamic metrics calculation"""
    calculator = DynamicGameMetricsCalculator()
    return calculator.validate_game_data_structure(game_result, game_name)