"""
Correlation Analysis - Tests specific hypotheses about relationships between MAgIC and performance metrics
Implements correlation testing methodology from experimental design document
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from metrics.metric_utils import MetricCalculator, MetricResult, PlayerMetrics, ExperimentResults


@dataclass
class CorrelationHypothesis:
    """Definition of a correlation hypothesis to test"""
    name: str
    description: str
    magic_metric: str
    performance_metric: str
    expected_direction: str  # 'positive', 'negative', 'any'
    game_name: str


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    hypothesis: CorrelationHypothesis
    correlation_coefficient: float
    p_value: float
    n_samples: int
    is_significant: bool
    significance_level: float
    data_points: List[Tuple[float, float]]  # (magic_score, performance_score) pairs
    interpretation: str


class CorrelationAnalyzer(MetricCalculator):
    """
    Analyzes correlations between MAgIC behavioral metrics and performance metrics
    Tests specific hypotheses outlined in experimental design
    """
    
    def __init__(self, significance_level: float = 0.05):
        super().__init__()
        self.significance_level = significance_level
        self.hypotheses = self._define_correlation_hypotheses()
    
    def _define_correlation_hypotheses(self) -> Dict[str, List[CorrelationHypothesis]]:
        """Define all correlation hypotheses from experimental design document"""
        
        hypotheses = {
            'salop': [
                CorrelationHypothesis(
                    name='judgment_vs_win_rate',
                    description='Tests if superior judgment leads to winning more often',
                    magic_metric='judgment_profitable_win_rate',
                    performance_metric='win_rate',
                    expected_direction='positive',
                    game_name='salop'
                ),
                CorrelationHypothesis(
                    name='self_awareness_vs_market_share',
                    description='Tests if self-awareness leads to better market positioning',
                    magic_metric='self_awareness_rate',
                    performance_metric='market_share',
                    expected_direction='positive',
                    game_name='salop'
                ),
                CorrelationHypothesis(
                    name='rationality_vs_average_profit',
                    description='Tests if rationality leads to higher profits',
                    magic_metric='rationality_rate',
                    performance_metric='average_profit',
                    expected_direction='positive',
                    game_name='salop'
                ),
                CorrelationHypothesis(
                    name='judgment_vs_profit_volatility',
                    description='Tests if good judgment reduces profit volatility',
                    magic_metric='judgment_profitable_win_rate',
                    performance_metric='profit_volatility',
                    expected_direction='negative',
                    game_name='salop'
                )
            ],
            'athey_bagwell': [
                CorrelationHypothesis(
                    name='rationality_vs_win_rate',
                    description='Tests if rational players win more often in repeated competition',
                    magic_metric='rationality_rate',
                    performance_metric='win_rate',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='cooperation_vs_reversion_frequency',
                    description='Tests if cooperative behavior affects reversion patterns',
                    magic_metric='cooperation_rate',
                    performance_metric='reversion_frequency',
                    expected_direction='negative',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='coordination_vs_industry_profit',
                    description='Tests if coordination ability improves industry outcomes',
                    magic_metric='coordination_rate',
                    performance_metric='industry_profit',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='rationality_vs_profit_margin',
                    description='Tests if rationality leads to better profit margins',
                    magic_metric='rationality_rate',
                    performance_metric='profit_margin',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='self_awareness_vs_market_capture',
                    description='Tests if self-awareness helps in market capture strategies',
                    magic_metric='self_awareness_rate',
                    performance_metric='market_capture_rate',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='cooperation_vs_average_profit',
                    description='Tests if cooperation leads to better average profits',
                    magic_metric='cooperation_rate',
                    performance_metric='average_profit',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='reasoning_vs_win_rate',
                    description='Tests if reasoning ability correlates with winning',
                    magic_metric='reasoning_rate',
                    performance_metric='win_rate',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='cooperation_vs_hhi',
                    description='Tests if cooperation affects market concentration',
                    magic_metric='cooperation_rate',
                    performance_metric='hhi',
                    expected_direction='negative',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='deception_vs_average_profit',
                    description='Tests if deception provides short-term profit advantages',
                    magic_metric='deception_rate',
                    performance_metric='average_profit',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                ),
                CorrelationHypothesis(
                    name='deception_vs_profit_volatility',
                    description='Tests if deception leads to boom-or-bust outcomes',
                    magic_metric='deception_rate',
                    performance_metric='profit_volatility',
                    expected_direction='positive',
                    game_name='athey_bagwell'
                )
            ]
        }
        
        return hypotheses
    
    def test_all_correlations(self, experiment_results: ExperimentResults) -> Dict[str, List[CorrelationResult]]:
        """
        Test all correlation hypotheses for given experiment results
        
        Args:
            experiment_results: Complete results across all challenger models and conditions
            
        Returns:
            Dictionary mapping game_name -> list of correlation results
        """
        all_results = {}
        
        game_name = experiment_results.game_name
        if game_name not in self.hypotheses:
            self.logger.warning(f"No correlation hypotheses defined for game: {game_name}")
            return {}
        
        game_hypotheses = self.hypotheses[game_name]
        correlation_results = []
        
        for hypothesis in game_hypotheses:
            try:
                result = self.test_correlation_hypothesis(hypothesis, experiment_results)
                # Only append if result is a valid CorrelationResult object
                if isinstance(result, CorrelationResult):
                    correlation_results.append(result)
                else:
                    self.logger.error(f"Invalid result type from hypothesis {hypothesis.name}: {type(result)}")
            except Exception as e:
                self.logger.error(f"Failed to test hypothesis {hypothesis.name}: {e}")
                # Note: Do not append anything to correlation_results on failure
        
        all_results[game_name] = correlation_results
        return all_results
    
    def test_correlation_hypothesis(self, hypothesis: CorrelationHypothesis, 
                                  experiment_results: ExperimentResults) -> CorrelationResult:
        """
        Test single correlation hypothesis using Pearson correlation
        
        Methodology from document:
        1. Extract paired observations (Agent_X_MAgIC_Score, Agent_X_Performance_Score)
        2. Calculate Pearson correlation coefficient
        3. Test for statistical significance
        
        CRITICAL: Only includes data from successful LLM responses, excluding default actions
        """
        
        # Extract paired data points across all challenger models and conditions
        data_points = []
        excluded_default_actions = 0
        total_potential_points = 0
        
        for challenger_model, conditions in experiment_results.results.items():
            for condition_name, player_metrics in conditions.items():
                total_potential_points += 1
                
                # CRITICAL CHECK: Skip data points from failed LLM responses (default actions)
                # Check if this data came from a default action due to LLM parsing failure
                if hasattr(player_metrics, 'parsing_success') and player_metrics.parsing_success is False:
                    excluded_default_actions += 1
                    self.logger.debug(f"Excluding default action data point from {challenger_model}/{condition_name}")
                    continue
    
    def _interpret_correlation_result(self, hypothesis: CorrelationHypothesis, 
                                    correlation: float, p_value: float, 
                                    is_significant: bool, n_samples: int) -> str:
        """Generate human-readable interpretation of correlation result"""
        
        # Correlation strength interpretation
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        # Direction
        direction = "positive" if correlation > 0 else "negative"
        
        # Statistical significance
        significance_text = "statistically significant" if is_significant else "not statistically significant"
        
        # Expected vs actual direction
        expectation_match = ""
        if hypothesis.expected_direction != 'any':
            expected_positive = hypothesis.expected_direction == 'positive'
            actual_positive = correlation > 0
            if expected_positive == actual_positive:
                expectation_match = " This matches the expected direction."
            else:
                expectation_match = " This contradicts the expected direction."
        
        interpretation = (
            f"Found a {strength} {direction} correlation (r = {correlation:.3f}) "
            f"between {hypothesis.magic_metric} and {hypothesis.performance_metric}. "
            f"This correlation is {significance_text} (p = {p_value:.3f}, n = {n_samples}).{expectation_match}"
        )
        
        return interpretation
    
    def calculate_correlation_summary(self, correlation_results: Dict[str, List[CorrelationResult]]) -> Dict[str, Any]:
        """Calculate summary statistics across all correlation tests"""
        
        summary = {
            'total_hypotheses_tested': 0,
            'significant_correlations': 0,
            'positive_correlations': 0,
            'negative_correlations': 0,
            'strong_correlations': 0,  # |r| > 0.5
            'confirmed_expectations': 0,
            'contradicted_expectations': 0,
            'by_game': {}
        }
        
        for game_name, results in correlation_results.items():
            game_summary = {
                'tested': 0,  # Count only valid results
                'significant': 0,
                'strong': 0,
                'confirmed': 0,
                'contradicted': 0
            }
            
            for result in results:
                # Skip non-CorrelationResult objects (like error strings)
                if not isinstance(result, CorrelationResult):
                    self.logger.warning(f"Skipping invalid result type: {type(result)}")
                    continue
                    
                summary['total_hypotheses_tested'] += 1
                game_summary['tested'] += 1
                
                if result.is_significant:
                    summary['significant_correlations'] += 1
                    game_summary['significant'] += 1
                
                if result.correlation_coefficient > 0:
                    summary['positive_correlations'] += 1
                else:
                    summary['negative_correlations'] += 1
                
                if abs(result.correlation_coefficient) > 0.5:
                    summary['strong_correlations'] += 1
                    game_summary['strong'] += 1
                
                # Check expectation matching
                if result.hypothesis.expected_direction != 'any':
                    expected_positive = result.hypothesis.expected_direction == 'positive'
                    actual_positive = result.correlation_coefficient > 0
                    if expected_positive == actual_positive:
                        summary['confirmed_expectations'] += 1
                        game_summary['confirmed'] += 1
                    else:
                        summary['contradicted_expectations'] += 1
                        game_summary['contradicted'] += 1
            
            summary['by_game'][game_name] = game_summary
        
        return summary
    
    def export_correlation_results(self, correlation_results: Dict[str, List[CorrelationResult]], 
                                 output_path: str):
        """Export correlation results to CSV for further analysis"""
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'game_name', 'hypothesis_name', 'magic_metric', 'performance_metric',
                'correlation_coefficient', 'p_value', 'n_samples', 'is_significant',
                'expected_direction', 'interpretation'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for game_name, results in correlation_results.items():
                for result in results:
                    # Only export valid CorrelationResult objects
                    if isinstance(result, CorrelationResult):
                        writer.writerow({
                            'game_name': game_name,
                            'hypothesis_name': result.hypothesis.name,
                            'magic_metric': result.hypothesis.magic_metric,
                            'performance_metric': result.hypothesis.performance_metric,
                            'correlation_coefficient': result.correlation_coefficient,
                            'p_value': result.p_value,
                            'n_samples': result.n_samples,
                            'is_significant': result.is_significant,
                            'expected_direction': result.hypothesis.expected_direction,
                            'interpretation': result.interpretation
                        })


# Convenience functions
def test_game_correlations(game_name: str, experiment_results: ExperimentResults) -> List[CorrelationResult]:
    """Test all correlations for a specific game"""
    analyzer = CorrelationAnalyzer()
    all_results = analyzer.test_all_correlations(experiment_results)
    return all_results.get(game_name, [])


def get_significant_correlations(correlation_results: Dict[str, List[CorrelationResult]], 
                               min_strength: float = 0.3) -> List[CorrelationResult]:
    """Filter for statistically significant correlations above minimum strength"""
    significant = []
    
    for game_results in correlation_results.values():
        for result in game_results:
            # Only process valid CorrelationResult objects
            if isinstance(result, CorrelationResult) and result.is_significant and abs(result.correlation_coefficient) >= min_strength:
                significant.append(result)
    
    return significant


def print_correlation_summary(correlation_results: Dict[str, List[CorrelationResult]]):
    """Print formatted summary of correlation analysis"""
    analyzer = CorrelationAnalyzer()
    summary = analyzer.calculate_correlation_summary(correlation_results)
    
    print("=== CORRELATION ANALYSIS SUMMARY ===")
    print(f"Total hypotheses tested: {summary['total_hypotheses_tested']}")
    if summary['total_hypotheses_tested'] > 0:
        print(f"Statistically significant: {summary['significant_correlations']} ({summary['significant_correlations']/summary['total_hypotheses_tested']*100:.1f}%)")
    else:
        print("Statistically significant: 0 (0.0%)")
    print(f"Strong correlations (|r| > 0.5): {summary['strong_correlations']}")
    print(f"Confirmed expectations: {summary['confirmed_expectations']}")
    print(f"Contradicted expectations: {summary['contradicted_expectations']}")
    print()
    
    for game_name, game_summary in summary['by_game'].items():
        print(f"{game_name.upper()}:")
        print(f"  Tested: {game_summary['tested']}")
        print(f"  Significant: {game_summary['significant']}")
        print(f"  Strong: {game_summary['strong']}")
        print(f"  Confirmed: {game_summary['confirmed']}")
        print(f"  Contradicted: {game_summary['contradicted']}")