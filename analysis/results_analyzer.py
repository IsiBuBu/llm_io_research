"""
Results Analyzer - Comprehensive analysis pipeline for LLM game theory experiments
Ties together all games, metrics, and correlation testing into publication-ready results
"""

import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from metrics.metric_utils import MetricStorage, ExperimentResults, PlayerMetrics, GameResult
from metrics.performance_metrics import PerformanceMetricsCalculator
from metrics.magic_metrics import MAgICMetricsCalculator
from metrics.correlation_analysis import CorrelationAnalyzer
from config import get_all_game_configs, GameConfig


@dataclass
class AnalysisReport:
    """Complete analysis report for publication"""
    experiment_metadata: Dict[str, Any]
    performance_summary: Dict[str, Any]
    magic_summary: Dict[str, Any]
    correlation_results: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    condition_comparisons: Dict[str, Any]
    export_timestamp: str


class ResultsAnalyzer:
    """
    Comprehensive analyzer that processes all experimental results and generates
    publication-ready analysis with statistical testing and correlation analysis
    """
    
    def __init__(self, output_dir: str = "analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.ResultsAnalyzer")
        
        # Initialize metric calculators
        self.performance_calculator = PerformanceMetricsCalculator()
        self.magic_calculator = MAgICMetricsCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
    
    def analyze_complete_experiment(self, results_dir: str, challenger_models: List[str],
                                  defender_model: str = "gpt-4") -> AnalysisReport:
        """
        Analyze complete experimental results across all games and conditions
        
        Args:
            results_dir: Directory containing experimental results
            challenger_models: List of challenger LLM models tested
            defender_model: Defender model used
            
        Returns:
            Complete analysis report
        """
        results_path = Path(results_dir)
        
        # Load all experimental results
        all_experiment_results = self._load_all_experiments(results_path, challenger_models, defender_model)
        
        # Calculate comprehensive metrics for all experiments
        analyzed_results = {}
        for game_name, experiment_results in all_experiment_results.items():
            analyzed_results[game_name] = self._analyze_single_game(experiment_results)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(analyzed_results, challenger_models)
        
        # Export results
        self._export_analysis_report(report)
        
        return report
    
    def _load_all_experiments(self, results_path: Path, challenger_models: List[str],
                            defender_model: str) -> Dict[str, ExperimentResults]:
        """Load experimental results for all games"""
        all_results = {}
        
        for game_name in ['salop', 'green_porter', 'spulber', 'athey_bagwell']:
            game_dir = results_path / game_name
            if game_dir.exists():
                experiment_results = ExperimentResults(
                    challenger_models=challenger_models,
                    defender_model=defender_model,
                    game_name=game_name
                )
                
                # Load results for each challenger and condition
                for challenger in challenger_models:
                    challenger_dir = game_dir / challenger
                    if challenger_dir.exists():
                        for condition_file in challenger_dir.glob("*_metrics.json"):
                            condition_name = condition_file.stem.replace("_metrics", "")
                            
                            try:
                                metrics = MetricStorage.load_player_metrics(condition_file)
                                experiment_results.add_player_metrics(challenger, condition_name, metrics)
                            except Exception as e:
                                self.logger.warning(f"Failed to load {condition_file}: {e}")
                
                all_results[game_name] = experiment_results
        
        return all_results
    
    def _analyze_single_game(self, experiment_results: ExperimentResults) -> Dict[str, Any]:
        """Comprehensive analysis for a single game"""
        game_name = experiment_results.game_name
        
        # Calculate correlation analysis
        correlation_results = self.correlation_analyzer.test_all_correlations(experiment_results)
        
        # Performance metric summaries
        performance_summary = self._summarize_performance_metrics(experiment_results)
        
        # MAgIC metric summaries  
        magic_summary = self._summarize_magic_metrics(experiment_results)
        
        # Statistical comparisons between conditions
        condition_comparisons = self._compare_conditions(experiment_results)
        
        # Model rankings
        model_rankings = self._rank_models(experiment_results)
        
        return {
            'game_name': game_name,
            'performance_summary': performance_summary,
            'magic_summary': magic_summary,
            'correlation_results': correlation_results,
            'condition_comparisons': condition_comparisons,
            'model_rankings': model_rankings,
            'participant_count': len(experiment_results.challenger_models)
        }
    
    def _summarize_performance_metrics(self, experiment_results: ExperimentResults) -> Dict[str, Any]:
        """Summarize performance metrics across all challengers and conditions"""
        summary = {
            'by_model': {},
            'by_condition': {},
            'overall_statistics': {}
        }
        
        all_metrics = []
        
        for challenger, conditions in experiment_results.results.items():
            model_metrics = {}
            
            for condition, player_metrics in conditions.items():
                condition_metrics = {}
                
                for metric_name, metric in player_metrics.performance_metrics.items():
                    condition_metrics[metric_name] = metric.value
                    all_metrics.append({
                        'model': challenger,
                        'condition': condition,
                        'metric': metric_name,
                        'value': metric.value
                    })
                
                model_metrics[condition] = condition_metrics
            
            summary['by_model'][challenger] = model_metrics
        
        # Condition-wise aggregation
        df = pd.DataFrame(all_metrics)
        if not df.empty:
            condition_summary = df.groupby(['condition', 'metric'])['value'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).round(4).to_dict('index')
            summary['by_condition'] = condition_summary
            
            # Overall statistics
            metric_summary = df.groupby('metric')['value'].agg([
                'mean', 'std', 'min', 'max'
            ]).round(4).to_dict('index')
            summary['overall_statistics'] = metric_summary
        
        return summary
    
    def _summarize_magic_metrics(self, experiment_results: ExperimentResults) -> Dict[str, Any]:
        """Summarize MAgIC behavioral metrics"""
        summary = {
            'by_model': {},
            'by_condition': {},
            'behavioral_profiles': {}
        }
        
        all_metrics = []
        
        for challenger, conditions in experiment_results.results.items():
            model_metrics = {}
            
            for condition, player_metrics in conditions.items():
                condition_metrics = {}
                
                for metric_name, metric in player_metrics.magic_metrics.items():
                    condition_metrics[metric_name] = metric.value
                    all_metrics.append({
                        'model': challenger,
                        'condition': condition,
                        'metric': metric_name,
                        'value': metric.value
                    })
                
                model_metrics[condition] = condition_metrics
            
            summary['by_model'][challenger] = model_metrics
        
        # Create behavioral profiles for each model
        df = pd.DataFrame(all_metrics)
        if not df.empty:
            for model in experiment_results.challenger_models:
                model_data = df[df['model'] == model]
                if not model_data.empty:
                    profile = model_data.groupby('metric')['value'].mean().to_dict()
                    summary['behavioral_profiles'][model] = profile
        
        return summary
    
    def _compare_conditions(self, experiment_results: ExperimentResults) -> Dict[str, Any]:
        """Statistical comparison between experimental conditions"""
        from scipy import stats
        
        comparisons = {}
        game_name = experiment_results.game_name
        
        # Get baseline condition and variation conditions
        all_conditions = set()
        for challenger_results in experiment_results.results.values():
            all_conditions.update(challenger_results.keys())
        
        baseline = 'baseline'
        variations = [c for c in all_conditions if c != baseline]
        
        if baseline in all_conditions:
            for variation in variations:
                comparison = self._statistical_comparison(
                    experiment_results, baseline, variation
                )
                comparisons[f"{baseline}_vs_{variation}"] = comparison
        
        return comparisons
    
    def _statistical_comparison(self, experiment_results: ExperimentResults,
                              condition1: str, condition2: str) -> Dict[str, Any]:
        """Perform statistical comparison between two conditions"""
        from scipy import stats
        
        comparison = {
            'condition1': condition1,
            'condition2': condition2,
            'performance_tests': {},
            'magic_tests': {},
            'effect_sizes': {}
        }
        
        # Collect data for both conditions
        data1 = {'performance': {}, 'magic': {}}
        data2 = {'performance': {}, 'magic': {}}
        
        for challenger, conditions in experiment_results.results.items():
            if condition1 in conditions and condition2 in conditions:
                metrics1 = conditions[condition1]
                metrics2 = conditions[condition2]
                
                # Performance metrics
                for metric_name, metric in metrics1.performance_metrics.items():
                    if metric_name not in data1['performance']:
                        data1['performance'][metric_name] = []
                        data2['performance'][metric_name] = []
                    
                    data1['performance'][metric_name].append(metric.value)
                    if metric_name in metrics2.performance_metrics:
                        data2['performance'][metric_name].append(
                            metrics2.performance_metrics[metric_name].value
                        )
                
                # MAgIC metrics
                for metric_name, metric in metrics1.magic_metrics.items():
                    if metric_name not in data1['magic']:
                        data1['magic'][metric_name] = []
                        data2['magic'][metric_name] = []
                    
                    data1['magic'][metric_name].append(metric.value)
                    if metric_name in metrics2.magic_metrics:
                        data2['magic'][metric_name].append(
                            metrics2.magic_metrics[metric_name].value
                        )
        
        # Perform t-tests
        for metric_type in ['performance', 'magic']:
            test_key = f"{metric_type}_tests"
            
            for metric_name in data1[metric_type]:
                if len(data1[metric_type][metric_name]) > 1 and len(data2[metric_type][metric_name]) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(
                            data1[metric_type][metric_name],
                            data2[metric_type][metric_name]
                        )
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            (np.var(data1[metric_type][metric_name], ddof=1) + 
                             np.var(data2[metric_type][metric_name], ddof=1)) / 2
                        )
                        effect_size = (np.mean(data1[metric_type][metric_name]) - 
                                     np.mean(data2[metric_type][metric_name])) / pooled_std
                        
                        comparison[test_key][metric_name] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'mean_condition1': float(np.mean(data1[metric_type][metric_name])),
                            'mean_condition2': float(np.mean(data2[metric_type][metric_name])),
                            'effect_size': float(effect_size)
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Statistical test failed for {metric_name}: {e}")
        
        return comparison
    
    def _rank_models(self, experiment_results: ExperimentResults) -> Dict[str, Any]:
        """Rank models by performance and behavioral metrics"""
        rankings = {
            'performance_rankings': {},
            'magic_rankings': {},
            'overall_ranking': {}
        }
        
        # Calculate average performance across all conditions for each model
        model_scores = {}
        
        for challenger in experiment_results.challenger_models:
            model_scores[challenger] = {
                'performance_scores': [],
                'magic_scores': []
            }
            
            challenger_results = experiment_results.results.get(challenger, {})
            for condition, metrics in challenger_results.items():
                # Average performance metrics
                perf_values = [m.value for m in metrics.performance_metrics.values()]
                if perf_values:
                    model_scores[challenger]['performance_scores'].extend(perf_values)
                
                # Average MAgIC metrics
                magic_values = [m.value for m in metrics.magic_metrics.values()]
                if magic_values:
                    model_scores[challenger]['magic_scores'].extend(magic_values)
        
        # Rank by average scores
        performance_averages = {
            model: np.mean(scores['performance_scores']) if scores['performance_scores'] else 0
            for model, scores in model_scores.items()
        }
        
        magic_averages = {
            model: np.mean(scores['magic_scores']) if scores['magic_scores'] else 0
            for model, scores in model_scores.items()
        }
        
        rankings['performance_rankings'] = sorted(
            performance_averages.items(), key=lambda x: x[1], reverse=True
        )
        
        rankings['magic_rankings'] = sorted(
            magic_averages.items(), key=lambda x: x[1], reverse=True
        )
        
        return rankings
    
    def _generate_comprehensive_report(self, analyzed_results: Dict[str, Any],
                                     challenger_models: List[str]) -> AnalysisReport:
        """Generate comprehensive analysis report"""
        
        # Aggregate correlation results across all games
        all_correlations = {}
        for game_name, analysis in analyzed_results.items():
            if 'correlation_results' in analysis:
                all_correlations[game_name] = analysis['correlation_results']
        
        # Create summary statistics
        correlation_summary = self.correlation_analyzer.calculate_correlation_summary(all_correlations)
        
        # Generate cross-game comparisons
        cross_game_analysis = self._cross_game_analysis(analyzed_results)
        
        return AnalysisReport(
            experiment_metadata={
                'challenger_models': challenger_models,
                'games_analyzed': list(analyzed_results.keys()),
                'analysis_date': datetime.now().isoformat(),
                'total_conditions': sum(
                    len(analysis.get('condition_comparisons', {})) 
                    for analysis in analyzed_results.values()
                )
            },
            performance_summary={
                game: analysis['performance_summary'] 
                for game, analysis in analyzed_results.items()
            },
            magic_summary={
                game: analysis['magic_summary'] 
                for game, analysis in analyzed_results.items()
            },
            correlation_results={
                'by_game': all_correlations,
                'summary': correlation_summary
            },
            statistical_tests={
                game: analysis['condition_comparisons'] 
                for game, analysis in analyzed_results.items()
            },
            condition_comparisons=cross_game_analysis,
            export_timestamp=datetime.now().isoformat()
        )
    
    def _cross_game_analysis(self, analyzed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across different games"""
        cross_analysis = {
            'model_consistency': {},
            'metric_correlations_across_games': {},
            'behavioral_patterns': {}
        }
        
        # Model consistency analysis - do models perform similarly across games?
        games = list(analyzed_results.keys())
        if len(games) > 1:
            for game1 in games:
                for game2 in games:
                    if game1 != game2:
                        # Compare model rankings between games
                        rankings1 = analyzed_results[game1].get('model_rankings', {})
                        rankings2 = analyzed_results[game2].get('model_rankings', {})
                        
                        key = f"{game1}_vs_{game2}"
                        cross_analysis['model_consistency'][key] = {
                            'performance_rank_correlation': self._rank_correlation(
                                rankings1.get('performance_rankings', []),
                                rankings2.get('performance_rankings', [])
                            ),
                            'magic_rank_correlation': self._rank_correlation(
                                rankings1.get('magic_rankings', []),
                                rankings2.get('magic_rankings', [])
                            )
                        }
        
        return cross_analysis
    
    def _rank_correlation(self, ranking1: List[Tuple], ranking2: List[Tuple]) -> float:
        """Calculate rank correlation between two rankings"""
        if not ranking1 or not ranking2:
            return 0.0
        
        # Convert to rank dictionaries
        rank1_dict = {model: rank for rank, (model, score) in enumerate(ranking1)}
        rank2_dict = {model: rank for rank, (model, score) in enumerate(ranking2)}
        
        # Find common models
        common_models = set(rank1_dict.keys()) & set(rank2_dict.keys())
        if len(common_models) < 2:
            return 0.0
        
        # Calculate Spearman correlation
        ranks1 = [rank1_dict[model] for model in common_models]
        ranks2 = [rank2_dict[model] for model in common_models]
        
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(ranks1, ranks2)
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _export_analysis_report(self, report: AnalysisReport):
        """Export comprehensive analysis report to multiple formats"""
        
        # Save complete report as JSON
        report_dict = asdict(report)
        with open(self.output_dir / "comprehensive_analysis_report.json", 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        # Export correlation results to CSV
        self._export_correlations_csv(report.correlation_results)
        
        # Export performance summaries to CSV
        self._export_performance_csv(report.performance_summary)
        
        # Export MAgIC summaries to CSV
        self._export_magic_csv(report.magic_summary)
        
        # Generate publication-ready summary
        self._generate_publication_summary(report)
        
        self.logger.info(f"Analysis report exported to {self.output_dir}")
    
    def _export_correlations_csv(self, correlation_results: Dict[str, Any]):
        """Export correlation results to CSV"""
        correlation_data = []
        
        for game_name, game_correlations in correlation_results.get('by_game', {}).items():
            for correlation_list in game_correlations.values():
                for result in correlation_list:
                    correlation_data.append({
                        'game': game_name,
                        'hypothesis': result.hypothesis.name,
                        'magic_metric': result.hypothesis.magic_metric,
                        'performance_metric': result.hypothesis.performance_metric,
                        'correlation': result.correlation_coefficient,
                        'p_value': result.p_value,
                        'significant': result.is_significant,
                        'n_samples': result.n_samples,
                        'expected_direction': result.hypothesis.expected_direction
                    })
        
        if correlation_data:
            df = pd.DataFrame(correlation_data)
            df.to_csv(self.output_dir / "correlation_analysis.csv", index=False)
    
    def _export_performance_csv(self, performance_summary: Dict[str, Any]):
        """Export performance metrics to CSV"""
        performance_data = []
        
        for game_name, game_summary in performance_summary.items():
            for model, conditions in game_summary.get('by_model', {}).items():
                for condition, metrics in conditions.items():
                    for metric_name, value in metrics.items():
                        performance_data.append({
                            'game': game_name,
                            'model': model,
                            'condition': condition,
                            'metric': metric_name,
                            'value': value
                        })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            df.to_csv(self.output_dir / "performance_metrics.csv", index=False)
    
    def _export_magic_csv(self, magic_summary: Dict[str, Any]):
        """Export MAgIC metrics to CSV"""
        magic_data = []
        
        for game_name, game_summary in magic_summary.items():
            for model, conditions in game_summary.get('by_model', {}).items():
                for condition, metrics in conditions.items():
                    for metric_name, value in metrics.items():
                        magic_data.append({
                            'game': game_name,
                            'model': model,
                            'condition': condition,
                            'metric': metric_name,
                            'value': value
                        })
        
        if magic_data:
            df = pd.DataFrame(magic_data)
            df.to_csv(self.output_dir / "magic_behavioral_metrics.csv", index=False)
    
    def _generate_publication_summary(self, report: AnalysisReport):
        """Generate human-readable publication summary"""
        summary_lines = [
            "# LLM Game Theory Experiment Analysis Summary",
            f"Generated: {report.export_timestamp}",
            "",
            "## Experiment Overview",
            f"- Challenger Models: {', '.join(report.experiment_metadata['challenger_models'])}",
            f"- Games Analyzed: {', '.join(report.experiment_metadata['games_analyzed'])}",
            f"- Total Conditions: {report.experiment_metadata['total_conditions']}",
            "",
            "## Key Findings",
            ""
        ]
        
        # Correlation summary
        corr_summary = report.correlation_results.get('summary', {})
        if corr_summary:
            summary_lines.extend([
                "### Correlation Analysis",
                f"- Total hypotheses tested: {corr_summary.get('total_hypotheses_tested', 0)}",
                f"- Significant correlations: {corr_summary.get('significant_correlations', 0)}",
                f"- Strong correlations (|r| > 0.5): {corr_summary.get('strong_correlations', 0)}",
                f"- Confirmed expectations: {corr_summary.get('confirmed_expectations', 0)}",
                ""
            ])
        
        # Game-specific highlights
        for game_name in report.experiment_metadata['games_analyzed']:
            summary_lines.extend([
                f"### {game_name.title()} Results",
                "- [Add key findings for this game]",
                ""
            ])
        
        summary_text = "\n".join(summary_lines)
        
        with open(self.output_dir / "publication_summary.md", 'w') as f:
            f.write(summary_text)


# Convenience functions for quick analysis
def analyze_experiment_results(results_dir: str, challenger_models: List[str],
                             output_dir: str = "analysis_output") -> AnalysisReport:
    """Quick analysis of complete experimental results"""
    analyzer = ResultsAnalyzer(output_dir)
    return analyzer.analyze_complete_experiment(results_dir, challenger_models)


def quick_game_analysis(game_results: List[GameResult], game_name: str,
                       output_dir: str = "quick_analysis") -> Dict[str, Any]:
    """Quick analysis of single game results"""
    analyzer = ResultsAnalyzer(output_dir)
    
    # Calculate all metrics
    performance_metrics = analyzer.performance_calculator.calculate_all_performance_metrics(game_results)
    magic_metrics = analyzer.magic_calculator.calculate_all_magic_metrics(game_results)
    
    return {
        'game_name': game_name,
        'performance_metrics': {name: metric.to_dict() for name, metric in performance_metrics.items()},
        'magic_metrics': {name: metric.to_dict() for name, metric in magic_metrics.items()},
        'summary_statistics': {
            'total_simulations': len(game_results),
            'avg_profit': np.mean([r.payoffs.get('challenger', 0) for r in game_results]),
            'win_rate': sum(1 for r in game_results 
                          if r.payoffs.get('challenger', 0) == max(r.payoffs.values())) / len(game_results)
        }
    }