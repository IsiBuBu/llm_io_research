# analysis/analyze_correlations.py

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from scipy.stats import pearsonr
import numpy as np

from metrics.metric_utils import ExperimentResults

@dataclass
class CorrelationHypothesis:
    """Defines a correlation hypothesis to be tested between a MAgIC and a performance metric."""
    name: str
    game_name: str
    magic_metric: str
    performance_metric: str
    expected_direction: str  # 'positive', 'negative', or 'any'

@dataclass
class CorrelationResult:
    """Stores the result of a single correlation hypothesis test."""
    hypothesis: CorrelationHypothesis
    correlation_coefficient: float
    p_value: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CorrelationAnalyzer:
    """
    Analyzes and tests the correlation hypotheses between MAgIC behavioral
    metrics and traditional performance metrics based on the experimental design.
    """

    def __init__(self, analysis_dir: str = "analysis_output"):
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hypotheses = self._define_hypotheses()

    def _define_hypotheses(self) -> List[CorrelationHypothesis]:
        """Dynamically generates all possible correlation hypotheses for each game."""
        hypotheses = []
        
        game_metrics = {
            'salop': {
                'magic': ['rationality', 'judgment', 'self_awareness'],
                'performance': ['win_rate', 'average_profit', 'profit_volatility', 'market_share']
            },
            'green_porter': {
                'magic': ['cooperation', 'coordination', 'rationality'],
                'performance': ['win_rate', 'average_profit', 'profit_volatility', 'reversion_frequency']
            },
            'spulber': {
                'magic': ['rationality', 'judgment', 'self_awareness'],
                'performance': ['win_rate', 'average_profit', 'profit_volatility', 'market_capture_rate']
            },
            'athey_bagwell': {
                'magic': ['deception', 'reasoning', 'cooperation'],
                'performance': ['win_rate', 'average_profit', 'profit_volatility', 'hhi']
            }
        }
        
        for game, metrics in game_metrics.items():
            for magic_metric in metrics['magic']:
                for perf_metric in metrics['performance']:
                    hypotheses.append(
                        CorrelationHypothesis(
                            name=f"{magic_metric.title()} vs. {perf_metric.replace('_', ' ').title()}",
                            game_name=game,
                            magic_metric=magic_metric,
                            performance_metric=perf_metric,
                            expected_direction='any'
                        )
                    )
        return hypotheses

    def analyze_all_correlations(self):
        """Runs correlation analysis for all games with analyzed metrics."""
        self.logger.info("Starting correlation analysis across all games.")
        all_correlation_results = []

        try:
            perf_df_raw = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
            magic_df_raw = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
        except FileNotFoundError:
            self.logger.error("Summary CSV files not found. Please ensure Steps 1 & 2 of the analysis ran correctly.")
            return

        # Pivot each dataframe so that metrics become columns
        perf_df = perf_df_raw.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='value').reset_index()
        magic_df = magic_df_raw.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='value').reset_index()

        # Merge the two pivoted dataframes
        merged_df = pd.merge(perf_df, magic_df, on=['game', 'model', 'condition'])

        for game_name in merged_df['game'].unique():
            self.logger.info(f"--- Analyzing correlations for: {game_name.upper()} ---")
            game_df = merged_df[merged_df['game'] == game_name]
            game_hypotheses = [h for h in self.hypotheses if h.game_name == game_name]

            for hypothesis in game_hypotheses:
                result = self._test_hypothesis(hypothesis, game_df)
                if result:
                    all_correlation_results.append(result.to_dict())

        self._save_results(all_correlation_results)

    def _test_hypothesis(self, hypothesis: CorrelationHypothesis, df: pd.DataFrame) -> Optional[CorrelationResult]:
        """Performs Pearson correlation test for a single hypothesis."""
        magic_col = hypothesis.magic_metric
        perf_col = hypothesis.performance_metric
        
        if magic_col not in df.columns or perf_col not in df.columns:
            self.logger.warning(f"Missing required columns '{magic_col}' or '{perf_col}' for hypothesis '{hypothesis.name}'. Skipping.")
            return None

        subset_df = df[[magic_col, perf_col]].dropna()
        if len(subset_df) < 3:
            self.logger.info(f"Not enough data points ({len(subset_df)}) for hypothesis '{hypothesis.name}'. Skipping.")
            return None
            
        corr, p_value = pearsonr(subset_df[magic_col], subset_df[perf_col])
        
        return CorrelationResult(
            hypothesis=hypothesis,
            correlation_coefficient=corr,
            p_value=p_value,
            n_samples=len(subset_df)
        )

    def _save_results(self, results: List[Dict[str, Any]]):
        """Saves the correlation results to a CSV file."""
        if not results:
            self.logger.warning("No correlation results were generated to save.")
            return

        output_path = self.analysis_dir / "correlations_analysis.csv"
        df = pd.DataFrame(results)
        # Flatten the nested hypothesis dataclass for easier CSV reading
        df = pd.concat([df.drop(['hypothesis'], axis=1), df['hypothesis'].apply(pd.Series)], axis=1)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Successfully saved correlation analysis to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    analyzer = CorrelationAnalyzer()
    analyzer.analyze_all_correlations()