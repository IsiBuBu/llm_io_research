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
        """Loads correlation hypotheses using the exact metric names from the analysis."""
        return [
            # Salop
            CorrelationHypothesis('Judgment vs. Win Rate', 'salop', 'judgment', 'win_rate', 'positive'),
            CorrelationHypothesis('Self-Awareness vs. Market Share', 'salop', 'self_awareness', 'market_share', 'positive'),
            CorrelationHypothesis('Rationality vs. Average Profit', 'salop', 'rationality', 'average_profit', 'positive'),
            CorrelationHypothesis('Judgment vs. Profit Volatility', 'salop', 'judgment', 'profit_volatility', 'negative'),
            # Green & Porter
            CorrelationHypothesis('Rationality vs. Win Rate', 'green_porter', 'rationality', 'win_rate', 'positive'),
            CorrelationHypothesis('Cooperation vs. Reversion Frequency', 'green_porter', 'cooperation', 'reversion_frequency', 'negative'),
            CorrelationHypothesis('Coordination vs. Average Profit', 'green_porter', 'coordination', 'average_profit', 'positive'),
            # Spulber
            CorrelationHypothesis('Judgment vs. Win Rate', 'spulber', 'judgment', 'win_rate', 'positive'),
            CorrelationHypothesis('Self-Awareness vs. Market Capture', 'spulber', 'self_awareness', 'market_capture_rate', 'positive'),
            CorrelationHypothesis('Rationality vs. Average Profit', 'spulber', 'rationality', 'average_profit', 'positive'),
            # Athey & Bagwell
            CorrelationHypothesis('Reasoning vs. Win Rate', 'athey_bagwell', 'reasoning', 'win_rate', 'positive'),
            CorrelationHypothesis('Cooperation vs. HHI', 'athey_bagwell', 'cooperation', 'hhi', 'negative'),
            CorrelationHypothesis('Deception vs. Average Profit', 'athey_bagwell', 'deception', 'average_profit', 'positive'),
            CorrelationHypothesis('Deception vs. Profit Volatility', 'athey_bagwell', 'deception', 'profit_volatility', 'positive'),
        ]

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
            
        # Check for constant input which makes correlation undefined
        if len(subset_df[magic_col].unique()) == 1 or len(subset_df[perf_col].unique()) == 1:
             self.logger.warning(f"One of the inputs for hypothesis '{hypothesis.name}' is constant. Correlation is not defined. Skipping.")
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