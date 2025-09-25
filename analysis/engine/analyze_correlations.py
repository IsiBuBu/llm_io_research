# analysis/engine/analyze_correlations.py

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from scipy.stats import pearsonr
import numpy as np
import sys

# Ensure the project root is in the Python path to allow for config import
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import load_config
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
    ci_95_low: float
    ci_95_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CorrelationAnalyzer:
    """
    Analyzes and tests the correlation hypotheses between MAgIC behavioral
    metrics and traditional performance metrics based on the experimental design.
    """

    def __init__(self, analysis_dir: str = "output/analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hypotheses = self._define_hypotheses()

    def _define_hypotheses(self) -> List[CorrelationHypothesis]:
        """
        Dynamically generates all possible correlation hypotheses for each game,
        testing each game-specific MAgIC metric against core performance outcomes.
        """
        hypotheses = []
        
        game_metrics = {
            'salop': ['rationality', 'self_awareness'],
            'green_porter': ['cooperation', 'coordination', 'judgment'],
            'spulber': ['rationality', 'self_awareness'],
            'athey_bagwell': ['deception', 'cooperation', 'reasoning']
        }
        
        performance_targets = ['win_rate', 'average_profit']

        for game, magic_metrics in game_metrics.items():
            for magic_metric in magic_metrics:
                for perf_metric in performance_targets:
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
        """
        Runs correlation analysis for baseline models, providing both a pooled result
        and separate results for each structural variation.
        """
        self.logger.info("Starting comprehensive correlation analysis.")
        all_correlation_results = []

        try:
            perf_df_raw = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
            magic_df_raw = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
            config = load_config()
        except FileNotFoundError:
            self.logger.error("Summary CSV files or config.json not found. Please ensure analysis steps ran correctly.")
            return

        # Dynamically determine which models to include (baseline, non-thinking models)
        models_to_include = []
        for model_name, model_conf in config.get('model_configs', {}).items():
            is_thinking_model = model_conf.get('thinking_available', False)
            if is_thinking_model:
                thinking_budget = model_conf.get('thinking_config', {}).get('thinking_budget', 0)
                if thinking_budget == 0:
                    models_to_include.append(model_name)
            else:
                models_to_include.append(model_name)
        
        # Filter dataframes to include only the desired models
        perf_df_filtered = perf_df_raw[perf_df_raw['model'].isin(models_to_include)]
        magic_df_filtered = magic_df_raw[magic_df_raw['model'].isin(models_to_include)]

        # --- CORRECTED LOGIC: Explicitly filter OUT ablation studies ---
        # A structural condition must contain 'players' but not be part of an ablation study.
        structural_conditions = [
            c for c in perf_df_filtered['condition'].unique() 
            if ('players' in c) and ('ablation' not in c) and ('low_persistence' not in c) and ('low_transport' not in c) and ('wide_cost_range' not in c) and ('low_discount_factor' not in c)
        ]
        
        perf_df_struct = perf_df_filtered[perf_df_filtered['condition'].isin(structural_conditions)]
        magic_df_struct = magic_df_filtered[magic_df_filtered['condition'].isin(structural_conditions)]

        # Pivot and merge the clean data for structural variations
        perf_df = perf_df_struct.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean').reset_index()
        magic_df = magic_df_struct.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean').reset_index()
        merged_df = pd.merge(perf_df, magic_df, on=['game', 'model', 'condition'])
        
        # Exclude random agent for meaningful correlation analysis
        merged_df = merged_df[merged_df['model'] != 'random_agent'].copy()

        if merged_df.empty:
            self.logger.warning("After filtering for structural variations, the dataset is empty. No correlations to analyze.")
            return

        # Define analysis scopes: pooled structural, and each structural condition separately
        analysis_scopes = {'pooled': merged_df}
        for condition_name in merged_df['condition'].unique():
            analysis_scopes[condition_name] = merged_df[merged_df['condition'] == condition_name]

        for scope_name, scope_df in analysis_scopes.items():
            self.logger.info(f"--- Analyzing correlations for scope: {scope_name.upper()} ---")
            for game_name in scope_df['game'].unique():
                game_df = scope_df[scope_df['game'] == game_name]
                game_hypotheses = [h for h in self.hypotheses if h.game_name == game_name]

                for hypothesis in game_hypotheses:
                    result = self._test_hypothesis(hypothesis, game_df)
                    if result:
                        result_dict = result.to_dict()
                        result_dict['condition_type'] = scope_name 
                        all_correlation_results.append(result_dict)

        self._save_results(all_correlation_results)

    def _test_hypothesis(self, hypothesis: CorrelationHypothesis, df: pd.DataFrame) -> Optional[CorrelationResult]:
        """Performs Pearson correlation test and calculates the 95% CI for a single hypothesis."""
        magic_col = hypothesis.magic_metric
        perf_col = hypothesis.performance_metric
        
        if magic_col not in df.columns or perf_col not in df.columns:
            return None

        subset_df = df[[magic_col, perf_col]].dropna()
        n_samples = len(subset_df)
        if n_samples < 3:
            return None

        if subset_df[magic_col].nunique() == 1 or subset_df[perf_col].nunique() == 1:
            corr, p_value, ci_low, ci_high = 0.0, 1.0, 0.0, 0.0
        else:
            if n_samples >= 3:
                result = pearsonr(subset_df[magic_col], subset_df[perf_col])
                corr, p_value = result.statistic, result.pvalue
                ci = result.confidence_interval(confidence_level=0.95)
                ci_low, ci_high = ci.low, ci.high
            else:
                corr, p_value = pearsonr(subset_df[magic_col], subset_df[perf_col])
                ci_low, ci_high = np.nan, np.nan

        return CorrelationResult(
            hypothesis=hypothesis,
            correlation_coefficient=corr,
            p_value=p_value,
            n_samples=n_samples,
            ci_95_low=ci_low,
            ci_95_high=ci_high
        )

    def _save_results(self, results: List[Dict[str, Any]]):
        """Saves the correlation results to a CSV file."""
        if not results:
            self.logger.warning("No correlation results were generated to save.")
            return

        output_path = self.analysis_dir / "correlations_analysis_structural.csv"
        df = pd.DataFrame(results)
        df = pd.concat([df.drop(['hypothesis'], axis=1), df['hypothesis'].apply(pd.Series)], axis=1)
        
        cols_order = [
            'condition_type', 'game_name', 'magic_metric', 'performance_metric',
            'correlation_coefficient', 'p_value', 'n_samples',
            'ci_95_low', 'ci_95_high'
        ]
        df = df[cols_order]
        df.to_csv(output_path, index=False)
        self.logger.info(f"Successfully saved structural correlation analysis to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    analyzer = CorrelationAnalyzer()
    analyzer.analyze_all_correlations()