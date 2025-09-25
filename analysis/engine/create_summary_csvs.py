# analysis/engine/create_summary_csvs.py

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
import numpy as np
import sys

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import get_experiments_dir
from metrics import (
    GameResult,
    PerformanceMetricsCalculator,
    MAgICMetricsCalculator,
)


# --- Helper function for confidence interval ---
def get_ci(data):
    """Calculates the lower and upper bounds of the 95% confidence interval for a mean."""
    if len(data) < 2:
        return 0, 0
    mean = np.mean(data)
    sem = stats.sem(data)
    if sem == 0:
        return mean, mean
    ci_half_width = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean - ci_half_width, mean + ci_half_width

class SummaryCreator:
    """
    Processes the raw experiment results, calculates metrics for each simulation,
    aggregates them to calculate mean, std, and confidence intervals, and saves
    it into flat CSV files.
    """

    def __init__(self, analysis_dir: str = "output/analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.experiments_dir = get_experiments_dir()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.perf_calc = PerformanceMetricsCalculator()
        self.magic_calc = MAgICMetricsCalculator()

    def create_all_summaries(self):
        """
        Loads all raw competition results, processes them, and saves the
        aggregated data to two main CSV files.
        """
        self.logger.info("Creating summary CSV files from raw experiment results...")
        all_perf_records = []
        all_magic_records = []

        json_files = list(self.experiments_dir.glob("*/*/*_competition_result.json"))
        if not json_files:
            self.logger.warning("No raw experiment result JSON files found. Cannot create summary CSVs.")
            return

        for result_file in json_files:
            with open(result_file, 'r') as f:
                data = json.load(f)

            if not data.get('simulation_results'):
                continue
                
            challenger_model = data.get('challenger_model')
            experiment_type = data.get('experiment_type')
            condition_name = data.get('condition_name')
            game_name = data.get('game_name')

            game_results = [GameResult(**sim) for sim in data.get('simulation_results', [])]
            
            if not game_results:
                continue

            # Get per-simulation raw values for universal performance metrics
            challenger_outcomes = [r.payoffs.get('challenger', 0.0) for r in game_results]

            wins = [1 if r.payoffs and r.payoffs.get('challenger', 0.0) == max(r.payoffs.values()) else 0 for r in game_results]

            # Create records for each simulation
            for i in range(len(game_results)):
                # Universal performance metrics
                all_perf_records.append({'game': game_name, 'model': challenger_model, 'condition': condition_name, 'metric': 'average_profit', 'value': challenger_outcomes[i]})
                all_perf_records.append({'game': game_name, 'model': challenger_model, 'condition': condition_name, 'metric': 'win_rate', 'value': wins[i]})

                # Game-specific performance metrics
                game_specific_perf = self.perf_calc.calculate_all_performance_metrics([game_results[i]], 'challenger')
                for name, metric_obj in game_specific_perf.items():
                    if name not in ['average_profit', 'win_rate', 'profit_volatility']:
                         all_perf_records.append({'game': game_name, 'model': challenger_model, 'condition': condition_name, 'metric': name, 'value': metric_obj.value})
                
                # MAgIC metrics
                magic_metrics = self.magic_calc.calculate_all_magic_metrics([game_results[i]], 'challenger')
                for name, metric_obj in magic_metrics.items():
                    all_magic_records.append({'game': game_name, 'model': challenger_model, 'condition': condition_name, 'metric': name, 'value': metric_obj.value})

        # Aggregate and save Performance Metrics
        perf_df = pd.DataFrame(all_perf_records)
        
        if not perf_df.empty:
            # Calculate volatility separately
            volatility_df = perf_df[perf_df['metric'] == 'average_profit'].groupby(['game', 'model', 'condition'])['value'].std().reset_index()
            volatility_df.rename(columns={'value': 'std'}, inplace=True)
            volatility_df['metric'] = 'profit_volatility'
            
            # Standard aggregation
            perf_agg_df = perf_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg(['mean', 'std', ('ci_95_low', lambda x: get_ci(x)[0]),('ci_95_high', lambda x: get_ci(x)[1])]).reset_index()

            # Merge volatility back in
            perf_agg_df = perf_agg_df[perf_agg_df['metric'] != 'profit_volatility']
            
            volatility_df_agg = perf_df[perf_df['metric'] == 'average_profit'].groupby(['game', 'model', 'condition'])['value'].agg(['mean', ('ci_95_low', lambda x: get_ci(x)[0]),('ci_95_high', lambda x: get_ci(x)[1])]).reset_index()
            
            volatility_df = pd.merge(volatility_df, volatility_df_agg, on=['game', 'model', 'condition'])
            volatility_df = volatility_df[['game', 'model', 'condition', 'metric', 'mean', 'std', 'ci_95_low', 'ci_95_high']]
            
            perf_agg_df = pd.concat([perf_agg_df, volatility_df], ignore_index=True)


            self._save_to_csv(perf_agg_df, "performance_metrics.csv")
        else:
            self.logger.warning("No performance records found to create summary CSV.")


        # Aggregate and save MAgIC Metrics
        magic_df = pd.DataFrame(all_magic_records)
        if not magic_df.empty:
            magic_agg_df = magic_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg(['mean','std',('ci_95_low', lambda x: get_ci(x)[0]),('ci_95_high', lambda x: get_ci(x)[1])]).reset_index()
            self._save_to_csv(magic_agg_df, "magic_behavioral_metrics.csv")
        else:
            self.logger.warning("No MAgIC records found to create summary CSV.")
            
        self.logger.info("Successfully created and saved aggregated summary CSV files.")

    def _save_to_csv(self, df: pd.DataFrame, filename: str):
        """Saves a DataFrame to a CSV file."""
        if df.empty:
            self.logger.warning(f"DataFrame is empty. Cannot save {filename}.")
            return
        
        output_path = self.analysis_dir / filename
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved data to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    creator = SummaryCreator()
    creator.create_all_summaries()