# analysis/create_summary_csvs.py

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
import numpy as np

# --- Helper function for confidence interval ---
def get_ci(data):
    """Calculates the lower and upper bounds of the 95% confidence interval for a mean."""
    if len(data) < 2:
        return 0, 0
    mean = np.mean(data)
    sem = stats.sem(data)
    ci_half_width = sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)
    return mean - ci_half_width, mean + ci_half_width

class SummaryCreator:
    """
    Processes the nested JSON output from MetricsAnalyzer, aggregates the data
    to calculate mean, std, and confidence intervals, and saves it into flat CSV files.
    """

    def __init__(self, analysis_dir: str = "analysis_output"):
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_all_summaries(self):
        """
        Loads all '*_metrics_analysis.json' files, processes them,
        and saves the aggregated data to two main CSV files.
        """
        self.logger.info("Creating summary CSV files from JSON analysis output...")
        all_perf_records = []
        all_magic_records = []

        json_files = list(self.analysis_dir.glob("*_metrics_analysis.json"))
        if not json_files:
            self.logger.warning("No metrics analysis JSON files found. Cannot create summary CSVs.")
            return

        for metrics_file in json_files:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            perf_records, magic_records = self._flatten_json_data(data)
            all_perf_records.extend(perf_records)
            all_magic_records.extend(magic_records)

        # Aggregate and save Performance Metrics
        perf_df = pd.DataFrame(all_perf_records)
        if not perf_df.empty:
            perf_agg_df = perf_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg([
                'mean', 
                'std', 
                ('ci_95_low', lambda x: get_ci(x)[0]),
                ('ci_95_high', lambda x: get_ci(x)[1])
            ]).reset_index()
            self._save_to_csv(perf_agg_df, "performance_metrics.csv")
        else:
            self.logger.warning("No performance records found to create summary CSV.")


        # Aggregate and save MAgIC Metrics
        magic_df = pd.DataFrame(all_magic_records)
        if not magic_df.empty:
            magic_agg_df = magic_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg([
                'mean', 
                'std',
                ('ci_95_low', lambda x: get_ci(x)[0]),
                ('ci_95_high', lambda x: get_ci(x)[1])
            ]).reset_index()
            self._save_to_csv(magic_agg_df, "magic_behavioral_metrics.csv")
        else:
            self.logger.warning("No MAgIC records found to create summary CSV.")
            
        self.logger.info("Successfully created and saved aggregated summary CSV files.")

    def _flatten_json_data(self, data: Dict[str, Any]) -> (List[Dict], List[Dict]):
        """Flattens the nested data from a single JSON file."""
        perf_records = []
        magic_records = []
        game_name = data.get("game_name")

        for model, conditions in data.get('results', {}).items():
            for condition, metrics in conditions.items():
                # Process performance metrics
                for name, metric_data in metrics.get('performance_metrics', {}).items():
                    perf_records.append({
                        'game': game_name,
                        'model': model,
                        'condition': condition,
                        'metric': name,
                        'value': metric_data.get('value')
                    })
                # Process MAgIC metrics
                for name, metric_data in metrics.get('magic_metrics', {}).items():
                    magic_records.append({
                        'game': game_name,
                        'model': model,
                        'condition': condition,
                        'metric': name,
                        'value': metric_data.get('value')
                    })
        return perf_records, magic_records

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