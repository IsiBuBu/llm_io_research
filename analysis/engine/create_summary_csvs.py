# analysis/create_summary_csvs.py

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

class SummaryCreator:
    """
    Processes the nested JSON output from MetricsAnalyzer into flat CSV files
    suitable for correlation and visualization tasks.
    """

    def __init__(self, analysis_dir: str = "analysis_output"):
        self.analysis_dir = Path(analysis_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_all_summaries(self):
        """
        Loads all '*_metrics_analysis.json' files, processes them,
        and saves the flattened data to two main CSV files.
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

        self._save_to_csv(all_perf_records, "performance_metrics.csv")
        self._save_to_csv(all_magic_records, "magic_behavioral_metrics.csv")
        self.logger.info("Successfully created and saved summary CSV files.")

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

    def _save_to_csv(self, records: List[Dict], filename: str):
        """Saves a list of records to a CSV file."""
        if not records:
            self.logger.warning(f"No records to save for {filename}.")
            return
        
        output_path = self.analysis_dir / filename
        pd.DataFrame(records).to_csv(output_path, index=False)
        self.logger.info(f"Saved data to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    creator = SummaryCreator()
    creator.create_all_summaries()