# analysis/analyze_metrics.py

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from config.config import get_challenger_models, get_defender_model
from metrics import (
    GameResult,
    PlayerMetrics,
    ExperimentResults,
    PerformanceMetricsCalculator,
    MAgICMetricsCalculator,
    DynamicGameMetricsCalculator
)

class MetricsAnalyzer:
    """
    Processes raw simulation results to calculate and save aggregate
    Performance, MAgIC, and per-round Dynamic metrics for each experimental condition.
    """

    def __init__(self, results_dir: str = "results", output_dir: str = "analysis_output"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Instantiate all metric calculators
        self.perf_calc = PerformanceMetricsCalculator()
        self.magic_calc = MAgICMetricsCalculator()
        self.dyn_calc = DynamicGameMetricsCalculator()

    def analyze_all_games(self):
        """
        Analyzes all available game results and saves the final metrics.
        """
        self.logger.info("Starting comprehensive metrics analysis...")
        game_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]

        for game_dir in game_dirs:
            game_name = game_dir.name
            self.logger.info(f"--- Analyzing game: {game_name.upper()} ---")

            try:
                all_sim_results = self._load_simulation_results(game_dir)
                if not all_sim_results:
                    self.logger.warning(f"No valid simulation results found for {game_name}. Skipping.")
                    continue

                experiment_results = self._calculate_all_metrics_for_game(game_name, all_sim_results)
                self._save_experiment_results(experiment_results)

            except Exception as e:
                self.logger.error(f"Failed to analyze {game_name}: {e}", exc_info=True)

        self.logger.info("Comprehensive metrics analysis complete.")

    def _load_simulation_results(self, game_dir: Path) -> Dict[str, List[GameResult]]:
        """Loads all raw simulation_results.json files for a specific game."""
        grouped_results = defaultdict(list)

        for json_file in game_dir.glob("*/*_competition_result*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)

            key = (
                data['challenger_model'],
                data['experiment_type'],
                data['condition_name']
            )
            for sim_data in data.get('simulation_results', []):
                grouped_results[key].append(GameResult(**sim_data))

        self.logger.info(f"Loaded {sum(len(v) for v in grouped_results.values())} total simulations across {len(grouped_results)} conditions.")
        return grouped_results

    def _calculate_all_metrics_for_game(self, game_name: str, all_sim_results: Dict[tuple, List[GameResult]]) -> ExperimentResults:
        """Calculates performance, MAgIC, and dynamic metrics for all loaded simulations."""
        exp_results = ExperimentResults(
            game_name=game_name,
            challenger_models=get_challenger_models(),
            defender_model=get_defender_model()
        )

        for (challenger, exp_type, cond_name), sim_results in all_sim_results.items():
            player_metrics = PlayerMetrics(
                player_id=challenger,
                game_name=game_name,
                experiment_type=exp_type,
                condition_name=cond_name
            )

            player_metrics.performance_metrics = self.perf_calc.calculate_all_performance_metrics(sim_results, 'challenger')
            player_metrics.magic_metrics = self.magic_calc.calculate_all_magic_metrics(sim_results, 'challenger')

            if game_name in ['green_porter', 'athey_bagwell']:
                player_metrics.dynamic_metrics = self._calculate_dynamic_metrics(sim_results)

            if challenger not in exp_results.results:
                exp_results.results[challenger] = {}
            exp_results.results[challenger][cond_name] = player_metrics

        return exp_results

    def _calculate_dynamic_metrics(self, sim_results: List[GameResult]) -> Dict[str, Any]:
        """Calculates per-round averages for dynamic game metrics across all simulations."""
        all_round_metrics = defaultdict(list)

        for result in sim_results:
            round_metrics = self.dyn_calc.calculate_round_metrics(result)
            for metric_name, values in round_metrics.items():
                all_round_metrics[metric_name].append(values)

        averaged_metrics = {}
        for metric_name, sim_runs in all_round_metrics.items():
            avg_per_round = np.mean(np.array(sim_runs), axis=0).round(4).tolist()

            # Add descriptions based on metric name
            description = f"Average value of {metric_name} for each round across all simulations."
            if "reversion_triggers" in metric_name:
                description = "Average rate of triggering a price war in each period."
            elif "cooperation_state" in metric_name:
                description = "Average proportion of simulations in the 'Collusive' state in each period."
            elif "deception_events" in metric_name:
                description = "Average rate of deceptive reports in each period."
            elif "hhi_per_round" in metric_name:
                description = "Average Herfindahl-Hirschman Index (HHI) in each period."

            averaged_metrics[f"per_round_{metric_name}"] = {
                "name": f"per_round_{metric_name}",
                "values": avg_per_round,
                "description": description
            }
        return averaged_metrics

    def _save_experiment_results(self, exp_results: ExperimentResults):
        """Saves the fully analyzed ExperimentResults object to a JSON file."""
        output_file = self.output_dir / f"{exp_results.game_name}_metrics_analysis.json"

        serializable_data = {
            "game_name": exp_results.game_name,
            "challenger_models": exp_results.challenger_models,
            "defender_model": exp_results.defender_model,
            "results": {
                challenger: {
                    cond: {
                        "performance_metrics": {k: v.to_dict() for k, v in metrics.performance_metrics.items()},
                        "magic_metrics": {k: v.to_dict() for k, v in metrics.magic_metrics.items()},
                        "dynamic_metrics": metrics.dynamic_metrics
                    } for cond, metrics in conditions.items()
                } for challenger, conditions in exp_results.results.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        self.logger.info(f"Successfully saved analyzed metrics to {output_file}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    analyzer = MetricsAnalyzer()
    analyzer.analyze_all_games()