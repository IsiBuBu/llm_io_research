# analysis/analyze_metrics.py

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from config.config import get_challenger_models, get_defender_model
from metrics import (
    GameResult, 
    PlayerMetrics, 
    ExperimentResults,
    PerformanceMetricsCalculator, 
    MAgICMetricsCalculator
)

class MetricsAnalyzer:
    """
    Processes raw simulation results to calculate and save aggregate
    Performance and MAgIC metrics for each experimental condition.
    """

    def __init__(self, results_dir: str = "results", output_dir: str = "analysis_output"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Instantiate metric calculators
        self.perf_calc = PerformanceMetricsCalculator()
        self.magic_calc = MAgICMetricsCalculator()

    def analyze_all_games(self):
        """
        Analyzes all available game results found in the results directory
        and saves the final metrics to the analysis output directory.
        """
        self.logger.info("Starting comprehensive metrics analysis...")
        game_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        for game_dir in game_dirs:
            game_name = game_dir.name
            self.logger.info(f"--- Analyzing game: {game_name.upper()} ---")
            
            try:
                # 1. Load all simulation results for the game
                all_sim_results = self._load_simulation_results(game_dir)
                if not all_sim_results:
                    self.logger.warning(f"No valid simulation results found for {game_name}. Skipping.")
                    continue

                # 2. Calculate metrics for each condition
                experiment_results = self._calculate_all_metrics_for_game(game_name, all_sim_results)

                # 3. Save the aggregated results
                self._save_experiment_results(experiment_results)

            except Exception as e:
                self.logger.error(f"Failed to analyze {game_name}: {e}", exc_info=True)
        
        self.logger.info("Comprehensive metrics analysis complete.")

    def _load_simulation_results(self, game_dir: Path) -> Dict[str, List[GameResult]]:
        """Loads all raw simulation_results.json files for a specific game."""
        # Group results by (challenger_model, experiment_type, condition_name)
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
        """Calculates performance and MAgIC metrics for all loaded simulations."""
        challenger_models = get_challenger_models()
        defender_model = get_defender_model()
        
        exp_results = ExperimentResults(
            game_name=game_name,
            challenger_models=challenger_models,
            defender_model=defender_model
        )

        for (challenger, exp_type, cond_name), sim_results in all_sim_results.items():
            player_metrics = PlayerMetrics(
                player_id=challenger,
                game_name=game_name,
                experiment_type=exp_type,
                condition_name=cond_name
            )
            
            # Calculate metrics using the dedicated calculators
            player_metrics.performance_metrics = self.perf_calc.calculate_all_performance_metrics(sim_results, 'challenger')
            player_metrics.magic_metrics = self.magic_calc.calculate_all_magic_metrics(sim_results, 'challenger')
            
            # Add to the main results container
            if challenger not in exp_results.results:
                exp_results.results[challenger] = {}
            exp_results.results[challenger][cond_name] = player_metrics
            
        return exp_results

    def _save_experiment_results(self, exp_results: ExperimentResults):
        """Saves the fully analyzed ExperimentResults object to a JSON file."""
        output_file = self.output_dir / f"{exp_results.game_name}_metrics_analysis.json"
        
        # Serialize the dataclasses to a dictionary format for JSON
        serializable_data = {
            "game_name": exp_results.game_name,
            "challenger_models": exp_results.challenger_models,
            "defender_model": exp_results.defender_model,
            "results": {
                challenger: {
                    cond: {
                        "performance_metrics": {k: v.to_dict() for k, v in metrics.performance_metrics.items()},
                        "magic_metrics": {k: v.to_dict() for k, v in metrics.magic_metrics.items()}
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