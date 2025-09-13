# analysis/visualize_results.py

import pandas as pd
import logging
from pathlib import Path
import json

# Optional: For plotting, libraries like matplotlib and seaborn are common.
# These would need to be added to your requirements.txt
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_LIBS_INSTALLED = True
except ImportError:
    PLOT_LIBS_INSTALLED = False

class ResultsVisualizer:
    """
    Generates and saves plots and tables from the analyzed experimental data.
    """

    def __init__(self, analysis_dir: str = "analysis_output"):
        self.analysis_dir = Path(analysis_dir)
        self.plots_dir = self.analysis_dir / "plots"
        self.tables_dir = self.analysis_dir / "tables"
        self.plots_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

        if not PLOT_LIBS_INSTALLED:
            self.logger.warning("Plotting libraries (matplotlib, seaborn) not found. Visualization will be skipped.")
            self.logger.warning("Please run: pip install matplotlib seaborn")

    def visualize_all(self):
        """Generates all standard visualizations for the project."""
        if not PLOT_LIBS_INSTALLED:
            return

        self.logger.info("Starting result visualization...")
        
        try:
            # Generate performance and MAgIC metric comparison plots
            perf_df = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
            magic_df = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
            self._plot_metric_comparisons(perf_df, "Performance Metrics Comparison")
            self._plot_metric_comparisons(magic_df, "MAgIC Metrics Comparison")
            self._create_summary_tables(perf_df, magic_df)

            # Generate correlation visualizations
            corr_df = pd.read_csv(self.analysis_dir / "correlations_analysis.csv")
            self._plot_correlation_heatmap(corr_df, "Correlation Analysis")
            self._create_correlation_tables(corr_df)

            # Generate dynamic game plots
            self._plot_dynamic_game_metrics()

            self.logger.info(f"Visualizations saved to '{self.plots_dir}' and tables to '{self.tables_dir}'")

        except FileNotFoundError as e:
            self.logger.error(f"Could not generate visualizations: an input file is missing. {e}")
            self.logger.error("Please ensure analyze_metrics.py and analyze_correlations.py have been run successfully.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during visualization: {e}", exc_info=True)

    def _plot_metric_comparisons(self, df: pd.DataFrame, title: str):
        """Creates and saves bar plots comparing models on key metrics for each game."""
        for game in df['game'].unique():
            game_df = df[df['game'] == game]
            
            # Create a plot for each metric in the game
            for metric in game_df['metric'].unique():
                plt.figure(figsize=(12, 8))
                metric_df = game_df[game_df['metric'] == metric]
                
                sns.barplot(data=metric_df, x='model', y='value', hue='condition', palette='viridis')
                
                plt.title(f"{game.title()}: {metric.replace('_', ' ').title()}", fontsize=16)
                plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                plt.xlabel("Model", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.legend(title='Condition')
                
                plot_filename = self.plots_dir / f"{game}_{metric}_comparison.png"
                plt.savefig(plot_filename)
                plt.close()

    def _plot_correlation_heatmap(self, df: pd.DataFrame, title: str):
        """Creates and saves a heatmap of the correlation results for each game."""
        for game in df['game_name'].unique():
            game_df = df[df['game_name'] == game]
            
            # Pivot the DataFrame to create a matrix for the heatmap
            pivot_df = game_df.pivot_table(
                index='magic_metric', 
                columns='performance_metric', 
                values='correlation_coefficient'
            )
            
            if pivot_df.empty:
                continue

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                pivot_df, 
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                linewidths=.5,
                fmt=".2f"
            )
            
            plt.title(f"{game.title()}: MAgIC vs. Performance Metrics Correlation", fontsize=16)
            plt.xlabel("Performance Metrics", fontsize=12)
            plt.ylabel("MAgIC Behavioral Metrics", fontsize=12)
            plt.tight_layout()

            plot_filename = self.plots_dir / f"{game}_correlation_heatmap.png"
            plt.savefig(plot_filename)
            plt.close()

    def _plot_dynamic_game_metrics(self):
        """Loads analysis JSONs and plots per-round dynamic metrics."""
        for json_file in self.analysis_dir.glob("*_metrics_analysis.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)

            game_name = data['game_name']
            if game_name not in ['green_porter', 'athey_bagwell']:
                continue

            for model, conditions in data['results'].items():
                for condition, metrics in conditions.items():
                    dynamic_metrics = metrics.get('dynamic_metrics', {})
                    if not dynamic_metrics:
                        continue

                    for metric_name, metric_data in dynamic_metrics.items():
                        plt.figure(figsize=(12, 7))
                        plt.plot(metric_data['values'], marker='o', linestyle='-')
                        plt.title(f"{game_name.title()} - {condition}\n{model}\n{metric_data['description']}", fontsize=14)
                        plt.xlabel("Game Round", fontsize=12)
                        plt.ylabel(metric_name.replace('per_round_', '').replace('_', ' ').title(), fontsize=12)
                        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                        plt.tight_layout()
                        
                        plot_filename = self.plots_dir / f"{game_name}_{model}_{condition}_{metric_name}.png"
                        plt.savefig(plot_filename)
                        plt.close()
    
    def _create_summary_tables(self, perf_df, magic_df):
        """Creates and saves summary tables for key metrics."""
        key_metrics = ['win_rate', 'average_profit', 'rationality']
        
        perf_summary = perf_df[perf_df['metric'].isin(key_metrics)].pivot_table(
            index='model', columns=['game', 'metric'], values='value'
        )
        magic_summary = magic_df[magic_df['metric'].isin(key_metrics)].pivot_table(
            index='model', columns=['game', 'metric'], values='value'
        )
        
        perf_summary.to_csv(self.tables_dir / "performance_summary.csv")
        magic_summary.to_csv(self.tables_dir / "magic_summary.csv")

    def _create_correlation_tables(self, corr_df):
        """Creates and saves tables of significant correlations."""
        significant_corrs = corr_df[corr_df['p_value'] < 0.05]
        significant_corrs.to_csv(self.tables_dir / "significant_correlations.csv", index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    visualizer = ResultsVisualizer()
    visualizer.visualize_all()