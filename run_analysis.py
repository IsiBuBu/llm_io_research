# run_analysis.py

import logging
import sys
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent))

from analysis.analyze_metrics import MetricsAnalyzer
from analysis.create_summary_csvs import SummaryCreator
from analysis.analyze_correlations import CorrelationAnalyzer
from analysis.visualize_results import ResultsVisualizer

def setup_logging():
    """Configures basic logging for the analysis pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
        stream=sys.stdout
    )

def main():
    """
    Runs the full post-experiment analysis pipeline:
    1. Calculates performance and MAgIC metrics from raw results.
    2. Flattens the JSON metrics into summary CSV files.
    3. Analyzes correlations between the calculated metrics.
    4. Generates and saves visualizations of the results.
    """
    setup_logging()
    logger = logging.getLogger("AnalysisPipeline")
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING FULL ANALYSIS PIPELINE 🚀")
    logger.info("=" * 80)

    try:
        # Step 1: Calculate all performance and MAgIC metrics from JSON results
        logger.info("[Step 1/4] Analyzing metrics from simulation results...")
        metrics_analyzer = MetricsAnalyzer()
        metrics_analyzer.analyze_all_games()
        logger.info("[Step 1/4] ✅ Metrics analysis complete.")

        # Step 2: Create flattened summary CSV files from the JSON analysis
        logger.info("-" * 80)
        logger.info("[Step 2/4] Creating summary CSV files...")
        summary_creator = SummaryCreator()
        summary_creator.create_all_summaries()
        logger.info("[Step 2/4] ✅ Summary CSV creation complete.")
        
        # Step 3: Analyze correlations between the metrics using the CSVs
        logger.info("-" * 80)
        logger.info("[Step 3/4] Analyzing correlations between metrics...")
        correlation_analyzer = CorrelationAnalyzer()
        correlation_analyzer.analyze_all_correlations()
        logger.info("[Step 3/4] ✅ Correlation analysis complete.")

        # Step 4: Generate visualizations from the CSVs
        logger.info("-" * 80)
        logger.info("[Step 4/4] Generating visualizations...")
        results_visualizer = ResultsVisualizer()
        results_visualizer.visualize_all()
        logger.info("[Step 4/4] ✅ Visualization generation complete.")
        
        logger.info("=" * 80)
        logger.info("🎉 ANALYSIS PIPELINE FINISHED SUCCESSFULLY! 🎉")
        logger.info("Check the 'analysis_output' directory for all generated files and plots.")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ An error occurred during the analysis pipeline: {e}", exc_info=True)
        logger.error("Please check the error message and ensure the raw result files exist in the 'results' directory.")

if __name__ == "__main__":
    main()