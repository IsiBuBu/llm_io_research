# run_analysis.py

import logging
import sys
from pathlib import Path

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent))

from analysis.analyze_metrics import MetricsAnalyzer
from analysis.create_summary_csvs import SummaryCreator
from analysis.analyze_correlations import CorrelationAnalyzer
from analysis.analyze_reliability import analyze_reliability # NEW
from analysis.visualize_results import main as visualize_all # UPDATED

def setup_logging():
    """Configures basic logging for the analysis pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
        stream=sys.stdout
    )

def main():
    """
    Runs the full post-experiment analysis pipeline.
    """
    setup_logging()
    logger = logging.getLogger("AnalysisPipeline")
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING FULL ANALYSIS PIPELINE 🚀")
    logger.info("=" * 80)

    try:
        # Step 1: Calculate metrics from raw results
        logger.info("[Step 1/5] Analyzing metrics from simulation results...")
        MetricsAnalyzer().analyze_all_games()
        logger.info("[Step 1/5] ✅ Metrics analysis complete.")

        # Step 2: Create flattened summary CSV files
        logger.info("-" * 80)
        logger.info("[Step 2/5] Creating summary CSV files...")
        SummaryCreator().create_all_summaries()
        logger.info("[Step 2/5] ✅ Summary CSV creation complete.")
        
        # Step 3: Analyze correlations between metrics
        logger.info("-" * 80)
        logger.info("[Step 3/5] Analyzing correlations between metrics...")
        CorrelationAnalyzer().analyze_all_correlations()
        logger.info("[Step 3/5] ✅ Correlation analysis complete.")

        # Step 4: Analyze reliability of judge evaluations
        logger.info("-" * 80)
        logger.info("[Step 4/5] Analyzing reliability of judge evaluations...")
        logger.info("NOTE: This step assumes `run_judge_evaluations.py` has been run.")
        analyze_reliability()
        logger.info("[Step 4/5] ✅ Reliability analysis complete.")

        # Step 5: Generate visualizations
        logger.info("-" * 80)
        logger.info("[Step 5/5] Generating visualizations...")
        visualize_all()
        logger.info("[Step 5/5] ✅ Visualization generation complete.")
        
        logger.info("=" * 80)
        logger.info("🎉 ANALYSIS PIPELINE FINISHED SUCCESSFULLY! 🎉")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ An error occurred during the analysis pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()