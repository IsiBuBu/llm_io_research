# run_analysis.py

import logging
import sys
from pathlib import Path

# --- Pre-flight Check for Dependencies ---
try:
    import pandas
    import numpy
    import scipy
    import matplotlib
    import seaborn
except ImportError as e:
    print("="*80)
    print(f"❌ CRITICAL ERROR: Missing essential Python package: {e.name}")
    print("Please install all required packages by running the following command:")
    print("pip install -r requirements.txt")
    print("="*80)
    sys.exit(1)


# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent))

from config.config import get_experiments_dir, get_analysis_dir
from analysis.engine.analyze_metrics import MetricsAnalyzer
from analysis.engine.create_summary_csvs import SummaryCreator
from analysis.engine.analyze_correlations import CorrelationAnalyzer
from analysis.visualization.visualize_results import main as visualize_all

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
    
    # Get directory paths from config
    experiments_dir = get_experiments_dir()
    analysis_dir = get_analysis_dir()
    analysis_dir.mkdir(exist_ok=True, parents=True)

    logger.info("=" * 80)
    logger.info("🚀 STARTING FULL ANALYSIS PIPELINE 🚀")
    logger.info(f"Reading experiment data from: {experiments_dir}")
    logger.info(f"Saving analysis output to:    {analysis_dir}")
    logger.info("=" * 80)

    # --- UPDATED LOGIC: Check for experiment results ---
    if not experiments_dir.exists() or not any(experiments_dir.iterdir()):
        logger.critical(f"❌ CRITICAL ERROR: Experiment results directory is missing or empty: '{experiments_dir}'")
        logger.critical("Please run the experiment script first by executing: python run_experiments.py")
        sys.exit(1)

    try:
        # Step 1: Calculate metrics from raw results
        logger.info("[Step 1/4] Analyzing metrics from simulation results...")
        MetricsAnalyzer(results_dir=str(experiments_dir), output_dir=str(analysis_dir)).analyze_all_games()
        logger.info("[Step 1/4] ✅ Metrics analysis complete.")

        # Step 2: Create flattened summary CSV files
        logger.info("-" * 80)
        logger.info("[Step 2/4] Creating summary CSV files...")
        SummaryCreator(analysis_dir=str(analysis_dir)).create_all_summaries()
        logger.info("[Step 2/4] ✅ Summary CSV creation complete.")
        
        # Step 3: Analyze correlations between metrics
        logger.info("-" * 80)
        logger.info("[Step 3/4] Analyzing correlations between metrics...")
        CorrelationAnalyzer(analysis_dir=str(analysis_dir)).analyze_all_correlations()
        logger.info("[Step 3/4] ✅ Correlation analysis complete.")

        # --- Check for necessary CSV files before visualization ---
        required_csvs = ["performance_metrics.csv", "magic_behavioral_metrics.csv", "correlations_analysis_structural.csv"]
        missing_csvs = [f for f in required_csvs if not (analysis_dir / f).exists()]

        if missing_csvs:
            logger.error("="*80)
            logger.error("❌ ERROR: Cannot generate visualizations because the following required data files are missing:")
            for f in missing_csvs:
                logger.error(f"  - {f}")
            logger.error("Please re-run the analysis pipeline to generate these files.")
            logger.error("="*80)
            sys.exit(1)


        # Step 4: Generate visualizations
        logger.info("-" * 80)
        logger.info("[Step 4/4] Generating visualizations...")
        visualize_all()
        logger.info("[Step 4/4] ✅ Visualization generation complete.")

        logger.info("=" * 80)
        logger.info("🎉 ANALYSIS PIPELINE FINISHED SUCCESSFULLY! 🎉")
        logger.info(f"Check the '{analysis_dir}' directory for all outputs.")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ An error occurred during the analysis pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    main()
