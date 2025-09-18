# analysis/visualization/visualize_results.py

import logging
import sys
from pathlib import Path

# Ensure the project root is in the Python path
# This allows imports from the top-level 'analysis' package
sys.path.append(str(Path(__file__).parent.parent.parent))

from .visualize_rq1 import visualize_rq1
from .visualize_rq2 import visualize_rq2
from .visualize_rq3 import visualize_rq3
from .visualize_rq4 import visualize_rq4
from .visualize_ablations import visualize_ablations

def setup_logging():
    """Configures basic logging for the visualization pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
        stream=sys.stdout
    )

def main():
    """
    Runs the full visualization pipeline by calling specialized scripts for each research question.
    """
    setup_logging()
    logger = logging.getLogger("MainVisualizer")

    logger.info("=" * 80)
    logger.info("🚀 STARTING FULL VISUALIZATION PIPELINE 🚀")
    logger.info("=" * 80)

    try:
        visualize_rq1()
        visualize_rq2()
        visualize_rq3()
        visualize_rq4()
        visualize_ablations()
        
        logger.info("=" * 80)
        logger.info("🎉 VISUALIZATION PIPELINE FINISHED SUCCESSFULLY! 🎉")
        logger.info("Check the 'output/analysis/plots' and 'output/analysis/tables' directories.")
        logger.info("=" * 80)

    except FileNotFoundError as e:
        logger.error(f"❌ A visualization script failed because an input file is missing: {e}")
        logger.error("Please ensure the full analysis pipeline (`run_analysis.py`) has been run successfully first.")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during the visualization pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()