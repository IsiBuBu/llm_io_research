# analysis/analyze_reliability.py

import pandas as pd
import logging
from pathlib import Path

try:
    import pingouin as pg
    import simpledorff as sd
    RELIABILITY_LIBS_INSTALLED = True
except ImportError:
    RELIABILITY_LIBS_INSTALLED = False

def analyze_reliability():
    """
    Calculates and prints reliability statistics for the LLM judge evaluations.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("ReliabilityAnalysis")
    
    if not RELIABILITY_LIBS_INSTALLED:
        logger.error("Reliability libraries not found. Please run: pip install pingouin simpledorff")
        return

    analysis_dir = Path("analysis_output")
    try:
        df = pd.read_csv(analysis_dir / "judge_evaluations.csv")
    except FileNotFoundError:
        logger.error("`judge_evaluations.csv` not found. Please run `run_judge_evaluations.py` first.")
        return

    logger.info("--- LLM as Judge Reliability Analysis ---")
    
    all_results = {}

    for (game, metric), data in df.groupby(['game', 'metric']):
        logger.info(f"\n--- Analyzing: {game.title()} - {metric.title()} ---")
        
        # Reshape data for reliability analysis (items in rows, raters in columns)
        pivot_df = data.pivot(index='sample_id', columns='replication_id', values='alignment_score')
        
        # (A) Internal-Consistency Reliability
        try:
            icc = pg.intraclass_corr(data=pivot_df.T, targets='index', raters='columns', ratings='values')
            cronbach_alpha = icc[icc['Type'] == 'ICC1']['ICC'].iloc[0]
            logger.info(f"Cronbach's Alpha (α): {cronbach_alpha:.4f}")
            # Note: McDonald's Omega often requires a factor analysis model, which can be complex.
            # Cronbach's Alpha is a very strong and standard substitute.
        except Exception as e:
            cronbach_alpha = None
            logger.error(f"Could not calculate Cronbach's Alpha: {e}")

        # (B) Inter-Rater Agreement
        try:
            kripp_alpha = sd.krippendorff_alpha(df_long=data,
                                               experiment_col='sample_id',
                                               annotator_col='replication_id',
                                               class_col='alignment_score')
            logger.info(f"Krippendorff's Alpha (ordinal): {kripp_alpha:.4f}")
        except Exception as e:
            kripp_alpha = None
            logger.error(f"Could not calculate Krippendorff's Alpha: {e}")
            
        all_results[f"{game}_{metric}"] = {
            "cronbach_alpha": cronbach_alpha,
            "krippendorff_alpha": kripp_alpha
        }

    # Save results
    output_path = analysis_dir / "reliability_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSuccessfully saved reliability results to {output_path}")


if __name__ == "__main__":
    analyze_reliability()
