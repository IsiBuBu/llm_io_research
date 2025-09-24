# analysis/visualize_rq4.py

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from scipy.stats import pearsonr, t

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_LIBS_INSTALLED = True
except ImportError:
    PLOT_LIBS_INSTALLED = False

# --- Helper Functions ---

def get_ci(data):
    """Calculates the 95% confidence interval for a given dataset."""
    if len(data) < 2:
        return 0
    mean, sem = np.mean(data), stats.sem(data)
    # Return half the width of the CI
    return sem * t.ppf((1 + 0.95) / 2., len(data)-1)

def format_with_ci(mean, ci, precision=2):
    """Formats a mean and confidence interval into a string."""
    return f"{mean:.{precision}f} [{mean-ci:.{precision}f}, {mean+ci:.{precision}f}]"

def _get_thinking_tokens(results_dir: Path) -> pd.DataFrame:
    """Loads and aggregates thinking token data from raw JSON results for each simulation."""
    records = []
    for file_path in results_dir.glob("*/*/*_competition_result*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for sim in data.get('simulation_results', []):
                challenger_meta = sim.get('game_data', {}).get('llm_metadata', {}).get('challenger', {})
                if 'thinking_medium' in sim['challenger_model']:
                    if 'rounds' in sim['game_data']:
                        thinking_tokens_per_round = [
                            r.get('llm_metadata', {}).get('challenger', {}).get('thinking_tokens', 0) 
                            for r in sim['game_data']['rounds']
                        ]
                        avg_thinking_tokens = np.mean([t for t in thinking_tokens_per_round if t is not None])
                    else:
                        avg_thinking_tokens = challenger_meta.get('thinking_tokens', 0)

                    records.append({
                        'game': sim['game_name'],
                        'model': sim['challenger_model'],
                        'condition': sim['condition_name'],
                        'simulation_id': sim['simulation_id'],
                        'thinking_tokens': avg_thinking_tokens
                    })
    return pd.DataFrame(records)


# --- Table and Plot Generation ---

def _create_comparative_analysis_table(magic_df, thinking_df, tables_dir):
    """Table 4.1: Creates a comparative table of thinking tokens and MAgIC scores with CIs."""
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("Creating Table 4.1: Comparative Analysis of Thinking Tokens and MAgIC Scores...")
    
    # Aggregate both magic scores and thinking tokens to get mean and CI
    magic_agg = magic_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg(['mean', get_ci]).reset_index()
    tokens_agg = thinking_df.groupby(['game', 'model', 'condition'])['thinking_tokens'].agg(['mean', get_ci]).reset_index()

    # Format with CIs
    magic_agg['MAgIC Score'] = magic_agg.apply(lambda row: format_with_ci(row['mean'], row['get_ci']), axis=1)
    tokens_agg['Avg. Thinking Tokens'] = tokens_agg.apply(lambda row: format_with_ci(row['mean'], row['get_ci'], precision=0), axis=1)

    # Pivot magic scores
    magic_pivot = magic_agg.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='MAgIC Score', aggfunc='first')
    
    # Merge with formatted tokens
    final_df = pd.merge(magic_pivot, tokens_agg[['game', 'model', 'condition', 'Avg. Thinking Tokens']], on=['game', 'model', 'condition']).reset_index()
    
    for game in final_df['game'].unique():
        game_df = final_df[final_df['game'] == game].copy()
        game_df['structural_variation'] = np.where(game_df['condition'].str.contains('5_players|more_players', regex=True), '5-Player', '3-Player')
        
        # Select relevant columns and reorder
        metrics = [col for col in game_df.columns if col in ['model', 'structural_variation', 'Avg. Thinking Tokens'] + list(magic_df['metric'].unique())]
        game_df = game_df[metrics]
        
        table_path = tables_dir / f"T4.1_{game}_comparative_analysis.csv"
        game_df.to_csv(table_path, index=False)
        logger.info(f"Saved table: {table_path}")

def _plot_dual_axis_chart(magic_df, thinking_df, plots_dir):
    """Plot 4.1: Creates a dual-axis chart comparing thinking tokens and MAgIC scores."""
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("Generating Plot 4.1: Dual-Axis Chart of Thinking Tokens and MAgIC Scores...")
    
    avg_tokens = thinking_df.groupby(['model', 'condition'])['thinking_tokens'].mean().reset_index()
    merged_df = pd.merge(magic_df, avg_tokens, on=['model', 'condition'])
    merged_df['structural_variation'] = np.where(merged_df['condition'].str.contains('5_players|more_players', regex=True), '5-Player', '3-Player')

    for (metric, variation), plot_data in merged_df.groupby(['metric', 'structural_variation']):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.barplot(data=plot_data, x='model', y='value', ax=ax1, palette='muted', alpha=0.8, ci=None)
        ax1.set_ylabel(f"{metric.title()} Score", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

        ax2 = ax1.twinx()
        sns.lineplot(data=plot_data, x='model', y='thinking_tokens', ax=ax2, color='r', marker='o', sort=False)
        ax2.set_ylabel("Average Thinking Tokens", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(f"Thinking Tokens vs. {metric.title()} Score ({variation})", fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / f"P4.1_dual_axis_{metric}_{variation}.png")
        plt.close()

# --- Main Visualization Function ---

def visualize_rq4():
    """
    Generates all tables and plots for Research Question 4.
    """
    if not PLOT_LIBS_INSTALLED:
        print("Plotting libraries not installed. Skipping visualization.")
        return
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("--- Generating visualizations for RQ4: Thinking & Strategic Capability ---")
    
    script_dir = Path(__file__).parent
    analysis_dir = script_dir
    results_dir = script_dir.parent / "results"

    plots_dir = analysis_dir / "plots" / "rq4"
    tables_dir = analysis_dir / "tables" / "rq4"
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    try:
        magic_df_full = pd.read_csv(analysis_dir / "magic_behavioral_metrics.csv")
        thinking_df = _get_thinking_tokens(results_dir)
        
        thinking_models = [m for m in magic_df_full['model'].unique() if 'thinking_medium' in m]
        magic_df_thinking = magic_df_full[magic_df_full['model'].isin(thinking_models)].copy()

        # --- Part 1: Quantitative Link ---
        _create_comparative_analysis_table(magic_df_thinking, thinking_df, tables_dir)
        _plot_dual_axis_chart(magic_df_thinking, thinking_df, plots_dir)
        
        logger.info("--- Finished RQ4 visualizations ---")
        
    except FileNotFoundError as e:
        logger.error(f"Failed to find necessary file for RQ4 visualizations: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during RQ4 visualization: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_rq4()