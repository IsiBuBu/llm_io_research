# analysis/visualization/visualize_ablations.py

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from scipy import stats

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
    return sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)

def format_with_ci(mean, ci, precision=2):
    """Formats a mean and confidence interval into a string."""
    return f"{mean:.{precision}f} [{mean-ci:.{precision}f}, {mean+ci:.{precision}f}]"

def _get_raw_results_ablation(results_dir: Path, game_name: str) -> pd.DataFrame:
    """Loads and structures raw JSON data specifically for ablation analysis."""
    records = []
    for file_path in results_dir.glob(f"{game_name}/*/*_competition_result*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Filter for 3-player baseline and ablation studies
            if '3_players' not in data['condition_name']:
                continue
            
            for sim in data.get('simulation_results', []):
                # Handle dynamic games
                if 'rounds' in sim['game_data']:
                    for i, round_data in enumerate(sim['game_data']['rounds']):
                        record = {
                            'game': sim['game_name'],
                            'model': sim['challenger_model'],
                            'condition': 'Baseline' if 'ablation' not in sim['condition_name'] else 'Ablation',
                            'simulation_id': sim['simulation_id'],
                            'round': i + 1,
                            'market_price': round_data.get('market_price'),
                            'market_state': round_data.get('market_state')
                        }
                        records.append(record)
                # Handle static games
                else:
                     for player, action in sim['actions'].items():
                        if player == 'challenger':
                            records.append({
                                'game': sim['game_name'],
                                'model': sim['challenger_model'],
                                'condition': 'Baseline' if 'ablation' not in sim['condition_name'] else 'Ablation',
                                'simulation_id': sim['simulation_id'],
                                'price': action.get('price') or action.get('bid')
                            })
    return pd.DataFrame(records)


# --- Table Generation ---

def _create_ablation_tables(perf_df, magic_df, tables_dir):
    """Creates the focused ablation study tables for each game."""
    logger = logging.getLogger("AblationVisualizer")
    logger.info("Creating Ablation Study summary tables...")
    
    # Combine performance and magic metrics
    combined_df = pd.concat([perf_df, magic_df])
    
    # Filter for 3-player baseline and ablation studies
    ablation_data = combined_df[combined_df['condition'].str.contains('3_players')].copy()
    ablation_data['condition_type'] = np.where(ablation_data['condition'].str.contains('ablation', case=False), 'Ablation', 'Baseline')
    
    agg_df = ablation_data.groupby(['game', 'model', 'condition_type', 'metric'])['value'].agg(['mean', get_ci]).reset_index()

    for game, metrics in [
        ('salop', ['average_profit', 'judgment']),
        ('green_porter', ['average_profit', 'reversion_frequency']), # Using average_profit for NPV
        ('spulber', ['average_profit', 'self_awareness']),
        ('athey_bagwell', ['average_profit', 'deception'])
    ]:
        game_df = agg_df[(agg_df['game'] == game) & (agg_df['metric'].isin(metrics))]
        
        game_df['formatted_metric'] = game_df.apply(lambda row: format_with_ci(row['mean'], row['get_ci']), axis=1)
        
        pivot = game_df.pivot_table(
            index='model',
            columns=['condition_type', 'metric'],
            values='formatted_metric',
            aggfunc='first'
        ).reindex(columns=['Baseline', 'Ablation'], level=0) # Ensure Baseline is first

        table_path = tables_dir / f"Ablation_Table_{game}.csv"
        pivot.to_csv(table_path)
        logger.info(f"Saved table: {table_path}")

# --- Plot Generation ---

def _plot_dumbbell(df: pd.DataFrame, game_name: str, metric: str, plots_dir: Path):
    """Helper function to create a dumbbell plot."""
    pivot_df = df.pivot(index='model', columns='condition', values=metric)
    
    plt.figure(figsize=(10, 8))
    for i, model in enumerate(pivot_df.index):
        baseline_val = pivot_df.loc[model, 'Baseline']
        ablation_val = pivot_df.loc[model, 'Ablation']
        plt.plot([baseline_val, ablation_val], [i, i], 'o-', color='grey', lw=3, markersize=10)
        plt.scatter(baseline_val, i, color='skyblue', s=150, zorder=5, label='Baseline' if i == 0 else "")
        plt.scatter(ablation_val, i, color='red', s=150, zorder=5, label='Ablation' if i == 0 else "")

    plt.yticks(range(len(pivot_df.index)), pivot_df.index)
    plt.title(f"{game_name.title()}: Impact of Ablation on {metric.replace('_', ' ').title()}", fontsize=16)
    plt.xlabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.ylabel("Challenger Model", fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"Ablation_{game_name}_dumbbell_{metric}.png")
    plt.close()

def _plot_salop_ablation(perf_df, results_dir, plots_dir):
    """Generates the performance and behavioral plots for the Salop ablation study."""
    logger = logging.getLogger("AblationVisualizer")
    logger.info("Generating Salop ablation plots...")
    
    # Performance Plot (Dumbbell)
    salop_profit = perf_df[(perf_df['game'] == 'salop') & (perf_df['metric'] == 'average_profit')]
    _plot_dumbbell(salop_profit, 'salop', 'average_profit', plots_dir)
    
    # Behavioral Plot (Violin)
    raw_df = _get_raw_results_ablation(results_dir, 'salop')
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=raw_df, x='model', y='price', hue='condition', split=True, inner='quart', palette='muted')
    plt.title("Salop: Price Distribution Under Increased Competition", fontsize=16)
    plt.xlabel("Challenger Model", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "Ablation_salop_violin_price.png")
    plt.close()

def _plot_green_porter_ablation(results_dir, plots_dir):
    """Generates the performance and behavioral plots for the Green & Porter ablation study."""
    logger = logging.getLogger("AblationVisualizer")
    logger.info("Generating Green & Porter ablation plots...")
    raw_df = _get_raw_results_ablation(results_dir, 'green_porter')
    
    # Performance Plot (Per-Round Market Price)
    market_price_df = raw_df.groupby(['condition', 'round'])['market_price'].mean().reset_index()
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=market_price_df, x='round', y='market_price', hue='condition', style='condition', markers=True, lw=2.5)
    plt.title("Green & Porter: Impact of Uncertainty on Market Price", fontsize=16)
    plt.xlabel("Game Round", fontsize=12)
    plt.ylabel("Average Market Price ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "Ablation_green_porter_line_market_price.png")
    plt.close()

    # Behavioral Plot (Collusion Stability)
    collusion_df = raw_df[raw_df['market_state'] == 'Collusive']
    collusion_by_round = collusion_df.groupby(['condition', 'round']).size().reset_index(name='count')
    total_sims = raw_df.groupby('condition')['simulation_id'].nunique().reset_index(name='total_sims')
    collusion_by_round = pd.merge(collusion_by_round, total_sims, on='condition')
    collusion_by_round['proportion'] = collusion_by_round['count'] / collusion_by_round['total_sims']
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=collusion_by_round, x='round', y='proportion', hue='condition', style='condition', markers=True, lw=2.5)
    plt.title("Green & Porter: Collusion Stability 'Survival Curve'", fontsize=16)
    plt.xlabel("Game Round", fontsize=12)
    plt.ylabel("Proportion of Games in Collusion", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "Ablation_green_porter_line_survival.png")
    plt.close()

def _plot_spulber_ablation(perf_df, magic_df, plots_dir):
    """Generates the performance and behavioral plots for the Spulber ablation study."""
    logger = logging.getLogger("AblationVisualizer")
    logger.info("Generating Spulber ablation plots...")
    
    # Performance Plot (Dumbbell)
    spulber_profit = perf_df[(perf_df['game'] == 'spulber') & (perf_df['metric'] == 'average_profit')]
    _plot_dumbbell(spulber_profit, 'spulber', 'average_profit', plots_dir)
    
    # Behavioral Plot (Grouped Bar)
    spulber_awareness = magic_df[(magic_df['game'] == 'spulber') & (magic_df['metric'] == 'self_awareness')]
    plt.figure(figsize=(12, 8))
    sns.barplot(data=spulber_awareness, x='model', y='value', hue='condition', palette='muted')
    plt.title("Spulber: Impact of Uncertainty on Bidding Strategy", fontsize=16)
    plt.xlabel("Challenger Model", fontsize=12)
    # UPDATED LABEL
    plt.ylabel("Bid Appropriateness Rate", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "Ablation_spulber_bar_self_awareness.png")
    plt.close()

def _plot_athey_bagwell_ablation(perf_df, magic_df, plots_dir):
    """Generates the performance and behavioral plots for the Athey & Bagwell ablation study."""
    logger = logging.getLogger("AblationVisualizer")
    logger.info("Generating Athey & Bagwell ablation plots...")

    # Performance Plot (Dumbbell for NPV, which is stored as average_profit)
    ab_profit = perf_df[(perf_df['game'] == 'athey_bagwell') & (perf_df['metric'] == 'average_profit')]
    _plot_dumbbell(ab_profit, 'athey_bagwell', 'average_profit', plots_dir)
    
    # Behavioral Plot (Grouped Bar for Deception Rate)
    ab_deception = magic_df[(magic_df['game'] == 'athey_bagwell') & (magic_df['metric'] == 'deception')]
    plt.figure(figsize=(12, 8))
    sns.barplot(data=ab_deception, x='model', y='value', hue='condition', palette='muted')
    plt.title("Athey & Bagwell: Impact of Reputational Stakes on Deception", fontsize=16)
    plt.xlabel("Challenger Model", fontsize=12)
    plt.ylabel("Deception Rate", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "Ablation_athey_bagwell_bar_deception.png")
    plt.close()


# --- Main Visualization Function ---

def visualize_ablations():
    """
    Generates all tables and plots for the Ablation Studies.
    """
    if not PLOT_LIBS_INSTALLED:
        print("Plotting libraries not installed. Skipping visualization.")
        return
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("AblationVisualizer")
    logger.info("--- Generating visualizations for Ablation Studies ---")
    
    script_dir = Path(__file__).parent
    analysis_dir = script_dir
    results_dir = script_dir.parent.parent / "results"

    plots_dir = analysis_dir / "plots" / "ablations"
    tables_dir = analysis_dir / "tables" / "ablations"
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    try:
        perf_df_full = pd.read_csv(analysis_dir.parent / "performance_metrics.csv")
        magic_df_full = pd.read_csv(analysis_dir.parent / "magic_behavioral_metrics.csv")
        
        # Prepare data by filtering and renaming conditions
        df_list = []
        for df in [perf_df_full, magic_df_full]:
            filtered_df = df[df['condition'].str.contains('3_players')].copy()
            filtered_df['condition'] = np.where(filtered_df['condition'].str.contains('ablation', case=False, regex=True), 'Ablation', 'Baseline')
            df_list.append(filtered_df)
        
        perf_df, magic_df = df_list

        # --- Generate Tables ---
        _create_ablation_tables(perf_df, magic_df, tables_dir)
        
        # --- Generate Plots ---
        _plot_salop_ablation(perf_df, results_dir, plots_dir)
        _plot_green_porter_ablation(results_dir, plots_dir)
        _plot_spulber_ablation(perf_df, magic_df, plots_dir)
        _plot_athey_bagwell_ablation(perf_df, magic_df, plots_dir)

        logger.info("--- Finished Ablation Study visualizations ---")

    except FileNotFoundError as e:
        logger.error(f"Failed to find necessary file for Ablation visualizations: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Ablation visualization: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_ablations()