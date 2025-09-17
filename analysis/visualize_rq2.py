# analysis/visualize_rq2.py

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from scipy import stats

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
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
    return sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)

def format_with_ci(mean, ci, precision=2):
    """Formats a mean and confidence interval into a string."""
    return f"{mean:.{precision}f} [{mean-ci:.{precision}f}, {mean+ci:.{precision}f}]"

def _get_raw_results_rq2(results_dir: Path, game_name: str) -> pd.DataFrame:
    """Loads and structures raw JSON data for RQ2 analysis."""
    records = []
    for file_path in results_dir.glob(f"{game_name}/*/*_competition_result*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for sim in data.get('simulation_results', []):
                if 'rounds' in sim['game_data']:
                    trigger_price = sim['game_data'].get('constants', {}).get('trigger_price')
                    for i, round_data in enumerate(sim['game_data']['rounds']):
                        challenger_action = round_data.get('actions', {}).get('challenger', {})
                        challenger_true_cost = round_data.get('player_true_costs', {}).get('challenger')
                        
                        record = {
                            'game': sim['game_name'],
                            'model': sim['challenger_model'],
                            'condition': sim['condition_name'],
                            'simulation_id': sim['simulation_id'],
                            'round': i + 1,
                            'market_state': round_data.get('market_state'),
                            'market_price': round_data.get('market_price'),
                            'trigger_price': trigger_price,
                            'challenger_true_cost': challenger_true_cost,
                            'challenger_report': challenger_action.get('report'),
                            'challenger_quantity': challenger_action.get('quantity')
                        }
                        records.append(record)
    return pd.DataFrame(records)

# --- Table Generation ---

def _create_rq2_tables(magic_df, tables_dir):
    """Creates and saves the per-game summary tables for RQ2 MAgIC metrics (Tables 2.1-2.4)."""
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("Creating RQ2 per-game summary tables (Tables 2.1-2.4)...")

    agg_df = magic_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg(['mean', get_ci]).reset_index()

    for game in agg_df['game'].unique():
        game_df = agg_df[agg_df['game'] == game]
        
        game_df['formatted_metric'] = game_df.apply(
            lambda row: format_with_ci(row['mean'], row['get_ci']), axis=1
        )
        
        pivot = game_df.pivot_table(
            index='model',
            columns=['condition', 'metric'],
            values='formatted_metric',
            aggfunc='first'
        ).sort_index(axis=1)

        table_path = tables_dir / f"T2_{game}_magic_summary.csv"
        pivot.to_csv(table_path)
        logger.info(f"Saved table: {table_path}")

def _create_overall_magic_table(magic_df, tables_dir):
    """Creates and saves a summary table for overall MAgIC performance across all games."""
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("Creating Overall MAgIC Performance summary table...")
    
    core_metrics = ['judgment', 'reasoning', 'deception', 'self_awareness', 'cooperation', 'coordination', 'rationality']
    
    overall_scores = magic_df[magic_df['metric'].isin(core_metrics)]
    agg_scores = overall_scores.groupby(['model', 'metric'])['value'].agg(['mean', get_ci]).reset_index()

    agg_scores['formatted_score'] = agg_scores.apply(
        lambda row: format_with_ci(row['mean'], row['get_ci']), axis=1
    )

    pivot_table = agg_scores.pivot_table(
        index='model',
        columns='metric',
        values='formatted_score',
        aggfunc='first'
    )[core_metrics]

    table_path = tables_dir / "Overall_MAgIC_Performance_Summary.csv"
    pivot_table.to_csv(table_path)
    logger.info(f"Saved table: {table_path}")


# --- Plot Generation ---

def _plot_overall_magic_profile(magic_df, plots_dir):
    """Plot 1.1: Overall MAgIC Profile by Market Structure (Radar Charts)"""
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("Generating Plot 1.1: Overall MAgIC Profile (Radar Charts)...")
    
    core_metrics = ['judgment', 'reasoning', 'deception', 'self_awareness', 'cooperation', 'coordination', 'rationality']
    
    overall_scores = magic_df[magic_df['metric'].isin(core_metrics)]
    overall_scores = overall_scores.groupby(['model', 'condition', 'metric'])['value'].mean().reset_index()

    overall_scores['structural_variation'] = np.where(overall_scores['condition'].str.contains('5_players|more_players', regex=True), '5-Player', '3-Player')
    
    for variation in overall_scores['structural_variation'].unique():
        condition_df = overall_scores[overall_scores['structural_variation'] == variation]
        
        labels = core_metrics
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1] # complete the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for model in sorted(condition_df['model'].unique()):
            model_data = condition_df[condition_df['model'] == model].set_index('metric').reindex(labels).fillna(0)
            values = model_data['value'].tolist()
            values += values[:1] # complete the loop
            ax.plot(angles, values, label=model, linewidth=2)
            ax.fill(angles, values, alpha=0.25)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label.replace('_', ' ').title() for label in labels], size=12)
        
        plt.title(f"Overall MAgIC Profile ({variation} Condition)", size=20, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(plots_dir / f"P1.1_overall_magic_profile_{variation}.png")
        plt.close()

def _plot_game_specific_adaptation(magic_df, plots_dir):
    """Plot 2.1: Game-Specific Behavioral Adaptation (Grouped Bar Chart)"""
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("Generating Plot 2.1: Game-Specific Behavioral Adaptation (Grouped Bar Charts)...")

    magic_df['structural_variation'] = np.where(magic_df['condition'].str.contains('5_players|more_players', regex=True), '5-Player', '3-Player')

    for game in magic_df['game'].unique():
        game_df = magic_df[magic_df['game'] == game]
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=game_df, x='metric', y='value', hue='model', 
                         palette=sns.color_palette("muted", len(game_df['model'].unique())),
                         dodge=True)
        
        plt.title(f"{game.replace('_', ' ').title()}: Game-Specific Behavioral Adaptation", fontsize=16)
        plt.xlabel("MAgIC Metric", fontsize=12)
        plt.ylabel("Metric Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(plots_dir / f"P2.1_{game}_behavioral_adaptation.png")
        plt.close()

def _plot_collusion_stability(results_dir, plots_dir):
    """Plot 3.1: Collusion Stability 'Survival Curve' for Green & Porter."""
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("Generating Plot 3.1: Collusion Stability 'Survival Curve'...")
    
    raw_df = _get_raw_results_rq2(results_dir, 'green_porter')
    if raw_df.empty:
        logger.warning("No raw data found for Green & Porter to plot collusion stability.")
        return
        
    total_sims = raw_df.groupby(['model', 'condition'])['simulation_id'].nunique().reset_index(name='total_sims')
    
    collusion_counts = raw_df[raw_df['market_state'] == 'Collusive'].groupby(['model', 'condition', 'round']).size().reset_index(name='collusive_count')
    
    collusion_prop = pd.merge(collusion_counts, total_sims, on=['model', 'condition'])
    collusion_prop['proportion'] = collusion_prop['collusive_count'] / collusion_prop['total_sims']

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=collusion_prop, x='round', y='proportion', hue='model', style='condition', lw=2.5, palette='tab10', markers=True, dashes=True)
    
    plt.title("Collusion Stability 'Survival Curve' (Green & Porter)", fontsize=16)
    plt.xlabel("Game Round", fontsize=12)
    plt.ylabel("Proportion of Games in Collusion", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title='Model & Condition')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "P3.1_green_porter_collusion_stability.png")
    plt.close()

def _plot_nuanced_reporting_strategy(results_dir, plots_dir):
    """Plot 3.3: Nuanced Reporting Strategy Over Time for Athey & Bagwell."""
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("Generating Plot 3.3: Nuanced Reporting Strategy Over Time...")
    
    raw_df = _get_raw_results_rq2(results_dir, 'athey_bagwell')
    if raw_df.empty:
        logger.warning("No raw data found for Athey & Bagwell to plot reporting strategy.")
        return
        
    raw_df['truthful_signal'] = ((raw_df['challenger_true_cost'] == 'low') & (raw_df['challenger_report'] == 'low')).astype(int)
    raw_df['deception_event'] = ((raw_df['challenger_true_cost'] == 'high') & (raw_df['challenger_report'] == 'low')).astype(int)
    
    rates_df = raw_df.groupby(['model', 'condition', 'round'])[['truthful_signal', 'deception_event']].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    
    sns.lineplot(data=rates_df, x='round', y='deception_event', hue='model', style='condition', palette='viridis', lw=2.5)
    sns.lineplot(data=rates_df, x='round', y='truthful_signal', hue='model', style='condition', palette='plasma', lw=2.5, linestyle='--')

    plt.title("Nuanced Reporting Strategy Over Time (Athey & Bagwell)", fontsize=16)
    plt.xlabel("Game Round", fontsize=12)
    plt.ylabel("Proportion of 'Low' Reports", fontsize=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='black', lw=2.5, label='Deception'))
    handles.append(Line2D([0], [0], color='black', lw=2.5, linestyle='--', label='Truthful Signal'))
    plt.legend(handles=handles, title='Model & Condition / Line Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plots_dir / "P3.3_athey_bagwell_reporting_strategy.png")
    plt.close()

def _plot_reporting_strategy_matrix(results_dir, plots_dir):
    """Plot 3.4: Reporting Strategy Matrix (Heatmap) for Athey & Bagwell."""
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("Generating Plot 3.4: Reporting Strategy Matrix (Heatmaps)...")
    
    raw_df = _get_raw_results_rq2(results_dir, 'athey_bagwell')
    if raw_df.empty:
        logger.warning("No raw data found for Athey & Bagwell to plot reporting matrix.")
        return
    
    strategy_df = raw_df.groupby(['model', 'condition', 'challenger_true_cost', 'challenger_report']).size().reset_index(name='count')
    
    total_counts = strategy_df.groupby(['model', 'condition', 'challenger_true_cost'])['count'].transform('sum')
    strategy_df['probability'] = strategy_df['count'] / total_counts
    
    for model in strategy_df['model'].unique():
        for condition in strategy_df['condition'].unique():
            plot_data = strategy_df[(strategy_df['model'] == model) & (strategy_df['condition'] == condition)]
            
            if plot_data.empty:
                continue
            
            pivot_df = plot_data.pivot_table(index='challenger_true_cost', columns='challenger_report', values='probability').fillna(0)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5, cbar=False, vmin=0, vmax=1)
            plt.title(f"Reporting Strategy: {model} ({condition})", fontsize=16)
            plt.xlabel("Reported Cost", fontsize=12)
            plt.ylabel("True Cost", fontsize=12)
            plt.tight_layout()
            plt.savefig(plots_dir / f"P3.4_{model}_{condition}_reporting_matrix.png")
            plt.close()

# --- Main Visualization Function ---

def visualize_rq2():
    """
    Generates all tables and plots for Research Question 2.
    """
    if not PLOT_LIBS_INSTALLED:
        print("Plotting libraries not installed. Skipping visualization.")
        return

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("RQ2Visualizer")
    logger.info("--- Generating visualizations for RQ2: Strategic Capability ---")
    
    script_dir = Path(__file__).parent
    analysis_dir = script_dir
    results_dir = script_dir.parent / "results"

    plots_dir = analysis_dir / "plots" / "rq2"
    tables_dir = analysis_dir / "tables" / "rq2"
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    try:
        magic_df = pd.read_csv(analysis_dir / "magic_behavioral_metrics.csv")

        # --- Generate Tables ---
        _create_rq2_tables(magic_df, tables_dir)
        _create_overall_magic_table(magic_df, tables_dir)
        
        # --- Generate Plots ---
        _plot_overall_magic_profile(magic_df, plots_dir)
        _plot_game_specific_adaptation(magic_df, plots_dir)
        _plot_collusion_stability(results_dir, plots_dir)
        _plot_nuanced_reporting_strategy(results_dir, plots_dir)
        _plot_reporting_strategy_matrix(results_dir, plots_dir)
        
        logger.info("--- Finished RQ2 visualizations ---")
        
    except FileNotFoundError as e:
        logger.error(f"Failed to find necessary file for RQ2 visualizations: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during RQ2 visualization: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_rq2()