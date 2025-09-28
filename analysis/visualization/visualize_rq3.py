# analysis/visualization/visualize_rq3.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_LIBS_INSTALLED = True
except ImportError:
    PLOT_LIBS_INSTALLED = False

# For advanced predictive modeling (SHAP plots)
try:
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor
    ADVANCED_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_LIBS_INSTALLED = False

from config.config import get_analysis_dir, get_experiments_dir


def _create_master_correlation_table(corr_df: pd.DataFrame, tables_dir: Path):
    """Saves the complete, unfiltered correlation results to a CSV file for the appendix."""
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("Creating Appendix Table A.1: Master Correlation Table...")
    
    master_table = corr_df[['game_name', 'magic_metric', 'performance_metric', 'correlation_coefficient', 'p_value', 'n_samples']]
    master_table.columns = ['Game', 'MAgIC Metric', 'Performance Metric', 'r-value', 'p-value', 'n']

    master_table_path = tables_dir / "A.1_master_correlation_table.csv"
    master_table.to_csv(master_table_path, index=False)
    logger.info(f"Saved master correlation table to {master_table_path}")


def _plot_correlation_heatmaps(corr_df: pd.DataFrame, plots_dir: Path):
    """Plot 3.1: Creates and saves a heatmap of the correlation results for each game."""
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("Generating Plot 3.1: Correlation Heatmaps...")
    for game in corr_df['game_name'].unique():
        game_df = corr_df[corr_df['game_name'] == game]
        
        pivot_df = game_df.pivot_table(
            index='magic_metric', 
            columns='performance_metric', 
            values='correlation_coefficient'
        )
        
        if pivot_df.empty:
            continue

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_df, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            linewidths=.5,
            fmt=".2f",
            vmin=-1, vmax=1
        )
        
        plt.title(f"{game.replace('_', ' ').title()}: MAgIC vs. Performance Correlation", fontsize=16)
        plt.xlabel("Performance Metrics", fontsize=12)
        plt.ylabel("MAgIC Behavioral Metrics", fontsize=12)
        plt.tight_layout()

        plot_filename = plots_dir / f"P3.1_{game}_correlation_heatmap.png"
        plt.savefig(plot_filename)
        plt.close()


def _plot_significant_scatter(corr_df: pd.DataFrame, perf_df: pd.DataFrame, magic_df: pd.DataFrame, plots_dir: Path):
    """Plot 3.2: Creates scatter plots for statistically significant correlations."""
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("Generating Plot 3.2: Scatter Plots for Significant Correlations...")
    significant_corrs = corr_df[corr_df['p_value'] < 0.05]
    
    perf_pivot = perf_df.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean')
    magic_pivot = magic_df.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean')
    merged_metrics = pd.merge(perf_pivot, magic_pivot, on=['game', 'model', 'condition']).reset_index()
    
    for _, row in significant_corrs.iterrows():
        game = row['game_name']
        magic_metric = row['magic_metric']
        perf_metric = row['performance_metric']
        
        plot_data = merged_metrics[merged_metrics['game'] == game].copy()
        
        if plot_data.empty or magic_metric not in plot_data.columns or perf_metric not in plot_data.columns:
            continue

        plot_data['Player Count'] = plot_data['condition'].apply(lambda x: '5-Player' if '5' in x else '3-Player')

        plt.figure(figsize=(11, 7))
        sns.scatterplot(data=plot_data, x=magic_metric, y=perf_metric, hue='Player Count', style='model', s=150, palette='viridis')
        sns.regplot(data=plot_data, x=magic_metric, y=perf_metric, scatter=False, color='red', line_kws={'linestyle':'--'})
        
        plt.title(f"Significant Correlation in {game.replace('_', ' ').title()}\n{magic_metric.replace('_', ' ').title()} vs. {perf_metric.replace('_', ' ').title()}", fontsize=16)
        plt.xlabel(f"{magic_metric.replace('_', ' ').title()} Score", fontsize=12)
        plt.ylabel(f"{perf_metric.replace('_', ' ').title()} Score", fontsize=12)
        plt.legend(title='Condition & Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plot_filename = plots_dir / f"P3.2_{game}_{magic_metric}_vs_{perf_metric}_scatter.png"
        plt.savefig(plot_filename)
        plt.close()

def _plot_composite_performance_summary(perf_df: pd.DataFrame, magic_df: pd.DataFrame, plots_dir: Path):
    """Generates the MAgIC paper-style composite radar and bar/line charts for both win rate and profit."""
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("Generating MAgIC-style composite performance plots for Win Rate and Profit...")

    # 1. Aggregate Data
    core_metrics = ['judgment', 'reasoning', 'deception', 'self_awareness', 'cooperation', 'coordination', 'rationality']
    magic_agg = magic_df[magic_df['metric'].isin(core_metrics)].groupby(['model', 'metric'])['mean'].mean().reset_index()
    magic_pivot = magic_agg.pivot_table(index='model', columns='metric', values='mean').reindex(columns=core_metrics)

    win_rate_agg = perf_df[perf_df['metric'] == 'win_rate'].groupby('model')['mean'].mean().reset_index()
    avg_profit_agg = perf_df[perf_df['metric'] == 'average_profit'].groupby('model')['mean'].mean().reset_index()

    # 2. Calculate Polygon Area
    n = len(core_metrics)
    areas = {model: 0.5 * np.sin(2 * np.pi / n) * sum(row.values[i] * row.values[(i + 1) % n] for i in range(n)) for model, row in magic_pivot.iterrows()}
    area_df = pd.DataFrame(list(areas.items()), columns=['model', 'area'])

    # 3. Generate Radar Chart (Plot A)
    labels = [m.replace('_', ' ').title() for m in core_metrics]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for model in magic_pivot.index:
        values = magic_pivot.loc[model].tolist() + [magic_pivot.loc[model].tolist()[0]]
        ax.plot(angles, values, label=model, linewidth=2)
        ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    plt.title("Overall MAgIC Capability Profile", size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(plots_dir / "P3.0a_composite_radar_chart.png")
    plt.close()

    # 4. Generate Bar/Line Chart for Win Rate (Plot B)
    plot_df_win = pd.merge(area_df, win_rate_agg, on='model').sort_values('area', ascending=False)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df_win, x='model', y='area', ax=ax1, color='skyblue')
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Radar Chart Area (Capability Score)", color='skyblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2 = ax1.twinx()
    sns.lineplot(data=plot_df_win, x='model', y='mean', ax=ax2, color='red', marker='o', sort=False, lw=2.5)
    ax2.set_ylabel("Average Win Rate", color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(bottom=0)
    plt.title("Capability Score vs. Win Rate", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(plots_dir / "P3.0b_area_vs_winrate_chart.png")
    plt.close()

    # 5. Generate Bar/Line Chart for Average Profit (Plot C)
    plot_df_profit = pd.merge(area_df, avg_profit_agg, on='model').sort_values('area', ascending=False)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df_profit, x='model', y='area', ax=ax1, color='lightgreen')
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Radar Chart Area (Capability Score)", color='green', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax2 = ax1.twinx()
    sns.lineplot(data=plot_df_profit, x='model', y='mean', ax=ax2, color='purple', marker='s', sort=False, lw=2.5)
    ax2.set_ylabel("Average Profit / NPV", color='purple', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='purple')
    plt.title("Capability Score vs. Average Profit/NPV", fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(plots_dir / "P3.0c_area_vs_profit_chart.png")
    plt.close()

def _plot_shap_summary(perf_df: pd.DataFrame, magic_df: pd.DataFrame, plots_dir: Path):
    """Plot 3.3: Trains predictive models and generates SHAP summary plots."""
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("Generating Plot 3.3: SHAP Summary Plots for Predictive Modeling...")
    if not ADVANCED_LIBS_INSTALLED:
        logger.warning("Advanced libraries (shap, scikit-learn) not found. Skipping SHAP plot generation.")
        logger.warning("Please run: pip install shap scikit-learn")
        return
    
    magic_pivot = magic_df.pivot_table(index=['model', 'condition', 'game'], columns='metric', values='mean').reset_index()
    
    # Model 1: Predicting Average Profit
    profit_df = perf_df[perf_df['metric'] == 'average_profit']
    df_profit = pd.merge(magic_pivot, profit_df[['model', 'condition', 'game', 'mean']], on=['model', 'condition', 'game']).rename(columns={'mean': 'average_profit'})
    
    X_profit = df_profit.drop(columns=['average_profit', 'model', 'condition', 'game']).fillna(0)
    y_profit = df_profit['average_profit']

    if len(X_profit) > 1:
        model_profit = GradientBoostingRegressor(random_state=42)
        model_profit.fit(X_profit, y_profit)
        explainer_profit = shap.Explainer(model_profit, X_profit)
        shap_values_profit = explainer_profit(X_profit)

        plt.figure()
        shap.summary_plot(shap_values_profit, X_profit, show=False, plot_type="bar")
        plt.title("SHAP Summary for Predicting Average Profit/NPV", fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / "P3.3_shap_summary_avg_profit.png")
        plt.close()

    # Model 2: Predicting Win Rate
    win_df = perf_df[perf_df['metric'] == 'win_rate']
    df_win = pd.merge(magic_pivot, win_df[['model', 'condition', 'game', 'mean']], on=['model', 'condition', 'game']).rename(columns={'mean': 'win_rate'})
    
    X_win = df_win.drop(columns=['win_rate', 'model', 'condition', 'game']).fillna(0)
    y_win = (df_win['win_rate'] > df_win['win_rate'].median()).astype(int)

    if len(X_win) > 1 and len(y_win.unique()) > 1:
        model_win = LogisticRegression(random_state=42, max_iter=1000)
        model_win.fit(X_win, y_win)
        explainer_win = shap.Explainer(model_win, X_win)
        shap_values_win = explainer_win(X_win)

        plt.figure()
        shap.summary_plot(shap_values_win, X_win, show=False, plot_type="bar")
        plt.title("SHAP Summary for Predicting High Win Rate", fontsize=16)
        plt.tight_layout()
        plt.savefig(plots_dir / "P3.3_shap_summary_win_rate.png")
        plt.close()

# --- Main Visualization Function ---

def visualize_rq3():
    """
    Generates all tables and plots for Research Question 3.
    """
    if not PLOT_LIBS_INSTALLED:
        print("Plotting libraries not installed. Skipping visualization.")
        return
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("--- Generating visualizations for RQ3: Correlations & Predictive Models ---")
    
    analysis_dir = get_analysis_dir()
    results_dir = get_experiments_dir()

    plots_dir = analysis_dir / "plots" / "rq3"
    tables_dir = analysis_dir / "tables" / "rq3"
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    try:
        corr_df = pd.read_csv(analysis_dir / "correlations_analysis_structural.csv")
        perf_df = pd.read_csv(analysis_dir / "performance_metrics.csv")
        magic_df = pd.read_csv(analysis_dir / "magic_behavioral_metrics.csv")

        # --- Generate MAgIC Paper-style Composite Plots ---
        _plot_composite_performance_summary(perf_df, magic_df, plots_dir)

        # --- Part 1: Foundational Correlation Analysis ---
        _create_master_correlation_table(corr_df, tables_dir)
        _plot_correlation_heatmaps(corr_df, plots_dir)
        _plot_significant_scatter(corr_df, perf_df, magic_df, plots_dir)

        # --- Part 2: Advanced Predictive Modeling ---
        _plot_shap_summary(perf_df, magic_df, plots_dir)
        
        logger.info("--- Finished RQ3 visualizations ---")

    except FileNotFoundError as e:
        logger.error(f"Failed to find necessary file for RQ3 visualizations: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during RQ3 visualization: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_rq3()