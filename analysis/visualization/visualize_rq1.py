# analysis/visualization/visualize_rq1.py

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

from config.config import get_analysis_dir, get_experiments_dir

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

def _get_raw_results(results_dir: Path, game_name: str) -> pd.DataFrame:
    """Loads and concatenates all raw JSON results for a specific game."""
    records = []
    for file_path in results_dir.glob(f"{game_name}/*/*_competition_result*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for sim in data.get('simulation_results', []):
                # Handle dynamic games with round data
                if 'rounds' in sim['game_data']:
                    trigger_price = sim['game_data'].get('constants', {}).get('trigger_price')
                    for i, round_data in enumerate(sim['game_data']['rounds']):
                        for player, action in round_data.get('actions', {}).items():
                            player_type = 'Challenger' if player == 'challenger' else 'Defender'
                            records.append({
                                'game': sim['game_name'],
                                'model': sim['challenger_model'],
                                'player_id': player,
                                'player_type': player_type,
                                'condition': sim['condition_name'],
                                'simulation_id': sim['simulation_id'],
                                'round': i + 1,
                                'action': action,
                                'profit': round_data.get('payoffs', {}).get(player),
                                'market_share': round_data.get('game_outcomes', {}).get('player_market_shares', {}).get(player),
                                'market_state': round_data.get('market_state'),
                                'market_price': round_data.get('market_price'),
                                'trigger_price': trigger_price
                            })
                # Handle static games
                else:
                    for player, action in sim['actions'].items():
                        player_type = 'Challenger' if player == 'challenger' else 'Defender'
                        records.append({
                            'game': sim['game_name'],
                            'model': sim['challenger_model'],
                            'player_id': player,
                            'player_type': player_type,
                            'condition': sim['condition_name'],
                            'simulation_id': sim['simulation_id'],
                            'round': 1,
                            'action': action,
                            'profit': sim['payoffs'].get(player)
                        })
    return pd.DataFrame(records)


# --- Table Generation ---

def _create_rq1_tables(perf_df, tables_dir):
    """Creates and saves summary tables for RQ1 performance metrics."""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Creating RQ1 summary tables (Tables 1.1-1.4)...")

    agg_df = perf_df.groupby(['game', 'model', 'condition', 'metric'])['mean'].agg(['mean', get_ci]).reset_index()

    for game in agg_df['game'].unique():
        game_df = agg_df[agg_df['game'] == game].copy()
        
        game_df['formatted_metric'] = game_df.apply(
            lambda row: format_with_ci(row['mean'], row['get_ci'], precision=2 if 'volatility' not in row['metric'] else 1), axis=1
        )
        
        pivot = game_df.pivot_table(
            index='model',
            columns=['condition', 'metric'],
            values='formatted_metric',
            aggfunc='first'
        ).sort_index(axis=1)

        table_path = tables_dir / f"T1_{game}_performance_summary.csv"
        pivot.to_csv(table_path)
        logger.info(f"Saved table: {table_path}")

# --- Plot Generation ---

def _plot_performance_heatmap(perf_df, plots_dir):
    """Plot 1.1: Performance Across Games (Heatmap)"""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Generating Plot 1.1: Performance Across Games (Heatmap)...")

    main_condition_df = perf_df[
        (perf_df['condition'].str.contains('3_players')) | 
        (perf_df['condition'].str.contains('few_players')) |
        (perf_df['condition'].str.contains('short_horizon'))
    ].copy()

    for metric in ['win_rate', 'average_profit']:
        metric_df = main_condition_df[main_condition_df['metric'] == metric].copy()
        
        metric_df['normalized_value'] = metric_df.groupby('game')['mean'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
        )

        pivot_df = metric_df.pivot_table(index='model', columns='game', values='normalized_value')

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title(f"Normalized {metric.replace('_', ' ').title()} Across All Games (3-Player/Short-Horizon)", fontsize=16)
        plt.xlabel("Game", fontsize=12)
        plt.ylabel("Challenger Model", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / f"P1.1_heatmap_{metric}.png")
        plt.close()

def _plot_action_distributions(results_dir, plots_dir):
    """Plot 2.1: Action Strategy vs. Economic Benchmarks for Salop, Spulber and Green & Porter."""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Generating Plot 2.1: Action Distribution (Violin Plots)...")

    for game in ['salop', 'spulber', 'green_porter']:
        raw_df = _get_raw_results(results_dir, game)
        
        if game == 'green_porter':
            action_col = 'quantity'
            raw_df[action_col] = raw_df['action'].apply(lambda x: x.get('quantity') if isinstance(x, dict) else None)
        else:
            action_col = 'price'
            raw_df[action_col] = raw_df['action'].apply(lambda x: x.get('price') if isinstance(x, dict) else x.get('bid') if isinstance(x, dict) else None)
        
        raw_df.dropna(subset=[action_col], inplace=True)
        challenger_df = raw_df[raw_df['player_type'] == 'Challenger']
        
        for condition in challenger_df['condition'].unique():
            plt.figure(figsize=(14, 8))
            plot_data = challenger_df[challenger_df['condition'] == condition]
            sns.violinplot(data=plot_data, x='model', y=action_col, palette='muted', inner='quartile')
            
            if game == 'green_porter':
                plt.axhline(y=17, color='g', linestyle='--', label='Collusive Quantity (17)')
                plt.axhline(y=25, color='r', linestyle='--', label='Cournot Quantity (25)')

            plt.title(f"{game.title()}: Action Strategy Distribution ({condition})", fontsize=16)
            plt.ylabel(f"{action_col.title()} Chosen", fontsize=12)
            plt.xlabel("Challenger Model", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.6)
            if game == 'green_porter':
                plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"P2.1_{game}_{condition}_{action_col}_violin.png")
            plt.close()

def _plot_risk_reward_profiles(perf_df, plots_dir):
    """Plot 2.2: Risk vs. Reward Profile for static games."""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Generating Plot 2.2: Risk vs. Reward Profiles (Scatter Plots)...")

    for game in ['salop', 'spulber']:
        game_df = perf_df[perf_df['game'] == game]
        pivot_df = game_df.pivot_table(
            index=['model', 'condition'],
            columns='metric',
            values='mean'
        ).reset_index()

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=pivot_df, x='profit_volatility', y='average_profit', hue='model', style='condition', s=200, palette='tab10')

        for model in pivot_df['model'].unique():
            model_data = pivot_df[pivot_df['model'] == model].sort_values('condition')
            if len(model_data) == 2:
                p1 = model_data.iloc[0]
                p2 = model_data.iloc[1]
                plt.arrow(p1['profit_volatility'], p1['average_profit'], p2['profit_volatility'] - p1['profit_volatility'], p2['average_profit'] - p1['average_profit'],
                          head_width=0.5, head_length=0.5, fc='gray', ec='gray', length_includes_head=True, linestyle='--')

        plt.title(f"{game.title()}: Risk vs. Reward Profile", fontsize=16)
        plt.xlabel("Profit Volatility (Risk)", fontsize=12)
        plt.ylabel("Average Profit (Reward)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Model & Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(plots_dir / f"P2.2_{game}_risk_reward.png")
        plt.close()
        
def _plot_cumulative_profit(results_dir, plots_dir):
    """Plot 3.1: Cumulative Profit Over Time for dynamic games."""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Generating Plot 3.1: Cumulative Profit Over Time (Line Plots)...")

    for game in ['green_porter', 'athey_bagwell']:
        raw_df = _get_raw_results(results_dir, game)
        challenger_df = raw_df[raw_df['player_type'] == 'Challenger'].copy()
        
        # --- FIXED LOGIC: Ensure profit is numeric before calculations ---
        challenger_df['profit'] = pd.to_numeric(challenger_df['profit'], errors='coerce')
        challenger_df.dropna(subset=['profit'], inplace=True)

        challenger_df['discounted_profit'] = challenger_df['profit'] * (0.95 ** (challenger_df['round'] - 1))
        
        challenger_df['cumulative_npv'] = challenger_df.groupby(['model', 'condition', 'simulation_id'])['discounted_profit'].cumsum()
        
        avg_cumulative_df = challenger_df.groupby(['model', 'condition', 'round'])['cumulative_npv'].mean().reset_index()

        for condition in avg_cumulative_df['condition'].unique():
            plt.figure(figsize=(14, 8))
            plot_data = avg_cumulative_df[avg_cumulative_df['condition'] == condition]
            sns.lineplot(data=plot_data, x='round', y='cumulative_npv', hue='model', palette='tab10', lw=2.5)
            
            plt.title(f"{game.replace('_', ' ').title()}: Cumulative NPV ({condition})", fontsize=16)
            plt.xlabel("Game Round", fontsize=12)
            plt.ylabel("Average Cumulative NPV", fontsize=12)
            plt.legend(title='Model')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(plots_dir / f"P3.1_{game}_{condition}_cumulative_profit.png")
            plt.close()

def _plot_challenger_vs_defender_profit(results_dir, plots_dir):
    """Plot 3.2: Challenger's Profit Trajectory vs. Defenders."""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Generating Plot 3.2: Challenger vs. Defender Profit (Line Plots)...")
    
    for game in ['green_porter', 'athey_bagwell']:
        raw_df = _get_raw_results(results_dir, game)
        
        avg_profit_df = raw_df.groupby(['model', 'condition', 'round', 'player_type'])['profit'].mean().unstack().reset_index()
        
        for model in avg_profit_df['model'].unique():
            for condition in avg_profit_df['condition'].unique():
                plot_data = avg_profit_df[(avg_profit_df['model'] == model) & (avg_profit_df['condition'] == condition)]
                
                if plot_data.empty or 'Challenger' not in plot_data.columns or 'Defender' not in plot_data.columns:
                    continue

                plt.figure(figsize=(14, 8))
                plt.plot(plot_data['round'], plot_data['Challenger'], label='Challenger', color='blue', lw=2.5)
                plt.plot(plot_data['round'], plot_data['Defender'], label='Avg. Defender', color='orange', lw=2.5)
                plt.fill_between(plot_data['round'], plot_data['Challenger'], plot_data['Defender'], 
                                 where=plot_data['Challenger'] >= plot_data['Defender'], 
                                 facecolor='green', alpha=0.3, interpolate=True, label='Challenger Advantage')
                plt.fill_between(plot_data['round'], plot_data['Challenger'], plot_data['Defender'], 
                                 where=plot_data['Challenger'] < plot_data['Defender'], 
                                 facecolor='red', alpha=0.3, interpolate=True, label='Defender Advantage')

                plt.title(f"{game.replace('_', ' ').title()}: {model} vs. Defenders ({condition})", fontsize=16)
                plt.xlabel("Game Round", fontsize=12)
                plt.ylabel("Average Profit per Round", fontsize=12)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.axhline(0, color='black', linewidth=0.5)
                plt.tight_layout()
                plt.savefig(plots_dir / f"P3.2_{game}_{model}_{condition}_profit_trajectory.png")
                plt.close()

def _plot_price_war_scatter(results_dir, plots_dir):
    """Plot 3.3: Price War Severity and Duration for Green & Porter."""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Generating Plot 3.3: Price War Severity and Duration (Scatter Plot)...")
    
    raw_df = _get_raw_results(results_dir, 'green_porter')
    
    wars = []
    for (model, condition, sim_id), group in raw_df.groupby(['model', 'condition', 'simulation_id']):
        in_war = False
        current_war = []
        for _, row in group.sort_values('round').iterrows():
            if row['market_state'] == 'Reversionary' and not in_war:
                in_war = True
                current_war.append(row)
            elif row['market_state'] == 'Reversionary' and in_war:
                current_war.append(row)
            elif row['market_state'] == 'Collusive' and in_war:
                in_war = False
                if current_war:
                    wars.append(pd.DataFrame(current_war))
                current_war = []
        if current_war:
            wars.append(pd.DataFrame(current_war))

    war_data = []
    for war_df in wars:
        trigger_price = war_df['trigger_price'].iloc[0]
        war_data.append({
            'model': war_df['model'].iloc[0],
            'condition': war_df['condition'].iloc[0],
            'duration': len(war_df),
            'severity': (trigger_price - war_df['market_price'].mean())
        })
    
    if war_data:
        war_df = pd.DataFrame(war_data)
        war_df['condition_type'] = war_df['condition'].apply(lambda x: '5-Player' if '5' in x else '3-Player')
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=war_df, x='duration', y='severity', hue='model', style='condition_type', s=150, palette='tab10')
        plt.title("Price War Severity vs. Duration", fontsize=16)
        plt.xlabel("Price War Duration (Rounds)", fontsize=12)
        plt.ylabel("Price War Severity (Avg. Price Drop)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Model & Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(plots_dir / "P3.3_green_porter_price_wars.png")
        plt.close()

def _plot_market_share_dominance(results_dir, plots_dir):
    """Plot 3.4: Market Share Dominance for Athey & Bagwell."""
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("Generating Plot 3.4: Market Share Dominance (Stacked Area Charts)...")
    
    raw_df = _get_raw_results(results_dir, 'athey_bagwell')
    
    for model in raw_df['model'].unique():
        for condition in raw_df['condition'].unique():
            plot_data = raw_df[(raw_df['model'] == model) & (raw_df['condition'] == condition)]
            
            if plot_data.empty:
                continue

            pivot = plot_data.groupby(['round', 'player_id'])['market_share'].mean().unstack().fillna(0)
            
            if pivot.empty:
                continue

            plt.figure(figsize=(14, 8))
            pivot.plot.area(stacked=True, colormap='viridis', ax=plt.gca())
            
            plt.title(f"Market Share Dominance: {model} ({condition})", fontsize=16)
            plt.xlabel("Game Round", fontsize=12)
            plt.ylabel("Market Share", fontsize=12)
            plt.ylim(0, 1)
            plt.legend(title='Player', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(plots_dir / f"P3.4_athey_bagwell_{model}_{condition}_market_share.png")
            plt.close()

# --- Main Visualization Function ---

def visualize_rq1():
    """
    Generates all tables and plots for Research Question 1.
    """
    if not PLOT_LIBS_INSTALLED:
        print("Plotting libraries not installed. Skipping visualization.")
        return
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("RQ1Visualizer")
    logger.info("--- Generating visualizations for RQ1: Economic Performance ---")
    
    analysis_dir = get_analysis_dir()
    results_dir = get_experiments_dir()

    plots_dir = analysis_dir / "plots" / "rq1"
    tables_dir = analysis_dir / "tables" / "rq1"
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    try:
        perf_df = pd.read_csv(analysis_dir / "performance_metrics.csv")

        # --- Generate Tables ---
        _create_rq1_tables(perf_df, tables_dir)
        
        # --- Generate Plots ---
        _plot_performance_heatmap(perf_df, plots_dir)
        _plot_action_distributions(results_dir, plots_dir)
        _plot_risk_reward_profiles(perf_df, plots_dir)
        _plot_cumulative_profit(results_dir, plots_dir)
        _plot_challenger_vs_defender_profit(results_dir, plots_dir)
        _plot_price_war_scatter(results_dir, plots_dir) 
        _plot_market_share_dominance(results_dir, plots_dir)

        logger.info("--- Finished RQ1 visualizations ---")

    except FileNotFoundError as e:
        logger.error(f"Failed to find necessary file for RQ1 visualizations: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during RQ1 visualization: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_rq1()