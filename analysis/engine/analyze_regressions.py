# analysis/engine/analyze_regressions.py

import pandas as pd
import numpy as np
import logging
import sys
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Imports for SHAP and Plotting ---
try:
    import shap
    import matplotlib.pyplot as plt
    ADVANCED_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_LIBS_INSTALLED = False

# --- Imports for the Alternative Beta Regression Method ---
try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    PYMC_INSTALLED = True
except ImportError:
    PYMC_INSTALLED = False

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import load_config

class RegressionAnalyzer:
    """
    Performs regression analysis to identify significant predictors of performance.
    - Uses Gradient Boosting for continuous outcomes (profit/NPV, HHI).
    - Uses Beta Regression via PyMC for proportional outcomes (win rate, market share, etc.).
    - Generates SHAP summary plots for Gradient Boosting models.
    - Saves all regression results to CSV files.
    """

    def __init__(self, analysis_dir: str = "output/analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.plots_dir = self.analysis_dir / "plots" / "regression_analysis"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.importance_results = []
        self.beta_summary_results = []

    def prepare_data(self) -> pd.DataFrame:
        """
        Loads, merges, and engineers features for the regression analysis.
        """
        self.logger.info("Preparing data for regression analysis...")
        try:
            perf_df = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
            magic_df = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
            config = load_config()
        except FileNotFoundError:
            self.logger.error("Summary CSV files not found. Please run the analysis pipeline first.")
            return pd.DataFrame()

        models_to_include = list(config.get('model_configs', {}).keys())
        perf_df = perf_df[perf_df['model'].isin(models_to_include)]
        magic_df = magic_df[magic_df['model'].isin(models_to_include)]

        perf_pivot = perf_df.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean').reset_index()
        magic_pivot = magic_df.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean').reset_index()
        merged_df = pd.merge(perf_pivot, magic_pivot, on=['game', 'model', 'condition'], how='left')

        # Feature Engineering
        merged_df['is_gen_2_5'] = merged_df['model'].apply(lambda x: 1 if '2_5' in x else 0)
        merged_df['is_lite'] = merged_df['model'].apply(lambda x: 1 if 'lite' in x else 0)
        merged_df['is_5_player'] = merged_df['condition'].apply(lambda x: 1 if 'more_players' in x else 0)
        
        merged_df['thinking'] = 'off'
        merged_df.loc[merged_df['model'].str.contains('low', na=False), 'thinking'] = 'low'
        merged_df.loc[merged_df['model'].str.contains('medium', na=False), 'thinking'] = 'medium'
        
        thinking_dummies = pd.get_dummies(merged_df['thinking'], prefix='thinking', drop_first=True).astype(int)
        merged_df = pd.concat([merged_df, thinking_dummies], axis=1)

        return merged_df

    def run_all_regressions(self):
        """
        Runs regression models for each game and each performance metric.
        """
        df = self.prepare_data()
        if df.empty: return

        for game in df['game'].unique():
            self.logger.info("-" * 80)
            self.logger.info(f"Running Regression Analysis for Game: {game.upper()}")
            game_df = df[df['game'] == game].dropna(axis=1, how='all')
            
            self.analyze_performance(game, game_df.copy(), 'average_profit', 'Average Profit/NPV')
            self.analyze_performance(game, game_df.copy(), 'win_rate', 'Win Rate')
            self.analyze_game_specific_metrics(game, game_df.copy())
            
        self.print_results()
        self.save_results_to_csv()

    def analyze_performance(self, game: str, game_df: pd.DataFrame, target_metric: str, target_name: str):
        """Dispatcher for running the correct regression model."""
        potential_predictors = [
            'is_gen_2_5', 'is_lite', 'is_5_player', 'thinking_low', 'thinking_medium',
            'rationality', 'self_awareness', 'cooperation', 'coordination', 
            'judgment', 'deception', 'reasoning'
        ]
        predictors = [p for p in potential_predictors if p in game_df.columns]
        
        X = game_df[predictors].fillna(0)
        y = game_df[target_metric]
        
        if y.isnull().all() or len(y) < 4:
            self.logger.warning(f"Target variable '{target_metric}' has insufficient data for {game}. Skipping.")
            return
            
        if target_metric in ['win_rate', 'market_share', 'market_capture_rate', 'reversion_frequency']:
             self.analyze_proportion_pymc(game, X, y, target_metric)
        else:
             self.analyze_continuous_sklearn(game, X, y, target_metric)


    def analyze_continuous_sklearn(self, game: str, X: pd.DataFrame, y: pd.Series, target_metric: str):
        """Analyzes a continuous target using Gradient Boosting and generates SHAP plot."""
        self.logger.info(f"\n--- Predicting {target_metric.replace('_', ' ').title()} for {game.upper()} ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
        importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        
        importances['game'] = game
        importances['target_metric'] = target_metric
        self.importance_results.append(importances)
        self.logger.info(f"Feature Importances:\n{importances.head()}")

        if X_test.empty:
            self.logger.warning(f"Test set is empty for {target_metric} in {game}. Skipping R-squared and SHAP plot.")
            self.results.setdefault(game, {})[target_metric] = {"r2": np.nan, "importances": importances}
            return

        r2 = r2_score(y_test, model.predict(X_test))
        self.results.setdefault(game, {})[target_metric] = {"r2": r2, "importances": importances}
        self.logger.info(f"Gradient Boosting R-squared: {r2:.4f}")

        if ADVANCED_LIBS_INSTALLED:
            try:
                self.logger.info("Generating SHAP summary plot...")
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)
                
                shap.summary_plot(shap_values, X_test, show=False)
                
                plt.title(f"SHAP Summary for Predicting {target_metric.replace('_', ' ').title()} in {game.title()}")
                plot_path = self.plots_dir / f"{game}_{target_metric}_shap_summary.png"
                
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Successfully saved SHAP plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"!!! FAILED to create SHAP plot for {target_metric} in {game} !!!")
                self.logger.error(f"Error details: {e}")

    def analyze_proportion_pymc(self, game: str, X: pd.DataFrame, y: pd.Series, target_metric: str):
        """Analyzes a proportional target using Beta Regression with PyMC."""
        self.logger.info(f"\n--- Predicting {target_metric.replace('_', ' ').title()} for {game.upper()} ---")
        if not PYMC_INSTALLED:
            self.logger.error("PyMC is not installed. Skipping Beta Regression. Please run 'pip install pymc arviz'.")
            return

        n = len(y)
        y_squeezed = (y * (n - 1) + 0.5) / n

        if y_squeezed.isnull().all() or y_squeezed.nunique() <= 1:
            self.logger.warning(f"Target variable '{target_metric}' has no variance or is all NaN. Skipping.")
            return

        with pm.Model() as beta_model:
            intercept = pm.Normal("intercept", mu=0, sigma=10)
            betas = pm.Normal("betas", mu=0, sigma=10, shape=X.shape[1])
            kappa = pm.HalfCauchy("kappa", beta=10)
            mu = pm.invlogit(intercept + pt.dot(X.values, betas))
            alpha = mu * kappa
            beta = (1 - mu) * kappa
            y_obs = pm.Beta("y_obs", alpha=alpha, beta=beta, observed=y_squeezed)
            
            # FIX: Increase max_treedepth to allow the sampler more time per step.
            trace = pm.sample(
                2000,
                tune=2000,
                chains=6,
                cores=min(6, os.cpu_count()),
                target_accept=0.99, # Keep this high for stability
                progressbar=False,
                random_seed=42,
                max_treedepth=15 # Increase from default of 10
            )

        posterior_pred = pm.sample_posterior_predictive(trace, model=beta_model, random_seed=42)
        y_pred = posterior_pred.posterior_predictive["y_obs"].mean(("chain", "draw"))
        pseudo_r2 = r2_score(y_squeezed, y_pred)
        
        summary = az.summary(trace, var_names=["intercept", "betas", "kappa"])
        summary.index = ['intercept'] + list(X.columns) + ['kappa']

        self.results.setdefault(game, {})[target_metric] = {"pseudo_r2": pseudo_r2, "summary": summary}
        
        summary['game'] = game
        summary['target_metric'] = target_metric
        self.beta_summary_results.append(summary)

        self.logger.info(f"PyMC Beta Regression (Pseudo) R-squared: {pseudo_r2:.4f}")
        self.logger.info(f"Model Summary:\n{summary}")

    def analyze_game_specific_metrics(self, game: str, game_df: pd.DataFrame):
        """Runs regressions for game-specific performance metrics."""
        game_specific_metrics = {
            'salop': 'market_share',
            'spulber': 'market_capture_rate',
            'green_porter': 'reversion_frequency',
            'athey_bagwell': 'hhi'
        }
        target_metric = game_specific_metrics.get(game)
        if target_metric and target_metric in game_df.columns:
            self.analyze_performance(game, game_df, target_metric, target_metric.replace('_', ' ').title())

    def print_results(self):
        """Prints a final summary of the R-squared values."""
        print("\n" + "="*80)
        print("REGRESSION ANALYSIS SUMMARY")
        print("="*80)
        for game, metrics in sorted(self.results.items()):
            for metric_type, value in sorted(metrics.items()):
                r2_type = "Pseudo R-squared" if "pseudo_r2" in value else "R-squared"
                r2_value = value.get('r2') or value.get('pseudo_r2')

                game_str = game.upper()
                metric_str = metric_type.replace('_', ' ').title()
                
                if pd.notna(r2_value):
                    print(f"Game: {game_str:<20} | Target: {metric_str:<25} | {r2_type}: {r2_value:.4f}")
                else:
                    print(f"Game: {game_str:<20} | Target: {metric_str:<25} | {r2_type}: Not calculated (small dataset)")
        print("="*80)

    def save_results_to_csv(self):
        """Saves the collected regression results to CSV files."""
        self.logger.info("Saving regression analysis results to CSV files...")
        
        if self.importance_results:
            importance_df = pd.concat(self.importance_results)
            importance_path = self.analysis_dir / "regression_feature_importances.csv"
            importance_df.to_csv(importance_path, index=False)
            self.logger.info(f"Saved feature importances to {importance_path}")

        if self.beta_summary_results:
            beta_summary_df = pd.concat(self.beta_summary_results)
            beta_summary_path = self.analysis_dir / "regression_beta_summary.csv"
            beta_summary_df.to_csv(beta_summary_path)
            self.logger.info(f"Saved Beta regression summaries to {beta_summary_path}")

if __name__ == '__main__':
    # Set PyMC logging to a higher level to reduce verbose output
    logging.getLogger('pymc').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s')
    analyzer = RegressionAnalyzer()
    analyzer.run_all_regressions()