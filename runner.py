# runner.py - Compact Version with Comprehensive Metrics
"""
Essential experiment runner with comprehensive behavioral metrics integration.
Provides streamlined access to LLM game theory experiments and behavioral analysis.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import (
    ExperimentConfig, GameConfig, 
    get_available_models, get_experiment_presets,
    ExperimentPresets, GameConfigs, validate_config
)
from competition import GameCompetition

class ExperimentRunner:
    """Compact experiment runner with comprehensive behavioral metrics"""
    
    def __init__(self, gemini_api_key: str = None, debug: bool = False):
        """Initialize experiment runner"""
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        if not validate_config():
            raise ValueError("Invalid configuration. Check config.json file.")
        
        # Setup API key
        if gemini_api_key:
            os.environ['GEMINI_API_KEY'] = gemini_api_key
        elif not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY required")
        
        # Initialize competition system
        self.competition = GameCompetition(debug=debug)
        self.debug = debug
        
        # Load configurations
        self.available_models = get_available_models()
        self.experiment_presets = get_experiment_presets()
        
        # Session tracking
        self.session_start_time = time.time()
        self.experiments_run = 0
        self.total_games_run = 0
        
        self.logger.info(f"ExperimentRunner initialized - {len(self.available_models)} models, {len(self.experiment_presets)} presets")

    def run_preset_experiment(self, preset_key: str, games: List[str] = None, 
                            output_dir: str = "results") -> Dict[str, Any]:
        """Run experiment using predefined preset with comprehensive metrics"""
        
        if preset_key not in self.experiment_presets:
            available = list(self.experiment_presets.keys())
            raise ValueError(f"Unknown preset: {preset_key}. Available: {available}")
        
        experiment_config = self.experiment_presets[preset_key]
        
        # Default to all games if not specified
        if not games:
            games = ['salop', 'spulber', 'green_porter', 'athey_bagwell']
        
        self.logger.info(f"=== RUNNING PRESET EXPERIMENT: {preset_key} ===")
        self.logger.info(f"Games: {games}")
        self.logger.info(f"Defender: {experiment_config.defender_model_key}")
        self.logger.info(f"Challengers: {experiment_config.challenger_model_keys}")
        
        all_results = {
            'preset_key': preset_key,
            'experiment_config': experiment_config.__dict__,
            'games_run': {},
            'comprehensive_analysis': {},
            'session_summary': {},
            'start_time': datetime.now().isoformat()
        }
        
        # Run each game
        for game_name in games:
            if game_name not in self.competition.games:
                self.logger.warning(f"Skipping unknown game: {game_name}")
                continue
            
            self.logger.info(f"\n--- Running {game_name.upper()} ---")
            
            # Create game-specific experiment config
            game_experiment_config = ExperimentConfig(
                defender_model_key=experiment_config.defender_model_key,
                challenger_model_keys=experiment_config.challenger_model_keys,
                game_name=game_name,
                num_players=experiment_config.num_players,
                num_rounds=experiment_config.num_rounds,
                num_games=experiment_config.num_games,
                include_thinking=experiment_config.include_thinking
            )
            
            try:
                # Run experiment with comprehensive metrics
                game_results = self.competition.run_experiment(game_experiment_config)
                all_results['games_run'][game_name] = game_results
                
                # Log key results
                self._log_game_results(game_name, game_results)
                
                self.experiments_run += 1
                self.total_games_run += len(game_results.get('game_results', []))
                
            except Exception as e:
                self.logger.error(f"Failed to run {game_name}: {e}")
                all_results['games_run'][game_name] = {'error': str(e)}
        
        # Generate comprehensive analysis
        all_results['comprehensive_analysis'] = self._analyze_all_results(all_results)
        all_results['session_summary'] = self._generate_session_summary(all_results)
        all_results['end_time'] = datetime.now().isoformat()
        
        # Export results
        self._export_results(all_results, output_dir, f"{preset_key}_{int(time.time())}")
        
        # Display summary
        self._display_summary(all_results)
        
        return all_results

    def run_single_game(self, game_name: str, defender_key: str, challenger_keys: List[str],
                       num_players: int = 2, num_rounds: int = 1, num_games: int = 1) -> Dict[str, Any]:
        """Run single game experiment with comprehensive metrics"""
        
        experiment_config = ExperimentConfig(
            defender_model_key=defender_key,
            challenger_model_keys=challenger_keys,
            game_name=game_name,
            num_players=num_players,
            num_rounds=num_rounds,
            num_games=num_games,
            include_thinking=False
        )
        
        self.logger.info(f"Running single {game_name} experiment")
        results = self.competition.run_experiment(experiment_config)
        
        # Log and display key metrics
        self._log_game_results(game_name, results)
        self._display_behavioral_insights(results.get('comprehensive_metrics', {}))
        
        return results

    def run_debug_test(self) -> Dict[str, Any]:
        """Quick debug test with comprehensive metrics"""
        self.logger.info("=== RUNNING DEBUG TEST ===")
        return self.run_preset_experiment('debug_test', games=['salop'])

    def run_main_experiment(self) -> Dict[str, Any]:
        """Run main experiment if available"""
        if 'main_experiment' in self.experiment_presets:
            return self.run_preset_experiment('main_experiment')
        else:
            self.logger.warning("main_experiment preset not found, running debug_test")
            return self.run_debug_test()

    def _log_game_results(self, game_name: str, results: Dict[str, Any]):
        """Log key results from game experiment"""
        
        summary = results.get('summary_metrics', {})
        exp_overview = summary.get('experiment_overview', {})
        
        self.logger.info(f"{game_name.upper()} Results:")
        self.logger.info(f"  ✓ Success Rate: {exp_overview.get('success_rate', 0):.1%}")
        self.logger.info(f"  ⏱️  Duration: {exp_overview.get('total_duration', 0):.1f}s")
        self.logger.info(f"  🎮 Games: {exp_overview.get('successful_games', 0)}")
        
        # Display behavioral insights
        behavioral = summary.get('behavioral_insights', {})
        key_findings = behavioral.get('key_findings', [])
        if key_findings:
            self.logger.info("  🧠 Key Behavioral Findings:")
            for finding in key_findings[:2]:  # Top 2 findings
                self.logger.info(f"    • {finding}")

    def _analyze_all_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results across all games"""
        
        analysis = {
            'cross_game_patterns': {},
            'model_performance_summary': {},
            'behavioral_consistency': {},
            'strategic_insights': []
        }
        
        # Extract metrics from all games
        all_metrics = {}
        model_performance = {}
        
        for game_name, game_result in all_results['games_run'].items():
            if 'error' in game_result:
                continue
            
            # Extract comprehensive metrics
            comp_metrics = game_result.get('comprehensive_metrics', {})
            if comp_metrics:
                all_metrics[game_name] = comp_metrics
            
            # Extract model performance
            model_perf = game_result.get('summary_metrics', {}).get('model_performance', {})
            for model_key, stats in model_perf.items():
                if model_key not in model_performance:
                    model_performance[model_key] = {}
                model_performance[model_key][game_name] = stats
        
        # Analyze cross-game patterns
        if all_metrics:
            analysis['cross_game_patterns'] = self._analyze_cross_game_patterns(all_metrics)
        
        # Summarize model performance across games
        analysis['model_performance_summary'] = self._summarize_model_performance(model_performance)
        
        # Generate insights
        analysis['strategic_insights'] = self._generate_strategic_insights(all_metrics, model_performance)
        
        return analysis

    def _analyze_cross_game_patterns(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral patterns across different games"""
        
        patterns = {
            'cooperation_by_game': {},
            'rationality_by_game': {},
            'consistency_analysis': {}
        }
        
        # Extract key behavioral metrics by game
        for game_name, game_metrics in all_metrics.items():
            aggregate = game_metrics.get('aggregate_analysis', {})
            behavioral_avg = aggregate.get('behavioral_averages', {})
            magic_metrics = behavioral_avg.get('magic_behavioral', {})
            
            if magic_metrics:
                if 'cooperation' in magic_metrics:
                    patterns['cooperation_by_game'][game_name] = magic_metrics['cooperation']['mean']
                if 'rationality' in magic_metrics:
                    patterns['rationality_by_game'][game_name] = magic_metrics['rationality']['mean']
        
        return patterns

    def _summarize_model_performance(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize model performance across all games"""
        
        summary = {}
        
        for model_key, game_stats in model_performance.items():
            total_games = sum(stats.get('total_games', 0) for stats in game_stats.values())
            total_wins = sum(stats.get('wins', 0) for stats in game_stats.values())
            avg_profit = sum(stats.get('avg_profit', 0) for stats in game_stats.values()) / len(game_stats) if game_stats else 0
            
            summary[model_key] = {
                'overall_win_rate': total_wins / total_games if total_games > 0 else 0,
                'average_profit': avg_profit,
                'games_played': len(game_stats),
                'total_games': total_games
            }
        
        return summary

    def _generate_strategic_insights(self, all_metrics: Dict[str, Any], 
                                   model_performance: Dict[str, Any]) -> List[str]:
        """Generate strategic insights from cross-game analysis"""
        
        insights = []
        
        # Analyze cooperation patterns
        coop_games = []
        for game_name, metrics in all_metrics.items():
            aggregate = metrics.get('aggregate_analysis', {})
            behavioral = aggregate.get('behavioral_averages', {}).get('magic_behavioral', {})
            if 'cooperation' in behavioral:
                coop_level = behavioral['cooperation']['mean']
                if coop_level > 0.6:
                    coop_games.append(game_name)
        
        if coop_games:
            insights.append(f"High cooperation observed in: {', '.join(coop_games)}")
        
        # Analyze model consistency
        model_count = len(model_performance)
        if model_count > 1:
            insights.append(f"Analyzed {model_count} different models across multiple games")
        
        # Add game-specific insights
        if 'spulber' in all_metrics:
            insights.append("Bertrand competition analysis includes rationality and judgment metrics")
        if 'green_porter' in all_metrics:
            insights.append("Collusion game analysis includes cooperation and coordination metrics")
        if 'athey_bagwell' in all_metrics:
            insights.append("Information collusion analysis includes deception detection")
        
        return insights

    def _generate_session_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate session summary statistics"""
        
        session_duration = time.time() - self.session_start_time
        successful_games = len([g for g in all_results['games_run'].values() if 'error' not in g])
        total_games = len(all_results['games_run'])
        
        return {
            'session_duration': session_duration,
            'experiments_run': self.experiments_run,
            'total_games_run': self.total_games_run,
            'successful_games': successful_games,
            'total_games': total_games,
            'success_rate': successful_games / total_games if total_games > 0 else 0,
            'comprehensive_metrics_enabled': True
        }

    def _display_summary(self, all_results: Dict[str, Any]):
        """Display comprehensive experiment summary"""
        
        print("\n" + "="*60)
        print("🧠 COMPREHENSIVE BEHAVIORAL ANALYSIS SUMMARY")
        print("="*60)
        
        # Session overview
        summary = all_results.get('session_summary', {})
        print(f"⏱️  Session Duration: {summary.get('session_duration', 0):.1f}s")
        print(f"🎮 Games Run: {summary.get('successful_games', 0)}/{summary.get('total_games', 0)}")
        print(f"✅ Success Rate: {summary.get('success_rate', 0):.1%}")
        
        # Cross-game insights
        analysis = all_results.get('comprehensive_analysis', {})
        insights = analysis.get('strategic_insights', [])
        if insights:
            print(f"\n🧠 Strategic Insights:")
            for insight in insights:
                print(f"   • {insight}")
        
        # Model performance summary
        model_summary = analysis.get('model_performance_summary', {})
        if model_summary:
            print(f"\n🤖 Model Performance Summary:")
            for model_key, stats in model_summary.items():
                print(f"   {model_key}: {stats['overall_win_rate']:.1%} win rate, ${stats['average_profit']:.2f} avg profit")
        
        # Cross-game patterns
        patterns = analysis.get('cross_game_patterns', {})
        coop_by_game = patterns.get('cooperation_by_game', {})
        if coop_by_game:
            print(f"\n🤝 Cooperation Levels by Game:")
            for game, coop_level in coop_by_game.items():
                print(f"   {game}: {coop_level:.2f}")
        
        print("="*60)

    def _display_behavioral_insights(self, comprehensive_metrics: Dict[str, Any]):
        """Display behavioral insights from comprehensive metrics"""
        
        if not comprehensive_metrics or 'error' in comprehensive_metrics:
            return
        
        print(f"\n🧠 Behavioral Insights:")
        
        # Display aggregate behavioral metrics
        aggregate = comprehensive_metrics.get('aggregate_analysis', {})
        behavioral_avg = aggregate.get('behavioral_averages', {})
        
        magic_metrics = behavioral_avg.get('magic_behavioral', {})
        if magic_metrics:
            print(f"   MAgIC Behavioral Metrics:")
            for metric_name, stats in magic_metrics.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"     {metric_name.title()}: {stats['mean']:.3f}")

    def _export_results(self, results: Dict[str, Any], output_dir: str, filename_prefix: str):
        """Export results to JSON file"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export main results
        results_file = output_path / f"{filename_prefix}_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"📄 Results exported to {results_file}")
            
            # Export summary CSV if possible
            self._export_summary_csv(results, output_path, filename_prefix)
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")

    def _export_summary_csv(self, results: Dict[str, Any], output_path: Path, filename_prefix: str):
        """Export summary to CSV format"""
        
        try:
            import pandas as pd
            
            # Create summary data
            summary_data = []
            
            for game_name, game_result in results.get('games_run', {}).items():
                if 'error' in game_result:
                    continue
                
                summary_metrics = game_result.get('summary_metrics', {})
                exp_overview = summary_metrics.get('experiment_overview', {})
                
                summary_data.append({
                    'game': game_name,
                    'success_rate': exp_overview.get('success_rate', 0),
                    'duration': exp_overview.get('total_duration', 0),
                    'games_run': exp_overview.get('successful_games', 0),
                    'comprehensive_metrics': 'available' if game_result.get('comprehensive_metrics') else 'unavailable'
                })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_file = output_path / f"{filename_prefix}_summary.csv"
                df.to_csv(csv_file, index=False)
                self.logger.info(f"📊 Summary exported to {csv_file}")
                
        except ImportError:
            self.logger.debug("pandas not available - skipping CSV export")
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")

    def list_available_presets(self):
        """List all available experiment presets"""
        print("\n📋 Available Experiment Presets:")
        for preset_key, config in self.experiment_presets.items():
            print(f"   {preset_key}: {config.game_name} - {config.defender_model_key} vs {len(config.challenger_model_keys)} challengers")

    def list_available_models(self):
        """List all available models"""
        print("\n🤖 Available Models:")
        for model_key, config in self.available_models.items():
            thinking_status = " (Thinking)" if config.thinking_enabled else " (Thinking Available)" if config.thinking_available else ""
            print(f"   {model_key}: {config.display_name}{thinking_status}")

# Convenience functions
def quick_debug_test(api_key: str = None) -> Dict[str, Any]:
    """Quick way to run debug test"""
    runner = ExperimentRunner(gemini_api_key=api_key, debug=True)
    return runner.run_debug_test()

def run_preset(preset_key: str, api_key: str = None) -> Dict[str, Any]:
    """Quick way to run any preset"""
    runner = ExperimentRunner(gemini_api_key=api_key)
    return runner.run_preset_experiment(preset_key)

if __name__ == "__main__":
    # Quick test if run directly
    print("🧠 LLM Game Theory Experiments with Comprehensive Behavioral Metrics")
    print("Run: python runner.py or use quick_debug_test()")
    
    # Example usage
    try:
        if os.getenv('GEMINI_API_KEY'):
            results = quick_debug_test()
            print("✅ Debug test completed successfully!")
        else:
            print("⚠️  Set GEMINI_API_KEY environment variable to run tests")
    except Exception as e:
        print(f"❌ Error: {e}")