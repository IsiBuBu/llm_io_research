import os
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from config import (
    ExperimentConfig, GameConfig, 
    get_available_models, get_experiment_presets, get_game_configs,
    get_api_config, get_logging_config, validate_config,
    ExperimentPresets, GameConfigs
)
from competition import GameCompetition, setup_competition_logging

class ExperimentRunner:
    """Enhanced experiment runner with JSON configuration and comprehensive debugging"""
    
    def __init__(self, gemini_api_key: str = None, debug: bool = True, 
                 config_file: str = "config.json"):
        
        # Setup logging first
        setup_competition_logging()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        if not validate_config():
            raise ValueError("Invalid configuration detected. Check config.json file.")
        
        # Setup API key
        api_config = get_api_config()
        api_key_env = api_config.get('gemini_api_key_env', 'GEMINI_API_KEY')
        
        if gemini_api_key:
            os.environ[api_key_env] = gemini_api_key
        elif not os.getenv(api_key_env):
            raise ValueError(f"Gemini API key required. Set {api_key_env} environment variable or pass gemini_api_key parameter")
        
        # Initialize competition system
        self.competition = GameCompetition(debug=debug)
        self.debug = debug
        
        # Load available models and presets
        self.available_models = get_available_models()
        self.experiment_presets = get_experiment_presets()
        
        # Session statistics
        self.session_start_time = time.time()
        self.experiments_run = 0
        self.total_games_run = 0
        self.session_id = f"session_{int(self.session_start_time)}"
        
        self.logger.info(f"=== EXPERIMENT RUNNER INITIALIZED ===")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Available models: {len(self.available_models)}")
        self.logger.info(f"Available presets: {list(self.experiment_presets.keys())}")
        self.logger.info(f"API key configured: {'✓' if os.getenv(api_key_env) else '✗'}")
    
    def run_your_main_experiment(self, output_dir: str = "your_main_results") -> Dict[str, Any]:
        """Run your main experimental setup - Gemini 2.0 Flash Lite defender vs all challengers"""
        
        self.logger.info("=== RUNNING YOUR MAIN EXPERIMENT ===")
        
        # Load your main setup from config.json
        experiment_config = self.experiment_presets['your_main_setup']
        
        self.logger.info(f"Defender: {experiment_config.defender_model_key}")
        self.logger.info(f"Challengers: {len(experiment_config.challenger_model_keys)} models")
        self.logger.info(f"Games per setup: {experiment_config.num_games}")
        
        # Run the experiment
        results = self.competition.run_experiment(experiment_config)
        
        # Export results
        self._export_experiment_results(results, output_dir, "main_experiment")
        
        self.experiments_run += 1
        self.total_games_run += experiment_config.num_games
        
        return results
    
    def run_thinking_comparison(self, output_dir: str = "thinking_results") -> Dict[str, Any]:
        """Run thinking capability comparison experiment"""
        
        self.logger.info("=== RUNNING THINKING COMPARISON EXPERIMENT ===")
        
        experiment_config = self.experiment_presets['thinking_comparison']
        
        # Log which models have thinking capabilities
        for challenger_key in experiment_config.challenger_model_keys:
            model = self.available_models[challenger_key]
            thinking_status = "WITH THINKING" if model.thinking_enabled else "NO THINKING"
            self.logger.info(f"  {model.display_name}: {thinking_status}")
        
        results = self.competition.run_experiment(experiment_config)
        
        # Additional analysis for thinking comparison
        self._analyze_thinking_performance(results)
        
        self._export_experiment_results(results, output_dir, "thinking_comparison")
        
        self.experiments_run += 1
        self.total_games_run += experiment_config.num_games
        
        return results
    
    def run_custom_experiment(self, defender_key: str, challenger_keys: List[str],
                            game_name: str = "custom", num_games: int = 10,
                            output_dir: str = "custom_results") -> Dict[str, Any]:
        """Run a custom experiment with specified models"""
        
        self.logger.info(f"=== RUNNING CUSTOM EXPERIMENT: {game_name} ===")
        
        # Validate model keys
        all_keys = [defender_key] + challenger_keys
        for key in all_keys:
            if key not in self.available_models:
                raise ValueError(f"Unknown model key: {key}. Available: {list(self.available_models.keys())}")
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            defender_model_key=defender_key,
            challenger_model_keys=challenger_keys,
            game_name=game_name,
            num_games=num_games,
            include_thinking=True
        )
        
        self.logger.info(f"Defender: {self.available_models[defender_key].display_name}")
        for challenger_key in challenger_keys:
            self.logger.info(f"Challenger: {self.available_models[challenger_key].display_name}")
        
        results = self.competition.run_experiment(experiment_config)
        self._export_experiment_results(results, output_dir, f"custom_{game_name}")
        
        self.experiments_run += 1
        self.total_games_run += experiment_config.num_games
        
        return results
    
    def run_thesis_experiments(self, output_dir: str = "thesis_results") -> Dict[str, Any]:
        """Run complete thesis experiments with all game types"""
        
        self.logger.info("=== RUNNING COMPLETE THESIS EXPERIMENTS ===")
        
        # Load game configurations from JSON
        static_games_configs = get_game_configs('static_games')
        dynamic_games_configs = get_game_configs('dynamic_games')
        
        all_results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'static_experiments': {},
            'dynamic_experiments': {},
            'thesis_summary': {}
        }
        
        # Run static game experiments (Salop, Spulber)
        self.logger.info("Running static game experiments...")
        for game_name, configs in static_games_configs.items():
            self.logger.info(f"Testing {game_name} with {len(configs)} configurations")
            
            for i, game_config in enumerate(configs):
                # Use your main experiment setup
                experiment_config = self.experiment_presets['your_main_setup']
                
                # Update with specific game settings
                updated_experiment = ExperimentConfig(
                    defender_model_key=experiment_config.defender_model_key,
                    challenger_model_keys=experiment_config.challenger_model_keys,
                    game_name=game_name,
                    num_players=game_config.number_of_players,
                    num_rounds=game_config.number_of_rounds,
                    num_games=game_config.num_games,
                    include_thinking=experiment_config.include_thinking
                )
                
                experiment_key = f"{game_name}_players_{game_config.number_of_players}_config_{i}"
                self.logger.info(f"Running {experiment_key}")
                
                try:
                    result = self.competition.run_experiment(updated_experiment)
                    all_results['static_experiments'][experiment_key] = result
                    self.total_games_run += game_config.num_games
                    
                except Exception as e:
                    self.logger.error(f"Static experiment {experiment_key} failed: {e}")
                    all_results['static_experiments'][experiment_key] = {
                        'error': str(e),
                        'experiment_config': asdict(updated_experiment)
                    }
        
        # Run dynamic game experiments (Green Porter, Athey Bagwell)
        self.logger.info("Running dynamic game experiments...")
        for game_name, configs in dynamic_games_configs.items():
            self.logger.info(f"Testing {game_name} with {len(configs)} configurations")
            
            for i, game_config in enumerate(configs):
                # Use thinking comparison for dynamic games (more relevant)
                experiment_config = self.experiment_presets['thinking_comparison']
                
                updated_experiment = ExperimentConfig(
                    defender_model_key=experiment_config.defender_model_key,
                    challenger_model_keys=experiment_config.challenger_model_keys,
                    game_name=game_name,
                    num_players=game_config.number_of_players,
                    num_rounds=game_config.number_of_rounds,
                    num_games=game_config.num_games,
                    include_thinking=experiment_config.include_thinking
                )
                
                experiment_key = f"{game_name}_rounds_{game_config.number_of_rounds}_config_{i}"
                self.logger.info(f"Running {experiment_key}")
                
                try:
                    result = self.competition.run_experiment(updated_experiment)
                    all_results['dynamic_experiments'][experiment_key] = result
                    self.total_games_run += game_config.num_games
                    
                except Exception as e:
                    self.logger.error(f"Dynamic experiment {experiment_key} failed: {e}")
                    all_results['dynamic_experiments'][experiment_key] = {
                        'error': str(e),
                        'experiment_config': asdict(updated_experiment)
                    }
        
        # Calculate thesis summary
        thesis_duration = time.time() - self.session_start_time
        
        # Count successful vs failed experiments
        successful_static = sum(1 for result in all_results['static_experiments'].values() 
                              if 'error' not in result)
        total_static = len(all_results['static_experiments'])
        
        successful_dynamic = sum(1 for result in all_results['dynamic_experiments'].values()
                               if 'error' not in result)
        total_dynamic = len(all_results['dynamic_experiments'])
        
        all_results['thesis_summary'] = {
            'session_id': self.session_id,
            'total_duration_hours': thesis_duration / 3600,
            'total_experiments_run': self.experiments_run,
            'total_games_run': self.total_games_run,
            'static_experiments': {
                'successful': successful_static,
                'total': total_static,
                'success_rate': successful_static / total_static if total_static > 0 else 0
            },
            'dynamic_experiments': {
                'successful': successful_dynamic,
                'total': total_dynamic, 
                'success_rate': successful_dynamic / total_dynamic if total_dynamic > 0 else 0
            },
            'models_tested': len(self.available_models),
            'completion_status': 'COMPLETED' if (successful_static + successful_dynamic) > 0 else 'FAILED'
        }
        
        all_results['end_time'] = datetime.now().isoformat()
        
        # Export comprehensive results
        self._export_thesis_results(all_results, output_dir)
        
        # Log final summary
        self.logger.info("=== THESIS EXPERIMENTS COMPLETED ===")
        self.logger.info(f"Duration: {thesis_duration/3600:.2f} hours")
        self.logger.info(f"Total experiments: {self.experiments_run}")
        self.logger.info(f"Total games: {self.total_games_run}")
        self.logger.info(f"Static experiments: {successful_static}/{total_static}")
        self.logger.info(f"Dynamic experiments: {successful_dynamic}/{total_dynamic}")
        
        return all_results
    
    def run_quick_debug_test(self) -> Dict[str, Any]:
        """Run quick debug test using the debug_test preset"""
        
        self.logger.info("=== RUNNING QUICK DEBUG TEST ===")
        
        experiment_config = self.experiment_presets['debug_test']
        
        # Log what we're testing
        defender = self.available_models[experiment_config.defender_model_key]
        self.logger.info(f"Debug defender: {defender.display_name}")
        
        for challenger_key in experiment_config.challenger_model_keys:
            challenger = self.available_models[challenger_key]
            self.logger.info(f"Debug challenger: {challenger.display_name}")
        
        results = self.competition.run_experiment(experiment_config)
        
        # Enhanced debug analysis
        self._debug_experiment_results(results)
        
        self.experiments_run += 1
        self.total_games_run += experiment_config.num_games
        
        return results
    
    def test_all_models_individually(self, test_prompt: str = None) -> Dict[str, Any]:
        """Test all available models individually for debugging"""
        
        if test_prompt is None:
            test_prompt = "You are in a 3-player economic game. Market demand is Q = 100 - P. Your marginal cost is 10. What quantity should you produce to maximize profit? Show your reasoning."
        
        self.logger.info("=== TESTING ALL MODELS INDIVIDUALLY ===")
        self.logger.info(f"Test prompt: {test_prompt[:100]}...")
        
        test_results = {
            'test_prompt': test_prompt,
            'start_time': datetime.now().isoformat(),
            'model_results': {},
            'summary': {}
        }
        
        successful_models = []
        failed_models = []
        
        for model_key, model_config in self.available_models.items():
            self.logger.info(f"Testing {model_config.display_name}...")
            
            try:
                from competition import GeminiLLMAgent
                agent = GeminiLLMAgent(model_config, debug=self.debug)
                response = agent.generate_response(test_prompt, include_thinking=True)
                
                test_results['model_results'][model_key] = {
                    'model_name': model_config.model_name,
                    'display_name': model_config.display_name,
                    'thinking_available': model_config.thinking_available,
                    'thinking_enabled': model_config.thinking_enabled,
                    'final_response': response.final_response,
                    'thinking_output': response.thinking_output,
                    'response_time': response.response_time,
                    'token_usage': response.token_usage,
                    'error': response.error,
                    'success': response.error is None
                }
                
                if response.error:
                    failed_models.append((model_key, response.error))
                    self.logger.warning(f"  FAILED: {response.error}")
                else:
                    successful_models.append(model_key)
                    self.logger.info(f"  SUCCESS: {response.response_time:.2f}s, {len(response.final_response)} chars")
                    if response.thinking_output:
                        self.logger.debug(f"  Thinking: {len(response.thinking_output)} chars")
                
            except Exception as e:
                error_msg = f"Exception testing {model_key}: {str(e)}"
                self.logger.error(error_msg)
                failed_models.append((model_key, error_msg))
                
                test_results['model_results'][model_key] = {
                    'model_name': model_config.model_name,
                    'display_name': model_config.display_name, 
                    'error': error_msg,
                    'success': False
                }
        
        # Create summary
        test_results['summary'] = {
            'total_models': len(self.available_models),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / len(self.available_models),
            'successful_model_keys': successful_models,
            'failed_model_details': failed_models
        }
        
        test_results['end_time'] = datetime.now().isoformat()
        
        # Log summary
        self.logger.info(f"=== MODEL TESTING SUMMARY ===")
        self.logger.info(f"Successful: {len(successful_models)}/{len(self.available_models)}")
        
        if failed_models:
            self.logger.warning("Failed models:")
            for model_key, error in failed_models:
                self.logger.warning(f"  {model_key}: {error}")
        
        # Export individual test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"individual_model_tests_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            self.logger.info(f"Individual test results exported to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to export individual test results: {e}")
        
        return test_results
    
    def run_preset_experiment(self, preset_key: str, output_dir: str = None) -> Dict[str, Any]:
        """Run an experiment using a predefined preset from config.json"""
        
        if preset_key not in self.experiment_presets:
            raise ValueError(f"Unknown preset: {preset_key}. Available: {list(self.experiment_presets.keys())}")
        
        experiment_config = self.experiment_presets[preset_key]
        
        self.logger.info(f"=== RUNNING PRESET EXPERIMENT: {preset_key} ===")
        self.logger.info(f"Game: {experiment_config.game_name}")
        self.logger.info(f"Defender: {experiment_config.defender_model_key}")
        self.logger.info(f"Challengers: {experiment_config.challenger_model_keys}")
        
        results = self.competition.run_experiment(experiment_config)
        
        # Export results
        if output_dir is None:
            output_dir = f"preset_{preset_key}_results"
        
        self._export_experiment_results(results, output_dir, f"preset_{preset_key}")
        
        self.experiments_run += 1
        self.total_games_run += experiment_config.num_games
        
        return results
    
    def run_all_presets(self, output_dir: str = "all_presets_results") -> Dict[str, Any]:
        """Run all available experiment presets"""
        
        self.logger.info("=== RUNNING ALL EXPERIMENT PRESETS ===")
        
        all_preset_results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'preset_results': {},
            'overall_summary': {}
        }
        
        for preset_key in self.experiment_presets.keys():
            self.logger.info(f"Running preset: {preset_key}")
            
            try:
                result = self.run_preset_experiment(preset_key, f"{output_dir}/{preset_key}")
                all_preset_results['preset_results'][preset_key] = result
                
            except Exception as e:
                error_msg = f"Preset {preset_key} failed: {str(e)}"
                self.logger.error(error_msg)
                all_preset_results['preset_results'][preset_key] = {'error': error_msg}
        
        # Calculate overall summary
        successful_presets = sum(1 for result in all_preset_results['preset_results'].values() 
                               if 'error' not in result)
        
        all_preset_results['overall_summary'] = {
            'total_presets': len(self.experiment_presets),
            'successful_presets': successful_presets,
            'total_experiments_run': self.experiments_run,
            'total_games_run': self.total_games_run,
            'session_duration': time.time() - self.session_start_time
        }
        
        all_preset_results['end_time'] = datetime.now().isoformat()
        
        # Export combined results
        self._export_thesis_results(all_preset_results, f"{output_dir}/combined")
        
        return all_preset_results
    
    def _analyze_thinking_performance(self, results: Dict[str, Any]):
        """Analyze performance differences between thinking and non-thinking models"""
        
        self.logger.info("=== THINKING PERFORMANCE ANALYSIS ===")
        
        thinking_models = {}
        no_thinking_models = {}
        
        if 'game_results' in results:
            for game_result in results['game_results']:
                if 'error' in game_result:
                    continue
                
                for player in game_result.get('players', []):
                    player_id = player.get('player_id', '')
                    
                    # Determine if this was a thinking or non-thinking model
                    # This would need to be tracked during game execution
                    if 'thinking' in player_id.lower():
                        if player_id not in thinking_models:
                            thinking_models[player_id] = []
                        thinking_models[player_id].append(player.get('profit', 0))
                    else:
                        if player_id not in no_thinking_models:
                            no_thinking_models[player_id] = []
                        no_thinking_models[player_id].append(player.get('profit', 0))
        
        # Log analysis results
        if thinking_models:
            self.logger.info("Thinking models performance:")
            for model, profits in thinking_models.items():
                avg_profit = sum(profits) / len(profits) if profits else 0
                self.logger.info(f"  {model}: {avg_profit:.2f} avg profit ({len(profits)} games)")
        
        if no_thinking_models:
            self.logger.info("Non-thinking models performance:")
            for model, profits in no_thinking_models.items():
                avg_profit = sum(profits) / len(profits) if profits else 0
                self.logger.info(f"  {model}: {avg_profit:.2f} avg profit ({len(profits)} games)")
    
    def _debug_experiment_results(self, results: Dict[str, Any]):
        """Enhanced debugging analysis of experiment results"""
        
        self.logger.debug("=== DETAILED EXPERIMENT DEBUG ===")
        
        if 'summary_metrics' in results:
            metrics = results['summary_metrics']
            self.logger.debug(f"Experiment duration: {metrics.get('total_duration', 0):.2f}s")
            self.logger.debug(f"API success rate: {metrics.get('api_success_rate', 0):.2%}")
            self.logger.debug(f"Total tokens used: {metrics.get('total_tokens_used', 0):,}")
            self.logger.debug(f"Games per minute: {metrics.get('games_per_minute', 0):.2f}")
        
        # Analyze response patterns
        if 'game_results' in results:
            response_lengths = []
            thinking_lengths = []
            response_times = []
            
            for game_result in results['game_results']:
                if 'additional_metrics' in game_result:
                    avg_time = game_result['additional_metrics'].get('average_response_time', 0)
                    if avg_time > 0:
                        response_times.append(avg_time)
            
            if response_times:
                self.logger.debug(f"Average response time: {sum(response_times)/len(response_times):.2f}s")
                self.logger.debug(f"Response time range: {min(response_times):.2f}s - {max(response_times):.2f}s")
    
    def _export_experiment_results(self, results: Dict[str, Any], output_dir: str, 
                                 experiment_name: str):
        """Export experiment results with enhanced formatting"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export detailed JSON results
            json_file = output_path / f"{experiment_name}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Detailed results exported to {json_file}")
            
            # Export summary CSV for easy analysis
            if 'game_results' in results and results['game_results']:
                summary_data = []
                
                for i, game_result in enumerate(results['game_results']):
                    if 'error' in game_result:
                        continue
                    
                    base_row = {
                        'game_index': i,
                        'game_name': game_result.get('game_name', ''),
                        'total_industry_profit': game_result.get('total_industry_profit', 0),
                        'market_price': game_result.get('market_price', 0)
                    }
                    
                    # Add metrics if available
                    if 'additional_metrics' in game_result:
                        base_row.update(game_result['additional_metrics'])
                    
                    # Add player results
                    for player in game_result.get('players', []):
                        row = base_row.copy()
                        row.update({
                            'player_id': player.get('player_id', ''),
                            'player_profit': player.get('profit', 0),
                            'player_win': player.get('win', False)
                        })
                        summary_data.append(row)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    csv_file = output_path / f"{experiment_name}_summary_{timestamp}.csv"
                    summary_df.to_csv(csv_file, index=False)
                    self.logger.info(f"Summary CSV exported to {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results for {experiment_name}: {e}")
    
    def _export_thesis_results(self, results: Dict[str, Any], output_dir: str):
        """Export comprehensive thesis results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export main thesis results
            thesis_file = output_path / f"thesis_experiments_{timestamp}.json"
            with open(thesis_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Export session summary
            summary_file = output_path / f"thesis_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Thesis Experiments Summary\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Session ID: {self.session_id}\n\n")
                
                if 'thesis_summary' in results:
                    summary = results['thesis_summary']
                    f.write(f"Total Duration: {summary.get('total_duration_hours', 0):.2f} hours\n")
                    f.write(f"Total Experiments: {summary.get('total_experiments_run', 0)}\n")
                    f.write(f"Total Games: {summary.get('total_games_run', 0)}\n")
                    f.write(f"Models Tested: {summary.get('models_tested', 0)}\n")
                    f.write(f"Status: {summary.get('completion_status', 'UNKNOWN')}\n\n")
                    
                    f.write("Static Game Results:\n")
                    static = summary.get('static_experiments', {})
                    f.write(f"  Success Rate: {static.get('success_rate', 0):.1%}\n")
                    f.write(f"  Successful: {static.get('successful', 0)}/{static.get('total', 0)}\n\n")
                    
                    f.write("Dynamic Game Results:\n")
                    dynamic = summary.get('dynamic_experiments', {})
                    f.write(f"  Success Rate: {dynamic.get('success_rate', 0):.1%}\n")
                    f.write(f"  Successful: {dynamic.get('successful', 0)}/{dynamic.get('total', 0)}\n")
            
            self.logger.info(f"Thesis results exported to {output_path}")
            self.logger.info(f"Main file: {thesis_file.name}")
            self.logger.info(f"Summary: {summary_file.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to export thesis results: {e}")
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get current session statistics"""
        
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_id': self.session_id,
            'session_duration': session_duration,
            'experiments_run': self.experiments_run,
            'total_games_run': self.total_games_run,
            'available_models': len(self.available_models),
            'available_presets': len(self.experiment_presets),
            'experiments_per_hour': self.experiments_run / (session_duration / 3600) if session_duration > 0 else 0
        }
    
    def list_available_configurations(self):
        """List all available configurations for easy reference"""
        
        self.logger.info("=== AVAILABLE CONFIGURATIONS ===")
        
        self.logger.info("Models:")
        for model_key, model_config in self.available_models.items():
            thinking_status = "Thinking ON" if model_config.thinking_enabled else "No Thinking" if model_config.thinking_available else "No Thinking Available"
            self.logger.info(f"  {model_key}: {model_config.display_name} ({thinking_status})")
        
        self.logger.info("\nExperiment Presets:")
        for preset_key, preset_config in self.experiment_presets.items():
            self.logger.info(f"  {preset_key}: {preset_config.game_name} (defender: {preset_config.defender_model_key}, {len(preset_config.challenger_model_keys)} challengers)")
        
        self.logger.info("\nGame Configurations:")
        for category in ['static_games', 'dynamic_games', 'quick_test_games']:
            configs = get_game_configs(category)
            self.logger.info(f"  {category}: {list(configs.keys())}")

# Convenience functions for easy usage
def quick_experiment_run(preset_key: str = "debug_test") -> Dict[str, Any]:
    """Quick way to run an experiment"""
    runner = ExperimentRunner(debug=True)
    return runner.run_preset_experiment(preset_key)

def debug_single_model(model_key: str, prompt: str = None) -> Dict[str, Any]:
    """Debug a single model with detailed logging"""
    if prompt is None:
        prompt = "Explain Nash equilibrium in a 2x2 game theory scenario."
    
    runner = ExperimentRunner(debug=True)
    
    if model_key not in runner.available_models:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(runner.available_models.keys())}")
    
    model_config = runner.available_models[model_key]
    
    from competition import GeminiLLMAgent
    agent = GeminiLLMAgent(model_config, debug=True)
    response = agent.generate_response(prompt, include_thinking=True)
    
    return {
        'model_key': model_key,
        'model_config': asdict(model_config),
        'prompt': prompt,
        'response': asdict(response),
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    """Example usage and testing"""
    
    try:
        # Initialize runner
        runner = ExperimentRunner(debug=True)
        
        # Show available configurations
        runner.list_available_configurations()
        
        # Test all models individually first
        print("\n" + "="*50)
        print("TESTING ALL MODELS INDIVIDUALLY")
        print("="*50)
        individual_results = runner.test_all_models_individually()
        
        # Run quick debug test
        print("\n" + "="*50) 
        print("RUNNING QUICK DEBUG TEST")
        print("="*50)
        debug_results = runner.run_quick_debug_test()
        
        # Run your main experiment
        print("\n" + "="*50)
        print("RUNNING YOUR MAIN EXPERIMENT") 
        print("="*50)
        main_results = runner.run_your_main_experiment()
        
        # Session summary
        stats = runner.get_session_statistics()
        print(f"\nSession completed: {stats['experiments_run']} experiments, {stats['total_games_run']} games in {stats['session_duration']:.1f}s")
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Runner execution failed: {e}")
        logging.getLogger(__name__).error(f"Traceback: {traceback.format_exc()}")