#!/usr/bin/env python3
"""
LLM Game Theory Experiment CLI Runner
Uses config.json and config.py for all configuration management
Supports two experiment types: main (full) and debug (quick)
"""

import os
import json
import logging
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import configuration system - FIXED: Only import functions that actually exist
from config import (
    ExperimentConfig, GameConfig, ModelConfig,
    get_available_models, validate_config, load_config_file
)
from competition import GameCompetition

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ExperimentCLI:
    """CLI runner that uses config.json and config.py"""
    
    def __init__(self, debug_mode: bool = False, output_base_dir: str = "results"):
        """Initialize CLI using configuration files"""
        
        # Set up logging
        log_level = logging.DEBUG if debug_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration first
        if not validate_config():
            raise ValueError("Invalid configuration. Please check config.json file.")
        
        # Set API key from config
        config = load_config_file()
        api_config = config.get('api', {})
        api_key_env = api_config.get('gemini_api_key_env', 'GEMINI_API_KEY')
        
        if not os.getenv(api_key_env):
            raise ValueError(f"API key required. Please set the {api_key_env} environment variable.")
        
        self.print_info(f"Using API key from environment variable: {api_key_env}")
        
        # Initialize competition system
        self.competition = GameCompetition(debug=debug_mode)
        
        # Load configurations from config files
        self.available_models = get_available_models()
        # REMOVED: self.experiment_presets = get_experiment_presets() - function doesn't exist
        self.config_data = config
        
        # Get available games from the competition system
        self.available_games = list(self.competition.games.keys())
        
        # Set up output directory
        self.output_dir = Path(output_base_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Session tracking
        self.experiments_run = 0
        self.total_games_run = 0
        self.session_start_time = datetime.now()
        
        self.logger.info(f"ExperimentCLI initialized with {len(self.available_models)} models, {len(self.available_games)} games")

    def print_header(self, text: str):
        """Print a colorful header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
        print(f"🧠 {text}")
        print(f"{'='*60}{Colors.END}")

    def print_info(self, text: str):
        """Print info message"""
        print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

    def print_success(self, text: str):
        """Print success message"""
        print(f"{Colors.GREEN}✅ {text}{Colors.END}")

    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

    def print_error(self, text: str):
        """Print error message"""
        print(f"{Colors.RED}❌ {text}{Colors.END}")

    def print_progress(self, current: int, total: int, text: str = ""):
        """Print progress indicator"""
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 40
        filled_length = int(bar_length * current // total) if total > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r{Colors.BLUE}📊 [{bar}] {percentage:.1f}% {text}{Colors.END}", end='', flush=True)
        if current == total:
            print()  # New line when complete

    def create_experiment_config(self, experiment_type: str, games: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create experiment configuration based on type (main or debug)"""
        
        # Get base configuration from config.json
        config = self.config_data
        
        # Get models from config
        defender_model = None
        challenger_models = []
        
        # Find a suitable defender and challengers from available models
        model_keys = list(self.available_models.keys())
        if len(model_keys) < 2:
            raise ValueError("Need at least 2 models configured for experiments")
        
        defender_model = model_keys[0]  # Use first model as defender
        challenger_models = model_keys[1:2]  # Use second model as challenger for debug, more for main
        
        # For main experiments, use more challengers if available
        if experiment_type == "main" and len(model_keys) > 2:
            challenger_models = model_keys[1:3]  # Use up to 2 challengers for main
        
        # Set number of players, rounds, and games based on experiment type
        if experiment_type == "debug":
            # Debug: Quick testing
            num_players = 2
            num_rounds = 1
            num_games = 1
        elif experiment_type == "main":
            # Main: Full testing
            num_players = 3
            num_rounds = 3
            num_games = 2
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}. Use 'main' or 'debug'")
        
        # Default to all games if not specified
        if games is None:
            games = self.available_games.copy()
        
        # Validate games
        invalid_games = [g for g in games if g not in self.available_games]
        if invalid_games:
            self.print_warning(f"Skipping invalid games: {invalid_games}")
            games = [g for g in games if g in self.available_games]
        
        if not games:
            raise ValueError("No valid games specified")
        
        return {
            'experiment_type': experiment_type,
            'defender_model': defender_model,
            'challenger_models': challenger_models,
            'games': games,
            'num_players': num_players,
            'num_rounds': num_rounds,
            'num_games': num_games,
            'include_thinking': True
        }

    def run_experiment(self, experiment_type: str, games: Optional[List[str]] = None, 
                      custom_output: Optional[str] = None) -> Dict[str, Any]:
        """Run experiment with specified type (main or debug)"""
        
        if experiment_type not in ['main', 'debug']:
            self.print_error(f"Unknown experiment type '{experiment_type}'. Use 'main' or 'debug'")
            return {}
        
        # Create experiment configuration
        try:
            experiment_config = self.create_experiment_config(experiment_type, games)
        except Exception as e:
            self.print_error(f"Failed to create experiment configuration: {e}")
            return {}
        
        self.print_header(f"{experiment_type.upper()} Experiment")
        
        print(f"{Colors.CYAN}🎮 Games: {', '.join(experiment_config['games'])}")
        print(f"🛡️  Defender: {experiment_config['defender_model']}")
        print(f"⚔️  Challengers: {', '.join(experiment_config['challenger_models'])}")
        print(f"👥 Players: {experiment_config['num_players']}")
        print(f"🔄 Rounds: {experiment_config['num_rounds']}")
        print(f"🎯 Games per experiment: {experiment_config['num_games']}{Colors.END}")
        
        # Initialize results
        all_results = {
            'experiment_type': experiment_type,
            'experiment_config': experiment_config,
            'games_run': {},
            'session_summary': {},
            'start_time': datetime.now().isoformat(),
            'games_tested': experiment_config['games'].copy()
        }
        
        # Calculate total experiments for progress tracking
        total_experiments = len(experiment_config['games']) * len(experiment_config['challenger_models'])
        current_experiment = 0
        
        # Run each game
        for i, game_name in enumerate(experiment_config['games']):
            print(f"\n{Colors.BLUE}{Colors.BOLD}--- 🎮 Running {game_name.upper()} ({i+1}/{len(experiment_config['games'])}) ---{Colors.END}")
            
            # Create ExperimentConfig object for this game
            game_experiment_config = ExperimentConfig(
                defender_model_key=experiment_config['defender_model'],
                challenger_model_keys=experiment_config['challenger_models'],
                game_name=game_name,
                num_players=experiment_config['num_players'],
                num_rounds=experiment_config['num_rounds'],
                num_games=experiment_config['num_games'],
                include_thinking=experiment_config['include_thinking']
            )
            
            try:
                # Show progress for challengers
                for j, challenger in enumerate(experiment_config['challenger_models']):
                    current_experiment += 1
                    self.print_progress(current_experiment, total_experiments, 
                                      f"Running {game_name} vs {challenger}")
                
                # Run the experiment
                experiment_result = self.competition.run_experiment(game_experiment_config)
                
                all_results['games_run'][game_name] = experiment_result
                self.experiments_run += 1
                
                # Quick summary for this game
                if 'summary_metrics' in experiment_result:
                    overview = experiment_result['summary_metrics'].get('experiment_overview', {})
                    success_rate = overview.get('success_rate', 0)
                    duration = overview.get('total_duration', 0)
                    
                    print(f"{Colors.GREEN}✅ {game_name}: {success_rate:.1%} success, {duration:.1f}s{Colors.END}")
                else:
                    print(f"{Colors.YELLOW}⚠️  {game_name}: Completed with warnings{Colors.END}")
                
                time.sleep(1)  # Brief pause between games
                
            except Exception as e:
                self.print_error(f"Failed to run {game_name}: {e}")
                all_results['games_run'][game_name] = {'error': str(e)}
        
        # Calculate session summary
        all_results['end_time'] = datetime.now().isoformat()
        all_results['session_summary'] = self._calculate_session_summary(all_results)
        
        # Export results
        self._export_results(all_results, experiment_type, custom_output)
        
        # Display final summary
        self._display_final_summary(all_results)
        
        return all_results

    def _calculate_session_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive session summary"""
        
        start_time = datetime.fromisoformat(all_results['start_time'])
        end_time = datetime.fromisoformat(all_results['end_time'])
        duration = (end_time - start_time).total_seconds()
        
        games_run = all_results['games_run']
        successful_games = sum(1 for result in games_run.values() if 'error' not in result)
        total_games = len(games_run)
        success_rate = successful_games / total_games if total_games > 0 else 0
        
        return {
            'session_duration': duration,
            'games_tested': list(games_run.keys()),
            'successful_games': successful_games,
            'total_games': total_games,
            'success_rate': success_rate,
            'total_experiments_run': self.experiments_run,
            'experiment_type': all_results['experiment_type']
        }

    def _export_results(self, results: Dict[str, Any], experiment_type: str, 
                       custom_output: Optional[str] = None):
        """Export experiment results to files"""
        
        # Determine output directory
        if custom_output:
            output_dir = Path(custom_output)
        else:
            output_dir = self.output_dir
        
        output_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"{experiment_type}_{timestamp}"
        
        # Export JSON results
        json_file = output_dir / f"{filename_prefix}_results.json"
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            self.print_success(f"Results exported to {json_file}")
        except Exception as e:
            self.print_error(f"Failed to export JSON: {e}")
        
        # Export CSV summary if pandas is available
        if PANDAS_AVAILABLE:
            self._export_csv_summary(results, output_dir, filename_prefix)

    def _export_csv_summary(self, results: Dict[str, Any], 
                           output_dir: Path, filename_prefix: str):
        """Export CSV summary"""
        
        try:
            summary_data = []
            games_run = results.get('games_run', {})
            
            for game_name, game_result in games_run.items():
                if game_result.get('error'):
                    summary_data.append({
                        'game': game_name,
                        'success_rate': 0.0,
                        'duration': 0.0,
                        'games_run': 0,
                        'comprehensive_metrics': 'error',
                        'error': game_result.get('error', 'Unknown error')
                    })
                else:
                    summary = game_result.get('summary_metrics', {})
                    exp_overview = summary.get('experiment_overview', {})
                    
                    summary_data.append({
                        'game': game_name,
                        'success_rate': exp_overview.get('success_rate', 0),
                        'duration': exp_overview.get('total_duration', 0),
                        'games_run': exp_overview.get('successful_games', 0),
                        'comprehensive_metrics': 'available' if game_result.get('comprehensive_metrics') else 'unavailable',
                        'error': None
                    })
            
            if summary_data:
                df = pd.DataFrame(summary_data)
                csv_file = output_dir / f"{filename_prefix}_summary.csv"
                df.to_csv(csv_file, index=False)
                self.print_success(f"Summary exported to {csv_file}")
                
        except Exception as e:
            self.print_error(f"Failed to export CSV: {e}")

    def _display_final_summary(self, all_results: Dict[str, Any]):
        """Display final comprehensive summary"""
        
        session_summary = all_results.get('session_summary', {})
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}")
        print("╭" + "─" * 58 + "╮")
        print("│" + " " * 12 + f"🎯 {session_summary.get('experiment_type', 'UNKNOWN').upper()} EXPERIMENT COMPLETED" + " " * 12 + "│")
        print("├" + "─" * 58 + "┤")
        print(f"│ ⏱️  Duration:     {session_summary.get('session_duration', 0):>8.1f}s" + " " * 21 + "│")
        print(f"│ 🎮 Games:        {', '.join(session_summary.get('games_tested', [])):<30}│")
        print(f"│ ✅ Success Rate: {session_summary.get('success_rate', 0):>8.1%}" + " " * 21 + "│")
        print(f"│ 🔢 Experiments:  {session_summary.get('total_experiments_run', 0):>8}" + " " * 21 + "│")
        print("╰" + "─" * 58 + "╯")
        print(f"{Colors.END}")

    # CLI Information Methods
    def list_models(self):
        """List all available models from config"""
        print(f"\n{Colors.HEADER}🤖 Available Models (from config.json):{Colors.END}")
        for model_key, model_config in self.available_models.items():
            thinking_status = ""
            if model_config.thinking_enabled:
                thinking_status = f"{Colors.GREEN} (🧠 Thinking Enabled){Colors.END}"
            elif model_config.thinking_available:
                thinking_status = f"{Colors.YELLOW} (🧠 Thinking Available){Colors.END}"
            print(f"  {Colors.CYAN}• {model_key:<20}{Colors.END} - {model_config.display_name}{thinking_status}")

    def list_games(self):
        """List all available games"""
        print(f"\n{Colors.HEADER}🎮 Available Games:{Colors.END}")
        for game in self.available_games:
            print(f"  {Colors.CYAN}• {game}{Colors.END}")

    def show_config(self):
        """Show current configuration"""
        print(f"\n{Colors.HEADER}⚙️  Current Configuration:{Colors.END}")
        
        config = self.config_data
        
        # Show experiment types
        print(f"\n{Colors.BLUE}📊 Experiment Types:{Colors.END}")
        print(f"  {Colors.CYAN}• debug{Colors.END} - Quick testing (2 players, 1 round, 1 game)")
        print(f"  {Colors.CYAN}• main{Colors.END}  - Full testing (3 players, 3 rounds, 2 games)")
        
        # Show models
        print(f"\n{Colors.BLUE}🤖 Models: {len(self.available_models)}{Colors.END}")
        for model_key in self.available_models.keys():
            print(f"  • {model_key}")
        
        # Show games
        print(f"\n{Colors.BLUE}🎮 Games: {len(self.available_games)}{Colors.END}")
        for game in self.available_games:
            print(f"  • {game}")

    def interactive_menu(self):
        """Interactive CLI menu"""
        while True:
            self.print_header("LLM Game Theory Experiment CLI")
            print(f"{Colors.BLUE}Select an option:{Colors.END}")
            print("1. Run Debug Experiment (quick: 2 players, 1 round, 1 game)")
            print("2. Run Main Experiment (full: 3 players, 3 rounds, 2 games)")
            print("3. List Available Models")
            print("4. List Available Games")
            print("5. Show Current Configuration")
            print("6. Exit")
            
            try:
                choice = input(f"\n{Colors.YELLOW}Enter your choice (1-6): {Colors.END}").strip()
                
                if choice == '1':
                    self.run_experiment('debug')
                elif choice == '2':
                    self.run_experiment('main')
                elif choice == '3':
                    self.list_models()
                elif choice == '4':
                    self.list_games()
                elif choice == '5':
                    self.show_config()
                elif choice == '6':
                    self.print_success("Goodbye!")
                    break
                else:
                    self.print_warning("Invalid choice, please try again")
                    
                if choice in ['1', '2']:
                    input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
                break
            except Exception as e:
                self.print_error(f"Error: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LLM Game Theory Experiment CLI - Uses config.json for all settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --debug                    # Run debug experiment (quick testing)
  python runner.py --main                     # Run main experiment (full testing)
  python runner.py --games salop spulber      # Run specific games only
  python runner.py --list-models              # List models from config.json
  python runner.py --show-config              # Show current configuration
  python runner.py --interactive              # Interactive menu mode
        """
    )
    
    # Experiment commands
    parser.add_argument('--debug', action='store_true',
                        help='Run debug experiment (quick: 2 players, 1 round, 1 game)')
    parser.add_argument('--main', action='store_true',
                        help='Run main experiment (full: 3 players, 3 rounds, 2 games)')
    parser.add_argument('--games', nargs='+', metavar='GAME',
                        help='Specify games to run (default: all available)')
    
    # Information commands
    parser.add_argument('--list-models', action='store_true',
                        help='List available models from config.json')
    parser.add_argument('--list-games', action='store_true',
                        help='List available games')
    parser.add_argument('--show-config', action='store_true',
                        help='Show current configuration')
    
    # Options
    parser.add_argument('--output', type=str, metavar='DIR',
                        help='Custom output directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive menu mode')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Initialize CLI
    try:
        cli = ExperimentCLI(debug_mode=args.verbose, output_base_dir=args.output or "results")
    except Exception as e:
        print(f"{Colors.RED}❌ Failed to initialize: {e}{Colors.END}")
        print(f"{Colors.YELLOW}💡 Make sure config.json exists and is valid{Colors.END}")
        return
    
    try:
        # Handle information commands
        if args.list_models:
            cli.list_models()
            return
        elif args.list_games:
            cli.list_games()
            return
        elif args.show_config:
            cli.show_config()
            return
        elif args.interactive:
            cli.interactive_menu()
            return
        
        # Handle experiment commands
        if args.debug:
            cli.run_experiment('debug', games=args.games, custom_output=args.output)
        elif args.main:
            cli.run_experiment('main', games=args.games, custom_output=args.output)
        else:
            cli.print_warning("No experiment command specified")
            parser.print_help()
            
    except KeyboardInterrupt:
        cli.print_warning("Interrupted by user")
    except Exception as e:
        cli.print_error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()