#!/usr/bin/env python3
"""
Updated LLM Game Theory Experiment Runner - Four-folder structure with dynamic metrics
Executes comprehensive experiments using the new competition API and experimental matrix
"""

import asyncio
import logging
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from config import (
    load_config_file, validate_config, get_challenger_models, get_defender_model,
    get_experiment_config, create_experiment_summary, get_model_display_name,
    is_thinking_enabled, get_model_config, get_all_game_configs, print_experiment_matrix
)
from competition import Competition

# Global flag for mock mode
MOCK_MODE = False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LLM Game Theory Experiment Runner - Four Folder Structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py                    # Run complete experimental matrix
  python runner.py --mock             # Run with mock responses (no API calls)
  python runner.py --analysis-only    # Skip experiments, run analysis only
  python runner.py --games salop      # Run only Salop experiments
  python runner.py --verbose          # Enable verbose logging
  python runner.py --matrix           # Show experimental matrix and exit
        """
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='Run experiments with mock LLM responses instead of real API calls'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true', 
        help='Skip experiments and run analysis only on existing results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--games',
        nargs='+',
        choices=['salop', 'green_porter', 'spulber', 'athey_bagwell'],
        help='Run only specific games (default: all games)'
    )
    
    parser.add_argument(
        '--matrix',
        action='store_true',
        help='Display experimental matrix and exit'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging based on config.json settings"""
    config = load_config_file()
    logging_config = config.get('logging', {})
    
    level = logging.DEBUG if verbose else getattr(logging, logging_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def setup_four_folder_structure():
    """Create four-folder results structure"""
    results_dir = Path("results")
    games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    
    for game_name in games:
        game_dir = results_dir / game_name
        game_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model subdirectories
        challenger_models = get_challenger_models()
        for model in challenger_models:
            model_dir = game_dir / model
            model_dir.mkdir(parents=True, exist_ok=True)


async def validate_system_setup() -> bool:
    """Validate system setup for experiments"""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate config
        if not validate_config():
            return False
        
        # Validate agent setup (skip in mock mode)
        if not MOCK_MODE:
            challenger_models = get_challenger_models()
            defender_model = get_defender_model()
            logger.info(f"Validating models: {challenger_models + [defender_model]}")
            # Agent validation logic would go here
        
        # Setup folder structure
        setup_four_folder_structure()
        
        return True
        
    except Exception as e:
        logger.error(f"System setup validation failed: {e}")
        return False


def display_experiment_overview(selected_games: Optional[List[str]] = None):
    """Display comprehensive experiment overview with new matrix"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧠 GEMINI THINKING LLM GAME THEORY EXPERIMENTS")
    logger.info("📁 Four-Folder Structure with Dynamic Metrics Integration")
    if MOCK_MODE:
        logger.info("🎭 MOCK MODE - Testing workflow without API calls")
    logger.info("=" * 80)
    
    # Model overview
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    logger.info(f"📱 MODELS ({len(challenger_models) + 1} total):")
    
    logger.info("  🏆 CHALLENGERS:")
    for model in challenger_models:
        display_name = get_model_display_name(model)
        thinking_status = "🧠 ON" if is_thinking_enabled(model) else "⚡ OFF"
        logger.info(f"    • {display_name} (Thinking: {thinking_status})")
    
    defender_display_name = get_model_display_name(defender_model)
    defender_thinking_status = "🧠 ON" if is_thinking_enabled(defender_model) else "⚡ OFF"
    logger.info(f"  🛡️ DEFENDER: {defender_display_name} (Thinking: {defender_thinking_status})")
    
    # Games overview with condition breakdown
    logger.info("🎮 EXPERIMENTAL MATRIX:")
    games_to_run = selected_games or ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    total_conditions = 0
    
    for game_name in games_to_run:
        game_configs = get_all_game_configs(game_name)
        game_conditions = len(game_configs)
        total_conditions += game_conditions
        
        # Show dynamic game info
        if game_name in ['green_porter', 'athey_bagwell']:
            logger.info(f"    📁 {game_name.upper()}: {game_conditions} conditions (3 players, dynamic metrics)")
        else:
            logger.info(f"    📁 {game_name.upper()}: {game_conditions} conditions (3/5 players)")
    
    # Detailed experimental summary
    total_competitions = total_conditions * len(challenger_models)
    experiment_config = get_experiment_config()
    sims_per_competition = experiment_config.get('main_experiment_simulations', 50)
    estimated_simulations = total_competitions * sims_per_competition
    
    logger.info("📊 EXPERIMENT SUMMARY:")
    logger.info(f"    • Games to run: {len(games_to_run)}")
    logger.info(f"    • Total conditions: {total_conditions}")
    logger.info(f"    • Total competitions: {total_competitions}")
    logger.info(f"    • Simulations per condition: {sims_per_competition}")
    logger.info(f"    • Total simulations: {estimated_simulations}")
    logger.info(f"    • Output structure: results/[game]/[model]/[condition]_game_output.json")
    
    if not MOCK_MODE:
        # More accurate time estimation
        estimated_seconds_per_sim = 15  # Conservative estimate with thinking
        estimated_time_hours = (estimated_simulations * estimated_seconds_per_sim) / 3600
        logger.info(f"    • Estimated time: ~{estimated_time_hours:.1f} hours")
        logger.info(f"    • With dynamic metrics integrated into game output")
    else:
        logger.info("    • Mock mode: Fast execution for testing")
    
    logger.info("=" * 80)


async def run_game_experiments(game_name: str, competition: Competition, 
                             selected_games: Optional[List[str]] = None) -> bool:
    """Run experiments for a single game using new competition API"""
    logger = logging.getLogger(__name__)
    
    # Skip if game not selected
    if selected_games and game_name not in selected_games:
        logger.info(f"⏭️ Skipping {game_name} (not in selected games)")
        return True
    
    logger.info(f"🎮 Starting {game_name.upper()} experiments...")
    
    try:
        # Get all configurations for this game using the matrix approach
        game_configs = get_all_game_configs(game_name)
        challenger_models = get_challenger_models()
        
        total_competitions = len(game_configs) * len(challenger_models)
        completed = 0
        successful = 0
        
        for game_config in game_configs:
            for challenger_model in challenger_models:
                
                logger.info(f"🚀 Running {completed+1}/{total_competitions}: "
                          f"{challenger_model} - {game_config.condition_name}")
                
                # Create output directory path
                output_dir = f"results/{game_name}/{challenger_model}"
                
                try:
                    # Use new competition API
                    success = await competition.run_single_competition(
                        game_config=game_config,
                        challenger_model=challenger_model,
                        output_dir=output_dir
                    )
                    
                    if success:
                        successful += 1
                        logger.info(f"✅ Competition completed successfully")
                    else:
                        logger.warning(f"⚠️ Competition completed with issues")
                    
                    completed += 1
                    
                except Exception as e:
                    logger.error(f"❌ Competition failed: {e}")
                    completed += 1
                    continue
        
        success_rate = successful / total_competitions if total_competitions > 0 else 0
        logger.info(f"🎯 {game_name.upper()} experiments completed! "
                   f"Success rate: {success_rate:.1%} ({successful}/{total_competitions})")
        
        # Consider it successful if at least 80% of competitions succeeded
        return success_rate >= 0.8
        
    except Exception as e:
        logger.error(f"💥 {game_name.upper()} experiments failed: {e}", exc_info=True)
        return False


async def run_all_experiments(selected_games: Optional[List[str]] = None) -> bool:
    """Run experiments for all games using four-folder structure"""
    logger = logging.getLogger(__name__)
    
    # Get models from config
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    # Initialize competition engine with new API
    competition = Competition(
        challenger_models=challenger_models,
        defender_model=defender_model,
        mock_mode=MOCK_MODE
    )
    
    # Run experiments for each game
    games = selected_games or ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    
    for game_name in games:
        success = await run_game_experiments(game_name, competition, selected_games)
        if not success:
            logger.error(f"❌ {game_name} experiments failed, stopping pipeline")
            return False
    
    logger.info("🎯 All game experiments completed successfully!")
    return True


async def run_comprehensive_analysis() -> bool:
    """Run comprehensive analysis with four-folder structure"""
    logger = logging.getLogger(__name__)
    
    # Skip analysis in mock mode since results are simulated
    if MOCK_MODE:
        logger.info("📊 Skipping analysis in mock mode")
        return True
    
    logger.info("📊 Starting comprehensive four-folder analysis...")
    logger.info("🔬 Including dynamic game metrics and correlation analysis")
    
    try:
        # Import the updated analyzer that handles four-folder structure
        from analysis.results_analyzer import analyze_four_folder_experiment
        
        challenger_models = get_challenger_models()
        
        # Run comprehensive analysis
        analysis_report = analyze_four_folder_experiment(
            results_dir="results",
            challenger_models=challenger_models,
            output_dir="analysis_output"
        )
        
        logger.info("✅ Comprehensive analysis completed!")
        logger.info("📁 Analysis outputs:")
        logger.info("    • comprehensive_analysis_report.json - Complete results")
        logger.info("    • analysis_summary.md - Publication summary")
        logger.info("    • results/[game]/[model]/ - Individual game outputs with dynamic metrics")
        
        # Show analysis breakdown
        games_analyzed = analysis_report.experiment_metadata.get('games_analyzed', [])
        for game_name in games_analyzed:
            if game_name in ['green_porter', 'athey_bagwell']:
                logger.info(f"    📊 {game_name}: Performance + Magic + Dynamic metrics")
            else:
                logger.info(f"    📊 {game_name}: Performance + Magic metrics")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Analysis failed: {e}", exc_info=True)
        return False


async def main() -> int:
    """Main entry point with four-folder structure support"""
    global MOCK_MODE
    
    # Parse command line arguments
    args = parse_arguments()
    MOCK_MODE = args.mock
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("🧠 Gemini Thinking LLM Game Theory Experiment Framework")
    logger.info("📁 Four-Folder Structure with Comprehensive Dynamic Metrics")
    logger.info("📊 Loading configuration and validating setup...")
    
    # Handle matrix display mode
    if args.matrix:
        if validate_config():
            print_experiment_matrix()
            return 0
        else:
            logger.error("❌ Configuration validation failed")
            return 1
    
    # Handle analysis-only mode
    if args.analysis_only:
        logger.info("📊 Running analysis only...")
        if validate_config():
            success = await run_comprehensive_analysis()
            if success:
                logger.info("✅ Analysis completed successfully!")
                return 0
            else:
                logger.error("❌ Analysis failed!")
                return 1
        else:
            logger.error("❌ Configuration validation failed")
            return 1
    
    # Validate configuration
    if not validate_config():
        logger.error("❌ Configuration validation failed. Please fix issues and retry.")
        return 1
    
    # Validate system setup
    if not await validate_system_setup():
        logger.error("❌ System setup validation failed. Please fix issues and retry.")
        return 1
    
    # Display experiment overview
    display_experiment_overview(args.games)
    
    # Calculate experiment scope for confirmation
    games = args.games or ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    total_conditions = sum(len(get_all_game_configs(game)) for game in games)
    challenger_models = get_challenger_models()
    total_competitions = total_conditions * len(challenger_models)
    experiment_config = get_experiment_config()
    total_simulations = total_competitions * experiment_config.get('main_experiment_simulations', 50)
    
    # Ask for user confirmation (optional in mock mode)
    if not MOCK_MODE:
        estimated_seconds_per_sim = 15
        estimated_time_hours = (total_simulations * estimated_seconds_per_sim) / 3600
        
        logger.info(f"⏱️ EXPERIMENT SCOPE:")
        logger.info(f"    • {total_competitions} competitions across {len(games)} games")
        logger.info(f"    • {total_simulations} total simulations")
        logger.info(f"    • Estimated time: ~{estimated_time_hours:.1f} hours")
        logger.info(f"    • Dynamic metrics integrated for Green & Porter and Athey & Bagwell")
        logger.info("🚀 Starting experiments in 5 seconds... (Press Ctrl+C to abort)")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            logger.info("🛑 Experiment aborted by user")
            return 0
    else:
        logger.info("🚀 Starting mock experiments with four-folder structure...")
    
    # Run all experiments
    try:
        success = await run_all_experiments(args.games)
        
        if success:
            # Run comprehensive analysis
            if not MOCK_MODE:
                analysis_success = await run_comprehensive_analysis()
                if not analysis_success:
                    logger.warning("⚠️ Analysis failed, but experiments completed successfully")
            
            logger.info("=" * 100)
            logger.info("🎉 GEMINI THINKING EXPERIMENTS COMPLETED SUCCESSFULLY!")
            if not MOCK_MODE:
                logger.info("📁 Results organized in four folders: results/[game]/[model]/")
                logger.info("📊 Analysis results available in analysis_output/")
                logger.info("🔬 Review analysis_summary.md for key findings")
                logger.info("📈 Dynamic game metrics integrated in game output files")
                logger.info("🔗 Correlation analysis between performance, magic, and dynamic metrics")
            else:
                logger.info("🎭 Mock experiment completed - four-folder workflow validated!")
                logger.info("📁 Check results/ folder structure")
            logger.info("=" * 100)
            return 0
        else:
            logger.error("❌ Experiments failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Experiments interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    print("🧠 Gemini Thinking LLM Game Theory Experiment Framework")
    print("📁 Four-Folder Structure with Comprehensive Dynamic Metrics")
    print("🚀 Running strategic reasoning experiments with round-by-round analysis...")
    print()
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)