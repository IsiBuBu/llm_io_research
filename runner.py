#!/usr/bin/env python3
"""
LLM Game Theory Experiment Runner - Executes Gemini thinking experiments from config.json
Orchestrates complete experimental pipeline with thinking analysis
NOW SUPPORTS MOCK MODE: Use --mock flag to run with simulated LLM responses
"""

import asyncio
import logging
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

from config import (
    load_config_file, validate_config, get_challenger_models, get_defender_model,
    get_experiment_config, create_experiment_summary, get_model_display_name,
    is_thinking_enabled, get_model_config, get_all_game_configs, get_output_dir
)
from competition import Competition
from analysis.results_analyzer import ResultsAnalyzer
from agents import create_agent, validate_agent_setup, test_agent_connectivity

# Global flag for mock mode
MOCK_MODE = False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LLM Game Theory Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py                    # Run normal experiments with real LLM calls
  python runner.py --mock             # Run with mock responses (no API calls)
  python runner.py --mock --quick     # Run quick mock tests
  python runner.py --verbose          # Enable verbose logging
        """
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='Run experiments with mock LLM responses instead of real API calls'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run with reduced simulation counts for quick testing'
    )
    
    parser.add_argument(
        '--games',
        nargs='+',
        choices=['salop', 'green_porter', 'spulber', 'athey_bagwell'],
        help='Run only specific games (default: all games)'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> tuple:
    """Setup logging to write all logs to logs folder"""
    global MOCK_MODE
    
    config = load_config_file()
    logging_config = config.get('logging', {})
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_mock" if MOCK_MODE else ""
    log_file = logs_dir / f"experiment_{timestamp}{mode_suffix}.log"
    
    # Configure logging level
    log_level = logging.DEBUG if verbose else getattr(logging, logging_config.get('level', 'INFO').upper())
    
    # Setup logging configuration
    log_format = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    
    return logger, log_file


async def test_all_agents():
    """Test connectivity for all configured agents"""
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    all_models = list(set(challenger_models + [defender_model]))
    test_results = {}
    
    for model_name in all_models:
        try:
            agent = create_agent(model_name, mock_mode=MOCK_MODE)
            result = await test_agent_connectivity(agent)
            test_results[model_name] = result
        except Exception as e:
            test_results[model_name] = {'success': False, 'error': str(e)}
    
    return test_results


async def validate_system_setup() -> bool:
    """Validate system setup before running experiments"""
    logger = logging.getLogger(__name__)
    
    # Validate configuration
    if not validate_config():
        logger.error("❌ Configuration validation failed")
        return False
    
    logger.info("✅ Configuration validation passed")
    
    # Test agent connectivity (skip in mock mode)
    if MOCK_MODE:
        logger.info("🎭 Skipping API connectivity tests in mock mode")
        return True
    
    logger.info("🔍 Testing API connectivity...")
    test_results = await test_all_agents()
    
    success_count = 0
    total_count = len(test_results)
    
    for model_name, result in test_results.items():
        if result.get('success', False):
            logger.info(f"  ✅ {model_name}: Connected successfully")
            success_count += 1
        else:
            logger.error(f"  ❌ {model_name}: {result.get('error', 'Unknown error')}")
    
    if success_count == total_count:
        logger.info(f"✅ All {total_count} agents connected successfully")
        return True
    else:
        logger.error(f"❌ Only {success_count}/{total_count} agents connected successfully")
        return False


def display_experiment_overview():
    """Display experiment configuration overview"""
    logger = logging.getLogger(__name__)
    
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    experiment_config = get_experiment_config()
    
    logger.info("=" * 100)
    logger.info("GEMINI THINKING EXPERIMENT OVERVIEW")
    logger.info("=" * 100)
    
    logger.info(f"🤖 Models:")
    logger.info(f"    • Challenger models: {len(challenger_models)}")
    for model in challenger_models:
        display_name = get_model_display_name(model)
        thinking_status = "✓" if is_thinking_enabled(model) else "✗"
        logger.info(f"      - {display_name} (thinking: {thinking_status})")
    
    defender_display = get_model_display_name(defender_model)
    thinking_status = "✓" if is_thinking_enabled(defender_model) else "✗"
    logger.info(f"    • Defender model: {defender_display} (thinking: {thinking_status})")
    
    logger.info(f"🎮 Games: salop, green_porter, spulber, athey_bagwell")
    logger.info(f"🔬 Simulations per condition: {experiment_config.get('main_experiment_simulations', 50)}")
    
    total_competitions = 0
    for game_name in ['salop', 'green_porter', 'spulber', 'athey_bagwell']:
        game_configs = get_all_game_configs(game_name)
        total_competitions += len(game_configs) * len(challenger_models)
    
    logger.info(f"📊 Total competitions: {total_competitions}")
    
    if MOCK_MODE:
        logger.info("🎭 Mode: MOCK MODE")
        logger.info("    • Mock mode: Fast execution, no API costs")
    
    logger.info("=" * 100)


async def run_game_experiments(game_name: str, competition: Competition) -> bool:
    """Run all experiments for a specific game"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎮 Starting {game_name.upper()} experiments...")
    
    try:
        challenger_models = get_challenger_models()
        defender_model = get_defender_model()
        
        # Get all configurations for this game
        game_configs = get_all_game_configs(game_name)
        
        total_competitions = len(game_configs) * len(challenger_models)
        completed = 0
        
        # Collect results for saving
        game_results = []
        
        for game_config in game_configs:
            for challenger_model in challenger_models:
                logger.info(f"🔄 Competition {completed+1}/{total_competitions}: "
                          f"{challenger_model} vs {defender_model} "
                          f"({game_config.experiment_type}:{game_config.condition_name})")
                
                try:
                    result = await competition.run_competition(
                        game_name=game_config.game_name,
                        experiment_type=game_config.experiment_type,
                        condition_name=game_config.condition_name,
                        challenger_model=challenger_model,
                        defender_model=defender_model
                    )
                    
                    if result:
                        logger.info(f"✅ Competition completed successfully")
                        game_results.append(result)  # Collect the result
                    else:
                        logger.warning(f"⚠️ Competition completed with issues")
                    
                    completed += 1
                    
                except Exception as e:
                    logger.error(f"❌ Competition failed: {e}")
                    return False
        
        # Save all results for this game
        if game_results:
            logger.info(f"💾 Saving {len(game_results)} competition results for {game_name.upper()}...")
            await competition._save_results(game_results)
        
        logger.info(f"🎉 {game_name.upper()} experiments completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ {game_name.upper()} experiments failed: {e}")
        return False

async def run_all_experiments() -> bool:
    """Run all game experiments"""
    logger = logging.getLogger(__name__)
    
    try:
        # Get models from config
        challenger_models = get_challenger_models()
        defender_model = get_defender_model()
        
        # CRITICAL FIX: Actually get the output directory from config
        output_dir = Path(get_output_dir())
        
        logger.info(f"📱 Models loaded: {len(challenger_models)} challengers, defender: {defender_model}")
        logger.info(f"📁 Output directory from config: {output_dir}")
        
        # ACTUALLY FIXED: Initialize competition with output_dir from config
        competition = Competition(challenger_models, defender_model, mock_mode=MOCK_MODE, output_dir=output_dir)
        
        # Run experiments for all games
        games_to_run = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
        logger.info(f"🎯 Running experiments for games: {', '.join(games_to_run)}")
        
        # Run experiments for each game
        all_success = True
        for game_name in games_to_run:
            try:
                success = await run_game_experiments(game_name, competition)
                if not success:
                    all_success = False
                    logger.error(f"❌ {game_name} experiments failed")
                else:
                    logger.info(f"✅ {game_name} experiments completed successfully")
            except Exception as e:
                logger.error(f"❌ {game_name} experiments failed with error: {e}")
                all_success = False
        
        if all_success:
            logger.info("🎯 All game experiments completed successfully!")
        else:
            logger.error("❌ Some experiments failed")
            
        return all_success
        
    except Exception as e:
        logger.error(f"💥 run_all_experiments failed: {e}", exc_info=True)
        return False


async def run_analysis() -> bool:
    """Run post-experiment analysis"""
    logger = logging.getLogger(__name__)
    
    if MOCK_MODE:
        logger.info("🎭 Skipping analysis in mock mode")
        return True
    
    logger.info("📊 Starting post-experiment analysis...")
    
    try:
        analyzer = ResultsAnalyzer()
        
        # Run comprehensive analysis
        await analyzer.run_full_analysis()
        
        logger.info("✅ Analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False


async def main() -> int:
    """Main execution function"""
    global MOCK_MODE
    
    # Parse command line arguments
    args = parse_arguments()
    MOCK_MODE = args.mock
    
    # Setup logging first
    logger, log_file = setup_logging(verbose=args.verbose)
    
    logger.info("🧠 Gemini Thinking LLM Game Theory Experiment Framework")
    logger.info("🚀 Running strategic reasoning experiments with thinking analysis...")
    logger.info(f"📝 Log file: {log_file}")
    
    # Validate configuration
    if not validate_config():
        logger.error("❌ Configuration validation failed. Please fix issues and retry.")
        return 1
    
    # Validate system setup
    if not await validate_system_setup():
        logger.error("❌ System setup validation failed. Please fix issues and retry.")
        return 1
    
    # Display experiment overview
    display_experiment_overview()
    
    # Ask for user confirmation (optional in mock mode)
    if not MOCK_MODE:
        experiment_config = get_experiment_config()
        estimated_time_hours = (experiment_config.get('main_experiment_simulations', 50) * 4 * 7) / 3600  # Rough estimate
        
        logger.info(f"⏱️ ESTIMATED EXPERIMENT TIME: ~{estimated_time_hours:.1f} hours")
        logger.info("🚀 Starting experiments in 5 seconds... (Press Ctrl+C to abort)")
        try:
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            logger.info("🛑 Experiment aborted by user")
            return 0
    else:
        logger.info("🚀 Starting mock experiments...")
    
    # Run all experiments
    try:
        success = await run_all_experiments()
        
        if success:
            # Run analysis (skipped in mock mode)
            if not MOCK_MODE:
                analysis_success = await run_analysis()
                if not analysis_success:
                    logger.warning("⚠️ Analysis failed, but experiments completed successfully")
            
            logger.info("=" * 100)
            logger.info("🎉 GEMINI THINKING EXPERIMENTS COMPLETED SUCCESSFULLY!")
            logger.info("=" * 100)
            
            if not MOCK_MODE:
                logger.info("📊 Check the analysis results for comprehensive insights")
                logger.info("📁 Raw results available in the configured output directory")
            
            return 0
        else:
            logger.error("❌ Some experiments failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Experiments interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)