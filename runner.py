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
    is_thinking_enabled, get_model_config
)
from competition import Competition
from analysis.results_analyzer import ResultsAnalyzer
from agents import create_agent

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


def setup_logging(verbose: bool = False) -> None:
    """Setup logging based on config.json settings"""
    config = load_config_file()
    logging_config = config.get('logging', {})
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_mock" if MOCK_MODE else ""
    log_file = logs_dir / f"gemini_thinking_experiment_{timestamp}{mode_suffix}.log"
    
    # Configure logging
    log_level = logging_config.get('level', 'DEBUG' if verbose else 'INFO')
    log_format = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w')
    ]
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Reduce external library noise unless verbose
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("google").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Gemini Thinking Experiment started. Log file: {log_file}")
    
    if MOCK_MODE:
        logger.info("🎭 MOCK MODE ENABLED - All LLM responses will be simulated")
        logger.info("💡 No API calls will be made, no costs will be incurred")


async def test_agent(model_name: str) -> dict:
    """Test a single agent configuration"""
    test_prompt = "You are playing an economic game. Choose a price between 0.1 and 2.0. Respond with a JSON format: {\"answer\": your_answer}"
    
async def test_agent(model_name: str) -> dict:
    """Test a single agent configuration"""
    test_prompt = "You are playing an economic game. Choose a price between 0.1 and 2.0. Respond with a JSON format: {\"answer\": your_answer}"
    
    try:
        agent = create_agent(model_name, "test_player")
        
        start_time = time.time()
        response = await agent.get_response(test_prompt, "test_call")
        end_time = time.time()
        
        return {
            'model_name': model_name,
            'success': response.success,
            'response_time': end_time - start_time,
            'content': response.content[:100] + "..." if len(response.content) > 100 else response.content,
            'tokens_used': response.tokens_used,
            'thinking_tokens': response.thinking_tokens,
            'error': response.error
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }


async def test_all_agents() -> dict:
    """Test all configured agents"""
    config = load_config_file()
    challenger_models = config.get('models', {}).get('challenger_models', [])
    defender_model = config.get('models', {}).get('defender_model')
    
    all_models = challenger_models + ([defender_model] if defender_model else [])
    
    results = {}
    
    for model in all_models:
        print(f"Testing {model}...")
        result = await test_agent(model)
        results[model] = result
        
        if result['success']:
            print(f"  ✅ Success in {result['response_time']:.2f}s")
        else:
            print(f"  ❌ Failed: {result['error']}")
    
    return results


def validate_agent_setup(model_name: str) -> bool:
    """Validate agent setup for a specific model"""
    try:
        model_config = get_model_config(model_name)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Model {model_name} configuration error: {e}")
        return False


async def validate_system_setup() -> bool:
    """Validate that the system is ready for experiments"""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating system setup...")
    
    # Validate configuration
    if not validate_config():
        logger.error("❌ Configuration validation failed. Please check config.json")
        return False
    
    # Get models from config
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    # Validate API setup for all models
    all_models = challenger_models + [defender_model]
    
    for model in all_models:
        if not validate_agent_setup(model):
            logger.error(f"❌ Agent setup validation failed for {model}")
            return False
    
    logger.info("✅ System setup validation passed")
    
    # Test API connectivity (skip in mock mode)
    if not MOCK_MODE:
        logger.info("Testing API connectivity with all models...")
        try:
            test_results = await test_all_agents()
            
            failed_models = [model for model, result in test_results.items() if not result['success']]
            if failed_models:
                logger.warning(f"⚠️ API test failed for models: {failed_models}")
                logger.warning("Proceeding anyway - errors may occur during experiments")
            else:
                logger.info("✅ All models responding correctly")
                
        except Exception as e:
            logger.warning(f"⚠️ API connectivity test failed: {e}")
            logger.warning("Proceeding anyway - errors may occur during experiments")
    else:
        logger.info("✅ Skipping API tests in mock mode")
    
    return True


def display_experiment_overview():
    """Display comprehensive overview of the experimental setup"""
    logger = logging.getLogger(__name__)
    
    summary = create_experiment_summary()
    
    logger.info("=" * 100)
    logger.info("🧠 GEMINI THINKING EXPERIMENT OVERVIEW")
    if MOCK_MODE:
        logger.info("🎭 MOCK MODE - Testing workflow and metrics")
    logger.info("=" * 100)
    
    # Get models from config directly since summary might not have all keys
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    
    # Model overview
    logger.info(f"📱 MODELS ({len(challenger_models) + 1} total):")
    
    logger.info("  🏆 CHALLENGERS:")
    for model in challenger_models:
        display_name = get_model_display_name(model)
        thinking_status = "🧠 ON" if is_thinking_enabled(model) else "⚡ OFF"
        logger.info(f"    • {display_name} (Thinking: {thinking_status})")
    
    defender_display_name = get_model_display_name(defender_model)
    defender_thinking_status = "🧠 ON" if is_thinking_enabled(defender_model) else "⚡ OFF"
    logger.info(f"  🛡️ DEFENDER: {defender_display_name} (Thinking: {defender_thinking_status})")
    
    # Games overview
    logger.info("🎮 GAMES (4):")
    games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    for game_name in games:
        from config import get_all_game_configs
        game_configs = get_all_game_configs(game_name)
        logger.info(f"    • {game_name.upper()}: {len(game_configs)} conditions")
    
    # Calculate experiment summary
    total_configs = 0
    for game_name in games:
        from config import get_all_game_configs
        game_configs = get_all_game_configs(game_name)
        total_configs += len(game_configs)
    
    total_competitions = total_configs * len(challenger_models)
    experiment_config = get_experiment_config()
    estimated_simulations = total_competitions * experiment_config.get('main_experiment_simulations', 50)
    
    # Experiment summary
    logger.info("📊 EXPERIMENT SUMMARY:")
    logger.info(f"    • Total configurations: {total_configs}")
    logger.info(f"    • Total competitions: {total_competitions}")
    logger.info(f"    • Estimated simulations: {estimated_simulations}")
    
    if not MOCK_MODE:
        estimated_time_hours = (estimated_simulations * 4 * 7) / 3600  # Rough estimate
        logger.info(f"    • Estimated time: ~{estimated_time_hours:.1f} hours")
        logger.info(f"    • Estimated cost: ~${estimated_simulations * 0.01:.2f}")  # Rough estimate
    else:
        logger.info(f"    • Mock mode: Fast execution, no API costs")
    
    logger.info("=" * 100)


async def run_game_experiments(game_name: str, competition: Competition) -> bool:
    """Run all experiments for a specific game"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎮 Starting {game_name.upper()} experiments...")
    
    try:
        challenger_models = get_challenger_models()
        defender_model = get_defender_model()
        
        # Get all configurations for this game
        from config import get_all_game_configs
        game_configs = get_all_game_configs(game_name)
        
        total_competitions = len(game_configs) * len(challenger_models)
        completed = 0
        
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
                    else:
                        logger.warning(f"⚠️ Competition completed with issues")
                    
                    completed += 1
                    
                except Exception as e:
                    logger.error(f"❌ Competition failed: {e}")
                    return False
        
        logger.info(f"🎉 {game_name.upper()} experiments completed! ({completed}/{total_competitions})")
        return True
        
    except Exception as e:
        logger.error(f"💥 {game_name.upper()} experiments failed: {e}", exc_info=True)
        return False


async def run_all_experiments() -> bool:
    """Run experiments for all games"""
    logger = logging.getLogger(__name__)
    
    # Initialize competition engine
    competition = Competition()
    
    # Run experiments for each game
    games = ['salop', 'green_porter', 'spulber', 'athey_bagwell']
    
    for game_name in games:
        success = await run_game_experiments(game_name, competition)
        if not success:
            logger.error(f"❌ {game_name} experiments failed, stopping pipeline")
            return False
    
    logger.info("🎯 All game experiments completed successfully!")
    
    # Run analysis
    logger.info("📊 Starting comprehensive analysis...")
    
    try:
        analyzer = ResultsAnalyzer()
        challenger_models = get_challenger_models()
        defender_model = get_defender_model()
        
        analysis_report = analyzer.analyze_complete_experiment(
            results_dir="results",
            challenger_models=challenger_models,
            defender_model=defender_model
        )
        
        logger.info("✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"💥 Analysis failed: {e}", exc_info=True)
        return False


async def main() -> int:
    """Main entry point"""
    global MOCK_MODE
    
    # Parse command line arguments
    args = parse_arguments()
    MOCK_MODE = args.mock
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("🧠 Gemini Thinking LLM Game Theory Experiment Framework")
    logger.info("📊 Loading configuration and validating setup...")
    
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
    
    # Ask for user confirmation (optional)
    experiment_config = get_experiment_config()
    estimated_time_hours = (experiment_config.get('main_experiment_simulations', 50) * 4 * 7) / 3600  # Rough estimate
    
    if not MOCK_MODE:
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
            logger.info("=" * 100)
            logger.info("🎉 GEMINI THINKING EXPERIMENTS COMPLETED SUCCESSFULLY!")
            logger.info("📊 Analysis results available in analysis_output/")
            logger.info("🔬 Review publication_summary.md for key findings")
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
    print("🚀 Running strategic reasoning experiments with thinking analysis...")
    print()
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)