#!/usr/bin/env python3
"""
LLM Game Theory Experiment Runner - Executes Gemini thinking experiments from config.json
Orchestrates complete experimental pipeline with thinking analysis
"""

import asyncio
import logging
import sys
import time
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


def setup_logging() -> None:
    """Setup logging based on config.json settings"""
    config = load_config_file()
    logging_config = config.get('logging', {})
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"gemini_thinking_experiment_{timestamp}.log"
    
    # Configure logging
    log_level = logging_config.get('level', 'INFO')
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
    
    # Reduce external library noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Gemini Thinking Experiment started. Log file: {log_file}")


def validate_agent_setup(model_name: str) -> bool:
    """Validate that an agent can be properly created"""
    try:
        # Check if model config exists
        model_config = get_model_config(model_name)
        
        # Check if API key is available
        config = load_config_file()
        api_config = config.get('api', {}).get('google', {})
        api_key_env = api_config.get('api_key_env', 'GEMINI_API_KEY')
        
        import os
        api_key = os.getenv(api_key_env)
        if not api_key:
            logging.getLogger(__name__).error(f"Missing API key: {api_key_env}")
            return False
        
        # Check if required packages are available
        try:
            from google import genai
        except ImportError:
            logging.getLogger(__name__).error("google-genai package not installed")
            logging.getLogger(__name__).error("Run: pip install google-genai")
            return False
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent setup validation failed: {e}")
        return False


async def test_agent(model_name: str) -> dict:
    """Test an agent with a simple prompt"""
    test_prompt = "What is 2 + 2? Respond with a JSON format: {\"answer\": your_answer}"
    
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
    
    # Test API connectivity (optional but recommended)
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
    
    return True


def display_experiment_overview():
    """Display comprehensive overview of the experimental setup"""
    logger = logging.getLogger(__name__)
    
    summary = create_experiment_summary()
    
    logger.info("=" * 100)
    logger.info("🧠 GEMINI THINKING EXPERIMENT OVERVIEW")
    logger.info("=" * 100)
    
    # Model overview
    logger.info(f"📱 MODELS ({summary['total_model_variants']} total):")
    
    logger.info("  🏆 CHALLENGERS:")
    for model in summary['challenger_models']:
        display_name = get_model_display_name(model)
        thinking_status = "🧠 ON" if is_thinking_enabled(model) else "⚡ OFF"
        logger.info(f"    • {display_name} (Thinking: {thinking_status})")
    
    defender_display_name = get_model_display_name(summary['defender_model'])
    defender_thinking_status = "🧠 ON" if is_thinking_enabled(summary['defender_model']) else "⚡ OFF"
    logger.info(f"  🛡️ DEFENDER: {defender_display_name} (Thinking: {defender_thinking_status})")
    
    # Games overview
    logger.info("🎮 GAMES (4):")
    for game_name, config_count in summary['configs_per_game'].items():
        logger.info(f"    • {game_name.upper()}: {config_count} conditions")
    
    # Experiment scope
    logger.info("⚙️ EXPERIMENT SCOPE:")
    logger.info(f"    • Total configurations: {summary['total_configurations']}")
    logger.info(f"    • Total competitions: {summary['total_competitions']}")
    logger.info(f"    • Estimated simulations: {summary['estimated_total_simulations']}")
    
    # Thinking analysis focus
    logger.info("🧠 THINKING ANALYSIS:")
    thinking_models = summary.get('thinking_enabled_models', [])
    logger.info(f"    • Models with thinking: {len(thinking_models)}")
    logger.info("    • Thinking vs non-thinking comparisons available")
    logger.info("    • Strategic reasoning depth analysis enabled")
    
    logger.info("=" * 100)


async def run_all_experiments() -> bool:
    """Run all experiments across all games and conditions"""
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize competition engine
        competition = Competition(output_dir="results")
        
        # Run competitions for all games
        logger.info("🎯 Starting game theory competitions...")
        
        success = await competition.run_all_competitions()
        
        if not success:
            logger.error("❌ Competitions failed!")
            return False
        
        # Run comprehensive analysis
        logger.info("📊 Starting comprehensive analysis...")
        analyzer = ResultsAnalyzer(output_dir="analysis_output")
        
        challenger_models = get_challenger_models()
        defender_model = get_defender_model()
        
        report = analyzer.analyze_complete_experiment(
            results_dir="results",
            challenger_models=challenger_models,
            defender_model=defender_model
        )
        
        logger.info("✅ Analysis completed successfully!")
        
        # Display final results summary
        display_final_results_summary()
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Experiment execution failed: {e}", exc_info=True)
        return False


def display_final_results_summary():
    """Display final summary of experiment results"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("📊 EXPERIMENT RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("📁 Results Location:")
    logger.info("    • Raw results: results/")
    logger.info("    • Analysis: analysis_output/")
    logger.info("📈 Key Output Files:")
    logger.info("    • analysis_output/correlation_analysis.csv")
    logger.info("    • analysis_output/performance_metrics.csv") 
    logger.info("    • analysis_output/magic_behavioral_metrics.csv")
    logger.info("    • analysis_output/publication_summary.md")
    logger.info("=" * 60)
    
    # Thinking-specific insights
    logger.info("🧠 KEY THINKING INSIGHTS:")
    logger.info("    • Compare thinking ON vs OFF performance in CSV files")
    logger.info("    • Analyze strategic sophistication differences")
    logger.info("    • Review correlation between thinking and MAgIC metrics")
    logger.info("    • Check publication_summary.md for detailed findings")


async def main():
    """Main entry point - runs complete Gemini thinking experiment suite"""
    
    # Setup logging first
    try:
        setup_logging()
    except Exception as e:
        print(f"Failed to setup logging: {e}")
        return 1
    
    logger = logging.getLogger(__name__)
    logger.info("🧠 Gemini Thinking LLM Game Theory Experiment Runner")
    
    # Display experiment overview
    display_experiment_overview()
    
    # Validate system setup
    if not await validate_system_setup():
        logger.error("❌ System validation failed. Please fix issues and retry.")
        return 1
    
    # Ask for user confirmation (optional)
    experiment_config = get_experiment_config()
    estimated_time_hours = (experiment_config.get('main_experiment_simulations', 50) * 4 * 7) / 3600  # Rough estimate
    
    logger.info(f"⏱️ ESTIMATED EXPERIMENT TIME: ~{estimated_time_hours:.1f} hours")
    logger.info("🚀 Starting experiments in 5 seconds... (Press Ctrl+C to abort)")
    
    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        logger.info("🛑 Experiment aborted by user")
        return 0
    
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