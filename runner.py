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
    is_thinking_enabled
)
from competition import Competition
from analysis.results_analyzer import ResultsAnalyzer
from agents import test_all_agents, validate_agent_setup


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
    
    defender_display = get_model_display_name(summary['defender_model'])
    defender_thinking = "🧠 ON" if is_thinking_enabled(summary['defender_model']) else "⚡ OFF"
    logger.info(f"  🛡️ DEFENDER: {defender_display} (Thinking: {defender_thinking})")
    
    # Games overview
    logger.info(f"🎮 GAMES ({len(summary['games_tested'])}):")
    for game, config_count in summary['configs_per_game'].items():
        logger.info(f"    • {game.upper()}: {config_count} conditions")
    
    # Experiment scope
    logger.info(f"⚙️ EXPERIMENT SCOPE:")
    logger.info(f"    • Total configurations: {summary['total_configurations']}")
    logger.info(f"    • Total competitions: {summary['total_competitions']}")
    logger.info(f"    • Estimated simulations: {summary['estimated_total_simulations']:,}")
    
    # Thinking analysis focus
    thinking_models = summary['thinking_enabled_models']
    logger.info(f"🧠 THINKING ANALYSIS:")
    logger.info(f"    • Models with thinking: {len(thinking_models)}")
    logger.info(f"    • Thinking vs non-thinking comparisons available")
    logger.info(f"    • Strategic reasoning depth analysis enabled")
    
    logger.info("=" * 100)


async def run_all_experiments() -> bool:
    """Run all experiments and ablations defined in config.json"""
    logger = logging.getLogger(__name__)
    
    # Get configuration
    challenger_models = get_challenger_models()
    defender_model = get_defender_model()
    experiment_config = get_experiment_config()
    
    logger.info("🚀 STARTING GEMINI THINKING EXPERIMENTS")
    logger.info("=" * 80)
    
    start_time = time.time()
    competition = Competition()
    
    # Generate all competition configurations from config.json
    from config import get_all_game_configs
    
    all_competitions = []
    
    for game_name in ['salop', 'green_porter', 'spulber', 'athey_bagwell']:
        # Get all configurations for this game (baseline + variations + ablations)
        game_configs = get_all_game_configs(game_name)
        
        logger.info(f"📋 {game_name.upper()}: {len(game_configs)} conditions × {len(challenger_models)} challengers = {len(game_configs) * len(challenger_models)} competitions")
        
        for game_config in game_configs:
            for challenger in challenger_models:
                all_competitions.append({
                    'game_name': game_name,
                    'experiment_type': game_config.experiment_type,
                    'condition_name': game_config.condition_name,
                    'challenger_model': challenger,
                    'defender_model': defender_model
                })
    
    total_competitions = len(all_competitions)
    logger.info("=" * 80)
    logger.info(f"📊 TOTAL COMPETITIONS TO RUN: {total_competitions}")
    logger.info("=" * 80)
    
    try:
        # Run all competitions with enhanced progress tracking
        results = await competition.run_batch_competitions(all_competitions)
        successful_competitions = len(results)
        
        duration = time.time() - start_time
        
        logger.info("=" * 100)
        logger.info("🎉 EXPERIMENTS COMPLETED")
        logger.info("=" * 100)
        logger.info(f"✅ Success: {successful_competitions}/{total_competitions} competitions ({successful_competitions/total_competitions*100:.1f}%)")
        logger.info(f"⏱️ Total duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"⚡ Average per competition: {duration/max(total_competitions, 1):.1f} seconds")
        
        if successful_competitions == 0:
            logger.error("❌ No experiments completed successfully!")
            return False
        
        # Run comprehensive analysis
        logger.info("=" * 80)
        logger.info("🔍 STARTING THINKING ANALYSIS")
        logger.info("=" * 80)
        
        analyzer = ResultsAnalyzer(output_dir="analysis_output")
        report = analyzer.analyze_complete_experiment(
            results_dir="results",
            challenger_models=challenger_models,
            defender_model=defender_model
        )
        
        # Display key findings
        display_analysis_summary(report)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Experiment suite failed: {e}", exc_info=True)
        return False


def display_analysis_summary(report):
    """Display key findings from the analysis"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info("📊 THINKING ANALYSIS RESULTS")
    logger.info("=" * 100)
    
    # Correlation analysis summary
    corr_summary = report.correlation_results.get('summary', {})
    if corr_summary:
        logger.info("🔗 CORRELATION ANALYSIS:")
        logger.info(f"    • Hypotheses tested: {corr_summary.get('total_hypotheses_tested', 0)}")
        logger.info(f"    • Significant correlations: {corr_summary.get('significant_correlations', 0)}")
        logger.info(f"    • Strong correlations (|r| > 0.5): {corr_summary.get('strong_correlations', 0)}")
        logger.info(f"    • Confirmed expectations: {corr_summary.get('confirmed_expectations', 0)}")
        logger.info(f"    • Contradicted expectations: {corr_summary.get('contradicted_expectations', 0)}")
    
    # Games analyzed
    games_analyzed = report.experiment_metadata.get('games_analyzed', [])
    logger.info(f"🎮 GAMES ANALYZED: {', '.join([g.upper() for g in games_analyzed])}")
    
    # Models tested
    models_tested = report.experiment_metadata.get('challenger_models', [])
    logger.info(f"🤖 MODELS TESTED: {len(models_tested)}")
    for model in models_tested:
        display_name = get_model_display_name(model)
        thinking_status = "🧠" if is_thinking_enabled(model) else "⚡"
        logger.info(f"    • {thinking_status} {display_name}")
    
    logger.info("=" * 60)
    logger.info("📁 OUTPUTS GENERATED:")
    logger.info("    • analysis_output/comprehensive_analysis_report.json")
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