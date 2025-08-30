#!/usr/bin/env python3
"""Quick test script to verify the comprehensive experiment system works"""

import os
from runner import ExperimentRunner
import logging

def main():
    """Run a quick test of the comprehensive system"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== QUICK COMPREHENSIVE TEST ===")
    
    try:
        # Initialize runner
        runner = ExperimentRunner(debug=False)  # Less verbose for quick test
        
        # Test just 2 games with 2 challengers for speed
        logger.info("Running quick comprehensive test...")
        
        # Temporarily modify the experiment config for faster testing
        original_config = runner.experiment_presets['your_main_setup']
        
        # Create a smaller test config
        quick_config = original_config.__class__(
            defender_model_key=original_config.defender_model_key,
            challenger_model_keys=["gemini-2.0-flash-lite", "gemini-2.5-flash"],  # Just 2 models
            game_name=original_config.game_name,
            num_players=3,
            num_rounds=1,
            num_games=1,  # Just 1 game per experiment for speed
            include_thinking=True
        )
        
        # Temporarily replace the config
        runner.experiment_presets['your_main_setup'] = quick_config
        
        # Run the comprehensive experiment
        results = runner.run_comprehensive_all_games_experiment("quick_test_results")
        
        # Print summary
        logger.info("=== QUICK TEST RESULTS ===")
        if 'comprehensive_summary' in results:
            summary = results['comprehensive_summary']
            logger.info(f"Games tested: {summary.get('games_tested', [])}")
            logger.info(f"Total experiments: {summary['overall_stats']['total_experiments']}")
            logger.info(f"Successful experiments: {summary['overall_stats']['successful_experiments']}")
            
            for model, perf in summary.get('models_performance', {}).items():
                logger.info(f"  {model}: {perf['successful_experiments']}/{perf['total_experiments']} successful ({perf['average_success_rate']:.1f}%)")
        
        logger.info("✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
