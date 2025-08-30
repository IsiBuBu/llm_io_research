#!/usr/bin/env python3
"""
Simple single model test to verify fixes
"""

import time
from runner import ExperimentRunner
from config import Config

def test_single_model():
    """Test just one model to verify the fixes work"""
    print("=== TESTING SINGLE MODEL ===")
    
    # Initialize
    config = Config()
    runner = ExperimentRunner(config)
    
    # Test a single model interaction
    print("Testing gemini-2.5-flash-thinking model...")
    
    # Create a simple experiment config
    experiment_config = type('ExperimentConfig', (), {
        'defender_model_key': 'gemini-2.5-flash-thinking',
        'challenger_model_keys': ['gemini-2.5-flash-thinking'],
        'game_type': 'salop',
        'number_of_games': 1,
        'include_thinking': True
    })()
    
    try:
        result = runner.competition.run_experiment(experiment_config)
        print(f"✓ SUCCESS: Experiment completed without errors")
        print(f"Games completed: {result.get('successful_games', 0)}")
        print(f"Total profit: {result.get('total_industry_profit', 0):.2f}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_single_model()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
