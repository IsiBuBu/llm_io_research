#!/usr/bin/env python3

import logging
import os
import json
from config import ExperimentConfig
from competition import GameCompetition

def test_simplified_output():
    """Test the simplified JSON output format"""
    
    # Set API key
    os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', 'AIzaSyChzMr2ggV_oqu_kwYKF6A6BtdKo6t2nJU')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple experiment config with 2 challengers and 2 games each (smaller for testing)
    experiment_config = ExperimentConfig(
        defender_model_key="gemini-2.0-flash-lite",
        challenger_model_keys=["gemini-2.5-pro", "gemini-2.0-flash"],
        game_name="salop",
        num_players=3,
        num_rounds=1,
        num_games=2,  # Just 2 games per challenger for quick test
        include_thinking=True
    )
    
    # Run experiment
    competition = GameCompetition(debug=True)
    results = competition.run_experiment(experiment_config)
    
    # Save results to file
    with open('simplified_output_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display first game result to show simplified format
    print("\n=== SIMPLIFIED JSON FORMAT SAMPLE ===")
    if results['game_results']:
        first_game = results['game_results'][0]
        print(json.dumps(first_game, indent=2))
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Total games: {len(results['game_results'])}")
    print(f"Expected: 4 (2 challengers × 2 games each)")
    
    # Check structure of results
    for i, game in enumerate(results['game_results']):
        print(f"Game {i+1}: {game['game_number']} - Challenger: {game['challenger_model']}")
        if 'challenger' in game and game['challenger']:
            print(f"  Challenger reasoning length: {len(game['challenger'].get('reasoning', '')) if game['challenger'].get('reasoning') else 0} chars")
        print(f"  Defenders: {len(game['defenders'])} (simplified)")

if __name__ == "__main__":
    test_simplified_output()
