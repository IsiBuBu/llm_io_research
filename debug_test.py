#!/usr/bin/env python3
"""
Debug test - Run a single game with maximum debugging
"""

import os
import logging
import json
from config import ExperimentPresets, GameConfigs
from competition import GameCompetition

def debug_single_game():
    """Run a single game with debug logging"""
    
    # Set up detailed logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STARTING DEBUG SINGLE GAME TEST")
    logger.info("=" * 60)
    
    # Set API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = "AIzaSyANLSAbDwH2UjLEjoeW7vZt-uYAeg9enZ0"
        os.environ['GEMINI_API_KEY'] = api_key
    
    logger.info(f"API key set: {api_key[:20]}...")
    
    # Create competition
    comp = GameCompetition()
    
    # Get minimal configs
    experiment_configs = ExperimentPresets.all_models_vs_baseline()[:1]  # Just first one
    game_configs = [GameConfigs.quick_test_games()['salop'][0]]  # Just salop, 2 games
    
    logger.info(f"Experiment config: {experiment_configs[0].challenger_model.display_name} vs {experiment_configs[0].defender_model.display_name}")
    logger.info(f"Game config: {game_configs[0].number_of_players} players, {game_configs[0].number_of_rounds} rounds, {game_configs[0].num_games} games")
    
    try:
        logger.info("Starting tournament...")
        results = comp.run_tournament(['salop'], experiment_configs, game_configs, 1)
        
        logger.info("Tournament completed!")
        logger.info(f"Results keys: {list(results.keys())}")
        
        if 'experiment_results' in results:
            logger.info(f"Experiment results: {len(results['experiment_results'])} experiments")
            for key, game_results in results['experiment_results'].items():
                logger.info(f"  {key}: {len(game_results)} games")
                for i, game_result in enumerate(game_results):
                    logger.info(f"    Game {i+1}: {len(game_result.players)} players")
                    for player in game_result.players:
                        logger.info(f"      Player {player.player_id}: profit={player.profit}, win={player.win}, actions={len(player.actions)}")
        
        if 'comprehensive_metrics' in results:
            logger.info(f"Comprehensive metrics: {len(results['comprehensive_metrics'])} calculated")
        
        logger.info("=" * 60)
        logger.info("DEBUG TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    debug_single_game()
