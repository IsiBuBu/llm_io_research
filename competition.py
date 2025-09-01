"""
Minimal competition system for LLM game theory experiments.
Only includes essential functionality for runner.py
"""

import os
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import minimal config system
from config import (
    ModelConfig, GameConfig, ExperimentConfig, PlayerResult, GameResult,
    get_model_config, get_api_config
)

# Import game classes
from games.salop_game import SalopGame
from games.spulber_game import SpulberGame  
from games.green_porter_game import GreenPorterGame
from games.athey_bagwell_game import AtheyBagwellGame


@dataclass
class LLMResponse:
    """Response from LLM API call"""
    final_response: str
    thinking_response: str = ""
    error: Optional[str] = None
    response_time: float = 0.0
    model_config: Optional[ModelConfig] = None
    call_id: str = ""


class GeminiLLMAgent:
    """Minimal Gemini LLM Agent"""
    
    def __init__(self, model_config: ModelConfig, debug: bool = False):
        self.model_config = model_config
        self.debug = debug
        self.logger = logging.getLogger(f"competition.{model_config.display_name}")
        
        # Get API key from config
        api_config = get_api_config()
        api_key_env = api_config.get('gemini_api_key_env', 'GEMINI_API_KEY')
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            raise ValueError(f"API key required: {api_key_env}")
        
        # Initialize Gemini client
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model_config.model_name)
        except ImportError:
            raise ImportError("google-generativeai package required")

    def call_llm(self, prompt: str, call_id: str = None) -> LLMResponse:
        """Make LLM API call"""
        call_id = call_id or f"{self.model_config.model_name}_{int(time.time())}"
        start_time = time.time()
        
        try:
            self.logger.debug(f"[{call_id}] Calling {self.model_config.model_name}")
            
            # Make API call
            response = self.client.generate_content(prompt)
            response_time = time.time() - start_time
            
            # Extract response text
            final_response = response.text if response.text else ""
            
            self.logger.info(f"[{call_id}] SUCCESS in {response_time:.2f}s")
            self.logger.info(f"[{call_id}] Output: {len(final_response)} chars")
            
            return LLMResponse(
                final_response=final_response,
                response_time=response_time,
                model_config=self.model_config,
                call_id=call_id
            )
            
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            response_time = time.time() - start_time
            
            self.logger.error(f"[{call_id}] {error_msg}")
            
            return LLMResponse(
                final_response="",
                error=error_msg,
                response_time=response_time,
                model_config=self.model_config,
                call_id=call_id
            )


class GameCompetition:
    """Minimal game competition system"""
    
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize game instances
        self.games = {
            'salop': SalopGame(),
            'spulber': SpulberGame(),
            'green_porter': GreenPorterGame(),
            'athey_bagwell': AtheyBagwellGame()
        }
        
        # Performance tracking
        self.total_api_calls = 0
        self.failed_calls = 0
        
        self.logger.info("GameCompetition initialized")

    def create_agents(self, experiment_config: ExperimentConfig, challenger_model_key: str = None) -> Dict[str, GeminiLLMAgent]:
        """Create agents for experiment"""
        agents = {}
        
        # Get defender model
        defender_model = get_model_config(experiment_config.defender_model_key)
        
        # Calculate number of defenders needed
        num_defenders = experiment_config.num_players - 1
        
        # Create defender agents
        for i in range(num_defenders):
            agent_key = f"defender_{i}" if num_defenders > 1 else "defender"
            agents[agent_key] = GeminiLLMAgent(defender_model, self.debug)
        
        # Create challenger agent
        challenger_key = challenger_model_key or experiment_config.challenger_model_keys[0]
        challenger_model = get_model_config(challenger_key)
        agents["challenger"] = GeminiLLMAgent(challenger_model, self.debug)
        
        self.logger.info(f"Created {len(agents)} agents for {experiment_config.num_players} players")
        return agents

    def run_single_game(self, game_name: str, experiment_config: ExperimentConfig, 
                       game_config: GameConfig, game_id: str, 
                       challenger_model_key: str = None) -> GameResult:
        """Run a single game instance"""
        
        self.logger.info(f"Starting game {game_id}: {game_name}")
        
        # Get game instance
        if game_name not in self.games:
            raise ValueError(f"Unknown game: {game_name}")
        game = self.games[game_name]
        
        # Create agents
        agents = self.create_agents(experiment_config, challenger_model_key)
        
        # Run game
        try:
            game_result = game.run_game(game_config, agents, game_id)
            
            self.logger.info(f"[{game_id}] Game completed successfully")
            return game_result
            
        except Exception as e:
            self.logger.error(f"[{game_id}] Game failed: {e}")
            raise

    def run_experiment(self, experiment_config: ExperimentConfig) -> Dict[str, Any]:
        """Run complete experiment with comprehensive metrics"""
        
        experiment_id = f"{experiment_config.game_name}_{int(time.time())}"
        start_time = time.time()
        
        self.logger.info(f"=== STARTING EXPERIMENT {experiment_id} ===")
        self.logger.info(f"Game: {experiment_config.game_name}")
        self.logger.info(f"Defender: {experiment_config.defender_model_key}")
        self.logger.info(f"Challengers: {experiment_config.challenger_model_keys}")
        
        # Reset statistics
        self.total_api_calls = 0
        self.failed_calls = 0
        
        # Initialize results
        experiment_results = {
            'experiment_config': asdict(experiment_config),
            'game_results': [],
            'summary_metrics': {},
            'start_time': datetime.now().isoformat(),
            'experiment_id': experiment_id
        }
        
        # Create game configuration
        game_config = GameConfig(
            number_of_players=experiment_config.num_players,
            number_of_rounds=experiment_config.num_rounds,
            num_games=experiment_config.num_games
        )
        
        successful_games = 0
        total_game_count = 0
        
        # Run games for each challenger model
        for challenger_idx, challenger_model_key in enumerate(experiment_config.challenger_model_keys):
            self.logger.info(f"Running games for challenger: {challenger_model_key}")
            
            # Run multiple game instances for this challenger
            for game_idx in range(experiment_config.num_games):
                total_game_count += 1
                game_instance_id = f"{experiment_id}_challenger_{challenger_idx}_game_{game_idx}"
                
                try:
                    game_result = self.run_single_game(
                        experiment_config.game_name,
                        experiment_config, 
                        game_config,
                        game_instance_id,
                        challenger_model_key
                    )
                    
                    experiment_results['game_results'].append(game_result)
                    successful_games += 1
                    
                except Exception as e:
                    self.logger.error(f"Game {game_instance_id} failed: {e}")
                    # Continue with other games
                    continue
        
        # Calculate experiment duration
        experiment_duration = time.time() - start_time
        
        # Generate summary metrics
        total_expected_games = len(experiment_config.challenger_model_keys) * experiment_config.num_games
        
        summary_metrics = {
            'experiment_overview': {
                'total_games': total_expected_games,
                'successful_games': successful_games,
                'failed_games': total_expected_games - successful_games,
                'success_rate': successful_games / total_expected_games if total_expected_games > 0 else 0,
                'total_duration': experiment_duration
            },
            'api_stats': {
                'total_calls': self.total_api_calls,
                'failed_calls': self.failed_calls,
                'success_rate': (self.total_api_calls - self.failed_calls) / self.total_api_calls if self.total_api_calls > 0 else 0
            }
        }
        
        experiment_results['summary_metrics'] = summary_metrics
        experiment_results['end_time'] = datetime.now().isoformat()
        
        # Generate comprehensive metrics if games were successful
        if successful_games > 0:
            experiment_results['comprehensive_metrics'] = self._generate_comprehensive_metrics(experiment_results)
        
        # Log experiment summary
        self.logger.info(f"=== EXPERIMENT {experiment_id} COMPLETED ===")
        self.logger.info(f"Duration: {experiment_duration:.1f}s")
        self.logger.info(f"Games: {successful_games}/{total_expected_games} successful")
        self.logger.info(f"API calls: {self.total_api_calls} total, {self.failed_calls} failed")
        
        return experiment_results

    def _generate_comprehensive_metrics(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive behavioral metrics from game results"""
        
        try:
            game_results = experiment_results.get('game_results', [])
            if not game_results:
                return {}
            
            # Model performance metrics
            model_performance = {}
            strategic_patterns = {
                'cooperation_rate': 0.0,
                'competition_rate': 0.0
            }
            
            # Analyze each game result
            total_games = len(game_results)
            cooperation_count = 0
            
            for game_result in game_results:
                players = game_result.players if hasattr(game_result, 'players') else []
                
                for player in players:
                    model_name = player.player_role if hasattr(player, 'player_role') else 'unknown'
                    
                    if model_name not in model_performance:
                        model_performance[model_name] = {
                            'total_games': 0,
                            'wins': 0,
                            'total_profit': 0.0
                        }
                    
                    # Track performance
                    model_performance[model_name]['total_games'] += 1
                    if hasattr(player, 'win') and player.win:
                        model_performance[model_name]['wins'] += 1
                    if hasattr(player, 'profit'):
                        model_performance[model_name]['total_profit'] += player.profit
                    
                    # Simple cooperation detection (placeholder)
                    if hasattr(player, 'profit') and player.profit > 0:
                        cooperation_count += 1
            
            # Calculate final metrics
            for model, perf in model_performance.items():
                if perf['total_games'] > 0:
                    perf['win_rate'] = perf['wins'] / perf['total_games']
                    perf['avg_profit'] = perf['total_profit'] / perf['total_games']
                else:
                    perf['win_rate'] = 0.0
                    perf['avg_profit'] = 0.0
            
            # Calculate strategic patterns
            total_player_instances = sum(perf['total_games'] for perf in model_performance.values())
            if total_player_instances > 0:
                strategic_patterns['cooperation_rate'] = cooperation_count / total_player_instances
                strategic_patterns['competition_rate'] = 1.0 - strategic_patterns['cooperation_rate']
            
            return {
                'model_performance': model_performance,
                'strategic_patterns': strategic_patterns,
                'game_insights': {
                    'total_games_analyzed': total_games,
                    'models_tested': len(model_performance)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate comprehensive metrics: {e}")
            return {'error': str(e)}