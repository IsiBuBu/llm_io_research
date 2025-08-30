import logging
import os
import time
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict, dataclass
import pandas as pd
from pathlib import Path

from config import (
    ModelConfig, GameConfig, ExperimentConfig, PlayerResult, GameResult,
    get_available_models, get_model_config, get_api_config, get_logging_config
)

# Import game modules
from games.salop_game import SalopGame
from games.spulber_game import SpulberGame  
from games.green_porter_game import GreenPorterGame
from games.athey_bagwell_game import AtheyBagwellGame

# Import the enhanced Gemini client
from google import genai
from google.genai import types as genai_types

@dataclass
class LLMResponse:
    """Enhanced response structure with debugging info"""
    final_response: str
    thinking_output: Optional[str] = None
    response_time: float = 0.0
    token_usage: Dict[str, int] = None
    thought_signatures: Optional[List] = None
    model_config: Optional[ModelConfig] = None
    error: Optional[str] = None
    call_id: str = ""
    
    def __post_init__(self):
        if self.token_usage is None:
            self.token_usage = {}

class GeminiLLMAgent:
    """Enhanced LLM Agent with comprehensive debugging and simple API calls"""
    
    def __init__(self, model_config: ModelConfig, debug: bool = True):
        self.model_config = model_config
        self.debug = debug
        self.logger = logging.getLogger(f"{__name__}.{model_config.display_name}")
        
        # Initialize Gemini client
        api_config = get_api_config()
        api_key = None
        if api_key_env := api_config.get('gemini_api_key_env'):
            api_key = os.environ.get(api_key_env)
            
        self.client = genai.Client(api_key=api_key)
        
        # Call counter for debugging
        self.call_count = 0
        
    def generate_response(self, prompt: str, include_thinking: bool = True) -> LLMResponse:
        """Generate response with comprehensive debugging"""
        
        self.call_count += 1
        call_id = f"{self.model_config.model_name}_{self.call_count}_{int(time.time())}"
        start_time = time.time()
        
        if self.debug:
            self.logger.debug(f"[{call_id}] Starting API call")
            self.logger.debug(f"[{call_id}] Model: {self.model_config.model_name}")
            self.logger.debug(f"[{call_id}] Thinking available: {self.model_config.thinking_available}")
            self.logger.debug(f"[{call_id}] Thinking enabled: {self.model_config.thinking_enabled}")
            self.logger.debug(f"[{call_id}] Include thinking: {include_thinking}")
            self.logger.debug(f"[{call_id}] Prompt length: {len(prompt)} chars")
            self.logger.debug(f"[{call_id}] Prompt preview: {prompt[:200]}...")
        
        try:
            # Build generation config
            config_dict = {}
            
            # Configure thinking based on model capabilities and documentation
            if self.model_config.thinking_available:
                # Special handling for Gemini 2.5 Pro - cannot disable thinking
                if "2.5-pro" in self.model_config.model_name:
                    config_dict['thinking_config'] = genai_types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=-1  # Dynamic thinking (always on for Pro)
                    )
                    if self.debug:
                        self.logger.debug(f"[{call_id}] Gemini 2.5 Pro: Always thinking enabled (cannot disable)")
                
                elif self.model_config.thinking_enabled and include_thinking:
                    config_dict['thinking_config'] = genai_types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=-1  # Dynamic thinking
                    )
                    if self.debug:
                        self.logger.debug(f"[{call_id}] Thinking enabled with dynamic budget")
                        
                else:
                    # Disable thinking for other models
                    config_dict['thinking_config'] = genai_types.ThinkingConfig(
                        thinking_budget=0  # Disable thinking
                    )
                    if self.debug:
                        self.logger.debug(f"[{call_id}] Thinking disabled (budget=0)")
            else:
                if self.debug:
                    self.logger.debug(f"[{call_id}] Model has no thinking capability")
            
            config = genai_types.GenerateContentConfig(**config_dict) if config_dict else None
            
            # Make API call with retries
            response = None
            last_error = None
            
            for attempt in range(self.model_config.max_retries):
                try:
                    if self.debug:
                        self.logger.debug(f"[{call_id}] Attempt {attempt + 1}/{self.model_config.max_retries}")
                    
                    response = self.client.models.generate_content(
                        model=self.model_config.model_name,
                        contents=prompt,
                        config=config
                    )
                    break
                    
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"[{call_id}] Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.model_config.max_retries - 1:
                        delay = get_api_config().get('backoff_factor', 2) ** attempt
                        time.sleep(delay)
                        if self.debug:
                            self.logger.debug(f"[{call_id}] Retrying in {delay}s...")
            
            if not response:
                error_msg = f"Failed after {self.model_config.max_retries} attempts: {last_error}"
                self.logger.error(f"[{call_id}] {error_msg}")
                return LLMResponse(
                    final_response="",
                    error=error_msg,
                    response_time=time.time() - start_time,
                    model_config=self.model_config,
                    call_id=call_id
                )
            
            # Extract response components with debugging
            final_text = ""
            thinking_text = ""
            thought_signatures = []
            
            if self.debug:
                self.logger.debug(f"[{call_id}] Processing response parts...")
            
            # Process response parts
            if response.candidates and response.candidates[0].content.parts:
                for i, part in enumerate(response.candidates[0].content.parts):
                    if self.debug:
                        self.logger.debug(f"[{call_id}] Part {i}: type={type(part).__name__}")
                    
                    if hasattr(part, 'thought') and part.thought:
                        thinking_text += part.text + "\n"
                        if self.debug:
                            self.logger.debug(f"[{call_id}] Found thinking part: {len(part.text)} chars")
                    elif hasattr(part, 'text'):
                        final_text += part.text
                        if self.debug:
                            self.logger.debug(f"[{call_id}] Found text part: {len(part.text)} chars")
                    
                    # Collect thought signatures if present
                    if hasattr(part, 'thought_signature') and part.thought_signature:
                        thought_signatures.append(part.thought_signature)
                        if self.debug:
                            self.logger.debug(f"[{call_id}] Found thought signature")
            
            # Extract token usage with debugging
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_usage = {
                    'input_tokens': getattr(usage, 'prompt_token_count', 0),
                    'output_tokens': getattr(usage, 'candidates_token_count', 0),
                    'thinking_tokens': getattr(usage, 'thoughts_token_count', 0),
                    'total_tokens': getattr(usage, 'total_token_count', 0)
                }
                if self.debug:
                    self.logger.debug(f"[{call_id}] Token usage: {token_usage}")
            
            response_time = time.time() - start_time
            
            # Create structured response
            llm_response = LLMResponse(
                final_response=final_text.strip(),
                thinking_output=thinking_text.strip() if thinking_text else None,
                response_time=response_time,
                token_usage=token_usage,
                thought_signatures=thought_signatures if thought_signatures else None,
                model_config=self.model_config,
                call_id=call_id
            )
            
            if self.debug:
                self.logger.info(f"[{call_id}] SUCCESS in {response_time:.2f}s")
                self.logger.info(f"[{call_id}] Output: {len(final_text)} chars")
                if thinking_text:
                    self.logger.info(f"[{call_id}] Thinking: {len(thinking_text)} chars")
            
            return llm_response
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"[{call_id}] {error_msg}")
            self.logger.error(f"[{call_id}] Traceback: {traceback.format_exc()}")
            
            return LLMResponse(
                final_response="",
                error=error_msg,
                response_time=time.time() - start_time,
                model_config=self.model_config,
                call_id=call_id
            )

class GameCompetition:
    """Enhanced game competition system with JSON-based configuration"""
    
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
        
        # Load debugging configuration
        logging_config = get_logging_config()
        self.verbose_logging = logging_config.get('verbose_logging', True)
        
        # Performance tracking
        self.experiment_start_time = None
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.failed_calls = 0
        
        self.logger.info("GameCompetition initialized with JSON configuration")
    
    def create_agents(self, experiment_config: ExperimentConfig, challenger_model_key: str = None) -> Dict[str, GeminiLLMAgent]:
        """Create agents based on experiment configuration with specified challenger model"""
        agents = {}
        
        # Get models
        defender_model = experiment_config.get_defender_model()
        challenger_models = experiment_config.get_challenger_models()
        
        # Calculate how many defenders we need (all players except 1 challenger)
        num_defenders = experiment_config.num_players - 1
        
        self.logger.info(f"Creating {num_defenders} defender agents and 1 challenger agent for {experiment_config.num_players} total players")
        
        # Create defender agents (multiple copies of same model)
        for i in range(num_defenders):
            agent_key = f"defender_{i}" if num_defenders > 1 else "defender"
            agents[agent_key] = GeminiLLMAgent(defender_model, debug=self.debug)
            
            self.logger.info(f"Created defender agent {i+1}/{num_defenders}: {defender_model.display_name}")
        
        # Create 1 challenger agent (use specified challenger model or first one)
        if challenger_model_key:
            # Get the specific challenger model by key
            challenger_model = get_model_config(challenger_model_key)
        else:
            challenger_model = challenger_models[0]
            
        agents['challenger'] = GeminiLLMAgent(challenger_model, debug=self.debug)
        
        self.logger.info(f"Created challenger agent: {challenger_model.display_name}")
        
        return agents
    
    def run_single_game(self, game_name: str, experiment_config: ExperimentConfig, 
                       config: GameConfig, game_instance_id: str = "", challenger_model_key: str = None) -> GameResult:
        """Run a single game with enhanced debugging"""
        
        if game_name not in self.games:
            raise ValueError(f"Unknown game: {game_name}")
        
        game = self.games[game_name]
        call_id = f"{game_name}_{game_instance_id}_{int(time.time())}"
        
        self.logger.info(f"[{call_id}] Starting {game_name} game")
        self.logger.debug(f"[{call_id}] Players: {config.number_of_players}, Rounds: {config.number_of_rounds}")
        
        # Create agents with specific challenger model
        agents = self.create_agents(experiment_config, challenger_model_key)
        
        # Initialize game state
        game_state = {
            'current_round': 1,
            'player_histories': {agent_key: {'actions': [], 'profits': [], 'responses': []} 
                               for agent_key in agents.keys()},
            'market_history': [],
            'call_id': call_id
        }
        
        player_results = []
        total_responses = 0
        failed_responses = 0
        
        try:
            # Run game rounds
            for round_num in range(1, config.number_of_rounds + 1):
                game_state['current_round'] = round_num
                
                self.logger.debug(f"[{call_id}] Round {round_num}/{config.number_of_rounds}")
                
                round_actions = {}
                round_responses = {}
                
                # Get actions from each agent
                for agent_key, agent in agents.items():
                    try:
                        # Create prompt for this agent
                        prompt = game.create_prompt(agent_key, game_state, config)
                        
                        self.logger.debug(f"[{call_id}] Getting action from {agent_key}")
                        
                        # Get response from agent
                        response = agent.generate_response(prompt, experiment_config.include_thinking)
                        round_responses[agent_key] = response
                        total_responses += 1
                        
                        # Track API call statistics
                        self.total_api_calls += 1
                        if response.token_usage:
                            self.total_tokens_used += response.token_usage.get('total_tokens', 0)
                        
                        if response.error:
                            failed_responses += 1
                            self.failed_calls += 1
                            self.logger.error(f"[{call_id}] {agent_key} failed: {response.error}")
                            # Use default action for failed responses
                            action = game.get_default_action(agent_key, game_state, config)
                        else:
                            # Parse action from response
                            action = game.parse_action(response.final_response, agent_key, game_state, config)
                            
                            if self.verbose_logging:
                                self.logger.debug(f"[{call_id}] {agent_key} response: {response.final_response[:100]}...")
                                if response.thinking_output:
                                    self.logger.debug(f"[{call_id}] {agent_key} thinking: {response.thinking_output[:100]}...")
                        
                        round_actions[agent_key] = action
                        
                        # Store response in game state for debugging
                        game_state['player_histories'][agent_key]['responses'].append({
                            'round': round_num,
                            'response': response.final_response,
                            'thinking': response.thinking_output,
                            'token_usage': response.token_usage,
                            'response_time': response.response_time,
                            'error': response.error
                        })
                        
                    except Exception as e:
                        error_msg = f"Exception in agent {agent_key}: {str(e)}"
                        self.logger.error(f"[{call_id}] {error_msg}")
                        self.logger.error(f"[{call_id}] Traceback: {traceback.format_exc()}")
                        
                        failed_responses += 1
                        self.failed_calls += 1
                        
                        # Use default action
                        round_actions[agent_key] = game.get_default_action(agent_key, game_state, config)
                
                # Update game state with actions
                game_state = game.update_game_state(game_state, round_actions, config)
                
                # Log round results
                if self.verbose_logging:
                    self.logger.debug(f"[{call_id}] Round {round_num} actions: {round_actions}")
            
            # Calculate final results
            player_results = game.calculate_final_results(game_state, config)
            market_price = game_state.get('final_market_price')
            total_industry_profit = sum(pr.profit for pr in player_results)
            
            # Separate challenger and defender results for enhanced metrics
            challenger_result = None
            defender_results = []
            
            for pr in player_results:
                if pr.player_id == "challenger":
                    challenger_result = pr
                else:
                    defender_results.append(pr)
            
            # Calculate defender summary metrics
            if defender_results:
                total_defender_profit = sum(pr.profit for pr in defender_results)
                defender_wins = sum(1 for pr in defender_results if pr.win)
                avg_defender_profit = total_defender_profit / len(defender_results)
            else:
                total_defender_profit = 0
                defender_wins = 0
                avg_defender_profit = 0
            
            # Enhanced logging of results
            self.logger.info(f"[{call_id}] Game completed successfully")
            self.logger.info(f"[{call_id}] API calls: {total_responses} total, {failed_responses} failed")
            self.logger.info(f"[{call_id}] Total industry profit: {total_industry_profit:.2f}")
            
            if self.verbose_logging:
                if challenger_result:
                    self.logger.debug(f"[{call_id}] Challenger: profit={challenger_result.profit:.2f}, win={challenger_result.win}")
                self.logger.debug(f"[{call_id}] Defenders: total_profit={total_defender_profit:.2f}, wins={defender_wins}/{len(defender_results)}")
            
            return GameResult(
                game_name=game_name,
                config=config,
                players=player_results,
                total_industry_profit=total_industry_profit,
                experiment_config=experiment_config,
                market_price=market_price,
                challenger_model_key=challenger_model_key,
                additional_metrics={
                    'total_api_calls': total_responses,
                    'failed_api_calls': failed_responses,
                    'success_rate': (total_responses - failed_responses) / total_responses if total_responses > 0 else 0,
                    'average_response_time': sum(resp['response_time'] for resp_list in game_state['player_histories'].values() 
                                               for resp in resp_list['responses']) / total_responses if total_responses > 0 else 0,
                    # Add challenger/defender summary
                    'challenger_profit': challenger_result.profit if challenger_result else 0,
                    'challenger_win': challenger_result.win if challenger_result else False,
                    'defender_count': len(defender_results),
                    'total_defender_profit': total_defender_profit,
                    'average_defender_profit': avg_defender_profit,
                    'defender_wins': defender_wins
                }
            )
            
        except Exception as e:
            error_msg = f"Game execution failed: {str(e)}"
            self.logger.error(f"[{call_id}] {error_msg}")
            self.logger.error(f"[{call_id}] Traceback: {traceback.format_exc()}")
            
            # Return failed game result
            return GameResult(
                game_name=game_name,
                config=config, 
                players=[],
                total_industry_profit=0.0,
                experiment_config=experiment_config,
                challenger_model_key=challenger_model_key,
                additional_metrics={
                    'error': error_msg,
                    'total_api_calls': total_responses,
                    'failed_api_calls': failed_responses
                }
            )
    
    def run_experiment(self, experiment_config: ExperimentConfig) -> Dict[str, Any]:
        """Run a complete experiment with enhanced tracking - runs num_games for each challenger model"""
        
        self.experiment_start_time = time.time()
        experiment_id = f"{experiment_config.game_name}_{int(time.time())}"
        
        self.logger.info(f"=== STARTING EXPERIMENT {experiment_id} ===")
        self.logger.info(f"Game: {experiment_config.game_name}")
        self.logger.info(f"Defender: {experiment_config.defender_model_key}")
        self.logger.info(f"Challengers: {experiment_config.challenger_model_keys}")
        self.logger.info(f"Games per challenger: {experiment_config.num_games}")
        
        total_expected_games = len(experiment_config.challenger_model_keys) * experiment_config.num_games
        self.logger.info(f"Total expected games: {total_expected_games}")
        
        # Reset statistics
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.failed_calls = 0
        
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
            self.logger.info(f"=== Running games for challenger {challenger_idx + 1}/{len(experiment_config.challenger_model_keys)}: {challenger_model_key} ===")
            
            # Run multiple game instances for this challenger
            for game_idx in range(experiment_config.num_games):
                total_game_count += 1
                game_instance_id = f"{experiment_id}_challenger_{challenger_idx}_game_{game_idx}"
                
                self.logger.info(f"Running game {total_game_count}/{total_expected_games} (Challenger: {challenger_model_key}, Game {game_idx + 1}/{experiment_config.num_games})")
                
                try:
                    game_result = self.run_single_game(
                        experiment_config.game_name,
                        experiment_config, 
                        game_config,
                        game_instance_id,
                        challenger_model_key
                    )
                    
                    # Create simplified result
                    simplified_result = self.create_simplified_game_result(
                        game_result, 
                        total_game_count, 
                        total_expected_games,
                        challenger_model_key
                    )
                    
                    experiment_results['game_results'].append(simplified_result)
                    
                    if not game_result.additional_metrics.get('error'):
                        successful_games += 1
                    
                    # Rate limiting
                    api_config = get_api_config()
                    delay = api_config.get('rate_limit_delay', 1)
                    time.sleep(delay)
                    
                except Exception as e:
                    error_msg = f"Game {total_game_count} failed: {str(e)}"
                    self.logger.error(error_msg)
                    experiment_results['game_results'].append({
                        'game_number': f"{total_game_count}/{total_expected_games}",
                        'challenger_model': challenger_model_key,
                        'error': error_msg,
                        'challenger': None,
                        'defenders': [],
                        'summary': {
                            'total_industry_profit': 0,
                            'challenger_profit': 0,
                            'challenger_win': False,
                            'defender_count': 0,
                            'total_defender_profit': 0,
                            'average_defender_profit': 0,
                            'defender_wins': 0
                        },
                        'performance': {
                            'api_calls': 0,
                            'success_rate': 0,
                            'avg_response_time': 0
                        }
                    })
        
        # Calculate summary metrics
        experiment_duration = time.time() - self.experiment_start_time
        
        summary_metrics = {
            'total_duration': experiment_duration,
            'successful_games': successful_games,
            'failed_games': total_expected_games - successful_games,
            'success_rate': successful_games / total_expected_games if total_expected_games > 0 else 0,
            'total_api_calls': self.total_api_calls,
            'failed_api_calls': self.failed_calls,
            'api_success_rate': (self.total_api_calls - self.failed_calls) / self.total_api_calls if self.total_api_calls > 0 else 0,
            'total_tokens_used': self.total_tokens_used,
            'average_tokens_per_call': self.total_tokens_used / self.total_api_calls if self.total_api_calls > 0 else 0,
            'games_per_minute': successful_games / (experiment_duration / 60) if experiment_duration > 0 else 0
        }
        
        experiment_results['summary_metrics'] = summary_metrics
        experiment_results['end_time'] = datetime.now().isoformat()
        
        # Log experiment summary
        self.logger.info(f"=== EXPERIMENT {experiment_id} COMPLETED ===")
        self.logger.info(f"Duration: {experiment_duration:.1f}s")
        self.logger.info(f"Games: {successful_games}/{total_expected_games} successful")
        self.logger.info(f"API calls: {self.total_api_calls} total, {self.failed_calls} failed")
        self.logger.info(f"Tokens used: {self.total_tokens_used:,}")
        
        return experiment_results
    
    def run_tournament(self, game_names: List[str], experiment_configs: List[ExperimentConfig]) -> Dict[str, Any]:
        """Run tournament across multiple games and configurations"""
        
        tournament_id = f"tournament_{int(time.time())}"
        self.logger.info(f"=== STARTING TOURNAMENT {tournament_id} ===")
        
        tournament_results = {
            'tournament_id': tournament_id,
            'start_time': datetime.now().isoformat(),
            'experiment_results': {},
            'tournament_summary': {}
        }
        
        total_experiments = len(game_names) * len(experiment_configs)
        completed_experiments = 0
        
        for game_name in game_names:
            if game_name not in self.games:
                self.logger.warning(f"Skipping unknown game: {game_name}")
                continue
                
            for experiment_config in experiment_configs:
                experiment_key = f"{game_name}_{experiment_config.game_name}_{experiment_config.defender_model_key}"
                
                self.logger.info(f"Running experiment {completed_experiments + 1}/{total_experiments}: {experiment_key}")
                
                # Update experiment config with game name
                updated_config = ExperimentConfig(
                    defender_model_key=experiment_config.defender_model_key,
                    challenger_model_keys=experiment_config.challenger_model_keys,
                    game_name=game_name,  # Use actual game name
                    num_players=experiment_config.num_players,
                    num_rounds=experiment_config.num_rounds,
                    num_games=experiment_config.num_games,
                    include_thinking=experiment_config.include_thinking
                )
                
                try:
                    experiment_result = self.run_experiment(updated_config)
                    tournament_results['experiment_results'][experiment_key] = experiment_result
                    completed_experiments += 1
                    
                except Exception as e:
                    error_msg = f"Experiment {experiment_key} failed: {str(e)}"
                    self.logger.error(error_msg)
                    tournament_results['experiment_results'][experiment_key] = {
                        'error': error_msg,
                        'experiment_config': asdict(updated_config)
                    }
        
        # Calculate tournament summary
        tournament_duration = time.time() - self.experiment_start_time if self.experiment_start_time else 0
        tournament_results['tournament_summary'] = {
            'total_experiments': completed_experiments,
            'total_duration': tournament_duration,
            'experiments_per_hour': completed_experiments / (tournament_duration / 3600) if tournament_duration > 0 else 0
        }
        tournament_results['end_time'] = datetime.now().isoformat()
        
        self.logger.info(f"=== TOURNAMENT {tournament_id} COMPLETED ===")
        self.logger.info(f"Experiments: {completed_experiments}/{total_experiments}")
        self.logger.info(f"Duration: {tournament_duration:.1f}s")
        
        return tournament_results
    
    def export_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """Export results with enhanced debugging information"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Export main results
            results_file = output_path / f"experiment_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results exported to {results_file}")
            
            # Export summary CSV for easy analysis
            if 'experiment_results' in results:
                summary_data = []
                
                for exp_key, exp_data in results['experiment_results'].items():
                    if 'summary_metrics' in exp_data:
                        summary_row = {
                            'experiment': exp_key,
                            'game_name': exp_data.get('experiment_config', {}).get('game_name', ''),
                            'defender_model': exp_data.get('experiment_config', {}).get('defender_model_key', ''),
                            'num_challengers': len(exp_data.get('experiment_config', {}).get('challenger_model_keys', [])),
                            'successful_games': exp_data['summary_metrics'].get('successful_games', 0),
                            'total_games': exp_data.get('experiment_config', {}).get('num_games', 0),
                            'success_rate': exp_data['summary_metrics'].get('success_rate', 0),
                            'total_api_calls': exp_data['summary_metrics'].get('total_api_calls', 0),
                            'api_success_rate': exp_data['summary_metrics'].get('api_success_rate', 0),
                            'total_tokens': exp_data['summary_metrics'].get('total_tokens_used', 0),
                            'duration_seconds': exp_data['summary_metrics'].get('total_duration', 0)
                        }
                        summary_data.append(summary_row)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_file = output_path / f"experiment_summary_{timestamp}.csv"
                    summary_df.to_csv(summary_file, index=False)
                    self.logger.info(f"Summary exported to {summary_file}")
            
            # Export debugging information
            debug_info = {
                'total_api_calls_session': self.total_api_calls,
                'total_tokens_session': self.total_tokens_used,
                'failed_calls_session': self.failed_calls,
                'session_success_rate': (self.total_api_calls - self.failed_calls) / self.total_api_calls if self.total_api_calls > 0 else 0,
                'available_models': [model.display_name for model in get_available_models().values()],
                'export_timestamp': datetime.now().isoformat()
            }
            
            debug_file = output_path / f"debug_info_{timestamp}.json"
            with open(debug_file, 'w') as f:
                json.dump(debug_info, f, indent=2)
            
            self.logger.info(f"Debug info exported to {debug_file}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    def create_simplified_game_result(self, game_result, game_number, total_games, challenger_model):
        """Create a simplified game result with focus on challenger and summary."""
        try:
            # Extract challenger data (with reasoning if available)
            challenger_data = None
            if hasattr(game_result, 'players') and game_result.players:
                for player in game_result.players:
                    if hasattr(player, 'player_role') and player.player_role == 'challenger':
                        # Extract reasoning from actions if available
                        reasoning = None
                        thinking = None
                        if player.actions:
                            reasoning = player.actions[0].get('reasoning', 'No reasoning available')
                            thinking = player.actions[0].get('thinking', None)
                        
                        challenger_data = {
                            "profit": player.profit,
                            "price": player.actions[0].get('action_data', {}).get('price') if player.actions else None,
                            "win": player.win,
                            "reasoning": reasoning,
                            "thinking": thinking
                        }
                        break
            
            # Extract simplified defender data (no reasoning)
            defenders = []
            if hasattr(game_result, 'players') and game_result.players:
                for player in game_result.players:
                    if hasattr(player, 'player_role') and player.player_role == 'defender':
                        defenders.append({
                            "profit": player.profit,
                            "price": player.actions[0].get('action_data', {}).get('price') if player.actions else None,
                            "win": player.win
                        })
            
            # Calculate summary metrics
            challenger_profit = challenger_data['profit'] if challenger_data else 0
            total_defender_profit = sum(d['profit'] for d in defenders)
            defender_wins = sum(1 for d in defenders if d['win'])
            
            simplified_result = {
                "game_number": f"{game_number}/{total_games}",
                "challenger_model": challenger_model,
                "challenger": challenger_data,
                "defenders": defenders,
                "summary": {
                    "total_industry_profit": game_result.total_industry_profit,
                    "challenger_profit": challenger_profit,
                    "challenger_win": challenger_data['win'] if challenger_data else False,
                    "defender_count": len(defenders),
                    "total_defender_profit": total_defender_profit,
                    "average_defender_profit": total_defender_profit / len(defenders) if defenders else 0,
                    "defender_wins": defender_wins
                },
                "performance": {
                    "api_calls": game_result.additional_metrics.get('total_api_calls', 0),
                    "success_rate": 1.0,
                    "avg_response_time": game_result.additional_metrics.get('average_response_time', 0)
                }
            }
            
            return simplified_result
            
        except Exception as e:
            # Return error format if something goes wrong
            return {
                "game_number": f"{game_number}/{total_games}",
                "challenger_model": challenger_model,
                "error": f"Game {game_number} failed: {str(e)}",
                "challenger": None,
                "defenders": [],
                "summary": {
                    "total_industry_profit": 0,
                    "challenger_profit": 0,
                    "challenger_win": False,
                    "defender_count": 0,
                    "total_defender_profit": 0,
                    "average_defender_profit": 0,
                    "defender_wins": 0
                },
                "performance": {
                    "api_calls": 0,
                    "success_rate": 0,
                    "avg_response_time": 0
                }
            }

def setup_competition_logging():
    """Setup logging based on config.json settings"""
    logging_config = get_logging_config()
    
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file', 'competition.log')
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_level = getattr(logging, logging_config.get('console_level', 'INFO').upper())
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.getLogger(__name__).info("Competition logging configured from config.json")

# Initialize logging on import
setup_competition_logging()

# Convenience functions for easy usage
def quick_model_test(model_key: str, prompt: str) -> LLMResponse:
    """Quick test of a single model"""
    model_config = get_model_config(model_key)
    agent = GeminiLLMAgent(model_config, debug=True)
    return agent.generate_response(prompt)

def compare_models(model_keys: List[str], prompt: str) -> Dict[str, LLMResponse]:
    """Compare multiple models on the same prompt"""
    logger = logging.getLogger(__name__)
    logger.info(f"Comparing {len(model_keys)} models")
    
    results = {}
    for model_key in model_keys:
        try:
            model_config = get_model_config(model_key)
            agent = GeminiLLMAgent(model_config, debug=True)
            response = agent.generate_response(prompt)
            results[model_key] = response
            
            status = "SUCCESS" if not response.error else f"FAILED: {response.error}"
            logger.info(f"{model_config.display_name}: {status} ({response.response_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Error testing {model_key}: {e}")
            results[model_key] = LLMResponse(
                final_response="",
                error=str(e),
                model_config=get_model_config(model_key),
                call_id=f"compare_{model_key}"
            )
    
    return results