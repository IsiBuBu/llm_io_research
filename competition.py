"""
Competition Engine - Core orchestration for LLM game theory experiments
Updated for Gemini-only experiments with thinking support and error response filtering
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from config import (
    GameConfig, get_game_config, get_experiment_config, get_challenger_models,
    get_defender_model, get_simulation_count, is_thinking_enabled, 
    get_model_display_name, get_all_game_configs
)
from games import create_game
from baselines.random_players import RandomPlayer
from metrics.metric_utils import GameResult, create_game_result
from agents import create_agent, BaseLLMAgent


@dataclass
class CompetitionResult:
    """Result of a single competition (multiple simulations)"""
    game_name: str
    experiment_type: str
    condition_name: str
    challenger_model: str
    challenger_display_name: str
    defender_model: str
    defender_display_name: str
    challenger_thinking_enabled: bool
    defender_thinking_enabled: bool
    simulation_results: List[GameResult]
    competition_metadata: Dict[str, Any]
    start_time: str
    end_time: str
    total_duration: float


class Competition:
    """
    Core competition engine that orchestrates game experiments between Gemini models
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.Competition")
        
        # Initialize game engines
        self.games = {
            'salop': create_game('salop'),
            'green_porter': create_game('green_porter'),
            'spulber': create_game('spulber'),
            'athey_bagwell': create_game('athey_bagwell')
        }
        
        # Load experiment configuration
        self.experiment_config = get_experiment_config()
        
        # Log system overview
        self._log_system_overview()
    
    def _log_system_overview(self):
        """Log overview of the experimental system"""
        challenger_models = get_challenger_models()
        defender_model = get_defender_model()
        
        self.logger.info("=" * 60)
        self.logger.info("LLM GAME THEORY COMPETITION SYSTEM")
        self.logger.info("=" * 60)
        self.logger.info(f"Challenger models: {len(challenger_models)}")
        
        for model in challenger_models:
            thinking_status = "ON" if is_thinking_enabled(model) else "OFF"
            display_name = get_model_display_name(model)
            self.logger.info(f"  • {display_name} (Thinking: {thinking_status})")
        
        defender_thinking = "ON" if is_thinking_enabled(defender_model) else "OFF"
        defender_display = get_model_display_name(defender_model)
        self.logger.info(f"Defender: {defender_display} (Thinking: {defender_thinking})")
        self.logger.info("=" * 60)

    def _clean_error_responses(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out error responses from actions before saving to JSON.
        
        This prevents cluttering result files with API error messages and quota exceeded errors.
        """
        cleaned_actions = {}
        
        for player_id, action in actions.items():
            cleaned_action = action.copy()
            
            # Check if raw_response contains an error
            if 'raw_response' in cleaned_action:
                raw_response = cleaned_action['raw_response']
                
                # Check for error indicators in the raw response
                if (isinstance(raw_response, str) and 
                    (raw_response.startswith('{"error":') or 
                     'API call failed:' in raw_response or
                     'RESOURCE_EXHAUSTED' in raw_response or
                     'quota' in raw_response.lower() or
                     'rate limit' in raw_response.lower() or
                     'exceeded your current quota' in raw_response)):
                    
                    # Replace error response with clean indicator
                    cleaned_action['raw_response'] = "[LLM Error Response Filtered]"
                    cleaned_action['llm_error'] = True
                    
                    self.logger.debug(f"Filtered error response for {player_id}")
                else:
                    # Keep successful responses as-is
                    cleaned_action['llm_error'] = False
            
            cleaned_actions[player_id] = cleaned_action
        
        return cleaned_actions

    async def _get_action_from_llm_agent(self, agent, player_id: str, prompt: str, 
                          game_engine, call_id: str) -> Dict[str, Any]:
        """Get action from LLM agent with parsing and error handling"""
        
        # Get raw response from LLM
        try:
            raw_response = await agent.get_action(prompt, call_id)
        except Exception as e:
            self.logger.error(f"[{call_id}] Agent {player_id} API call failed: {e}")
            raw_response = f'{{"error": "API call failed: {str(e)}"}}'
        
        # Parse response using game-specific parser
        parsed_action = game_engine.parse_llm_response(raw_response, player_id, call_id)
        
        if parsed_action is None:
            # Use default action if parsing fails
            self.logger.warning(f"[{call_id}] Parsing failed for {player_id}, using default action")
            parsed_action = game_engine.get_default_action(player_id, {}, 
                                                         get_game_config(game_engine.game_name, 'baseline'))
        
        # Add metadata
        parsed_action.update({
            'player_id': player_id,
            'raw_response': raw_response,
            'parsing_success': parsed_action is not None
        })
        
        return parsed_action
    
    def _summarize_action(self, action: Dict[str, Any]) -> str:
        """Create brief summary of action for logging"""
        if 'price' in action:
            return f"Price: {action['price']:.2f}"
        elif 'quantity' in action:
            return f"Quantity: {action['quantity']:.1f}"
        elif 'report' in action:
            return f"Report: {action['report']}"
        elif 'error' in action:
            return f"Error: {action['error']}"
        else:
            return f"Action: {str(action)[:50]}..."

    async def run_competition(self, game_name: str, experiment_type: str, condition_name: str,
                            challenger_model: str, defender_model: Optional[str] = None) -> CompetitionResult:
        """
        Run complete competition between challenger and defender models
        
        Args:
            game_name: Name of game to play ('salop', 'green_porter', 'spulber', 'athey_bagwell')
            experiment_type: Type of experiment ('baseline', 'structural_variations', etc.)
            condition_name: Specific condition name
            challenger_model: Model name for challenger
            defender_model: Model name for defender (defaults to config)
            
        Returns:
            Complete competition result with all simulations
        """
        
        if defender_model is None:
            defender_model = get_defender_model()
        
        # Get display names for logging
        challenger_display = get_model_display_name(challenger_model)
        defender_display = get_model_display_name(defender_model)
        
        # Check thinking configuration
        challenger_thinking = is_thinking_enabled(challenger_model)
        defender_thinking = is_thinking_enabled(defender_model)
        
        self.logger.info("=" * 80)
        self.logger.info(f"COMPETITION: {game_name.upper()} - {experiment_type} - {condition_name}")
        self.logger.info("=" * 80)
        self.logger.info(f"Challenger: {challenger_display} (Thinking: {'ON' if challenger_thinking else 'OFF'})")
        self.logger.info(f"Defender: {defender_display} (Thinking: {'ON' if defender_thinking else 'OFF'})")
        
        # Load game configuration
        game_config = get_game_config(game_name, condition_name)
        game_engine = self.games[game_name]
        
        # Get simulation count from config
        num_simulations = get_simulation_count(experiment_type)
        self.logger.info(f"Simulations: {num_simulations}")
        
        start_time = datetime.now()
        simulation_results = []
        successful_sims = 0
        failed_sims = 0
        
        # Run simulations
        for sim_id in range(num_simulations):
            try:
                call_id = f"{game_name}_{condition_name}_{challenger_model}_sim{sim_id}"
                
                self.logger.debug(f"[{call_id}] Starting simulation {sim_id + 1}/{num_simulations}")
                
                # Run single game simulation
                result = await self._run_single_simulation(
                    sim_id, game_engine, game_config, challenger_model, defender_model, call_id
                )
                
                simulation_results.append(result)
                successful_sims += 1
                
                # Progress logging
                if (sim_id + 1) % 10 == 0 or (sim_id + 1) == num_simulations:
                    progress = ((sim_id + 1) / num_simulations) * 100
                    self.logger.info(f"Progress: {sim_id + 1}/{num_simulations} ({progress:.1f}%) - "
                                   f"Success: {successful_sims}, Failed: {failed_sims}")
                    
            except Exception as e:
                failed_sims += 1
                self.logger.error(f"Simulation {sim_id} failed: {e}")
                # Continue with other simulations
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create competition result
        competition_result = CompetitionResult(
            game_name=game_name,
            experiment_type=experiment_type,
            condition_name=condition_name,
            challenger_model=challenger_model,
            challenger_display_name=challenger_display,
            defender_model=defender_model,
            defender_display_name=defender_display,
            challenger_thinking_enabled=challenger_thinking,
            defender_thinking_enabled=defender_thinking,
            simulation_results=simulation_results,
            competition_metadata={
                'num_simulations_planned': num_simulations,
                'num_simulations_successful': successful_sims,
                'num_simulations_failed': failed_sims,
                'success_rate': successful_sims / num_simulations if num_simulations > 0 else 0,
                'game_config': game_config.to_dict()
            },
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=duration
        )
        
        # Save results
        await self._save_competition_result(competition_result)
        
        # Log completion
        success_rate = (successful_sims / num_simulations) * 100 if num_simulations > 0 else 0
        self.logger.info("=" * 60)
        self.logger.info("COMPETITION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Results: {successful_sims}/{num_simulations} successful ({success_rate:.1f}%)")
        self.logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        if failed_sims > 0:
            self.logger.warning(f"Failed simulations: {failed_sims}")
        
        return competition_result

    async def _run_single_simulation(self, sim_id: int, game_engine, game_config: GameConfig,
                                   challenger_model: str, defender_model: str, call_id: str) -> GameResult:
        """Run a single game simulation"""
        
        # Create agents
        challenger_agent = create_agent(challenger_model, "challenger")
        defender_agents = {}
        
        # Create defender agents (multiple defenders for games like Salop)
        num_players = game_config.constants.get('number_of_players', 2)
        players = ['challenger']
        
        for i in range(1, num_players):
            defender_id = f"defender_{i}" if num_players > 2 else "defender"
            defender_agents[defender_id] = create_agent(defender_model, defender_id)
            players.append(defender_id)
        
        # Initialize game state
        game_state = game_engine.initialize_game_state(game_config)
        
        # Run game rounds
        if hasattr(game_engine, 'run_dynamic_game'):
            # Dynamic games (Green-Porter, Athey-Bagwell)
            actions, payoffs, game_data, round_data = await game_engine.run_dynamic_game(
                challenger_agent, defender_agents, game_config, call_id
            )
        else:
            # Static games (Salop, Spulber)
            actions, payoffs, game_data = await self._run_static_game(
                game_engine, challenger_agent, defender_agents, game_config, game_state, call_id
            )
            round_data = []
        
        # Create game result
        result = create_game_result(
            simulation_id=sim_id,
            game_name=game_config.game_name,
            experiment_type=game_config.experiment_type,
            condition_name=game_config.condition_name,
            players=players,
            actions=actions,
            payoffs=payoffs,
            game_data=game_data
        )
        
        # Add round data for dynamic games
        if round_data:
            result.round_data = round_data
        
        self.logger.debug(f"[{call_id}] Simulation completed - Payoffs: {payoffs}")
        
        return result

    async def _run_static_game(self, game_engine, challenger_agent, defender_agents: Dict,
                             game_config: GameConfig, game_state: Dict, call_id: str) -> Tuple[Dict, Dict, Dict]:
        """Run a static (single-round) game"""
        
        all_agents = {'challenger': challenger_agent}
        all_agents.update(defender_agents)
        
        # Get actions from all players
        actions = {}
        
        for player_id, agent in all_agents.items():
            # Generate prompt
            prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
            
            # Get action from agent
            action = await self._get_action_from_llm_agent(agent, player_id, prompt, game_engine, call_id)
            actions[player_id] = action
            
            # Log action summary
            action_summary = self._summarize_action(action)
            self.logger.debug(f"[{call_id}] {player_id}: {action_summary}")
        
        # Calculate payoffs
        payoffs = game_engine.calculate_payoffs(actions, game_config, game_state)
        
        # Get game data for analysis
        game_data = game_engine.get_game_data_for_logging(actions, payoffs, game_config, game_state)
        
        return actions, payoffs, game_data

    async def _save_competition_result(self, result: CompetitionResult):
        """Save competition result to disk with error response filtering"""
        
        # Create directory structure
        output_path = (self.output_dir / result.game_name / result.challenger_model)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete result
        filename = f"{result.condition_name}_competition_result.json"
        
        # Convert to serializable format with error filtering
        cleaned_simulation_results = []
        for sim in result.simulation_results:
            cleaned_sim = {
                'simulation_id': sim.simulation_id,
                'players': sim.players,
                'actions': self._clean_error_responses(sim.actions),  # Filter errors here
                'payoffs': sim.payoffs,
                'game_data': sim.game_data,
                'round_data': getattr(sim, 'round_data', [])
            }
            
            # Also clean any actions within game_data if they exist
            if 'game_data' in cleaned_sim and isinstance(cleaned_sim['game_data'], dict):
                if 'actions' in cleaned_sim['game_data']:
                    cleaned_sim['game_data']['actions'] = self._clean_error_responses(
                        cleaned_sim['game_data']['actions']
                    )
            
            cleaned_simulation_results.append(cleaned_sim)
        
        # Convert to serializable format
        result_dict = {
            'game_name': result.game_name,
            'experiment_type': result.experiment_type,
            'condition_name': result.condition_name,
            'challenger_model': result.challenger_model,
            'challenger_display_name': result.challenger_display_name,
            'defender_model': result.defender_model,
            'defender_display_name': result.defender_display_name,
            'challenger_thinking_enabled': result.challenger_thinking_enabled,
            'defender_thinking_enabled': result.defender_thinking_enabled,
            'competition_metadata': result.competition_metadata,
            'start_time': result.start_time,
            'end_time': result.end_time,
            'total_duration': result.total_duration,
            'simulation_results': cleaned_simulation_results
        }
        
        with open(output_path / filename, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Results saved (errors filtered): {output_path / filename}")
    
    async def run_batch_competitions(self, competitions: List[Dict[str, str]]) -> List[CompetitionResult]:
        """Run multiple competitions in batch with enhanced progress tracking"""
        
        results = []
        total = len(competitions)
        
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING BATCH COMPETITIONS: {total} total")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        for i, comp_config in enumerate(competitions):
            comp_start_time = time.time()
            
            # Enhanced logging
            challenger_display = get_model_display_name(comp_config['challenger_model'])
            self.logger.info(f"[{i+1}/{total}] {comp_config['game_name']} - "
                           f"{comp_config['experiment_type']} - {comp_config['condition_name']}")
            self.logger.info(f"  Challenger: {challenger_display}")
            
            try:
                result = await self.run_competition(**comp_config)
                results.append(result)
                
                comp_duration = time.time() - comp_start_time
                success_rate = result.competition_metadata.get('success_rate', 0) * 100
                
                self.logger.info(f"  ✅ Completed in {comp_duration:.1f}s (Success rate: {success_rate:.1f}%)")
                
            except Exception as e:
                comp_duration = time.time() - comp_start_time
                self.logger.error(f"  ❌ Failed in {comp_duration:.1f}s: {e}")
        
        total_duration = time.time() - start_time
        
        self.logger.info("=" * 80)
        self.logger.info("BATCH COMPETITIONS COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Success: {len(results)}/{total} competitions")
        self.logger.info(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        
        return results

    async def run_all_competitions(self) -> bool:
        """Run all competitions across all games and conditions"""
        
        try:
            # Get challenger models and defender model from config
            challenger_models = get_challenger_models()
            defender_model = get_defender_model()
            
            self.logger.info(f"Challenger models: {len(challenger_models)}")
            self.logger.info(f"Defender model: {defender_model}")
            
            # Generate all competition configurations
            competitions = []
            
            for game_name in ['salop', 'spulber', 'green_porter', 'athey_bagwell']:
                # Get all conditions for this game
                game_configs = get_all_game_configs(game_name)
                
                for game_config in game_configs:
                    for challenger in challenger_models:
                        competitions.append({
                            'game_name': game_name,
                            'experiment_type': game_config.experiment_type,
                            'condition_name': game_config.condition_name,
                            'challenger_model': challenger,
                            'defender_model': defender_model
                        })
            
            self.logger.info(f"Total competitions to run: {len(competitions)}")
            
            # Run all competitions using existing batch method
            results = await self.run_batch_competitions(competitions)
            
            # Check if all competitions were successful
            success_count = len([r for r in results if r.competition_metadata.get('success_rate', 0) > 0])
            
            self.logger.info(f"Completed: {success_count}/{len(competitions)} competitions successful")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to run all competitions: {e}")
            return False


# Convenience functions
async def run_single_competition(game_name: str, experiment_type: str, condition_name: str,
                               challenger_model: str, defender_model: Optional[str] = None) -> CompetitionResult:
    """Run a single competition with specified parameters"""
    competition = Competition()
    return await competition.run_competition(
        game_name, experiment_type, condition_name, challenger_model, defender_model
    )


async def run_full_experimental_suite(challenger_models: Optional[List[str]] = None, 
                                    defender_model: Optional[str] = None) -> List[CompetitionResult]:
    """Run complete experimental suite for all games and conditions"""
    
    if challenger_models is None:
        challenger_models = get_challenger_models()
    
    if defender_model is None:
        defender_model = get_defender_model()
    
    competition = Competition()
    
    # Generate all competition configurations
    competitions = []
    
    for game_name in ['salop', 'green_porter', 'spulber', 'athey_bagwell']:
        # Get all conditions for this game
        game_configs = get_all_game_configs(game_name)
        
        for game_config in game_configs:
            for challenger in challenger_models:
                competitions.append({
                    'game_name': game_name,
                    'experiment_type': game_config.experiment_type,
                    'condition_name': game_config.condition_name,
                    'challenger_model': challenger,
                    'defender_model': defender_model
                })
    
    return await competition.run_batch_competitions(competitions)