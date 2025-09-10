"""
Competition Engine - Core orchestration for LLM game theory experiments
Updated for Gemini-only experiments with thinking support and error response filtering
NOW SUPPORTS MOCK MODE for testing workflow and metrics
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
    Core competition engine that orchestrates game experiments between LLM models
    NOW SUPPORTS MOCK MODE for testing workflow without API calls
    """
    
    def __init__(self, challenger_models: list, defender_model: str, mock_mode: bool = False, output_dir: str = "results"):
        """
        Initialize competition system
        
        Args:
            challenger_models: List of challenger model names
            defender_model: Name of defender model
            mock_mode: Whether to use mock agents instead of real ones
            output_dir: Directory to save results
        """
        self.challenger_models = challenger_models
        self.defender_model = defender_model
        self.mock_mode = mock_mode
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
        """Log competition system setup"""
        self.logger.info("=" * 60)
        self.logger.info("LLM GAME THEORY COMPETITION SYSTEM")
        self.logger.info("=" * 60)
        
        if self.mock_mode:
            self.logger.info("🎭 MOCK MODE - Using simulated agents")
        
        self.logger.info(f"Challenger models: {len(self.challenger_models)}")
        for model in self.challenger_models:
            thinking_status = "ON" if is_thinking_enabled(model) else "OFF"
            self.logger.info(f"  • {get_model_display_name(model)} (Thinking: {thinking_status})")
        
        thinking_status = "ON" if is_thinking_enabled(self.defender_model) else "OFF"
        self.logger.info(f"Defender: {get_model_display_name(self.defender_model)} (Thinking: {thinking_status})")
        self.logger.info("=" * 60)

    async def _create_agents(self, challenger_model: str, defender_model: str, 
                           num_players: int, call_id: str) -> Dict[str, Any]:
        """Create agents for a single simulation"""
        agents = {}
        
        try:
            # Create challenger agent (always player 1)
            challenger_agent = create_agent(
                challenger_model, 
                "challenger", 
                mock_mode=self.mock_mode  # Pass mock_mode here
            )
            agents["challenger"] = challenger_agent
            
            # Create defender agents for remaining players
            for i in range(2, num_players + 1):
                player_id = f"defender_{i}"
                defender_agent = create_agent(
                    defender_model, 
                    player_id,
                    mock_mode=self.mock_mode  # Pass mock_mode here
                )
                agents[player_id] = defender_agent
                
        except Exception as e:
            self.logger.error(f"[{call_id}] Failed to create agents: {e}")
            raise
            
        return agents

    async def _run_single_simulation(self, simulation_id: int, game_engine, game_config: GameConfig,
                                   challenger_model: str, defender_model: str, call_id: str) -> GameResult:
        """Run a single game simulation"""
        
        # Determine number of players from game configuration
        num_players = game_config.constants.get('num_players', 2)
        
        # Create agents
        agents = await self._create_agents(challenger_model, defender_model, num_players, call_id)
        
        # Separate challenger and defender agents
        challenger_agent = agents["challenger"]
        defender_agents = {k: v for k, v in agents.items() if k != "challenger"}
        
        # Initialize game state
        game_state = game_engine.initialize_game_state(game_config)
        
        # Check if this is a dynamic game (multiple rounds)
        if hasattr(game_engine, 'is_dynamic_game') and game_engine.is_dynamic_game():
            # Run dynamic (multi-round) game
            actions, payoffs, game_data = await self._run_dynamic_game(
                game_engine, game_config, game_state, challenger_agent, defender_agents, call_id
            )
        else:
            # Run static (single-round) game
            actions, payoffs, game_data = await self._run_static_game(
                game_engine, game_config, game_state, challenger_agent, defender_agents, call_id
            )
        
        # Create player list
        players = ["challenger"] + list(defender_agents.keys())
        
        # Create and return game result
        return create_game_result(
            simulation_id=simulation_id,
            players=players,
            actions=actions,
            payoffs=payoffs,
            game_data=game_data
        )

    async def _run_dynamic_game(self, game_engine, game_config: GameConfig, game_state: Dict,
                              challenger_agent: BaseLLMAgent, defender_agents: Dict[str, BaseLLMAgent],
                              call_id: str) -> Tuple[Dict, Dict, Dict]:
        """Run a multi-round dynamic game"""
        
        all_agents = {'challenger': challenger_agent}
        all_agents.update(defender_agents)
        
        # Initialize tracking for all rounds
        all_round_actions = []
        all_round_payoffs = []
        
        num_rounds = game_config.constants.get('num_rounds', 10)
        
        for round_num in range(num_rounds):
            round_call_id = f"{call_id}_r{round_num}"
            
            # Get actions from all players for this round
            round_actions = {}
            
            for player_id, agent in all_agents.items():
                # Generate prompt with current game state and history
                prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
                
                # Get action from agent
                action = await self._get_action_from_llm_agent(agent, player_id, prompt, game_engine, round_call_id)
                round_actions[player_id] = action
                
                # Log action summary
                action_summary = self._summarize_action(action)
                self.logger.debug(f"[{round_call_id}] {player_id}: {action_summary}")
            
            # Update game state with actions
            game_state = game_engine.update_game_state(game_state, round_actions, game_config)
            
            # Calculate round payoffs
            round_payoffs = game_engine.calculate_round_payoffs(round_actions, game_config, game_state)
            
            # Store round data
            all_round_actions.append(round_actions)
            all_round_payoffs.append(round_payoffs)
            
            # Check for early termination conditions
            if game_engine.should_terminate_early(game_state, round_num, game_config):
                self.logger.debug(f"[{call_id}] Game terminated early at round {round_num + 1}")
                break
        
        # Calculate final payoffs
        final_payoffs = game_engine.calculate_final_payoffs(all_round_payoffs, game_config, game_state)
        
        # Aggregate actions (for compatibility with static games)
        final_actions = game_engine.aggregate_round_actions(all_round_actions)
        
        # Get comprehensive game data
        game_data = game_engine.get_game_data_for_logging(final_actions, final_payoffs, game_config, game_state)
        game_data['round_data'] = {
            'round_actions': all_round_actions,
            'round_payoffs': all_round_payoffs,
            'num_rounds_played': len(all_round_actions)
        }
        
        return final_actions, final_payoffs, game_data

    async def _run_static_game(self, game_engine, game_config: GameConfig, game_state: Dict,
                             challenger_agent: BaseLLMAgent, defender_agents: Dict[str, BaseLLMAgent],
                             call_id: str) -> Tuple[Dict, Dict, Dict]:
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

    def _clean_error_responses(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mark error responses in actions to prevent JSON issues"""
        cleaned_actions = {}
        
        for player_id, action in actions.items():
            cleaned_action = dict(action)  # Copy the action
            
            # Check if this action contains an error
            if 'error' in cleaned_action:
                # Mark as error but clean the message
                cleaned_action['error'] = "LLM response error (cleaned for export)"
                cleaned_action['llm_error'] = True
            else:
                # Mark as successful
                cleaned_action['llm_error'] = False
            
            cleaned_actions[player_id] = cleaned_action
        
        return cleaned_actions

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
            defender_model = self.defender_model
        
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
                    self.logger.info(f"Progress: {sim_id + 1}/{num_simulations} simulations completed")
                    
            except Exception as e:
                failed_sims += 1
                self.logger.error(f"Simulation {sim_id} failed: {e}")
                continue
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate success rate
        success_rate = successful_sims / num_simulations if num_simulations > 0 else 0
        
        # Create competition metadata
        competition_metadata = {
            'total_simulations': num_simulations,
            'successful_simulations': successful_sims,
            'failed_simulations': failed_sims,
            'success_rate': success_rate,
            'game_config': game_config.__dict__,
            'mock_mode': self.mock_mode
        }
        
        # Create competition result
        result = CompetitionResult(
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
            competition_metadata=competition_metadata,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration=total_duration
        )
        
        # Save results to disk
        await self._save_competition_result(result)
        
        # Log completion
        self.logger.info("=" * 60)
        self.logger.info("COMPETITION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Results: {successful_sims}/{num_simulations} successful ({success_rate*100:.1f}%)")
        self.logger.info(f"Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        
        if failed_sims > 0:
            self.logger.warning(f"Failed simulations: {failed_sims}")
        
        return result

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
            self.logger.info(f"Challenger models: {len(self.challenger_models)}")
            self.logger.info(f"Defender model: {self.defender_model}")
            
            # Generate all competition configurations
            competitions = []
            
            for game_name in ['salop', 'spulber', 'green_porter', 'athey_bagwell']:
                # Get all conditions for this game
                game_configs = get_all_game_configs(game_name)
                
                for game_config in game_configs:
                    for challenger in self.challenger_models:
                        competitions.append({
                            'game_name': game_name,
                            'experiment_type': game_config.experiment_type,
                            'condition_name': game_config.condition_name,
                            'challenger_model': challenger,
                            'defender_model': self.defender_model
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
                               challenger_model: str, defender_model: Optional[str] = None,
                               mock_mode: bool = False) -> CompetitionResult:
    """Run a single competition with specified parameters"""
    challenger_models = [challenger_model]
    if defender_model is None:
        defender_model = get_defender_model()
    
    competition = Competition(challenger_models, defender_model, mock_mode=mock_mode)
    return await competition.run_competition(
        game_name, experiment_type, condition_name, challenger_model, defender_model
    )


async def run_full_experimental_suite(challenger_models: Optional[List[str]] = None, 
                                    defender_model: Optional[str] = None,
                                    mock_mode: bool = False) -> List[CompetitionResult]:
    """Run complete experimental suite for all games and conditions"""
    
    if challenger_models is None:
        challenger_models = get_challenger_models()
    
    if defender_model is None:
        defender_model = get_defender_model()
    
    competition = Competition(challenger_models, defender_model, mock_mode=mock_mode)
    
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