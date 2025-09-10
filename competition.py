"""
Competition Engine - Core orchestration for LLM game theory experiments
Updated for Gemini-only experiments with thinking support
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
from games.salop_game import SalopGame
from games.green_porter_game import GreenPorterGame
from games.spulber_game import SpulberGame
from games.athey_bagwell_game import AtheyBagwellGame
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
            'salop': SalopGame(),
            'green_porter': GreenPorterGame(),
            'spulber': SpulberGame(),
            'athey_bagwell': AtheyBagwellGame()
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
    
    async def run_competition(self, game_name: str, experiment_type: str, 
                            condition_name: str, challenger_model: str,
                            defender_model: Optional[str] = None) -> CompetitionResult:
        """
        Run complete competition for given configuration
        
        Args:
            game_name: 'salop', 'green_porter', 'spulber', 'athey_bagwell'
            experiment_type: 'baseline', 'structural_variations', 'ablation_studies'
            condition_name: Specific condition within experiment type
            challenger_model: Gemini model for challenger
            defender_model: Gemini model for defenders (defaults to config setting)
        """
        if defender_model is None:
            defender_model = get_defender_model()
        
        start_time = datetime.now()
        
        # Get display names and thinking status
        challenger_display = get_model_display_name(challenger_model)
        defender_display = get_model_display_name(defender_model)
        challenger_thinking = is_thinking_enabled(challenger_model)
        defender_thinking = is_thinking_enabled(defender_model)
        
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING COMPETITION: {game_name.upper()}")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {experiment_type} - {condition_name}")
        self.logger.info(f"Challenger: {challenger_display} (Thinking: {'ON' if challenger_thinking else 'OFF'})")
        self.logger.info(f"Defender: {defender_display} (Thinking: {'ON' if defender_thinking else 'OFF'})")
        
        # Load game configuration
        game_config = get_game_config(game_name, experiment_type, condition_name)
        
        # Determine number of simulations
        num_simulations = get_simulation_count(experiment_type)
        self.logger.info(f"Simulations: {num_simulations}")
        
        # Run all simulations
        simulation_results = []
        successful_sims = 0
        failed_sims = 0
        
        for sim_id in range(num_simulations):
            try:
                result = await self._run_single_simulation(
                    game_name, game_config, challenger_model, defender_model, sim_id
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
    
    async def _run_single_simulation(self, game_name: str, game_config: GameConfig,
                                   challenger_model: str, defender_model: str, 
                                   simulation_id: int) -> GameResult:
        """Run a single game simulation"""
        
        game_engine = self.games[game_name]
        call_id = f"{game_name}_{simulation_id:04d}"
        
        # Create player agents
        num_players = game_config.constants.get('number_of_players', 3)
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]
        
        agents = {}
        
        # Create challenger agent
        try:
            agents['challenger'] = create_agent(challenger_model, 'challenger')
            thinking_status = "ON" if is_thinking_enabled(challenger_model) else "OFF"
            self.logger.debug(f"[{call_id}] Created challenger: {challenger_model} (Thinking: {thinking_status})")
        except Exception as e:
            self.logger.error(f"[{call_id}] Failed to create challenger agent: {e}")
            raise
        
        # Create defender agents
        for i in range(1, num_players):
            defender_id = f'defender_{i}'
            try:
                if defender_model == "random":
                    agents[defender_id] = RandomPlayer(defender_id)
                    self.logger.debug(f"[{call_id}] Created defender: Random baseline")
                else:
                    agents[defender_id] = create_agent(defender_model, defender_id)
                    thinking_status = "ON" if is_thinking_enabled(defender_model) else "OFF"
                    self.logger.debug(f"[{call_id}] Created defender: {defender_model} (Thinking: {thinking_status})")
            except Exception as e:
                self.logger.error(f"[{call_id}] Failed to create defender {defender_id}: {e}")
                raise
        
        # Initialize game state for dynamic games
        game_state = {}
        if hasattr(game_engine, 'initialize_game_state'):
            game_state = game_engine.initialize_game_state(game_config, simulation_id)
            self.logger.debug(f"[{call_id}] Initialized game state for dynamic game")
        
        # Run game (static vs dynamic)
        if game_name in ['salop', 'spulber']:
            # Static games - single round
            actions = await self._run_static_game(game_engine, game_config, agents, game_state, call_id)
            payoffs = game_engine.calculate_payoffs(actions, game_config, game_state)
            
            # Create game result
            game_data = game_engine.get_game_data_for_logging(actions, payoffs, game_config, game_state)
            
            return create_game_result(
                simulation_id=simulation_id,
                game_name=game_name,
                experiment_type=game_config.experiment_type,
                condition_name=game_config.condition_name,
                players=player_ids,
                actions=actions,
                payoffs=payoffs,
                **game_data
            )
        
        else:
            # Dynamic games - multiple rounds
            return await self._run_dynamic_game(
                game_engine, game_config, agents, game_state, simulation_id, call_id
            )
    
    async def _run_static_game(self, game_engine, game_config: GameConfig, 
                             agents: Dict[str, Any], game_state: Dict, 
                             call_id: str) -> Dict[str, Any]:
        """Run static game (single round)"""
        
        actions = {}
        
        # Get actions from all agents
        action_tasks = []
        for player_id, agent in agents.items():
            if isinstance(agent, RandomPlayer):
                # Handle random baseline synchronously
                action = agent.get_action(game_config.game_name, game_state, game_config)
                actions[player_id] = action
                self.logger.debug(f"[{call_id}] {player_id}: Random action")
            else:
                # Handle LLM agents asynchronously
                prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
                task = self._get_agent_action(agent, prompt, player_id, game_engine, call_id)
                action_tasks.append((player_id, task))
        
        # Wait for all LLM agent responses
        for player_id, task in action_tasks:
            action = await task
            actions[player_id] = action
            thinking_status = "ON" if is_thinking_enabled(agent.model_name) else "OFF"
            action_summary = self._summarize_action(action)
            self.logger.debug(f"[{call_id}] {player_id} (Thinking: {thinking_status}): {action_summary}")
        
        return actions
    
    async def _run_dynamic_game(self, game_engine, game_config: GameConfig,
                              agents: Dict[str, Any], initial_game_state: Dict,
                              simulation_id: int, call_id: str) -> GameResult:
        """Run dynamic game (multiple rounds)"""
        
        time_horizon = game_config.constants.get('time_horizon', 50)
        game_state = initial_game_state.copy()
        
        round_history = []
        all_actions = {}
        all_payoffs = {}
        
        self.logger.debug(f"[{call_id}] Starting dynamic game: {time_horizon} rounds")
        
        for round_num in range(1, time_horizon + 1):
            round_call_id = f"{call_id}_r{round_num:03d}"
            
            # Update game state for current round
            game_state['current_round'] = round_num
            
            # Get actions for this round
            round_actions = {}
            action_tasks = []
            
            for player_id, agent in agents.items():
                if isinstance(agent, RandomPlayer):
                    action = agent.get_action(game_config.game_name, game_state, game_config)
                    round_actions[player_id] = action
                else:
                    prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
                    task = self._get_agent_action(agent, prompt, player_id, game_engine, round_call_id)
                    action_tasks.append((player_id, task))
            
            # Wait for LLM responses
            for player_id, task in action_tasks:
                action = await task
                round_actions[player_id] = action
            
            # Calculate round payoffs
            round_payoffs = game_engine.calculate_payoffs(round_actions, game_config, game_state)
            
            # Log round progress (every 10 rounds or final round)
            if round_num % 10 == 0 or round_num == time_horizon:
                challenger_action = self._summarize_action(round_actions.get('challenger', {}))
                challenger_payoff = round_payoffs.get('challenger', 0)
                self.logger.debug(f"[{round_call_id}] Challenger: {challenger_action} → Payoff: {challenger_payoff:.2f}")
            
            # Store round data
            round_data = {
                'round': round_num,
                'actions': round_actions,
                'payoffs': round_payoffs,
                'game_state': game_state.copy()
            }
            round_history.append(round_data)
            
            # Accumulate total actions and payoffs
            for player_id in round_actions:
                if player_id not in all_actions:
                    all_actions[player_id] = []
                    all_payoffs[player_id] = []
                
                all_actions[player_id].append(round_actions[player_id])
                all_payoffs[player_id].append(round_payoffs[player_id])
            
            # Update game state for next round
            game_state = game_engine.update_game_state(game_state, round_actions, game_config)
        
        # Calculate final aggregate payoffs (NPV for dynamic games)
        final_payoffs = {}
        for player_id in all_payoffs:
            discount_factor = game_config.constants.get('discount_factor', 0.95)
            npv = sum(
                (discount_factor ** t) * payoff 
                for t, payoff in enumerate(all_payoffs[player_id])
            )
            final_payoffs[player_id] = npv
        
        challenger_npv = final_payoffs.get('challenger', 0)
        self.logger.debug(f"[{call_id}] Final NPV - Challenger: {challenger_npv:.2f}")
        
        # Create comprehensive game data
        game_data = game_engine.get_game_data_for_logging(
            all_actions, final_payoffs, game_config, game_state
        )
        game_data['round_history'] = round_history
        
        return create_game_result(
            simulation_id=simulation_id,
            game_name=game_config.game_name,
            experiment_type=game_config.experiment_type,
            condition_name=game_config.condition_name,
            players=list(agents.keys()),
            actions=all_actions,
            payoffs=final_payoffs,
            round_data=round_history,
            **game_data
        )
    
    async def _get_agent_action(self, agent: BaseLLMAgent, prompt: str, player_id: str,
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
    
    async def _save_competition_result(self, result: CompetitionResult):
        """Save competition result to disk"""
        
        # Create directory structure
        output_path = (self.output_dir / result.game_name / result.challenger_model)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete result
        filename = f"{result.condition_name}_competition_result.json"
        
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
            'simulation_results': [
                {
                    'simulation_id': sim.simulation_id,
                    'players': sim.players,
                    'actions': sim.actions,
                    'payoffs': sim.payoffs,
                    'game_data': sim.game_data,
                    'round_data': getattr(sim, 'round_data', [])
                }
                for sim in result.simulation_results
            ]
        }
        
        with open(output_path / filename, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Results saved: {output_path / filename}")
    
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