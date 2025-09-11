"""
Competition Framework - Fixed version with proper parameter handling
Orchestrates economic game competitions between LLM agents with correct state management
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from config import (
    GameConfig, get_game_config, get_challenger_models, get_defender_model,
    get_model_display_name, is_thinking_enabled, get_all_game_configs,
    get_simulation_count  # Import the correct function from config.py
)
from games import create_game
from agents import create_agent, BaseLLMAgent
from metrics.metric_utils import GameResult, create_game_result


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
    """Manages competitions between challenger and defender LLM agents"""
    
    def __init__(self, challenger_models: List[str], defender_model: str, 
                 mock_mode: bool = False, output_dir: Optional[Path] = None):
        """Initialize competition framework"""
        self.challenger_models = challenger_models
        self.defender_model = defender_model
        self.mock_mode = mock_mode
        self.output_dir = output_dir or Path("results")
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize games
        self.games = {
            'salop': create_game('salop'),
            'spulber': create_game('spulber'),
            'green_porter': create_game('green_porter'),
            'athey_bagwell': create_game('athey_bagwell')
        }
        
        self.logger.info(f"Competition initialized with {len(self.challenger_models)} challengers")
        if mock_mode:
            self.logger.info("🎭 MOCK MODE: Using simulated responses")

    async def run_competition(self, game_name: str, experiment_type: str, condition_name: str,
                            challenger_model: str, defender_model: Optional[str] = None) -> CompetitionResult:
        """
        Run a complete competition for a specific game configuration
        
        Args:
            game_name: Name of the game ('salop', 'green_porter', etc.)
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
        game_config = get_game_config(game_name, experiment_type, condition_name)
        game_engine = self.games[game_name]
        
        # Get simulation count from config.json (FIXED: Now uses correct function)
        num_simulations = get_simulation_count(experiment_type)
        self.logger.info(f"Simulations: {num_simulations}")
        
        start_time = datetime.now()
        simulation_results = []
        
        # Track competition metadata
        competition_metadata = {
            'game_config': game_config.to_dict(),
            'challenger_model_config': challenger_model,
            'defender_model_config': defender_model,
            'num_simulations_planned': num_simulations,
            'mock_mode': self.mock_mode
        }
        
        # Run all simulations for this competition
        successful_simulations = 0
        for sim_id in range(num_simulations):
            try:
                self.logger.info(f"  🎯 Simulation {sim_id + 1}/{num_simulations}")
                
                call_id = f"comp_{game_name}_{sim_id}"
                
                # Run single simulation (FIXED: Use correct method signature)
                game_result = await self._run_single_simulation(
                    sim_id, game_engine, game_config, challenger_model, defender_model, call_id
                )
                
                simulation_results.append(game_result)
                successful_simulations += 1
                
                self.logger.info(f"    ✅ Simulation {sim_id + 1} completed")
                
            except Exception as e:
                self.logger.error(f"    ❌ Simulation {sim_id + 1} failed: {e}")
                continue
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Update metadata with results
        competition_metadata.update({
            'num_simulations_completed': successful_simulations,
            'success_rate': successful_simulations / num_simulations if num_simulations > 0 else 0,
            'total_duration_seconds': total_duration
        })
        
        self.logger.info(f"Competition completed: {successful_simulations}/{num_simulations} simulations successful")
        self.logger.info(f"Total duration: {total_duration:.2f} seconds")
        
        # Create and return competition result
        return CompetitionResult(
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

    async def _run_single_simulation(self, simulation_id: int, game_engine, game_config: GameConfig,
                                   challenger_model: str, defender_model: str, call_id: str) -> GameResult:
        """Run a single game simulation (FIXED: Correct method signature from project knowledge)"""
        
        # Initialize game state - CRITICAL: This returns a Dict
        game_state = game_engine.initialize_game_state(game_config, simulation_id)
        
        # DEBUG: Verify types
        self.logger.debug(f"[{call_id}] game_state type: {type(game_state)}")
        self.logger.debug(f"[{call_id}] game_config type: {type(game_config)}")
        
        # Create agents - FIXED: Use the _create_agents method from project knowledge
        num_of_players = game_config.constants.get('number_of_players', 2)
        agents = self._create_agents(challenger_model, defender_model, num_of_players, call_id)

        # Determine if this is a dynamic or static game
        if hasattr(game_engine, 'update_game_state'):
            # Dynamic game (multi-round) - FIXED: Correct parameter order
            actions, payoffs, game_data = await self._run_dynamic_game(
                game_engine, game_config, game_state, agents, call_id
            )
        else:
            # Static game (single-round) - FIXED: Correct parameter order  
            actions, payoffs, game_data = await self._run_static_game(
                game_engine, game_config, game_state, agents, call_id
            )
        
        # Create GameResult
        player_ids = list(agents.keys())
        
        result = create_game_result(
            simulation_id=simulation_id,
            game_name=game_config.game_name,
            experiment_type=game_config.experiment_type,
            condition_name=game_config.condition_name,
            players=player_ids,  # Fixed: use 'players' instead of 'player_ids'
            actions=actions,
            payoffs=payoffs,
            **game_data  # Fixed: unpack game_data as keyword arguments
        )
        
        return result

    def _create_agents(self, challenger_model: str, defender_model: str, 
                       num_of_players: int, call_id: str) -> Dict[str, BaseLLMAgent]:
        """Create agents for the game (FIXED: From project knowledge)"""
        
        agents = {}
        
        try:
            # Create challenger agent
            challenger_agent = create_agent(challenger_model, 'challenger', mock_mode=self.mock_mode)
            agents['challenger'] = challenger_agent
            
            # Create defender agents
            for i in range(1, num_of_players):
                player_id = f'defender_{i}'
                defender_agent = create_agent(defender_model, player_id, mock_mode=self.mock_mode)
                agents[player_id] = defender_agent
                
        except Exception as e:
            self.logger.error(f"[{call_id}] Failed to create agents: {e}")
            raise
        
        return agents

    async def _run_static_game(self, game_engine, game_config: GameConfig, game_state: Dict,
                             agents: Dict[str, BaseLLMAgent], call_id: str) -> Tuple[Dict, Dict, Dict]:
        """Run a static (single-round) game (FIXED: From project knowledge)"""
        
        actions = {}
        
        # Get actions from all players
        for player_id, agent in agents.items():
            try:
                # Generate prompt for this player (CRITICAL: Correct parameter order)
                prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
                
                # Get action from agent
                action = await self._get_action_from_llm_agent(agent, player_id, prompt, game_engine, call_id)
                actions[player_id] = action
                
            except Exception as e:
                self.logger.error(f"[{call_id}] Error getting action from {player_id}: {e}")
                # Use fallback action
                actions[player_id] = self._get_fallback_action(game_config.game_name)
        
        # Calculate payoffs
        payoffs = game_engine.calculate_payoffs(actions, game_config, game_state)
        
        # Get game data for analysis
        game_data = game_engine.get_game_data_for_logging(actions, payoffs, game_config, game_state)
        
        return actions, payoffs, game_data

    async def _run_dynamic_game(self, game_engine, game_config: GameConfig, game_state: Dict,
                              agents: Dict[str, BaseLLMAgent], call_id: str) -> Tuple[Dict, Dict, Dict]:
        """Run a dynamic (multi-round) game (FIXED: From project knowledge)"""
        
        # FIXED: Ensure we're accessing constants correctly
        num_rounds = game_config.constants.get('time_horizon', 10)
        all_round_actions = []
        final_actions = {}
        final_payoffs = {}
        
        for round_num in range(1, num_rounds + 1):
            self.logger.debug(f"[{call_id}] Round {round_num}/{num_rounds}")
            
            # Get actions from all players for this round
            round_actions = {}
            
            for player_id, agent in agents.items():
                # CRITICAL: Ensure correct parameter order (player_id, game_state, game_config)
                prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
                
                # Get action from agent
                action = await self._get_action_from_llm_agent(agent, player_id, prompt, game_engine, call_id)
                round_actions[player_id] = action
                
                # Log action summary
                action_summary = self._summarize_action(action)
                self.logger.debug(f"[{call_id}] Round {round_num} - {player_id}: {action_summary}")
            
            # Calculate payoffs for this round
            round_payoffs = game_engine.calculate_payoffs(round_actions, game_config, game_state)
            
            # Store round data
            all_round_actions.append({
                'round': round_num,
                'actions': round_actions,
                'payoffs': round_payoffs,
                'game_state': dict(game_state)  # Copy current state
            })
            
            # CRITICAL: Update game state for next round - ensure correct parameter order
            if round_num < num_rounds:
                # FIXED: Make sure update_game_state gets (game_state, actions, game_config)
                game_state = game_engine.update_game_state(game_state, round_actions, game_config)
            
            # Keep last round's data for final result
            final_actions = round_actions
            final_payoffs = round_payoffs
        
        # Create comprehensive game data
        game_data = {
            'all_rounds': all_round_actions,
            'num_rounds': num_rounds,
            'final_round_actions': final_actions,
            'final_round_payoffs': final_payoffs
        }
        
        return final_actions, final_payoffs, game_data

    async def _get_action_from_llm_agent(self, agent, player_id: str, prompt: str, 
                          game_engine, call_id: str) -> Dict[str, Any]:
        """Get action from LLM agent with parsing and error handling (FIXED: From project knowledge)"""
        
        # Get raw response from LLM
        try:
            response = await agent.get_response(prompt, call_id)
            if not response.success:
                self.logger.warning(f"[{call_id}] Agent response failed for {player_id}: {response.error}")
                return self._get_fallback_action(game_engine.game_name)
            
            raw_response = response.content
        except Exception as e:
            self.logger.error(f"[{call_id}] Agent {player_id} API call failed: {e}")
            return self._get_fallback_action(game_engine.game_name)
        
        # Parse response using game-specific parser
        parsed_action = game_engine.parse_llm_response(raw_response, player_id, call_id)
        
        if parsed_action is None:
            self.logger.warning(f"[{call_id}] Parsing failed for {player_id}, returning fallback action")
            return self._get_fallback_action(game_engine.game_name)
        
        # Add metadata
        parsed_action.update({
            'player_id': player_id,
            'parsing_success': True
        })
        
        return parsed_action

    def _get_fallback_action(self, game_name: str) -> Dict[str, Any]:
        """Get fallback action based on game type"""
        fallbacks = {
            'salop': {'price': 15.0},
            'spulber': {'price': 15.0},
            'green_porter': {'quantity': 20},
            'athey_bagwell': {'report': 'high'}
        }
        return fallbacks.get(game_name, {'price': 1.0})

    def _summarize_action(self, action: Dict[str, Any]) -> str:
        """Create a summary string for an action"""
        if 'error' in action:
            return f"ERROR: {action['error']}"
        
        # Extract key action fields
        summary_parts = []
        for key in ['price', 'quantity', 'report', 'bid']:
            if key in action:
                summary_parts.append(f"{key}={action[key]}")
        
        return ", ".join(summary_parts) if summary_parts else "No action parsed"

    async def _save_competition_result(self, result: CompetitionResult) -> None:
        """Save individual competition result to JSON file"""
        
        self.logger.info(f"🔄 _save_competition_result called for {result.game_name}")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
        self.logger.info(f"📊 Simulation results count: {len(result.simulation_results)}")
        
        try:
            # Create output directory structure
            game_dir = self.output_dir / result.game_name
            challenger_dir = game_dir / result.challenger_model
            
            self.logger.info(f"📁 Creating directories: {challenger_dir}")
            challenger_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Directories created successfully")
            
            # Create filename
            safe_condition = result.condition_name.replace(" ", "_").replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mock_suffix = "_mock" if self.mock_mode else ""
            filename = f"{safe_condition}_competition_result{mock_suffix}_{timestamp}.json"
            
            filepath = challenger_dir / filename
            self.logger.info(f"📄 Target file path: {filepath}")
            
            # Filter out error simulations for cleaner results
            filtered_results = [
                sim for sim in result.simulation_results 
                if not any('error' in str(action) for action in sim.actions.values())
            ]
            
            self.logger.info(f"📊 Filtered results: {len(filtered_results)}/{len(result.simulation_results)}")
            
            # Create filtered result
            filtered_result = CompetitionResult(
                game_name=result.game_name,
                experiment_type=result.experiment_type,
                condition_name=result.condition_name,
                challenger_model=result.challenger_model,
                challenger_display_name=result.challenger_display_name,
                defender_model=result.defender_model,
                defender_display_name=result.defender_display_name,
                challenger_thinking_enabled=result.challenger_thinking_enabled,
                defender_thinking_enabled=result.defender_thinking_enabled,
                simulation_results=filtered_results,
                competition_metadata=result.competition_metadata,
                start_time=result.start_time,
                end_time=result.end_time,
                total_duration=result.total_duration
            )
            
            # Convert to dict for JSON serialization
            self.logger.info(f"🔄 Converting to JSON dict...")
            result_dict = {
                'game_name': filtered_result.game_name,
                'experiment_type': filtered_result.experiment_type,
                'condition_name': filtered_result.condition_name,
                'challenger_model': filtered_result.challenger_model,
                'challenger_display_name': filtered_result.challenger_display_name,
                'defender_model': filtered_result.defender_model,
                'defender_display_name': filtered_result.defender_display_name,
                'challenger_thinking_enabled': filtered_result.challenger_thinking_enabled,
                'defender_thinking_enabled': filtered_result.defender_thinking_enabled,
                'simulation_results': [
                    {
                        'simulation_id': sim.simulation_id,
                        'game_name': sim.game_name,
                        'experiment_type': sim.experiment_type,
                        'condition_name': sim.condition_name,
                        'players': sim.players,
                        'actions': sim.actions,
                        'payoffs': sim.payoffs,
                        'game_data': sim.game_data
                    } for sim in filtered_result.simulation_results
                ],
                'competition_metadata': filtered_result.competition_metadata,
                'start_time': filtered_result.start_time,
                'end_time': filtered_result.end_time,
                'total_duration': filtered_result.total_duration
            }
            
            # Save to file
            self.logger.info(f"💾 Writing JSON to file...")
            import json
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            # Verify file was created
            if filepath.exists():
                file_size = filepath.stat().st_size
                self.logger.info(f"✅ Competition result saved: {filepath}")
                self.logger.info(f"📏 File size: {file_size} bytes")
                self.logger.info(f"📊 Simulations: {len(filtered_result.simulation_results)}/{len(result.simulation_results)} (after filtering)")
            else:
                self.logger.error(f"❌ File was not created: {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save competition result: {e}")
            self.logger.error(f"📁 Attempted path: {self.output_dir}")
            import traceback
            self.logger.error(f"📋 Full traceback: {traceback.format_exc()}")

    async def run_batch_competitions(self, competitions: List[Dict[str, str]]) -> List[CompetitionResult]:
        """Run a batch of competitions with progress tracking"""
        results = []
        total = len(competitions)
        
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING BATCH: {total} COMPETITIONS")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        for i, comp in enumerate(competitions):
            comp_start = time.time()
            self.logger.info(f"🔄 Running competition {i+1}/{total}: "
                           f"{comp['game_name']} - {comp['experiment_type']} - {comp['condition_name']}")
            
            try:
                result = await self.run_competition(**comp)
                results.append(result)
                
                comp_duration = time.time() - comp_start
                self.logger.info(f"  ✅ {i+1}/{total} completed in {comp_duration:.1f}s")
                
            except Exception as e:
                comp_duration = time.time() - comp_start
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
            
            # Calculate expected simulations
            total_expected_simulations = 0
            for comp in competitions:
                sim_count = get_simulation_count(comp['experiment_type'])
                total_expected_simulations += sim_count
            
            self.logger.info(f"Expected total simulations: {total_expected_simulations}")
            
            # Run all competitions using existing batch method
            results = await self.run_batch_competitions(competitions)
            
            # Calculate actual simulations completed
            total_actual_simulations = 0
            successful_simulations = 0
            for result in results:
                total_actual_simulations += result.competition_metadata['num_simulations_planned']
                successful_simulations += result.competition_metadata['num_simulations_completed']
            
            self.logger.info("=" * 80)
            self.logger.info("ALL COMPETITIONS SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Competitions: {len(results)}/{len(competitions)} successful")
            self.logger.info(f"Total simulations: {successful_simulations}/{total_actual_simulations} successful")
            self.logger.info("=" * 80)
            
            # Save results if output directory specified
            if self.output_dir:
                await self._save_results(results)
            
            return len(results) == len(competitions)
            
        except Exception as e:
            self.logger.error(f"run_all_competitions failed: {e}")
            return False

    async def _save_results(self, results: List[CompetitionResult]):
        """Save competition results to files"""
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual competition results
            self.logger.info(f"💾 Saving {len(results)} individual competition results...")
            for result in results:
                await self._save_competition_result(result)
            
            # Save summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.output_dir / f"competition_summary_{timestamp}.json"
            
            summary = {
                'timestamp': timestamp,
                'total_competitions': len(results),
                'mock_mode': self.mock_mode,
                'challenger_models': self.challenger_models,
                'defender_model': self.defender_model,
                'results_summary': []
            }
            
            for result in results:
                summary['results_summary'].append({
                    'game_name': result.game_name,
                    'experiment_type': result.experiment_type,
                    'condition_name': result.condition_name,
                    'challenger_model': result.challenger_model,
                    'simulations_completed': len(result.simulation_results),
                    'duration': result.total_duration
                })
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"📋 Summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def get_competition_statistics(self) -> Dict[str, Any]:
        """Get statistics about the competition setup"""
        total_competitions = 0
        total_simulations = 0
        game_breakdown = {}
        
        for game_name in ['salop', 'spulber', 'green_porter', 'athey_bagwell']:
            game_configs = get_all_game_configs(game_name)
            game_competitions = len(game_configs) * len(self.challenger_models)
            total_competitions += game_competitions
            
            game_simulations = 0
            for config in game_configs:
                sim_count = get_simulation_count(config.experiment_type)
                game_simulations += sim_count * len(self.challenger_models)
            
            total_simulations += game_simulations
            
            game_breakdown[game_name] = {
                'configurations': len(game_configs),
                'competitions': game_competitions,
                'simulations': game_simulations
            }
        
        return {
            'total_competitions': total_competitions,
            'total_simulations': total_simulations,
            'challenger_models': len(self.challenger_models),
            'defender_model': self.defender_model,
            'game_breakdown': game_breakdown,
            'mock_mode': self.mock_mode
        }