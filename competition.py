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
    get_model_display_name, is_thinking_enabled, get_all_game_configs
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


def get_simulation_count(experiment_type: str) -> int:
    """Get number of simulations based on experiment type"""
    counts = {
        'baseline': 20,
        'structural_variations': 15,
        'ablation_studies': 10
    }
    return counts.get(experiment_type, 20)


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
                    sim_id, game_engine, game_config,
                    challenger_model, defender_model, call_id
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
        
        return result

    async def _run_single_simulation(self, simulation_id: int, game_engine, game_config: GameConfig,
                                   challenger_model: str, defender_model: str, call_id: str) -> GameResult:
        """Run a single game simulation"""
        
        # Initialize game state - CRITICAL: This returns a Dict
        game_state = game_engine.initialize_game_state(game_config, simulation_id)
        
        # DEBUG: Verify types
        self.logger.debug(f"[{call_id}] game_state type: {type(game_state)}")
        self.logger.debug(f"[{call_id}] game_config type: {type(game_config)}")
        
        # Create agents
        num_of_players = game_config.constants.get('number_of_players', 2)
        agents = await self._create_agents(challenger_model, defender_model, num_of_players, call_id)

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
            players=player_ids,
            actions=actions,
            payoffs=payoffs,
            game_data=game_data
        )
        
        return result

    async def _run_dynamic_game(self, game_engine, game_config: GameConfig, game_state: Dict,
                              agents: Dict[str, BaseLLMAgent], call_id: str) -> Tuple[Dict, Dict, Dict]:
        """Run a dynamic (multi-round) game - FIXED parameter handling"""
        
        # DEBUG: Verify we have correct types
        self.logger.debug(f"[{call_id}] _run_dynamic_game: game_state type = {type(game_state)}")
        self.logger.debug(f"[{call_id}] _run_dynamic_game: game_config type = {type(game_config)}")
        
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
                updated_state = game_engine.update_game_state(game_state, round_actions, game_config)
                
                # DEBUG: Verify the updated state is still a Dict
                self.logger.debug(f"[{call_id}] Updated game_state type: {type(updated_state)}")
                
                # CRITICAL: Ensure we're assigning the right type
                if not isinstance(updated_state, dict):
                    self.logger.error(f"[{call_id}] update_game_state returned {type(updated_state)}, expected dict")
                    raise TypeError(f"update_game_state must return Dict, got {type(updated_state)}")
                
                game_state = updated_state
            
            # Update final results (last round wins, or could be cumulative)
            final_actions = round_actions
            final_payoffs = round_payoffs
        
        # Get comprehensive game data for analysis
        game_data = game_engine.get_game_data_for_logging(final_actions, final_payoffs, game_config, game_state)
        game_data.update({
            'round_data': all_round_actions,
            'num_rounds_played': len(all_round_actions)
        })
        
        return final_actions, final_payoffs, game_data

    async def _run_static_game(self, game_engine, game_config: GameConfig, game_state: Dict,
                             agents: Dict[str, BaseLLMAgent], call_id: str) -> Tuple[Dict, Dict, Dict]:
        """Run a static (single-round) game - FIXED parameter handling"""
        
        # DEBUG: Verify we have correct types
        self.logger.debug(f"[{call_id}] _run_static_game: game_state type = {type(game_state)}")
        self.logger.debug(f"[{call_id}] _run_static_game: game_config type = {type(game_config)}")
        
        # Get actions from all players
        actions = {}
        
        for player_id, agent in agents.items():
            # CRITICAL: Ensure correct parameter order (player_id, game_state, game_config)
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
            return {
                'error': f"API call failed: {str(e)}",
                'player_id': player_id,
                'raw_response': None,
                'parsing_success': False
            }
        
        # Parse response using game-specific parser
        parsed_action = game_engine.parse_llm_response(raw_response, player_id, call_id)
        
        if parsed_action is None:
            self.logger.warning(f"[{call_id}] Parsing failed for {player_id}, returning error action")
            return {
                'error': f"LLM response parsing failed",
                'player_id': player_id,
                'raw_response': raw_response,
                'parsing_success': False
            }
        
        # Add metadata
        parsed_action.update({
            'player_id': player_id,
            'parsing_success': True
        })
        
        return parsed_action

    async def _create_agents(self, challenger_model: str, defender_model: str, 
                           num_of_players: int, call_id: str) -> Dict[str, BaseLLMAgent]:
        """Create agents for the game"""
        
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
        """Save competition result to JSON file with error filtering"""
        
        # Create output directory structure
        game_dir = self.output_dir / result.game_name
        challenger_dir = game_dir / result.challenger_model
        challenger_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        safe_condition = result.condition_name.replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mock_suffix = "_mock" if self.mock_mode else ""
        filename = f"{safe_condition}_competition_result{mock_suffix}_{timestamp}.json"
        
        filepath = challenger_dir / filename
        
        try:
            # Filter out error simulations for cleaner results
            filtered_results = [
                sim for sim in result.simulation_results 
                if not any('error' in action for action in sim.actions.values())
            ]
            
            # Create filtered result
            filtered_result = result.__class__(
                **{**result.__dict__, 'simulation_results': filtered_results}
            )
            
            # Save to file
            with open(filepath, 'w') as f:
                import json
                json.dump(filtered_result.__dict__, f, indent=2, default=str)
            
            self.logger.info(f"Results saved (errors filtered): {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    async def run_batch_competitions(self, competitions: List[Dict[str, str]]) -> List[CompetitionResult]:
        """Run multiple competitions in batch"""
        
        results = []
        total = len(competitions)
        start_time = time.time()
        
        self.logger.info(f"Starting batch of {total} competitions...")
        
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