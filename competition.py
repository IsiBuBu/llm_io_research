"""
Updated Competition Framework - Four-folder structure with dynamic metrics integration
Orchestrates economic game competitions between LLM agents with comprehensive output
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from config import (
    GameConfig, get_game_config, get_challenger_models, get_defender_model,
    get_model_display_name, is_thinking_enabled, get_all_game_configs,
    get_simulation_count
)
from games import create_game
from agents import create_agent, BaseLLMAgent
from metrics.metric_utils import GameResult, create_game_result
from metrics.dynamic_game_metrics import DynamicGameMetricsCalculator


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
        
        # Initialize dynamic metrics calculator
        self.dynamic_calculator = DynamicGameMetricsCalculator()
        
        self.logger.info(f"Competition initialized with {len(self.challenger_models)} challengers")
        if mock_mode:
            self.logger.info("🎭 MOCK MODE: Using simulated responses")

    async def run_single_competition(self, game_config: GameConfig, challenger_model: str, 
                                   output_dir: str) -> bool:
        """
        Run a single competition and save to four-folder structure
        
        Args:
            game_config: Game configuration with experiment details
            challenger_model: Challenger model name
            output_dir: Output directory (should be results/[game]/[model]/)
            
        Returns:
            Success status
        """
        
        try:
            # Get model details
            challenger_display = get_model_display_name(challenger_model)
            defender_display = get_model_display_name(self.defender_model)
            challenger_thinking = is_thinking_enabled(challenger_model)
            defender_thinking = is_thinking_enabled(self.defender_model)
            
            self.logger.info("=" * 80)
            self.logger.info(f"COMPETITION: {game_config.game_name.upper()} - {game_config.condition_name}")
            self.logger.info("=" * 80)
            self.logger.info(f"Challenger: {challenger_display} (Thinking: {'ON' if challenger_thinking else 'OFF'})")
            self.logger.info(f"Defender: {defender_display} (Thinking: {'ON' if defender_thinking else 'OFF'})")
            
            # Get simulation count from config
            num_simulations = get_simulation_count(game_config.experiment_type)
            self.logger.info(f"Simulations: {num_simulations}")
            
            # Run simulations
            start_time = datetime.now()
            simulation_results = []
            successful_sims = 0
            failed_sims = 0
            
            game_engine = self.games[game_config.game_name]
            
            for sim_id in range(num_simulations):
                try:
                    call_id = f"{game_config.game_name}_{game_config.condition_name}_{challenger_model}_sim{sim_id}"
                    
                    self.logger.debug(f"[{call_id}] Starting simulation {sim_id + 1}/{num_simulations}")
                    
                    # Run single game simulation
                    result = await self._run_single_simulation(
                        sim_id, game_engine, game_config,
                        challenger_model, self.defender_model, call_id
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
            
            # Generate comprehensive game output with dynamic metrics integration
            game_output = await self._generate_game_output(
                game_config, challenger_model, simulation_results, 
                challenger_thinking, successful_sims, failed_sims, success_rate
            )
            
            # Save to four-folder structure
            await self._save_game_output(game_output, output_dir, game_config.condition_name)
            
            self.logger.info(f"✅ Competition completed: {success_rate:.1%} success rate")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Competition failed: {e}")
            return False

    async def _run_single_simulation(self, simulation_id: int, game_engine, game_config: GameConfig,
                                   challenger_model: str, defender_model: str, call_id: str) -> GameResult:
        """Run a single game simulation"""
        
        # Initialize game state
        game_state = game_engine.initialize_game_state(game_config, simulation_id)
        
        # Create agents
        num_of_players = game_config.constants.get('number_of_players', 2)
        agents = await self._create_agents(challenger_model, defender_model, num_of_players, call_id)

        # Determine if this is a dynamic or static game
        if hasattr(game_engine, 'update_game_state'):
            # Dynamic game (multi-round)
            actions, payoffs, game_data = await self._run_dynamic_game(
                game_engine, game_config, game_state, agents, call_id
            )
        else:
            # Static game (single-round)
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
        """Run a dynamic (multi-round) game"""
        
        num_rounds = game_config.constants.get('time_horizon', 10)
        all_round_data = []
        final_actions = {}
        final_payoffs = {}
        
        for round_num in range(1, num_rounds + 1):
            self.logger.debug(f"[{call_id}] Round {round_num}/{num_rounds}")
            
            # Get actions from all agents for this round
            round_actions = {}
            for player_id, agent in agents.items():
                try:
                    # Generate prompt and get action
                    prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
                    
                    if self.mock_mode:
                        action = await self._get_mock_action(game_config.game_name, player_id)
                    else:
                        action = await agent.get_action(prompt, game_config.game_name, game_state, game_config)
                    
                    round_actions[player_id] = action
                    
                except Exception as e:
                    self.logger.error(f"[{call_id}] Failed to get action for {player_id}: {e}")
                    round_actions[player_id] = {"error": str(e)}
            
            # Update game state and calculate round outcomes
            try:
                updated_state, round_payoffs = game_engine.update_game_state(
                    game_state, round_actions, game_config, round_num
                )
                game_state = updated_state
                
                # Store round data for dynamic metrics
                round_data = {
                    'round': round_num,
                    'actions': round_actions.copy(),
                    'payoffs': round_payoffs.copy(),
                    **game_state  # Include all game state for metrics calculation
                }
                all_round_data.append(round_data)
                
                # Update final actions and payoffs
                final_actions.update(round_actions)
                for player_id, payoff in round_payoffs.items():
                    final_payoffs[player_id] = final_payoffs.get(player_id, 0) + payoff
                
            except Exception as e:
                self.logger.error(f"[{call_id}] Round {round_num} state update failed: {e}")
                break
        
        # Prepare game data with round information
        game_data = {
            'config': game_config.constants,
            'total_rounds': num_rounds,
            'round_data': all_round_data,
            'final_state': game_state
        }
        
        return final_actions, final_payoffs, game_data

    async def _run_static_game(self, game_engine, game_config: GameConfig, game_state: Dict,
                             agents: Dict[str, BaseLLMAgent], call_id: str) -> Tuple[Dict, Dict, Dict]:
        """Run a static (single-round) game"""
        
        # Get actions from all agents
        actions = {}
        for player_id, agent in agents.items():
            try:
                prompt = game_engine.generate_player_prompt(player_id, game_state, game_config)
                
                if self.mock_mode:
                    action = await self._get_mock_action(game_config.game_name, player_id)
                else:
                    action = await agent.get_action(prompt, game_config.game_name, game_state, game_config)
                
                actions[player_id] = action
                
            except Exception as e:
                self.logger.error(f"[{call_id}] Failed to get action for {player_id}: {e}")
                actions[player_id] = {"error": str(e)}
        
        # Calculate payoffs
        try:
            payoffs = game_engine.calculate_payoffs(actions, game_config)
        except Exception as e:
            self.logger.error(f"[{call_id}] Payoff calculation failed: {e}")
            payoffs = {player_id: 0.0 for player_id in actions.keys()}
        
        # Prepare game data
        game_data = {
            'config': game_config.constants,
            'final_state': game_state
        }
        
        return actions, payoffs, game_data

    async def _create_agents(self, challenger_model: str, defender_model: str, 
                           num_players: int, call_id: str) -> Dict[str, BaseLLMAgent]:
        """Create agents for the game"""
        
        agents = {}
        
        # Create challenger agent
        agents['challenger'] = await create_agent(challenger_model, 'challenger', self.mock_mode)
        
        # Create defender agents
        for i in range(1, num_players):
            defender_id = f'defender_{i}'
            agents[defender_id] = await create_agent(defender_model, defender_id, self.mock_mode)
        
        return agents

    def _create_simple_mock_agent(self, player_id: str):
        """Create a simple mock agent for fallback"""
        class SimpleMockAgent:
            def __init__(self, player_id):
                self.player_id = player_id
            
            def get_action(self, prompt, game_name, game_state, game_config):
                return self._get_mock_action_sync(game_name)
            
            def _get_mock_action_sync(self, game_name):
                mock_actions = {
                    'salop': {'price': 15.0, 'reasoning': 'Mock strategic pricing'},
                    'spulber': {'price': 12.0, 'reasoning': 'Mock bid under uncertainty'},
                    'green_porter': {'quantity': 20, 'reasoning': 'Mock quantity choice'},
                    'athey_bagwell': {'report': 'low', 'reasoning': 'Mock cost report'}
                }
                action = mock_actions.get(game_name, {'action': 'default', 'reasoning': 'Mock action'})
                action['player_type'] = 'simple_mock'
                return action
        
        return SimpleMockAgent(player_id)

    async def _get_mock_action(self, game_name: str, player_id: str) -> Dict[str, Any]:
        """Generate mock action for testing"""
        
        mock_actions = {
            'salop': {'price': 15.0, 'reasoning': 'Mock strategic pricing'},
            'spulber': {'price': 12.0, 'reasoning': 'Mock bid under uncertainty'},
            'green_porter': {'quantity': 20, 'reasoning': 'Mock quantity choice'},
            'athey_bagwell': {'report': 'low', 'reasoning': 'Mock cost report'}
        }
        
        base_action = mock_actions.get(game_name, {'action': 'default', 'reasoning': 'Mock action'})
        base_action['player_type'] = 'mock'
        
        return base_action

    async def _generate_game_output(self, game_config: GameConfig, challenger_model: str,
                                  simulation_results: List[GameResult], challenger_thinking: bool,
                                  successful_sims: int, failed_sims: int, success_rate: float) -> Dict[str, Any]:
        """Generate comprehensive game output with integrated dynamic metrics"""
        
        # Extract experimental setup details
        setup_details = self._extract_experimental_setup(game_config, simulation_results)
        
        # Base game output structure
        game_output = {
            'metadata': {
                'model': challenger_model,
                'condition': game_config.condition_name,
                'experiment_type': game_config.experiment_type,
                'player_type': 'challenger',
                'thinking_enabled': challenger_thinking,
                'total_simulations': len(simulation_results),
                'successful_simulations': successful_sims,
                'failed_simulations': failed_sims,
                'success_rate': success_rate,
                'timestamp': datetime.now().isoformat()
            },
            'experimental_setup': setup_details,
            'simulations': []
        }
        
        # Process each simulation
        for result in simulation_results:
            sim_output = {
                'game_number': result.simulation_id,
                'player_id': 'challenger',
                'model': challenger_model,
                'player_type': 'challenger',
                'llm_response': result.actions.get('challenger', {}),
                'payoff': result.payoffs.get('challenger', 0),
                'game_data': result.game_data
            }
            
            # Add thoughts if available and thinking enabled
            challenger_action = result.actions.get('challenger', {})
            if challenger_thinking and 'thoughts' in challenger_action:
                sim_output['thoughts'] = challenger_action['thoughts']
            
            # **INTEGRATE DYNAMIC METRICS HERE**
            if game_config.game_name in ['green_porter', 'athey_bagwell']:
                dynamic_metrics = await self._calculate_dynamic_metrics(result, game_config.game_name)
                if dynamic_metrics:
                    sim_output['dynamic_metrics'] = dynamic_metrics
            
            game_output['simulations'].append(sim_output)
        
        return game_output

    async def _calculate_dynamic_metrics(self, game_result: GameResult, game_name: str) -> Dict[str, Any]:
        """Calculate dynamic metrics for a single simulation"""
        
        try:
            if game_name == 'green_porter':
                metrics = self.dynamic_calculator.calculate_green_porter_metrics([game_result])
            elif game_name == 'athey_bagwell':
                metrics = self.dynamic_calculator.calculate_athey_bagwell_metrics([game_result])
            else:
                return {}
            
            # Format for integration into game output
            formatted_metrics = {
                'round_by_round': {},
                'aggregate': metrics.get('aggregate_metrics', {})
            }
            
            # Extract round-by-round data
            for metric_type, rounds in metrics.items():
                if metric_type != 'aggregate_metrics' and isinstance(rounds, list):
                    formatted_metrics['round_by_round'][metric_type] = [
                        {'round': r.round_number, 'value': r.value, 'description': r.description}
                        for r in rounds
                    ]
            
            return formatted_metrics
            
        except Exception as e:
            self.logger.error(f"Dynamic metrics calculation failed: {e}")
            return {}

    def _extract_experimental_setup(self, game_config: GameConfig, 
                                   simulation_results: List[GameResult]) -> Dict[str, Any]:
        """Extract experimental setup details"""
        
        setup = {
            'game_name': game_config.game_name,
            'condition_name': game_config.condition_name,
            'experiment_type': game_config.experiment_type,
            'parameters': game_config.constants.copy()
        }
        
        # Add game-specific setup details
        if game_config.game_name in ['green_porter', 'athey_bagwell']:
            # Extract lists used from first simulation
            if simulation_results:
                first_sim = simulation_results[0]
                game_data = first_sim.game_data
                
                if game_config.game_name == 'green_porter':
                    setup['demand_shock_list'] = game_data.get('demand_shocks', [])
                elif game_config.game_name == 'athey_bagwell':
                    setup['cost_type_lists'] = game_data.get('cost_sequences', {})
        
        # Identify which lists were used
        setup['list_identification'] = self._identify_experimental_lists(game_config)
        
        return setup

    def _identify_experimental_lists(self, game_config: GameConfig) -> Dict[str, Any]:
        """Identify which demand shock or cost type lists were used"""
        
        identification = {
            'structural_variation': None,
            'ablation_condition': None
        }
        
        # Parse condition name to identify components
        condition = game_config.condition_name
        
        if game_config.experiment_type == 'baseline':
            identification['structural_variation'] = 'baseline'
            identification['ablation_condition'] = 'none'
        elif game_config.experiment_type == 'structural_variations':
            identification['structural_variation'] = condition
            identification['ablation_condition'] = 'none'
        elif game_config.experiment_type == 'combined':
            # Format: "structural_ablation"
            if '_' in condition:
                parts = condition.split('_', 1)
                identification['structural_variation'] = parts[0]
                identification['ablation_condition'] = parts[1]
        
        return identification

    async def _save_game_output(self, game_output: Dict[str, Any], output_dir: str, 
                              condition_name: str) -> None:
        """Save game output to four-folder structure"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        safe_condition = condition_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_condition}_game_output.json"
        
        filepath = output_path / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(game_output, f, indent=2, default=str)
            
            self.logger.info(f"📁 Game output saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save game output: {e}")

    async def run_batch_competitions(self, competitions: List[Dict[str, str]]) -> List[bool]:
        """Run multiple competitions in batch for four-folder structure"""
        
        results = []
        total = len(competitions)
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting batch of {total} competitions...")
        
        for i, comp in enumerate(competitions):
            comp_start = time.time()
            
            # Create game config
            game_config = get_game_config(
                comp['game_name'], 
                comp['experiment_type'], 
                comp['condition_name']
            )
            
            # Create output directory path
            output_dir = f"results/{comp['game_name']}/{comp['challenger_model']}"
            
            self.logger.info(f"🔄 Running {i+1}/{total}: "
                           f"{comp['game_name']} - {comp['condition_name']} - {comp['challenger_model']}")
            
            try:
                success = await self.run_single_competition(
                    game_config, comp['challenger_model'], output_dir
                )
                results.append(success)
                
                comp_duration = time.time() - comp_start
                status = "✅" if success else "❌"
                self.logger.info(f"  {status} {i+1}/{total} completed in {comp_duration:.1f}s")
                
            except Exception as e:
                results.append(False)
                comp_duration = time.time() - comp_start
                self.logger.error(f"  ❌ Failed in {comp_duration:.1f}s: {e}")
        
        total_duration = time.time() - start_time
        successful = sum(results)
        
        self.logger.info("=" * 80)
        self.logger.info("🎯 BATCH COMPETITIONS COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Success: {successful}/{total} competitions")
        self.logger.info(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        
        return results


# Convenience functions
async def run_single_competition_standalone(game_name: str, experiment_type: str, condition_name: str,
                                          challenger_model: str, defender_model: Optional[str] = None,
                                          mock_mode: bool = False) -> bool:
    """Run a single competition standalone"""
    
    challenger_models = [challenger_model]
    if defender_model is None:
        defender_model = get_defender_model()
    
    competition = Competition(challenger_models, defender_model, mock_mode=mock_mode)
    
    game_config = get_game_config(game_name, experiment_type, condition_name)
    output_dir = f"results/{game_name}/{challenger_model}"
    
    return await competition.run_single_competition(game_config, challenger_model, output_dir)


async def run_full_experimental_suite(challenger_models: Optional[List[str]] = None, 
                                    defender_model: Optional[str] = None,
                                    mock_mode: bool = False) -> List[bool]:
    """Run complete experimental suite for all games and conditions"""
    
    if challenger_models is None:
        challenger_models = get_challenger_models()
    
    if defender_model is None:
        defender_model = get_defender_model()
    
    competition = Competition(challenger_models, defender_model, mock_mode=mock_mode)
    
    # Generate all competition configurations using the matrix approach
    competitions = []
    
    for game_name in ['salop', 'green_porter', 'spulber', 'athey_bagwell']:
        # Get all conditions for this game (includes baseline + structural + combined)
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