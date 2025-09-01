"""
Minimal base game system - only what's needed for comprehensive_metrics.py
Provides essential game execution and data structures for metrics calculation
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Import minimal config system
from config import GameConfig, GameConstants, PlayerResult, GameResult


class EconomicGame(ABC):
    """
    Minimal base class for economic games.
    Only includes what's needed to generate data for comprehensive_metrics.py
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"games.{name}.{self.__class__.__name__}")
        
        self.logger.info(f"Initialized {name} game")

    def run_game(self, config: GameConfig, agents: Dict[str, Any], call_id: str) -> GameResult:
        """
        Run complete game - main entry point called by competition.py
        Returns GameResult object needed by comprehensive_metrics.py
        """
        self.logger.info(f"[{call_id}] Starting {self.name} game")
        
        # Initialize game state
        game_state = {
            'current_round': 1,
            'total_rounds': config.number_of_rounds,
            'number_of_players': config.number_of_players,
            'game_name': self.name,
            'call_id': call_id,
            'player_histories': {}
        }
        
        # Initialize player histories
        for agent_id in agents.keys():
            game_state['player_histories'][agent_id] = {
                'actions': [],
                'profits': [],
                'cumulative_profit': 0.0
            }
        
        # Get game constants
        constants = GameConstants()
        
        try:
            # Run game rounds
            for round_num in range(1, config.number_of_rounds + 1):
                game_state['current_round'] = round_num
                self.logger.info(f"[{call_id}] Round {round_num}/{config.number_of_rounds}")
                
                # Get actions from all players
                actions = self._collect_actions(agents, game_state, config, constants)
                
                # Calculate payoffs (implemented by subclasses)
                payoffs = self.calculate_payoffs(actions, config, game_state)
                
                # Update player histories
                self._update_histories(game_state, actions, payoffs)
                
                # Update game state for next round (if multi-round)
                game_state = self.update_game_state(game_state, actions, config)
            
            # Calculate final results
            player_results = self._create_player_results(game_state, config, call_id)
            
            # Calculate total industry profit
            total_profit = sum(pr.profit for pr in player_results)
            
            # Create game result
            game_result = GameResult(
                game_name=self.name,
                config=config,
                players=player_results,
                total_industry_profit=total_profit,
                game_id=call_id
            )
            
            # Log final results
            self._log_final_results(player_results, call_id)
            
            self.logger.info(f"[{call_id}] Game completed successfully")
            return game_result
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Game failed: {e}")
            raise

    def _collect_actions(self, agents: Dict[str, Any], game_state: Dict, 
                        config: GameConfig, constants: GameConstants) -> Dict[str, Any]:
        """Collect actions from all players"""
        actions = {}
        
        for agent_id, agent in agents.items():
            try:
                # Generate prompt
                prompt = self.generate_player_prompt(agent_id, game_state, config, constants)
                
                # Get LLM response
                response = agent.call_llm(prompt, f"{game_state['call_id']}_{agent_id}")
                
                # Parse action
                if response.error:
                    self.logger.warning(f"Agent {agent_id} failed: {response.error}")
                    action = self.get_default_action(agent_id, game_state, config)
                else:
                    action = self.parse_llm_response(response.final_response, agent_id, game_state['call_id'])
                    if action is None:
                        action = self.get_default_action(agent_id, game_state, config)
                
                # Add metadata needed for metrics
                action['player_id'] = agent_id
                action['round'] = game_state['current_round']
                action['game_name'] = self.name
                
                actions[agent_id] = action
                
            except Exception as e:
                self.logger.error(f"Error collecting action from {agent_id}: {e}")
                actions[agent_id] = self.get_default_action(agent_id, game_state, config)
        
        return actions

    def _update_histories(self, game_state: Dict, actions: Dict[str, Any], payoffs: Dict[str, float]):
        """Update player histories with actions and payoffs"""
        for agent_id in actions.keys():
            if agent_id in game_state['player_histories']:
                history = game_state['player_histories'][agent_id]
                history['actions'].append(actions[agent_id])
                history['profits'].append(payoffs.get(agent_id, 0.0))
                history['cumulative_profit'] += payoffs.get(agent_id, 0.0)

    def _create_player_results(self, game_state: Dict, config: GameConfig, call_id: str) -> List[PlayerResult]:
        """Create PlayerResult objects needed by comprehensive_metrics.py"""
        player_results = []
        
        for agent_id, history in game_state['player_histories'].items():
            # Create enhanced actions list for metrics
            enhanced_actions = []
            actions = history.get('actions', [])
            profits = history.get('profits', [])
            
            for i, action in enumerate(actions):
                enhanced_action = {
                    'round': i + 1,
                    'action_data': action,
                    'profit': profits[i] if i < len(profits) else 0.0,
                    'reasoning': action.get('reasoning', 'No reasoning provided')
                }
                # Also include raw action data for direct access
                enhanced_action.update(action)
                enhanced_actions.append(enhanced_action)
            
            # Determine player role for metrics
            player_role = "challenger" if "challenger" in agent_id else "defender"
            
            # Create player result
            player_result = PlayerResult(
                player_id=agent_id,
                profit=history.get('cumulative_profit', 0.0),
                actions=enhanced_actions,
                win=False,  # Will be set below
                player_role=player_role
            )
            
            player_results.append(player_result)
        
        # Determine winners (highest profit)
        if player_results:
            max_profit = max(pr.profit for pr in player_results)
            for pr in player_results:
                pr.win = abs(pr.profit - max_profit) < 0.01
        
        return player_results

    def _log_final_results(self, player_results: List[PlayerResult], call_id: str):
        """Log final game results"""
        sorted_results = sorted(player_results, key=lambda x: x.profit, reverse=True)
        
        self.logger.info(f"[{call_id}] Calculating final results for {self.name}")
        self.logger.info(f"[{call_id}] Final standings for {self.name}:")
        
        for i, pr in enumerate(sorted_results):
            status = "🏆 WINNER" if pr.win else f"#{i+1}"
            self.logger.info(f"[{call_id}]   {status}: {pr.player_id} - Profit: ${pr.profit:.2f}")

    def basic_json_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """Basic JSON parsing utility"""
        try:
            # Look for JSON patterns
            json_patterns = [
                r'\{[^{}]*\}',  # Simple JSON object
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```'  # JSON in any code blocks
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        action = json.loads(match)
                        if isinstance(action, dict):
                            return action
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception:
            return None

    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             config: GameConfig, constants: GameConstants) -> str:
        """Generate prompt for player - implemented by specific games"""
        pass
    
    @abstractmethod
    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into action - implemented by specific games"""
        pass
    
    @abstractmethod
    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Get default action when parsing fails - implemented by specific games"""
        pass
    
    @abstractmethod
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate payoffs for all players - implemented by specific games"""
        pass
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """Update game state after round - default implementation"""
        return game_state


class StaticGame(EconomicGame):
    """
    Base class for static (single-round) games like Salop and Spulber.
    No state updates needed between rounds.
    """
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """Static games don't change state between rounds"""
        return game_state


class DynamicGame(EconomicGame):
    """
    Base class for dynamic (multi-round) games like Green-Porter and Athey-Bagwell.
    State evolves based on actions and outcomes.
    """
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """Store round history for dynamic games"""
        
        # Store round information for metrics analysis
        if 'round_history' not in game_state:
            game_state['round_history'] = []
        
        round_info = {
            'round': game_state['current_round'],
            'actions': actions.copy(),
            'total_players': len(actions)
        }
        
        game_state['round_history'].append(round_info)
        return game_state