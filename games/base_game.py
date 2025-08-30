import logging
import json
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import asdict

from config import GameConfig, GameConstants, PlayerResult

class EconomicGame(ABC):
    """
    Enhanced abstract base class for economic games with JSON configuration support
    and comprehensive debugging capabilities.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Enhanced tracking
        self.game_statistics = {}
        self.action_history = []
        self.parsing_failures = 0
        self.total_actions_parsed = 0
        
        self.logger.debug(f"Initialized {name} game with enhanced capabilities")
    
    @abstractmethod
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        """
        Create strategic prompt for player decision-making.
        
        Args:
            player_id: Unique identifier for the player
            game_state: Current game state with history and context
            config: Game configuration from JSON
            
        Returns:
            Formatted prompt string with strategic context
        """
        pass
    
    def parse_action(self, response: str, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Enhanced action parsing with multiple fallback methods and comprehensive logging.
        
        This method provides a standardized interface for parsing LLM responses across all games,
        with robust error handling and detailed debugging information.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            self.total_actions_parsed += 1
            self.logger.debug(f"[{call_id}] Parsing action for {player_id} in {self.name}")
            
            # Try JSON parsing first (preferred method)
            action = self._parse_json_response(response, player_id, call_id)
            if action:
                action['parsing_method'] = 'json'
                action['parsing_success'] = True
                return self._validate_and_enhance_action(action, player_id, game_state, config)
            
            # Try pattern-based parsing
            action = self._parse_pattern_response(response, player_id, call_id)
            if action:
                action['parsing_method'] = 'pattern'
                action['parsing_success'] = True
                return self._validate_and_enhance_action(action, player_id, game_state, config)
            
            # Try intelligent text parsing
            action = self._parse_intelligent_response(response, player_id, call_id)
            if action:
                action['parsing_method'] = 'intelligent'
                action['parsing_success'] = True
                return self._validate_and_enhance_action(action, player_id, game_state, config)
            
            # Final fallback to default action
            self.parsing_failures += 1
            self.logger.warning(f"[{call_id}] All parsing methods failed for {player_id}, using default")
            
            return self.get_default_action(player_id, game_state, config)
            
        except Exception as e:
            self.parsing_failures += 1
            self.logger.error(f"[{call_id}] Critical error parsing action for {player_id}: {e}")
            return self.get_default_action(player_id, game_state, config)
    
    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Provide intelligent default action when parsing fails.
        Subclasses should override for game-specific defaults.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        default_action = {
            'parsing_method': 'default',
            'parsing_success': False,
            'raw_response': 'PARSING_FAILED',
            'player_id': player_id,
            'round': game_state.get('current_round', 1),
            'reasoning': f'Default action due to parsing failure in {self.name}'
        }
        
        self.logger.warning(f"[{call_id}] Using base default action for {player_id} in {self.name}")
        return default_action
    
    @abstractmethod
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Union[Dict[str, float], Tuple[Dict[str, float], Any]]:
        """
        Calculate payoffs for all players based on their actions.
        
        Args:
            actions: Dictionary of player actions
            config: Game configuration
            game_state: Current game state (optional)
            
        Returns:
            Dictionary of player payoffs, or tuple of (payoffs, additional_info)
        """
        pass
    
    @abstractmethod
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """
        Update game state after actions are resolved.
        
        Args:
            game_state: Current game state
            actions: Player actions from this round
            config: Game configuration
            
        Returns:
            Updated game state
        """
        pass
    
    def calculate_final_results(self, game_state: Dict, config: GameConfig) -> List[PlayerResult]:
        """
        Calculate final game results with enhanced analysis.
        
        This method provides comprehensive result calculation including win determination,
        performance metrics, and strategic analysis.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            self.logger.info(f"[{call_id}] Calculating final results for {self.name}")
            
            player_results = []
            player_histories = game_state.get('player_histories', {})
            
            if not player_histories:
                self.logger.warning(f"[{call_id}] No player histories found in game state")
                return []
            
            # Calculate results for each player
            for player_id, history in player_histories.items():
                player_result = self._calculate_player_result(
                    player_id, history, game_state, config, call_id
                )
                player_results.append(player_result)
            
            # Determine winners and rankings
            player_results = self._determine_winners(player_results, call_id)
            
            # Log final standings
            self._log_final_standings(player_results, call_id)
            
            # Store game statistics for analysis
            self._update_game_statistics(player_results, game_state)
            
            return player_results
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error calculating final results: {e}")
            return []
    
    def initialize_game_state(self, config: GameConfig, experiment_config: Optional[Any] = None) -> Dict:
        """
        Enhanced game state initialization with comprehensive tracking.
        """
        game_state = {
            'current_round': 1,
            'total_rounds': config.number_of_rounds,
            'number_of_players': config.number_of_players,
            'game_name': self.name,
            'player_histories': {},
            'round_history': [],
            'game_statistics': {},
            'config': asdict(config),
            'start_timestamp': None,  # Will be set by competition system
            'call_id': None  # Will be set by competition system
        }
        
        # Initialize player histories
        for i in range(config.number_of_players):
            player_id = f"player_{i+1}"
            game_state['player_histories'][player_id] = {
                'actions': [],
                'profits': [],
                'cumulative_profit': 0.0,
                'reasoning': [],
                'performance_metrics': {}
            }
        
        self.logger.debug(f"Initialized game state for {self.name} with {config.number_of_players} players")
        return game_state
    
    def validate_action(self, action: Dict[str, Any], player_id: str, config: GameConfig) -> bool:
        """
        Enhanced action validation with game-specific constraints.
        Subclasses should override for specific validation rules.
        """
        try:
            # Basic validation - ensure action is a dictionary
            if not isinstance(action, dict):
                self.logger.warning(f"Action for {player_id} is not a dictionary: {type(action)}")
                return False
            
            # Check for required fields (game-specific, override in subclasses)
            required_fields = self.get_required_action_fields()
            for field in required_fields:
                if field not in action:
                    self.logger.warning(f"Missing required field '{field}' in action for {player_id}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating action for {player_id}: {e}")
            return False
    
    def get_required_action_fields(self) -> List[str]:
        """
        Get required fields for actions. Override in subclasses.
        """
        return ['reasoning']  # All games should provide reasoning
    
    def calculate_game_metrics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Calculate comprehensive game-level metrics for analysis.
        """
        try:
            player_histories = game_state.get('player_histories', {})
            
            if not player_histories:
                return {}
            
            # Basic metrics
            total_rounds_played = len(next(iter(player_histories.values())).get('profits', []))
            total_players = len(player_histories)
            
            # Profit analysis
            all_profits = []
            for history in player_histories.values():
                all_profits.extend(history.get('profits', []))
            
            profit_stats = {
                'total_industry_profit': sum(all_profits),
                'average_profit_per_round': np.mean(all_profits) if all_profits else 0,
                'profit_variance': np.var(all_profits) if len(all_profits) > 1 else 0,
                'max_individual_profit': max(all_profits) if all_profits else 0,
                'min_individual_profit': min(all_profits) if all_profits else 0
            }
            
            # Strategic analysis
            strategic_metrics = self._calculate_strategic_metrics(game_state, config)
            
            # Parsing statistics
            parsing_stats = {
                'total_actions_parsed': self.total_actions_parsed,
                'parsing_failures': self.parsing_failures,
                'parsing_success_rate': (self.total_actions_parsed - self.parsing_failures) / self.total_actions_parsed if self.total_actions_parsed > 0 else 0
            }
            
            return {
                'basic_metrics': {
                    'total_rounds_played': total_rounds_played,
                    'total_players': total_players,
                    'game_name': self.name
                },
                'profit_statistics': profit_stats,
                'strategic_metrics': strategic_metrics,
                'parsing_statistics': parsing_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating game metrics: {e}")
            return {}
    
    # Protected helper methods for parsing
    
    def _parse_json_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse JSON-formatted response with enhanced error handling"""
        try:
            # Try to find JSON object in response
            json_patterns = [
                r'\{[^{}]*\}',  # Simple JSON object
                r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Nested JSON
                r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
                r'```\s*(\{.*?\})\s*```'  # JSON in generic code blocks
            ]
            
            for pattern in json_patterns:
                matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        json_str = match.group(1) if match.lastindex else match.group(0)
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            self.logger.debug(f"[{call_id}] Successfully parsed JSON for {player_id}")
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Try parsing entire response
            try:
                parsed = json.loads(response.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            
            return None
            
        except Exception as e:
            self.logger.debug(f"[{call_id}] JSON parsing error for {player_id}: {e}")
            return None
    
    def _parse_pattern_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse response using pattern matching. Override in subclasses for game-specific patterns."""
        return None  # Base implementation returns None
    
    def _parse_intelligent_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Intelligent text parsing using NLP-like techniques.
        This method tries to extract meaningful decisions from free-form text.
        """
        try:
            response_lower = response.lower()
            
            # Look for decision indicators
            decision_patterns = [
                r'(?:i choose|i select|i decide|my choice is|i will)\s+([^.!?\n]+)',
                r'(?:decision|choice|action):\s*([^.!?\n]+)',
                r'(?:therefore|thus|so)\s+(?:i|my)\s+([^.!?\n]+)'
            ]
            
            for pattern in decision_patterns:
                match = re.search(pattern, response_lower)
                if match:
                    decision_text = match.group(1).strip()
                    
                    # Extract reasoning (everything before the decision)
                    decision_pos = response_lower.find(match.group(0))
                    reasoning = response[:decision_pos].strip() if decision_pos > 0 else response[:100]
                    
                    parsed_action = {
                        'decision_text': decision_text,
                        'reasoning': reasoning,
                        'raw_response': response[:500]  # First 500 chars for debugging
                    }
                    
                    # Try to extract specific game elements (override in subclasses)
                    parsed_action = self._enhance_intelligent_parsing(parsed_action, response_lower)
                    
                    if parsed_action:
                        self.logger.debug(f"[{call_id}] Successfully parsed intelligent response for {player_id}")
                        return parsed_action
            
            return None
            
        except Exception as e:
            self.logger.debug(f"[{call_id}] Intelligent parsing error for {player_id}: {e}")
            return None
    
    def _enhance_intelligent_parsing(self, parsed_action: Dict[str, Any], response_lower: str) -> Optional[Dict[str, Any]]:
        """
        Enhance intelligent parsing with game-specific logic.
        Override in subclasses for game-specific enhancements.
        """
        return parsed_action
    
    def _validate_and_enhance_action(self, action: Dict[str, Any], player_id: str, 
                                   game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Validate and enhance parsed action with additional metadata.
        """
        # Add standard metadata
        action['player_id'] = player_id
        action['round'] = game_state.get('current_round', 1)
        action['game_name'] = self.name
        action['parsing_timestamp'] = game_state.get('call_id', 'unknown')
        
        # Ensure raw_response is limited in size
        if 'raw_response' in action:
            action['raw_response'] = action['raw_response'][:500]  # Limit to 500 chars
        
        # Validate using game-specific rules
        is_valid = self.validate_action(action, player_id, config)
        action['validation_passed'] = is_valid
        
        if not is_valid:
            self.logger.warning(f"Action validation failed for {player_id}, but proceeding with action")
        
        return action
    
    def _calculate_player_result(self, player_id: str, history: Dict, game_state: Dict, 
                               config: GameConfig, call_id: str) -> PlayerResult:
        """
        Calculate comprehensive result for a single player.
        Override in subclasses for game-specific result calculation.
        """
        profits = history.get('profits', [])
        actions = history.get('actions', [])
        
        # Calculate total profit (subclasses may override for discounting)
        total_profit = sum(profits)
        
        # Create player result with enhanced action tracking
        enhanced_actions = []
        for i, action in enumerate(actions):
            enhanced_action = {
                'round': i + 1,
                'action_data': action,
                'profit': profits[i] if i < len(profits) else 0,
                'reasoning': action.get('reasoning', 'No reasoning provided')
            }
            enhanced_actions.append(enhanced_action)
        
        # Determine player role
        player_role = "challenger" if player_id == "challenger" else "defender"
        
        player_result = PlayerResult(
            player_id=player_id,
            profit=total_profit,
            actions=enhanced_actions,
            win=False,  # Will be determined later
            player_role=player_role
        )
        
        return player_result
    
    def _determine_winners(self, player_results: List[PlayerResult], call_id: str) -> List[PlayerResult]:
        """Determine winners based on profit."""
        if not player_results:
            return player_results
        
        max_profit = max(pr.profit for pr in player_results)
        
        for player_result in player_results:
            # Allow for small floating-point differences
            player_result.win = abs(player_result.profit - max_profit) < 0.01
        
        winners = [pr for pr in player_results if pr.win]
        self.logger.debug(f"[{call_id}] Determined {len(winners)} winner(s) in {self.name}")
        
        return player_results
    
    def _log_final_standings(self, player_results: List[PlayerResult], call_id: str):
        """Log final game standings."""
        sorted_results = sorted(player_results, key=lambda x: x.profit, reverse=True)
        
        self.logger.info(f"[{call_id}] Final standings for {self.name}:")
        for i, pr in enumerate(sorted_results):
            status = "🏆 WINNER" if pr.win else f"#{i+1}"
            self.logger.info(f"[{call_id}]   {status}: {pr.player_id} - Profit: ${pr.profit:.2f}")
    
    def _calculate_strategic_metrics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Calculate strategic metrics. Override in subclasses for game-specific metrics.
        """
        return {
            'strategic_analysis': 'Base implementation - override in subclasses'
        }
    
    def _update_game_statistics(self, player_results: List[PlayerResult], game_state: Dict):
        """Update internal game statistics for analysis."""
        self.game_statistics = {
            'total_players': len(player_results),
            'winner_count': sum(1 for pr in player_results if pr.win),
            'total_profit_generated': sum(pr.profit for pr in player_results),
            'parsing_success_rate': (self.total_actions_parsed - self.parsing_failures) / self.total_actions_parsed if self.total_actions_parsed > 0 else 0,
            'game_completed': True
        }


class StaticGame(EconomicGame):
    """
    Enhanced base class for static (single-round) economic games.
    
    Static games involve simultaneous decisions with immediate payoff resolution,
    such as Cournot competition, auctions, or public goods games.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        self.logger.debug(f"Initialized static game: {name}")
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """
        Enhanced game state update for static games.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            # Ensure player_histories exists
            if 'player_histories' not in game_state:
                game_state['player_histories'] = {}
            
            # Calculate payoffs
            payoff_result = self.calculate_payoffs(actions, config, game_state)
            
            # Handle both single dict and tuple returns
            if isinstance(payoff_result, tuple):
                profits, additional_info = payoff_result
                game_state['additional_info'] = additional_info
            else:
                profits = payoff_result
            
            # Update player histories
            round_num = game_state.get('current_round', 1)
            
            for player_id, action in actions.items():
                if player_id not in game_state['player_histories']:
                    game_state['player_histories'][player_id] = {
                        'actions': [],
                        'profits': [],
                        'cumulative_profit': 0.0,
                        'reasoning': []
                    }
                
                profit = profits.get(player_id, 0.0)
                reasoning = action.get('reasoning', 'No reasoning provided')
                
                # Update histories
                player_hist = game_state['player_histories'][player_id]
                player_hist['actions'].append(action)
                player_hist['profits'].append(profit)
                
                # Safely update cumulative profit with fallback
                if 'cumulative_profit' not in player_hist:
                    player_hist['cumulative_profit'] = 0.0
                player_hist['cumulative_profit'] += profit
                
                # Safely update reasoning with fallback
                if 'reasoning' not in player_hist:
                    player_hist['reasoning'] = []
                player_hist['reasoning'].append(reasoning)
                
                self.logger.debug(f"[{call_id}] Updated {player_id}: profit=${profit:.2f}")
            
            # Ensure round_history exists
            if 'round_history' not in game_state:
                game_state['round_history'] = []
            
            # Store round information
            game_state['round_history'].append({
                'round': round_num,
                'actions': actions.copy(),
                'profits': profits.copy(),
                'timestamp': game_state.get('call_id', 'unknown')
            })
            
            return game_state
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error updating static game state: {e}")
            return game_state


class DynamicGame(EconomicGame):
    """
    Enhanced base class for dynamic (multi-round) economic games.
    
    Dynamic games involve repeated interactions with state evolution,
    such as repeated prisoner's dilemma, dynamic oligopolies, or reputation games.
    """
    
    def __init__(self, name: str, default_rounds: int = 10):
        super().__init__(name)
        self.default_rounds = default_rounds
        self.round_statistics = []
        self.state_evolution_history = []
        
        self.logger.debug(f"Initialized dynamic game: {name} with default {default_rounds} rounds")
    
    def calculate_npv(self, payoffs_history: List[Dict[str, float]], 
                     player_id: str, discount_factor: float) -> float:
        """
        Enhanced NPV calculation with comprehensive logging.
        """
        try:
            npv = 0.0
            for round_num, payoffs in enumerate(payoffs_history):
                if player_id in payoffs:
                    discounted_payoff = payoffs[player_id] * (discount_factor ** round_num)
                    npv += discounted_payoff
            
            self.logger.debug(f"Calculated NPV for {player_id}: ${npv:.2f} (discount={discount_factor})")
            return npv
            
        except Exception as e:
            self.logger.error(f"Error calculating NPV for {player_id}: {e}")
            return 0.0
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """
        Enhanced game state update for dynamic games with state evolution tracking.
        """
        call_id = game_state.get('call_id', 'unknown')
        
        try:
            # Calculate payoffs for this round
            payoff_result = self.calculate_payoffs(actions, config, game_state)
            
            # Handle both single dict and tuple returns
            if isinstance(payoff_result, tuple):
                profits, additional_info = payoff_result
                if additional_info is not None:
                    game_state[f'round_{game_state.get("current_round", 1)}_info'] = additional_info
            else:
                profits = payoff_result
            
            # Update player histories with comprehensive tracking
            current_round = game_state.get('current_round', 1)
            
            for player_id, action in actions.items():
                if player_id not in game_state['player_histories']:
                    game_state['player_histories'][player_id] = {
                        'actions': [],
                        'profits': [],
                        'cumulative_profit': 0.0,
                        'reasoning': [],
                        'round_by_round_analysis': []
                    }
                
                profit = profits.get(player_id, 0.0)
                reasoning = action.get('reasoning', 'No reasoning provided')
                
                # Enhanced history tracking
                player_hist = game_state['player_histories'][player_id]
                player_hist['actions'].append(action)
                player_hist['profits'].append(profit)
                
                # Safely update cumulative profit with fallback
                if 'cumulative_profit' not in player_hist:
                    player_hist['cumulative_profit'] = 0.0
                player_hist['cumulative_profit'] += profit
                
                player_hist['reasoning'].append(reasoning)
                
                # Round-by-round analysis
                round_analysis = {
                    'round': current_round,
                    'profit': profit,
                    'cumulative_profit': player_hist['cumulative_profit'],
                    'action_summary': self._summarize_action(action),
                    'strategic_classification': self._classify_strategy(action, game_state, player_id)
                }
                player_hist['round_by_round_analysis'].append(round_analysis)
                
                self.logger.debug(f"[{call_id}] Round {current_round} - {player_id}: profit=${profit:.2f}, cumulative=${player_hist['cumulative_profit']:.2f}")
            
            # Track state evolution
            round_summary = {
                'round': current_round,
                'total_profits': sum(profits.values()),
                'profit_distribution': profits.copy(),
                'actions_summary': {pid: self._summarize_action(action) for pid, action in actions.items()},
                'market_conditions': self._analyze_market_conditions(game_state, actions, profits)
            }
            
            if 'state_evolution' not in game_state:
                game_state['state_evolution'] = []
            game_state['state_evolution'].append(round_summary)
            
            # Store detailed round information
            game_state['round_history'].append({
                'round': current_round,
                'actions': actions.copy(),
                'profits': profits.copy(),
                'round_analysis': round_summary,
                'timestamp': call_id
            })
            
            return game_state
            
        except Exception as e:
            self.logger.error(f"[{call_id}] Error updating dynamic game state: {e}")
            return game_state
    
    def _calculate_player_result(self, player_id: str, history: Dict, game_state: Dict, 
                               config: GameConfig, call_id: str) -> PlayerResult:
        """
        Enhanced player result calculation for dynamic games with NPV calculation.
        """
        profits = history.get('profits', [])
        actions = history.get('actions', [])
        
        # Calculate NPV for dynamic games
        payoffs_history = [{player_id: profit} for profit in profits]
        discounted_profit = self.calculate_npv(payoffs_history, player_id, config.discount_factor)
        
        # Create enhanced actions with temporal analysis
        enhanced_actions = []
        for i, action in enumerate(actions):
            round_profit = profits[i] if i < len(profits) else 0
            discounted_round_profit = round_profit * (config.discount_factor ** i)
            
            enhanced_action = {
                'round': i + 1,
                'action_data': action,
                'round_profit': round_profit,
                'discounted_profit': discounted_round_profit,
                'reasoning': action.get('reasoning', 'No reasoning provided'),
                'strategic_classification': self._classify_strategy(action, game_state, player_id),
                'cumulative_profit': sum(profits[:i+1])
            }
            enhanced_actions.append(enhanced_action)
        
        # Determine player role
        player_role = "challenger" if player_id == "challenger" else "defender"
        
        # Create player result with NPV
        player_result = PlayerResult(
            player_id=player_id,
            profit=discounted_profit,  # Use NPV for dynamic games
            actions=enhanced_actions,
            win=False,
            player_role=player_role
        )
        
        # Add dynamic game specific metrics
        player_result.additional_metrics = {
            'undiscounted_total_profit': sum(profits),
            'average_round_profit': np.mean(profits) if profits else 0,
            'profit_volatility': np.std(profits) if len(profits) > 1 else 0,
            'strongest_round': max(enumerate(profits), key=lambda x: x[1])[0] + 1 if profits else 0,
            'weakest_round': min(enumerate(profits), key=lambda x: x[1])[0] + 1 if profits else 0,
            'strategic_consistency': self._calculate_strategic_consistency(actions)
        }
        
        self.logger.debug(f"[{call_id}] Calculated NPV for {player_id}: ${discounted_profit:.2f} (undiscounted: ${sum(profits):.2f})")
        
        return player_result
    
    def _summarize_action(self, action: Dict[str, Any]) -> str:
        """
        Create a brief summary of an action. Override in subclasses for game-specific summaries.
        """
        if 'quantity' in action:
            return f"Q={action['quantity']}"
        elif 'price' in action:
            return f"P=${action['price']}"
        elif 'bid' in action:
            return f"Bid=${action['bid']}"
        else:
            return "Action taken"
    
    def _classify_strategy(self, action: Dict[str, Any], game_state: Dict, player_id: str) -> str:
        """
        Classify the strategic nature of an action. Override in subclasses for game-specific classification.
        """
        return "Strategic action"  # Base implementation
    
    def _analyze_market_conditions(self, game_state: Dict, actions: Dict[str, Any], 
                                 profits: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze current market conditions. Override in subclasses for game-specific analysis.
        """
        total_profit = sum(profits.values())
        avg_profit = total_profit / len(profits) if profits else 0
        
        return {
            'total_market_profit': total_profit,
            'average_player_profit': avg_profit,
            'profit_concentration': max(profits.values()) / total_profit if total_profit > 0 else 0,
            'active_players': len([p for p in profits.values() if p > 0])
        }
    
    def _calculate_strategic_consistency(self, actions: List[Dict[str, Any]]) -> float:
        """
        Calculate how consistent a player's strategy has been over time.
        Override in subclasses for game-specific consistency measures.
        """
        if len(actions) < 2:
            return 1.0  # Perfect consistency with insufficient data
        
        # Basic implementation - can be overridden for more sophisticated analysis
        return 0.5  # Neutral consistency score
    
    def _calculate_strategic_metrics(self, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """
        Calculate strategic metrics specific to dynamic games.
        """
        try:
            state_evolution = game_state.get('state_evolution', [])
            
            if not state_evolution:
                return {'error': 'No state evolution data'}
            
            # Calculate market evolution metrics
            total_profits_over_time = [round_data['total_profits'] for round_data in state_evolution]
            
            metrics = {
                'market_evolution': {
                    'initial_total_profit': total_profits_over_time[0] if total_profits_over_time else 0,
                    'final_total_profit': total_profits_over_time[-1] if total_profits_over_time else 0,
                    'peak_profit_round': np.argmax(total_profits_over_time) + 1 if total_profits_over_time else 0,
                    'market_volatility': np.std(total_profits_over_time) if len(total_profits_over_time) > 1 else 0,
                    'profit_trend': self._calculate_trend(total_profits_over_time)
                },
                'strategic_dynamics': {
                    'cooperation_periods': self._identify_cooperation_periods(state_evolution),
                    'competitive_periods': self._identify_competitive_periods(state_evolution),
                    'market_regime_changes': len(self._detect_regime_changes(total_profits_over_time))
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating strategic metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction of a time series."""
        if len(values) < 3:
            return 'Insufficient data'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
        
        if slope > 0.1:
            return 'Rising'
        elif slope < -0.1:
            return 'Falling'
        else:
            return 'Stable'
    
    def _identify_cooperation_periods(self, state_evolution: List[Dict]) -> int:
        """
        Identify periods of apparent cooperation. Override for game-specific logic.
        """
        # Basic implementation - count periods with above-average profits
        if len(state_evolution) < 2:
            return 0
        
        profits = [round_data['total_profits'] for round_data in state_evolution]
        avg_profit = np.mean(profits)
        
        return sum(1 for profit in profits if profit > avg_profit * 1.1)
    
    def _identify_competitive_periods(self, state_evolution: List[Dict]) -> int:
        """
        Identify periods of intense competition. Override for game-specific logic.
        """
        # Basic implementation - count periods with below-average profits
        if len(state_evolution) < 2:
            return 0
        
        profits = [round_data['total_profits'] for round_data in state_evolution]
        avg_profit = np.mean(profits)
        
        return sum(1 for profit in profits if profit < avg_profit * 0.9)
    
    def _detect_regime_changes(self, values: List[float]) -> List[int]:
        """
        Detect significant regime changes in the time series.
        """
        if len(values) < 3:
            return []
        
        changes = []
        threshold = np.std(values) * 1.5  # 1.5 standard deviations
        
        for i in range(1, len(values) - 1):
            if abs(values[i] - values[i-1]) > threshold:
                changes.append(i)
        
        return changes