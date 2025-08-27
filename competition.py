# competition.py
import time
import json
import logging
from typing import Dict, List, Any
from config import GameConfig, PlayerResult, GameResult
from agents import LLMAgent
from games.salop_game import SalopGame
from games.spulber_game import SpulberGame
from games.green_porter_game import GreenPorterGame
from games.athey_bagwell_game import AtheyBagwellGame
from metrics.comprehensive_metrics import ComprehensiveMetricsCalculator

class GameCompetition:
    def __init__(self):
        self.games = {
            'salop': SalopGame(),
            'spulber': SpulberGame(),
            'green_porter': GreenPorterGame(),
            'athey_bagwell': AtheyBagwellGame()
        }
        self.metrics_calculator = ComprehensiveMetricsCalculator()
        self.logger = logging.getLogger(__name__)
    
    def run_single_game(self, game_name: str, challenger_agent: Any, 
                       defender_agents: List[Any], config: GameConfig) -> GameResult:
        if game_name not in self.games:
            raise ValueError(f"Unknown game: {game_name}")
        
        game = self.games[game_name]
        game_state = game.initialize_game_state(config)
        all_agents = {'1': challenger_agent}
        
        # Add defender agents
        for i, agent in enumerate(defender_agents):
            all_agents[str(i + 2)] = agent
        
        # Initialize player results
        player_results = []
        for player_id in all_agents.keys():
            player_results.append(PlayerResult(
                player_id=player_id,
                profit=0.0,
                actions=[],
                reasoning=[]
            ))
        
        # Run game rounds
        for round_num in range(1, config.number_of_rounds + 1):
            round_actions = {}
            round_reasoning = {}
            
            # Get decisions from all agents
            for player_id, agent in all_agents.items():
                try:
                    prompt = game.create_prompt(player_id, game_state, config)
                    decision = agent.make_decision(prompt, max_retries=config.max_retries)
                    
                    if 'error' in decision:
                        decision = self._get_default_action(game_name)
                    
                    round_actions[player_id] = decision
                    round_reasoning[player_id] = decision.get('reasoning', '')
                    
                except Exception as e:
                    self.logger.error(f"Error from player {player_id}: {e}")
                    round_actions[player_id] = self._get_default_action(game_name)
                    round_reasoning[player_id] = f"Error: {str(e)}"
            
            # Calculate payoffs
            try:
                if game_name == 'green_porter':
                    round_payoffs, market_price = game.calculate_payoffs(round_actions, config, game_state)
                else:
                    round_payoffs = game.calculate_payoffs(round_actions, config, game_state)
                    market_price = None
            except Exception as e:
                self.logger.error(f"Error calculating payoffs: {e}")
                round_payoffs = {pid: 0.0 for pid in round_actions.keys()}
                market_price = None
            
            # Update game state
            try:
                game_state = game.update_game_state(game_state, round_actions, round_num)
            except Exception as e:
                self.logger.error(f"Error updating game state: {e}")
            
            # Update player results
            for player_id in all_agents.keys():
                player_result = next(pr for pr in player_results if pr.player_id == player_id)
                player_result.actions.append(round_actions.get(player_id, {}))
                player_result.reasoning.append(round_reasoning.get(player_id, ''))
                
                round_profit = round_payoffs.get(player_id, 0)
                discounted_profit = round_profit * (config.discount_factor ** (round_num - 1))
                player_result.profit += discounted_profit
        
        # Determine winners
        max_profit = max(pr.profit for pr in player_results)
        for pr in player_results:
            pr.win = (abs(pr.profit - max_profit) < 0.01)
        
        return GameResult(
            game_name=game_name,
            config=config,
            players=player_results,
            total_industry_profit=sum(pr.profit for pr in player_results),
            market_price=market_price
        )
    
    def run_tournament(self, game_names: List[str], challenger_models: List[str], 
                      configs: List[GameConfig], num_games_per_config: int = 10) -> Dict[str, Any]:
        tournament_results = {
            'experiment_results': {},
            'comprehensive_metrics': {}
        }
        
        for game_name in game_names:
            for config in configs:
                for challenger_model in challenger_models:
                    experiment_key = f"{game_name}_{challenger_model}_p{config.number_of_players}_r{config.number_of_rounds}"
                    
                    self.logger.info(f"Running experiment: {experiment_key}")
                    
                    # Run games for this configuration
                    game_results = []
                    for game_idx in range(num_games_per_config):
                        try:
                            # Create agents
                            challenger = LLMAgent(challenger_model)
                            defenders = [LLMAgent("gemini-1.5-flash") 
                                       for _ in range(config.number_of_players - 1)]
                            
                            # Run single game
                            result = self.run_single_game(game_name, challenger, defenders, config)
                            game_results.append(result)
                            
                            time.sleep(1)  # Rate limiting
                            
                        except Exception as e:
                            self.logger.error(f"Game {game_idx} failed: {e}")
                            continue
                    
                    if game_results:
                        # Store results
                        tournament_results['experiment_results'][experiment_key] = game_results
                        
                        # Calculate comprehensive metrics
                        try:
                            comprehensive_metrics = self.metrics_calculator.calculate_game_metrics(
                                game_results, player_id='1'
                            )
                            tournament_results['comprehensive_metrics'][experiment_key] = comprehensive_metrics
                        except Exception as e:
                            self.logger.error(f"Failed to calculate metrics for {experiment_key}: {e}")
        
        return tournament_results
    
    def export_results(self, tournament_results: Dict[str, Any], output_dir: str) -> None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export comprehensive metrics
        metrics_data = {}
        for exp_key, game_metrics in tournament_results['comprehensive_metrics'].items():
            metrics_dict = {}
            
            # Convert GameMetrics to dict
            for category in ['primary_behavioral', 'core_performance', 'advanced_strategic']:
                category_metrics = getattr(game_metrics, category, {})
                for metric_name, metric_result in category_metrics.items():
                    metrics_dict[f"{category}_{metric_name}"] = {
                        'value': metric_result.value,
                        'formula': metric_result.formula
                    }
            
            metrics_data[exp_key] = metrics_dict
        
        with open(os.path.join(output_dir, "comprehensive_metrics.json"), 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Results exported to {output_dir}")
    
    def _get_default_action(self, game_name: str) -> Dict[str, Any]:
        defaults = {
            'salop': {'price': 12.0, 'reasoning': 'Default price'},
            'spulber': {'price': 10.0, 'reasoning': 'Default bid'},
            'green_porter': {'quantity': 25.0, 'reasoning': 'Default quantity'},
            'athey_bagwell': {'report': 'high', 'reasoning': 'Default report'}
        }
        return defaults.get(game_name, {'error': 'Unknown game'})