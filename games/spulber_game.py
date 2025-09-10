"""
Spulber Bertrand Competition Game - Compact implementation with full config integration
Implements Winner Determination Algorithm from t.txt for comprehensive metrics analysis
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import StaticGame, extract_numeric_value, validate_action_bounds
from config import GameConfig, get_prompt_variables


class SpulberGame(StaticGame):
    """
    Spulber Bertrand Competition - winner-take-all price auction with incomplete information
    Implements Winner Determination Algorithm from t.txt
    """
    
    def __init__(self):
        super().__init__("spulber")
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template from markdown file"""
        prompt_path = Path("prompts/spulber.md")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract template between ``` blocks
        template_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if not template_match:
            raise ValueError("No template found in spulber.md")
        
        return template_match.group(1)

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt using template and config"""
        
        # Get all template variables from config
        variables = get_prompt_variables(game_config, player_id=player_id)
        
        # Format template with variables
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse bidding decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.basic_json_parse(response)
        if json_action and 'price' in json_action:
            price = json_action['price']
            if isinstance(price, (int, float)) and price >= 0:
                return {'price': float(price), 'raw_response': response}
        
        # Try numeric extraction for 'price' and 'bid'
        price = extract_numeric_value(response, 'price')
        if price > 0:
            return {'price': price, 'parsing_method': 'regex', 'raw_response': response}
        
        bid = extract_numeric_value(response, 'bid')
        if bid > 0:
            return {'price': bid, 'parsing_method': 'regex', 'raw_response': response}
        
        # Try simple number extraction - first reasonable number found
        number_patterns = [r'\$?(\d+\.?\d*)', r'(\d+\.?\d*)', r'(\d+\.?\d+)']
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    price = float(matches[0])
                    if price >= 0:
                        return {'price': price, 'parsing_method': 'number', 'raw_response': response}
                except ValueError:
                    continue
        
        self.logger.warning(f"[{call_id}] Could not parse price from {player_id}")
        return None

    def get_default_action(self, player_id: str, game_state: Dict, 
                         game_config: GameConfig) -> Dict[str, Any]:
        """Default pricing action when parsing fails"""
        # Use player's own cost + small markup as default
        private_values = game_config.constants.get('private_values', {})
        if player_id == 'challenger':
            own_cost = private_values.get('challenger_cost', 8)
        else:
            own_cost = private_values.get('defender_cost', 10)
        
        default_price = own_cost + 2.0
        
        return {
            'price': default_price,
            'reasoning': 'Default pricing due to parsing failure',
            'parsing_success': False,
            'player_id': player_id
        }

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Spulber Bertrand payoffs using Winner Determination Algorithm from t.txt
        
        Algorithm from t.txt:
        1. Find the minimum bid: p_min = min(p_1, p_2, ..., p_N)
        2. Identify all players (K) who bid p_min
        3. Tie-Breaking Rule: market_share for each winner = 1/K, others = 0
        4. Payoff = (price - cost) * market_share * market_size
        """
        constants = game_config.constants
        market_size = constants.get('market_size', 100)
        private_values = constants.get('private_values', {})
        
        # Extract prices and costs
        players = list(actions.keys())
        prices = {}
        costs = {}
        
        for player_id, action in actions.items():
            price = action.get('price', 0)
            prices[player_id] = max(0.0, price)  # Ensure non-negative
            
            # Get player's private cost
            if player_id == 'challenger':
                costs[player_id] = private_values.get('challenger_cost', 8)
            else:
                costs[player_id] = private_values.get('defender_cost', 10)
        
        # Winner Determination Algorithm
        min_price = min(prices.values())
        winners = [pid for pid, price in prices.items() if price == min_price]
        market_share_per_winner = 1.0 / len(winners)
        
        # Calculate payoffs
        payoffs = {}
        for player_id in players:
            if player_id in winners:
                market_share = market_share_per_winner
            else:
                market_share = 0.0
            
            profit_margin = prices[player_id] - costs[player_id]
            payoff = profit_margin * market_share * market_size
            payoffs[player_id] = payoff
        
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get structured data for metrics calculation and logging"""
        constants = game_config.constants
        private_values = constants.get('private_values', {})
        
        # Extract prices and calculate market outcomes
        prices = {pid: action.get('price', 0) for pid, action in actions.items()}
        min_price = min(prices.values())
        winners = [pid for pid, price in prices.items() if price == min_price]
        
        # Calculate challenger-specific metrics
        challenger_price = prices.get('challenger', 0)
        challenger_cost = private_values.get('challenger_cost', 8)
        challenger_won = 'challenger' in winners
        
        # Calculate profit margins
        profit_margins = {}
        for player_id, price in prices.items():
            if player_id == 'challenger':
                cost = private_values.get('challenger_cost', 8)
            else:
                cost = private_values.get('defender_cost', 10)
            profit_margins[player_id] = (price - cost) / price if price > 0 else 0
        
        return {
            'game_name': game_config.game_name,
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'actions': actions,
            'payoffs': payoffs,
            'constants': constants,
            'game_state': game_state,
            
            # Spulber-specific metrics data
            'spulber_metrics': {
                'prices': prices,
                'min_price': min_price,
                'winners': winners,
                'challenger_won': challenger_won,
                'challenger_price': challenger_price,
                'challenger_cost': challenger_cost,
                'challenger_profit_margin': profit_margins.get('challenger', 0),
                'profit_margins': profit_margins,
                'num_players': len(actions),
                'market_concentration': len(actions)  # For structural variation analysis
            }
        }