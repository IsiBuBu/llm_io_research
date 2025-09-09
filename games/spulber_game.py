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
        
        # Add player-specific cost information
        private_values = game_config.constants.get('private_values', {})
        if player_id == 'challenger':
            variables['your_cost'] = private_values.get('challenger_cost', 8)
        else:
            variables['your_cost'] = private_values.get('defender_cost', 10)
        
        # Format template with variables
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse pricing decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.basic_json_parse(response)
        if json_action and 'price' in json_action:
            price = json_action['price']
            if isinstance(price, (int, float)) and price >= 0:
                return {'price': float(price), 'raw_response': response}
        
        # Try numeric extraction
        price = extract_numeric_value(response, 'price')
        if price > 0:
            return {'price': price, 'parsing_method': 'regex', 'raw_response': response}
        
        # Try simple number extraction
        number_patterns = [r'\$?(\d+\.?\d*)', r'(\d+\.?\d+)']
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
        4. For each winner: quantity_sold = (demand_intercept - p_min) × market_share
                           profit = (p_min - private_cost) × quantity_sold
        """
        constants = game_config.constants
        demand_intercept = constants.get('demand_intercept', 100)
        private_values = constants.get('private_values', {})
        
        # Extract prices and get private costs
        players = list(actions.keys())
        prices = {}
        private_costs = {}
        
        for player_id, action in actions.items():
            price = action.get('price', 50)
            prices[player_id] = max(0.0, price)  # Ensure non-negative
            
            # Get private cost for each player
            if player_id == 'challenger':
                private_costs[player_id] = private_values.get('challenger_cost', 8)
            else:
                private_costs[player_id] = private_values.get('defender_cost', 10)
        
        # Step 1: Find minimum bid
        p_min = min(prices.values())
        
        # Step 2: Identify all players who bid p_min
        winners = [pid for pid, price in prices.items() if price == p_min]
        K = len(winners)
        
        # Step 3 & 4: Calculate payoffs using tie-breaking rule
        payoffs = {}
        quantities = {}
        market_shares = {}
        win_status = {}
        
        for player_id in players:
            if player_id in winners:
                # Winner gets share 1/K
                market_share = 1.0 / K
                quantity_sold = (demand_intercept - p_min) * market_share
                profit = (p_min - private_costs[player_id]) * quantity_sold
                win_status[player_id] = 1.0 / K  # For tie accounting
            else:
                # Loser gets nothing
                market_share = 0.0
                quantity_sold = 0.0
                profit = 0.0
                win_status[player_id] = 0.0
            
            market_shares[player_id] = market_share
            quantities[player_id] = quantity_sold
            payoffs[player_id] = profit
        
        # Store additional data in game_state for logging
        if game_state is not None:
            game_state.update({
                'prices': prices,
                'private_costs': private_costs,
                'quantities': quantities,
                'market_shares': market_shares,
                'p_min': p_min,
                'winners': winners,
                'num_winners': K
            })
        
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for metrics calculation as specified in t.txt"""
        
        # Get additional calculated data from game_state
        prices = game_state.get('prices', {}) if game_state else {}
        private_costs = game_state.get('private_costs', {}) if game_state else {}
        quantities = game_state.get('quantities', {}) if game_state else {}
        market_shares = game_state.get('market_shares', {}) if game_state else {}
        p_min = game_state.get('p_min', 0) if game_state else 0
        winners = game_state.get('winners', []) if game_state else []
        
        # Calculate status indicators for MAgIC metrics
        win_status = {}
        profitable_status = {}
        rational_bids = {}
        market_capture = {}
        
        constants = game_config.constants
        rival_cost_mean = constants.get('rival_cost_mean', 10)
        
        for player_id in payoffs:
            # Win status (accounting for ties)
            win_status[player_id] = 1 if player_id in winners else 0
            
            # Profitable status
            profitable_status[player_id] = 1 if payoffs[player_id] > 0 else 0
            
            # Rational bids (price >= own cost)
            own_cost = private_costs.get(player_id, 0)
            player_price = prices.get(player_id, 0)
            rational_bids[player_id] = 1 if player_price >= own_cost else 0
            
            # Market capture rate (for ties, gets fractional credit)
            if player_id in winners:
                K = len(winners)
                market_capture[player_id] = 1.0 / K
            else:
                market_capture[player_id] = 0.0
            
        # Calculate bid appropriateness for challenger (MAgIC Self-awareness metric)
        challenger_appropriate = 0
        if 'challenger' in private_costs and 'challenger' in prices:
            challenger_cost = private_costs['challenger']
            challenger_price = prices['challenger']
            
            # Appropriate if: (cost < rival_mean AND bid < rival_mean) OR (cost > rival_mean AND bid > rival_mean)
            if ((challenger_cost < rival_cost_mean and challenger_price < rival_cost_mean) or
                (challenger_cost > rival_cost_mean and challenger_price > rival_cost_mean)):
                challenger_appropriate = 1
        
        return {
            'game_name': 'spulber',
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'constants': game_config.constants,
            
            # Core data for metrics (from t.txt requirements)
            'actions': actions,
            'payoffs': payoffs,
            'prices': prices,
            'private_costs': private_costs,
            'quantities': quantities,
            'market_shares': market_shares,
            
            # Winner determination results
            'p_min': p_min,
            'winners': winners,
            'num_winners': len(winners),
            
            # Status indicators for MAgIC metrics
            'win_status': win_status,
            'profitable_status': profitable_status,
            'rational_bids': rational_bids,
            'market_capture': market_capture,
            'challenger_bid_appropriate': challenger_appropriate,
            
            # Additional metadata
            'game_state': game_state,
            'num_players': game_config.constants.get('number_of_players', 3)
        }