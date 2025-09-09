"""
Salop Spatial Competition Game - Compact implementation with full config integration
Implements all algorithms from t.txt for comprehensive metrics analysis
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import StaticGame, extract_numeric_value, validate_action_bounds
from config import GameConfig, get_prompt_variables


class SalopGame(StaticGame):
    """
    Salop Spatial Competition - firms compete on prices in circular market
    Implements Market Share and Quantity Calculation Algorithm from t.txt
    """
    
    def __init__(self):
        super().__init__("salop")
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template from markdown file"""
        prompt_path = Path("prompts/salop.md")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract template between ``` blocks
        template_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if not template_match:
            raise ValueError("No template found in salop.md")
        
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
        # Use marginal cost + small markup as default
        marginal_cost = game_config.constants.get('marginal_cost', 8)
        default_price = marginal_cost + 2.0
        
        return {
            'price': default_price,
            'reasoning': 'Default pricing due to parsing failure',
            'parsing_success': False,
            'player_id': player_id
        }

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Salop spatial competition payoffs using Market Share Algorithm from t.txt
        
        Algorithm from t.txt:
        1. Calculate Competitive Boundaries (x_right, x_left)
        2. Calculate Monopoly Boundary (x_max)  
        3. Determine Final Market Reach
        4. Calculate Quantity and Share
        """
        constants = game_config.constants
        num_players = constants.get('number_of_players', 3)
        market_size = constants.get('market_size', 1000)
        transport_cost = constants.get('transport_cost', 1.5)
        marginal_cost = constants.get('marginal_cost', 8)
        fixed_cost = constants.get('fixed_cost', 100)
        v = constants.get('v', 30)
        circumference = constants.get('circumference', 1.0)
        
        # Extract and validate prices
        players = list(actions.keys())
        prices = {}
        for player_id, action in actions.items():
            price = action.get('price', marginal_cost + 3)
            # Validate price bounds
            prices[player_id] = max(0.0, min(price, v))
        
        # Calculate payoffs using Market Share and Quantity Calculation Algorithm
        payoffs = {}
        quantities = {}
        market_shares = {}
        
        for i, player_id in enumerate(players):
            # Get neighbors in circular arrangement
            left_neighbor = players[(i - 1) % num_players]
            right_neighbor = players[(i + 1) % num_players]
            
            p_i = prices[player_id]
            p_left = prices[left_neighbor]
            p_right = prices[right_neighbor]
            
            # Step 1: Calculate Competitive Boundaries (from t.txt)
            x_right = (p_right - p_i) / (2 * transport_cost) + circumference / (2 * num_players)
            x_left = (p_left - p_i) / (2 * transport_cost) + circumference / (2 * num_players)
            
            # Step 2: Calculate Monopoly Boundary  
            x_max = (v - p_i) / transport_cost
            
            # Step 3: Determine Final Market Reach
            reach_right = max(0, min(x_right, x_max))
            reach_left = max(0, min(x_left, x_max))
            
            # Step 4: Calculate Quantity and Share
            total_reach = reach_right + reach_left
            quantity_sold = total_reach * market_size
            market_share = quantity_sold / market_size if market_size > 0 else 0
            
            quantities[player_id] = quantity_sold
            market_shares[player_id] = market_share
            
            # Calculate profit
            profit = (p_i - marginal_cost) * quantity_sold - fixed_cost
            payoffs[player_id] = profit
        
        # Store additional data in game_state for logging
        if game_state is not None:
            game_state.update({
                'prices': prices,
                'quantities': quantities,
                'market_shares': market_shares
            })
        
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for metrics calculation as specified in t.txt"""
        
        # Get additional calculated data from game_state
        prices = game_state.get('prices', {}) if game_state else {}
        quantities = game_state.get('quantities', {}) if game_state else {}
        market_shares = game_state.get('market_shares', {}) if game_state else {}
        
        # Calculate win status and profitability
        max_profit = max(payoffs.values()) if payoffs else 0
        win_status = {pid: (1 if payoffs[pid] == max_profit else 0) for pid in payoffs}
        profitable_status = {pid: (1 if payoffs[pid] >= 0 else 0) for pid in payoffs}
        
        return {
            'game_name': 'salop',
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'constants': game_config.constants,
            
            # Core data for metrics (from t.txt requirements)
            'actions': actions,
            'payoffs': payoffs,
            'prices': prices,
            'quantities': quantities,
            'market_shares': market_shares,
            
            # Win and profitability status for MAgIC metrics
            'win_status': win_status,
            'profitable_status': profitable_status,
            'max_profit': max_profit,
            
            # Additional metadata
            'game_state': game_state,
            'num_players': game_config.constants.get('number_of_players', 3)
        }