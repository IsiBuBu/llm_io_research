"""
Green & Porter Dynamic Oligopoly Game - Compact implementation with full config integration
Implements State Transition Algorithm from t.txt for comprehensive metrics analysis
"""

import json
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import DynamicGame, extract_numeric_value, validate_action_bounds
from config import GameConfig, get_prompt_variables


class GreenPorterGame(DynamicGame):
    """
    Green & Porter Dynamic Oligopoly - Cournot competition with demand uncertainty
    Implements State Transition Algorithm and NPV calculations from t.txt
    """
    
    def __init__(self):
        super().__init__("green_porter")
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template from markdown file"""
        prompt_path = Path("prompts/green_porter.md")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract template between ``` blocks
        template_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
        if not template_match:
            raise ValueError("No template found in green_porter.md")
        
        return template_match.group(1)

    def initialize_game_state(self, game_config: GameConfig, 
                            simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state for new simulation"""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        demand_shock_std = constants.get('demand_shock_std', 5)
        
        # Generate demand shocks for entire simulation
        np.random.seed(simulation_id)  # Reproducible shocks per simulation
        demand_shocks = np.random.normal(0, demand_shock_std, time_horizon).tolist()
        
        return {
            'current_round': 1,
            'market_state': 'Collusive',
            'punishment_timer': 0,
            'demand_shocks': demand_shocks,
            'price_history': [],
            'quantity_history': {},
            'profit_history': {},
            'state_history': [],
            'total_rounds': time_horizon
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt using template and config"""
        
        # Get template variables from config
        variables = get_prompt_variables(
            game_config, 
            player_id=player_id,
            current_round=game_state.get('current_round', 1),
            current_market_state=game_state.get('market_state', 'Collusive'),
            price_history=game_state.get('price_history', [])
        )
        
        # Format template with variables
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse quantity decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.basic_json_parse(response)
        if json_action and 'quantity' in json_action:
            quantity = json_action['quantity']
            if isinstance(quantity, (int, float)) and quantity >= 0:
                return {'quantity': float(quantity), 'raw_response': response}
        
        # Try numeric extraction
        quantity = extract_numeric_value(response, 'quantity')
        if quantity >= 0:
            return {'quantity': quantity, 'parsing_method': 'regex', 'raw_response': response}
        
        # Try simple number extraction
        number_patterns = [r'(\d+\.?\d*)', r'(\d+\.?\d+)']
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    quantity = float(matches[0])
                    if quantity >= 0:
                        return {'quantity': quantity, 'parsing_method': 'number', 'raw_response': response}
                except ValueError:
                    continue
        
        self.logger.warning(f"[{call_id}] Could not parse quantity from {player_id}")
        return None

    def get_default_action(self, player_id: str, game_state: Dict, 
                         game_config: GameConfig) -> Dict[str, Any]:
        """Default quantity action when parsing fails"""
        # Use collusive quantity as safe default
        collusive_quantity = game_config.constants.get('collusive_quantity', 17)
        
        return {
            'quantity': collusive_quantity,
            'reasoning': 'Default collusive quantity due to parsing failure',
            'parsing_success': False,
            'player_id': player_id
        }

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Green-Porter payoffs with market price and profit calculation
        """
        constants = game_config.constants
        base_demand = constants.get('base_demand', 120)
        marginal_cost = constants.get('marginal_cost', 20)
        
        current_round = game_state.get('current_round', 1) if game_state else 1
        demand_shocks = game_state.get('demand_shocks', [0]) if game_state else [0]
        
        # Get demand shock for current period (1-indexed)
        shock_index = min(current_round - 1, len(demand_shocks) - 1)
        demand_shock = demand_shocks[shock_index]
        
        # Extract quantities
        players = list(actions.keys())
        quantities = {}
        for player_id, action in actions.items():
            quantity = action.get('quantity', 17)
            quantities[player_id] = max(0.0, quantity)  # Ensure non-negative
        
        # Calculate market price: P = base_demand - Total_Quantity + Demand_Shock
        total_quantity = sum(quantities.values())
        market_price = base_demand - total_quantity + demand_shock
        
        # Calculate individual profits
        payoffs = {}
        for player_id, quantity in quantities.items():
            profit = (market_price - marginal_cost) * quantity
            payoffs[player_id] = profit
        
        # Store market data in game_state for state transition
        if game_state is not None:
            game_state.update({
                'current_market_price': market_price,
                'current_quantities': quantities,
                'current_profits': payoffs,
                'demand_shock': demand_shock,
                'total_quantity': total_quantity
            })
        
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """
        Update game state using State Transition Algorithm from t.txt
        """
        constants = game_config.constants
        trigger_price = constants.get('trigger_price', 55)
        punishment_duration = constants.get('punishment_duration', 3)
        
        current_round = game_state.get('current_round', 1)
        current_market_state = game_state.get('market_state', 'Collusive')
        punishment_timer = game_state.get('punishment_timer', 0)
        current_market_price = game_state.get('current_market_price', 0)
        
        # State Transition Algorithm from t.txt
        if current_market_state == 'Collusive':
            if current_market_price < trigger_price:
                # Trigger punishment
                new_market_state = 'Price War'
                new_punishment_timer = punishment_duration
            else:
                # Stay collusive
                new_market_state = 'Collusive'
                new_punishment_timer = 0
        elif current_market_state == 'Price War':
            # Decrement punishment timer
            new_punishment_timer = punishment_timer - 1
            if new_punishment_timer > 0:
                new_market_state = 'Price War'
            else:
                new_market_state = 'Collusive'
                new_punishment_timer = 0
        else:
            # Default case
            new_market_state = 'Collusive'
            new_punishment_timer = 0
        
        # Update histories
        price_history = game_state.get('price_history', [])
        price_history.append(current_market_price)
        
        state_history = game_state.get('state_history', [])
        state_history.append(current_market_state)
        
        # Update player histories
        quantity_history = game_state.get('quantity_history', {})
        profit_history = game_state.get('profit_history', {})
        current_quantities = game_state.get('current_quantities', {})
        current_profits = game_state.get('current_profits', {})
        
        for player_id in current_quantities:
            if player_id not in quantity_history:
                quantity_history[player_id] = []
                profit_history[player_id] = []
            
            quantity_history[player_id].append(current_quantities[player_id])
            profit_history[player_id].append(current_profits[player_id])
        
        # Update state
        game_state.update({
            'current_round': current_round + 1,
            'market_state': new_market_state,
            'punishment_timer': new_punishment_timer,
            'price_history': price_history,
            'state_history': state_history,
            'quantity_history': quantity_history,
            'profit_history': profit_history
        })
        
        return game_state

    def calculate_npv(self, profit_stream: List[float], discount_factor: float) -> float:
        """Calculate Net Present Value from profit stream"""
        npv = 0.0
        for t, profit in enumerate(profit_stream):
            npv += (discount_factor ** t) * profit
        return npv

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for metrics calculation as specified in t.txt"""
        
        constants = game_config.constants
        discount_factor = constants.get('discount_factor', 0.95)
        collusive_quantity = constants.get('collusive_quantity', 17)
        
        # Get game state data
        current_round = game_state.get('current_round', 1) if game_state else 1
        market_state = game_state.get('market_state', 'Collusive') if game_state else 'Collusive'
        price_history = game_state.get('price_history', []) if game_state else []
        state_history = game_state.get('state_history', []) if game_state else []
        quantity_history = game_state.get('quantity_history', {}) if game_state else {}
        profit_history = game_state.get('profit_history', {}) if game_state else {}
        
        # Calculate NPVs and status indicators for MAgIC metrics
        npvs = {}
        cooperation_status = {}
        coordination_status = {}
        rationality_status = {}
        
        for player_id in payoffs:
            # Calculate NPV if we have profit history
            if player_id in profit_history and profit_history[player_id]:
                npv = self.calculate_npv(profit_history[player_id], discount_factor)
                npvs[player_id] = npv
            else:
                npvs[player_id] = payoffs[player_id]  # Single period fallback
            
            # Cooperation status (periods in collusive state)
            collusive_periods = sum(1 for state in state_history if state == 'Collusive')
            cooperation_status[player_id] = collusive_periods
            
            # Coordination status (cooperated actions in collusive periods)
            if player_id in quantity_history:
                quantities = quantity_history[player_id]
                constructive_actions = 0
                collusive_opportunities = 0
                
                for i, state in enumerate(state_history):
                    if state == 'Collusive' and i < len(quantities):
                        collusive_opportunities += 1
                        if abs(quantities[i] - collusive_quantity) < 0.1:  # Close to collusive quantity
                            constructive_actions += 1
                
                coordination_status[player_id] = constructive_actions
            else:
                coordination_status[player_id] = 0
            
            # Rationality status (periods cooperated)
            if player_id in quantity_history:
                quantities = quantity_history[player_id]
                cooperated_periods = sum(1 for q in quantities if abs(q - collusive_quantity) < 0.1)
                rationality_status[player_id] = cooperated_periods
            else:
                rationality_status[player_id] = 0
        
        # Calculate win status based on NPV
        max_npv = max(npvs.values()) if npvs else 0
        win_status = {pid: (1 if npvs[pid] == max_npv else 0) for pid in npvs}
        
        # Calculate reversion frequency
        reversions = 0
        for i in range(1, len(state_history)):
            if state_history[i] == 'Price War' and state_history[i-1] == 'Collusive':
                reversions += 1
        
        reversion_frequency = reversions / max(1, len(state_history) - 1)
        
        return {
            'game_name': 'green_porter',
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'constants': game_config.constants,
            
            # Core data for metrics (from t.txt requirements)
            'actions': actions,
            'payoffs': payoffs,
            'npvs': npvs,
            'price_history': price_history,
            'state_history': state_history,
            'quantity_history': quantity_history,
            'profit_history': profit_history,
            
            # Status indicators for MAgIC metrics
            'win_status': win_status,
            'cooperation_status': cooperation_status,
            'coordination_status': coordination_status,
            'rationality_status': rationality_status,
            'reversion_frequency': reversion_frequency,
            
            # Current state info
            'current_round': current_round,
            'market_state': market_state,
            'total_rounds': game_state.get('total_rounds', 50) if game_state else 50,
            
            # Additional metadata
            'game_state': game_state
        }