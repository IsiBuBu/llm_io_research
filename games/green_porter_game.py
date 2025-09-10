"""
Green & Porter Dynamic Oligopoly Game - Updated implementation with t.txt algorithms only
Implements State Transition Algorithm and NPV calculations from t.txt specification
"""

import json
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from games.base_game import DynamicGame, QuantityParsingMixin, extract_numeric_value
from config import GameConfig, get_prompt_variables


class GreenPorterGame(DynamicGame, QuantityParsingMixin):
    """
    Green & Porter Dynamic Oligopoly - Cournot competition with demand uncertainty
    Implements State Transition Algorithm and NPV calculations from t.txt
    """
    
    def __init__(self):
        super().__init__("green_porter")

    def initialize_game_state(self, game_config: GameConfig, 
                            simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state for new simulation (from t.txt specification)"""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        demand_shock_std = constants.get('demand_shock_std', 5)
        
        # Generate demand shocks for entire simulation (reproducible per simulation)
        np.random.seed(simulation_id)
        demand_shocks = np.random.normal(0, demand_shock_std, time_horizon).tolist()
        
        # Initialize state as specified in t.txt
        return {
            'current_period': 1,
            'market_state': 'Collusive',  # t.txt: Start in "Collusive" state
            'punishment_timer': 0,        # t.txt: Initialize punishment_timer = 0
            'demand_shocks': demand_shocks,
            'price_history': [],
            'quantity_history': {},
            'profit_history': {},
            'state_history': [],
            'total_periods': time_horizon
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt using template and config"""
        
        # Get template variables from config (includes current_market_state and histories)
        variables = get_prompt_variables(
            game_config, 
            player_id=player_id,
            current_round=game_state.get('current_period', 1),
            current_market_state=game_state.get('market_state', 'Collusive'),
            price_history=game_state.get('price_history', [])
        )
        
        try:
            return self.prompt_template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse quantity decision from LLM response using inherited mixin"""
        
        # Use the QuantityParsingMixin method
        result = self.parse_quantity_response(response, player_id, call_id)
        
        if result:
            self.logger.debug(f"[{call_id}] Successfully parsed quantity: {result.get('quantity', 'N/A')} for {player_id}")
            return result
        
        self.logger.warning(f"[{call_id}] Could not parse quantity from {player_id}")
        return None

    def get_default_action(self, player_id: str, game_state: Dict, 
                         game_config: GameConfig) -> Dict[str, Any]:
        """Default quantity action when parsing fails"""
        constants = game_config.constants
        
        # Default to collusive quantity if in collusive state, otherwise competitive
        if game_state.get('market_state') == 'Collusive':
            default_quantity = constants.get('collusive_quantity', 17)
        else:
            default_quantity = constants.get('competitive_quantity', 25)
        
        return {
            'quantity': default_quantity,
            'reasoning': 'Default quantity due to parsing failure',
            'parsing_success': False,
            'player_id': player_id
        }

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Green & Porter payoffs using t.txt specification
        
        From t.txt Outcome Calculation:
        - Total_Quantity_t = Σq_i,t
        - Market_Price_t = base_demand - Total_Quantity_t + Demand_Shock_t
        - Profit_i,t = (Market_Price_t - marginal_cost) × q_i,t
        """
        
        constants = game_config.constants
        
        # Extract constants from t.txt specification
        base_demand = constants.get('base_demand', 100)
        marginal_cost = constants.get('marginal_cost', 8)
        
        # Get current period and demand shock
        current_period = game_state.get('current_period', 1) if game_state else 1
        demand_shocks = game_state.get('demand_shocks', []) if game_state else []
        
        # Get demand shock for current period (index is period - 1)
        shock_index = min(current_period - 1, len(demand_shocks) - 1) if demand_shocks else 0
        demand_shock = demand_shocks[shock_index] if demand_shocks else 0
        
        # Extract quantities from actions
        quantities = {}
        for player_id, action in actions.items():
            quantity = action.get('quantity', 0)
            quantities[player_id] = max(0, quantity)  # Non-negative quantities
        
        # Step 1: Calculate Total_Quantity_t = Σq_i,t (from t.txt)
        total_quantity = sum(quantities.values())
        
        # Step 2: Calculate Market_Price_t = base_demand - Total_Quantity_t + Demand_Shock_t (from t.txt)
        market_price = base_demand - total_quantity + demand_shock
        market_price = max(0, market_price)  # Non-negative price
        
        # Step 3: Calculate Profit_i,t = (Market_Price_t - marginal_cost) × q_i,t (from t.txt)
        payoffs = {}
        for player_id, quantity in quantities.items():
            profit = (market_price - marginal_cost) * quantity
            payoffs[player_id] = profit
            
            self.logger.debug(f"Player {player_id}: quantity={quantity:.2f}, "
                            f"market_price={market_price:.2f}, profit={profit:.2f}")
        
        # Store data needed for state transition and logging
        if game_state is not None:
            game_state.update({
                'last_market_price': market_price,
                'last_total_quantity': total_quantity,
                'last_quantities': quantities.copy()
            })
        
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Update game state using State Transition Algorithm from t.txt"""
        
        constants = game_config.constants
        trigger_price = constants.get('trigger_price', 20)
        punishment_duration = constants.get('punishment_duration', 5)
        
        # Get market price from last calculation
        market_price = game_state.get('last_market_price', trigger_price + 1)
        current_state = game_state.get('market_state', 'Collusive')
        punishment_timer = game_state.get('punishment_timer', 0)
        current_period = game_state.get('current_period', 1)
        
        # Execute State Transition Algorithm (exactly from t.txt)
        if current_state == 'Collusive':
            if market_price < trigger_price:
                # Trigger punishment phase
                new_state = 'Reversionary'  # t.txt uses "Reversionary" for price war
                new_punishment_timer = punishment_duration
                self.logger.info(f"Period {current_period}: Price war triggered! "
                               f"Price {market_price:.2f} < trigger {trigger_price}")
            else:
                # Stay in collusive state
                new_state = 'Collusive'
                new_punishment_timer = 0
        else:  # current_state == 'Reversionary'
            # Decrement punishment timer
            new_punishment_timer = punishment_timer - 1
            if new_punishment_timer > 0:
                # Continue punishment
                new_state = 'Reversionary'
            else:
                # Return to collusive state
                new_state = 'Collusive'
                new_punishment_timer = 0
                self.logger.info(f"Period {current_period}: Returning to collusive state")
        
        # Update histories for metrics calculation
        game_state['price_history'].append(market_price)
        game_state['state_history'].append(current_state)
        
        # Update quantity and profit histories
        for player_id, action in actions.items():
            if player_id not in game_state['quantity_history']:
                game_state['quantity_history'][player_id] = []
            if player_id not in game_state['profit_history']:
                game_state['profit_history'][player_id] = []
            
            quantity = action.get('quantity', 0)
            game_state['quantity_history'][player_id].append(quantity)
        
        # Update state for next period
        game_state.update({
            'current_period': current_period + 1,
            'market_state': new_state,
            'punishment_timer': new_punishment_timer
        })
        
        return game_state

    def calculate_npv(self, profit_stream: List[float], discount_factor: float) -> float:
        """Calculate Net Present Value as specified in t.txt"""
        npv = 0.0
        for t, profit in enumerate(profit_stream):
            npv += (discount_factor ** t) * profit
        return npv

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for t.txt metrics calculation only"""
        
        if not game_state:
            return {}
        
        constants = game_config.constants
        discount_factor = constants.get('discount_factor', 0.95)
        collusive_quantity = constants.get('collusive_quantity', 17)
        
        # Get game histories
        current_period = game_state.get('current_period', 1)
        price_history = game_state.get('price_history', [])
        state_history = game_state.get('state_history', [])
        quantity_history = game_state.get('quantity_history', {})
        profit_history = game_state.get('profit_history', {})
        
        # Add current period's profits to history
        for player_id, profit in payoffs.items():
            if player_id not in profit_history:
                profit_history[player_id] = []
            profit_history[player_id].append(profit)
        
        # Calculate NPVs for win status determination
        npvs = {}
        strategic_inertia = {}
        cooperation_periods = {}
        coordination_actions = {}
        rationality_periods = {}
        
        for player_id in payoffs.keys():
            # Calculate NPV from profit stream
            player_profits = profit_history.get(player_id, [])
            if player_profits:
                npvs[player_id] = self.calculate_npv(player_profits, discount_factor)
            else:
                npvs[player_id] = payoffs[player_id]
            
            # Calculate Strategic Inertia (t.txt metric)
            player_quantities = quantity_history.get(player_id, [])
            if len(player_quantities) > 1:
                repeats = sum(1 for i in range(1, len(player_quantities)) 
                            if abs(player_quantities[i] - player_quantities[i-1]) < 0.1)
                strategic_inertia[player_id] = repeats / (len(player_quantities) - 1)
            else:
                strategic_inertia[player_id] = 0
            
            # Calculate cooperation periods (periods in collusive state)
            cooperation_periods[player_id] = sum(1 for state in state_history if state == 'Collusive')
            
            # Calculate coordination (constructive actions in collusive periods)
            constructive_actions = 0
            collusive_opportunities = 0
            
            for i, state in enumerate(state_history):
                if state == 'Collusive' and i < len(player_quantities):
                    collusive_opportunities += 1
                    if abs(player_quantities[i] - collusive_quantity) < 0.1:
                        constructive_actions += 1
            
            coordination_actions[player_id] = constructive_actions
            
            # Calculate rationality (periods cooperating with collusive quantity)
            rationality_periods[player_id] = sum(1 for q in player_quantities 
                                               if abs(q - collusive_quantity) < 0.1)
        
        # Calculate Reversion Frequency (t.txt metric)
        reversions = 0
        for i in range(1, len(state_history)):
            if state_history[i] == 'Reversionary' and state_history[i-1] == 'Collusive':
                reversions += 1
        
        reversion_frequency = reversions / max(1, len(state_history) - 1) if len(state_history) > 1 else 0
        
        # Calculate win status based on NPV
        max_npv = max(npvs.values()) if npvs else 0
        win_status = {pid: (1 if npvs[pid] == max_npv else 0) for pid in npvs}
        
        return {
            # Core identifiers
            'game_name': 'green_porter',
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'constants': game_config.constants,
            
            # Required data for t.txt metrics calculation
            'actions': actions,
            'payoffs': payoffs,
            'npvs': npvs,
            
            # Game history data (required for t.txt metrics)
            'price_history': price_history,
            'state_history': state_history,
            'quantity_history': quantity_history,
            'profit_history': profit_history,
            
            # t.txt specific metrics data
            'win_status': win_status,
            'strategic_inertia': strategic_inertia,
            'cooperation_periods': cooperation_periods,
            'coordination_actions': coordination_actions,
            'rationality_periods': rationality_periods,
            'reversion_frequency': reversion_frequency,
            
            # Current state information
            'current_period': current_period,
            'market_state': game_state.get('market_state', 'Collusive'),
            'total_periods': game_state.get('total_periods', 50),
            
            # Additional metadata
            'game_state': game_state
        }