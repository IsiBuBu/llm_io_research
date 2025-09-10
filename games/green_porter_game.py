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

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate Green & Porter payoffs using t.txt specification
        
        Algorithm from t.txt:
        1. Calculate Market Price: p = demand_intercept - demand_slope × total_quantity + demand_shock
        2. For each player: profit = (price - marginal_cost) × player_quantity
        3. Apply State Transition Algorithm to update market state
        """
        
        # Extract game constants
        constants = game_config.constants
        demand_intercept = constants.get('demand_intercept', 100)
        demand_slope = constants.get('demand_slope', 1)
        marginal_cost = constants.get('marginal_cost', 10)
        
        # Get current period and demand shock
        current_period = game_state.get('current_period', 1) if game_state else 1
        demand_shocks = game_state.get('demand_shocks', []) if game_state else []
        demand_shock = demand_shocks[current_period - 1] if current_period <= len(demand_shocks) else 0
        
        # Extract quantities from actions
        quantities = {}
        for player_id, action in actions.items():
            if isinstance(action, dict) and 'quantity' in action:
                quantities[player_id] = action['quantity']
            else:
                self.logger.warning(f"Invalid action format for {player_id}: {action}")
                quantities[player_id] = constants.get('cournot_quantity', 25)  # Default fallback
        
        # Calculate total quantity and market price
        total_quantity = sum(quantities.values())
        market_price = demand_intercept - demand_slope * total_quantity + demand_shock
        
        # Ensure non-negative price
        market_price = max(0, market_price)
        
        # Calculate individual payoffs
        payoffs = {}
        for player_id, quantity in quantities.items():
            profit = (market_price - marginal_cost) * quantity
            payoffs[player_id] = profit
            
            self.logger.debug(f"Player {player_id}: quantity={quantity:.1f}, price={market_price:.2f}, "
                           f"profit={profit:.2f}")
        
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Update game state using Green & Porter State Transition Algorithm from t.txt"""
        
        constants = game_config.constants
        trigger_price = constants.get('trigger_price', 60)
        punishment_periods = constants.get('punishment_periods', 5)
        
        # Calculate current market price for state transition
        demand_intercept = constants.get('demand_intercept', 100)
        demand_slope = constants.get('demand_slope', 1)
        
        current_period = game_state.get('current_period', 1)
        demand_shocks = game_state.get('demand_shocks', [])
        demand_shock = demand_shocks[current_period - 1] if current_period <= len(demand_shocks) else 0
        
        # Calculate total quantity and market price
        total_quantity = sum(action.get('quantity', 0) for action in actions.values() if isinstance(action, dict))
        market_price = max(0, demand_intercept - demand_slope * total_quantity + demand_shock)
        
        # Update histories
        game_state.setdefault('price_history', []).append(market_price)
        game_state.setdefault('state_history', []).append(game_state.get('market_state', 'Collusive'))
        
        for player_id, action in actions.items():
            if isinstance(action, dict) and 'quantity' in action:
                game_state.setdefault('quantity_history', {}).setdefault(player_id, []).append(action['quantity'])
        
        # State Transition Algorithm from t.txt
        current_state = game_state.get('market_state', 'Collusive')
        punishment_timer = game_state.get('punishment_timer', 0)
        
        if current_state == 'Collusive':
            # Check if price fell below trigger (indicating deviation)
            if market_price < trigger_price:
                # Transition to Reversionary (punishment phase)
                new_state = 'Reversionary'
                new_timer = punishment_periods
                self.logger.debug(f"Period {current_period}: Price {market_price:.2f} < trigger {trigger_price:.2f}, "
                               f"switching to Reversionary for {punishment_periods} periods")
            else:
                # Stay in Collusive
                new_state = 'Collusive'
                new_timer = 0
        else:  # current_state == 'Reversionary'
            if punishment_timer > 1:
                # Continue punishment
                new_state = 'Reversionary'
                new_timer = punishment_timer - 1
                self.logger.debug(f"Period {current_period}: Continuing punishment, {new_timer} periods remaining")
            else:
                # End punishment, return to Collusive
                new_state = 'Collusive'
                new_timer = 0
                self.logger.debug(f"Period {current_period}: Punishment ended, returning to Collusive")
        
        # Update state
        game_state['market_state'] = new_state
        game_state['punishment_timer'] = new_timer
        game_state['current_period'] = current_period + 1
        
        return game_state

    def calculate_npv(self, profit_stream: List[float], discount_factor: float) -> float:
        """Calculate Net Present Value of profit stream as specified in t.txt"""
        npv = 0.0
        for t, profit in enumerate(profit_stream):
            npv += (discount_factor ** t) * profit
        return npv

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get data needed for t.txt metrics calculation only"""
        
        if not game_state:
            return {
                'game_name': 'green_porter',
                'experiment_type': game_config.experiment_type,
                'condition_name': game_config.condition_name,
                'actions': actions,
                'payoffs': payoffs,
                'constants': game_config.constants,
                'game_state': {}
            }
        
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
                npvs[player_id] = 0.0
            
            # Calculate other t.txt metrics
            player_quantities = quantity_history.get(player_id, [])
            if player_quantities:
                # Strategic Inertia: Standard deviation of quantities
                strategic_inertia[player_id] = np.std(player_quantities) if len(player_quantities) > 1 else 0.0
                
                # Cooperation Periods: Count of periods with collusive quantity
                cooperation_periods[player_id] = sum(1 for q in player_quantities if abs(q - collusive_quantity) < 1.0)
                
                # Coordination Actions: Fraction of periods with coordinated behavior
                coordination_actions[player_id] = cooperation_periods[player_id] / len(player_quantities)
                
                # Rationality Periods: Count of periods with quantity > 0 (basic rationality check)
                rationality_periods[player_id] = sum(1 for q in player_quantities if q > 0)
            else:
                strategic_inertia[player_id] = 0.0
                cooperation_periods[player_id] = 0
                coordination_actions[player_id] = 0.0
                rationality_periods[player_id] = 0
        
        # Calculate reversion frequency
        reversions = 0
        if len(state_history) > 1:
            for i in range(1, len(state_history)):
                if state_history[i] == 'Reversionary' and state_history[i-1] == 'Collusive':
                    reversions += 1
        
        reversion_frequency = reversions / max(1, len(state_history) - 1) if len(state_history) > 1 else 0
        
        # Calculate win status based on NPV
        max_npv = max(npvs.values()) if npvs else 0
        win_status = {pid: (1 if npvs[pid] == max_npv else 0) for pid in npvs}
        
        return {
            # Core identifiers - REQUIRED for create_game_result
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