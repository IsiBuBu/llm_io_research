"""
Green-Porter Dynamic Oligopoly Game - Minimal implementation for comprehensive_metrics.py
Focuses only on core Cournot competition with demand uncertainty for metrics analysis
"""

import json
import re
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from games.base_game import DynamicGame
from config import GameConfig, GameConstants


class GreenPorterGame(DynamicGame):
    """
    Green-Porter Dynamic Oligopoly - Cournot competition with demand uncertainty
    Generates data needed by comprehensive_metrics.py for behavioral analysis
    """
    
    def __init__(self):
        super().__init__("Green Porter Dynamic Oligopoly")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             config: GameConfig, constants: GameConstants) -> str:
        """Generate strategic prompt for Green-Porter quantity decision"""
        
        current_round = game_state.get('current_round', 1)
        price_history = game_state.get('price_history', [])
        player_history = game_state.get('player_histories', {}).get(player_id, {})
        
        # Calculate strategic benchmarks
        collusive_quantity = self._calculate_collusive_quantity(constants, config)
        cournot_quantity = self._calculate_cournot_quantity(constants, config)
        
        prompt = f"""**GREEN-PORTER DYNAMIC OLIGOPOLY**

You are Firm {player_id} in a {config.number_of_players}-firm oligopoly, Round {current_round}/{config.number_of_rounds}.

**MARKET STRUCTURE:**
- **Competition Type**: Cournot quantity competition (choose quantities, market sets price)
- **Market Price Formula**: P = ${constants.GP_BASE_DEMAND_INTERCEPT} - (Total Industry Quantity) + Demand Shock
- **Demand Uncertainty**: Each round has random demand shock ~ Normal(0, 5.0)
- **Your Marginal Cost**: ${constants.GP_MARGINAL_COST} per unit

**THE GREEN-PORTER DILEMMA:**
When market prices are low, is it because:
1. **Competitors are cheating** (producing more than agreed)
2. **Bad demand shock** (random market downturn)
3. **Both** (impossible to distinguish)

This uncertainty makes cartel enforcement extremely difficult.

**PROFIT CALCULATION:**
- **Your Profit**: (Market Price - ${constants.GP_MARGINAL_COST}) × Your Quantity
- **Discount Factor**: {config.discount_factor} (future rounds worth {int(config.discount_factor*100)}% of current)

**STRATEGIC BENCHMARKS:**
- **Collusive Quantity**: {collusive_quantity:.1f} units (joint profit maximizing)
- **Cournot Quantity**: {cournot_quantity:.1f} units (Nash equilibrium)
- **Competitive Quantity**: Higher quantities (approaching marginal cost pricing)

**MARKET HISTORY:**
- **Recent Prices**: {price_history[-5:] if len(price_history) >= 5 else price_history}
- **Your Recent Quantities**: {player_history.get('quantities', [])[-3:]}
- **Average Recent Price**: ${np.mean(price_history[-3:]) if len(price_history) >= 3 else 'N/A'}

**STRATEGIC CONSIDERATIONS:**
1. **Cooperation vs Competition**: Lower quantities support higher prices but reduce your output
2. **Punishment Strategy**: High quantities can punish suspected cheaters but hurt everyone
3. **Market Signals**: Use price history to infer if others are cooperating or competing
4. **End Game Effects**: {config.number_of_rounds - current_round + 1} rounds remaining affects incentives

**QUANTITY GUIDANCE:**
- Collusive: {collusive_quantity:.1f} units (supports cartel)
- Moderate: {(collusive_quantity + cournot_quantity)/2:.1f} units (mixed strategy)  
- Aggressive: {cournot_quantity:.1f}+ units (competitive)

**REQUIRED FORMAT:**
{{"quantity": <your_quantity>, "reasoning": "<explanation of your quantity strategy>"}}

Choose your quantity to maximize long-term discounted profits."""
        
        return prompt

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse quantity decision from LLM response"""
        try:
            # Try JSON parsing first
            action = self._parse_json_response(response)
            if action and 'quantity' in action:
                quantity = float(action['quantity'])
                if 0 <= quantity <= 100:  # Reasonable bounds
                    return action
            
            # Pattern-based fallback for quantity extraction
            quantity_patterns = [
                r'quantity["\']?\s*:\s*([0-9]+\.?[0-9]*)',
                r'produce\s+([0-9]+\.?[0-9]*)',
                r'output\s+([0-9]+\.?[0-9]*)',
                r'([0-9]+\.?[0-9]*)\s+units'
            ]
            
            for pattern in quantity_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    quantity = float(match.group(1))
                    if 0 <= quantity <= 100:
                        return {
                            "quantity": quantity,
                            "reasoning": "Pattern-parsed quantity decision"
                        }
            
            # Default to collusive quantity if no valid quantity found
            constants = GameConstants()
            collusive_qty = self._calculate_collusive_quantity(constants, GameConfig())
            return {
                "quantity": collusive_qty,
                "reasoning": "Default collusive quantity"
            }
                
        except Exception as e:
            self.logger.warning(f"Parsing error for {player_id}: {e}")
            return None

    def get_default_action(self, player_id: str, game_state: Dict, config: GameConfig) -> Dict[str, Any]:
        """Default action when parsing fails - use collusive quantity"""
        constants = GameConstants(config)
        collusive_qty = self._calculate_collusive_quantity(constants, config)
        return {
            "quantity": collusive_qty,
            "reasoning": "Default cooperative quantity"
        }

    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate payoffs based on Cournot competition with demand uncertainty"""
        constants = GameConstants(config)
        
        # Extract quantities
        quantities = {}
        for player_id, action in actions.items():
            quantity = action.get('quantity', 25.0)
            quantities[player_id] = max(0, float(quantity))
        
        # Calculate market dynamics
        total_quantity = sum(quantities.values())
        demand_shock = np.random.normal(0, 5.0)  # Standard demand shock
        market_price = max(0, constants.GP_BASE_DEMAND_INTERCEPT - total_quantity + demand_shock)
        
        # Store market price in game state for history
        if game_state:
            if 'price_history' not in game_state:
                game_state['price_history'] = []
            game_state['price_history'].append(market_price)
            game_state['last_demand_shock'] = demand_shock
        
        # Calculate profits
        profits = {}
        for player_id, quantity in quantities.items():
            profit = max(0, (market_price - constants.GP_MARGINAL_COST) * quantity)
            profits[player_id] = profit
        
        return profits

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         config: GameConfig) -> Dict:
        """Update game state with quantity history tracking"""
        # Call parent method for basic round history
        game_state = super().update_game_state(game_state, actions, config)
        
        # Track player quantity histories for analysis
        for player_id, action in actions.items():
            if player_id not in game_state['player_histories']:
                game_state['player_histories'][player_id] = {'quantities': []}
            
            quantity = action.get('quantity', 25.0)
            game_state['player_histories'][player_id]['quantities'].append(quantity)
        
        return game_state

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response"""
        try:
            json_patterns = [
                r'\{[^{}]*"quantity"[^{}]*\}',  # Simple JSON with quantity
                r'```json\s*(\{.*?\})\s*```',   # JSON in code blocks
                r'```\s*(\{.*?\})\s*```'       # JSON in any code blocks
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    action = json.loads(match)
                    if isinstance(action, dict):
                        return action
        except:
            pass
        return None

    def _calculate_collusive_quantity(self, constants: GameConstants, config: GameConfig) -> float:
        """Calculate monopoly quantity divided by number of firms"""
        monopoly_quantity = (constants.GP_BASE_DEMAND_INTERCEPT - constants.GP_MARGINAL_COST) / 2
        return monopoly_quantity / config.number_of_players

    def _calculate_cournot_quantity(self, constants: GameConstants, config: GameConfig) -> float:
        """Calculate Cournot Nash equilibrium quantity"""
        n = config.number_of_players
        return (constants.GP_BASE_DEMAND_INTERCEPT - constants.GP_MARGINAL_COST) / (n + 1)

    def _analyze_cost_position(self, constants: GameConstants) -> str:
        """Analyze cost position for compatibility"""
        return "Standard oligopolist"

    def _estimate_winning_probability(self, constants: GameConstants, config: GameConfig) -> float:
        """Estimate probability of achieving high profits"""
        return 1.0 / config.number_of_players  # Equal chance in symmetric game

    def _calculate_bid_guidance(self, constants: GameConstants) -> Dict[str, float]:
        """Calculate guidance for quantity decisions"""
        return {
            'min': 10.0,  # Conservative quantity
            'max': 40.0   # Aggressive quantity
        }