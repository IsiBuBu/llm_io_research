# games/green_porter_game.py
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .base_game import DynamicGame
from config import GameConfig, GameConstants

class GreenPorterGame(DynamicGame):
    def __init__(self):
        super().__init__("Green Porter Dynamic Oligopoly", default_rounds=25)
        
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        constants = GameConstants(config)
        history = game_state.get('price_history', [])
        player_history = game_state.get('player_histories', {}).get(player_id, {})
        current_round = game_state.get('current_round', 1)
        
        return f"""**Context:** You are Firm {player_id} in a {config.number_of_players}-firm oligopoly producing a homogeneous product over {config.number_of_rounds} periods. Market conditions are uncertain - demand shocks affect the market price in ways that make it difficult to distinguish between low demand and competitor cheating.

**Market Dynamics:** 
- Market price = ${constants.GP_DEMAND_INTERCEPT} - (Total Industry Quantity) + Demand Shock
- Demand shock is random each period: Normal(0, ${constants.GP_DEMAND_SHOCK_STD})
- Your quantity decision affects both current profit and future market conditions
- Other firms observe market prices but not individual quantities or demand shocks

**Economic Information:**
- Your marginal cost: ${constants.GP_MARGINAL_COST} per unit
- Discount factor: {config.discount_factor} (future profits are worth {int(config.discount_factor*100)}% of current profits)
- Current period profit: (Market Price - ${constants.GP_MARGINAL_COST}) × Your Quantity
- Total payoff: Sum of discounted profits across all periods

**Strategic Environment:** This market presents a complex intertemporal trade-off. Higher production increases current profits but may signal aggressive competition to rivals, potentially triggering competitive responses. The presence of demand uncertainty means that low prices could result from either competitor aggression or poor market conditions.

**Historical Context:** Industry participants have learned that sustainable profits require some degree of market discipline. However, each firm faces ongoing temptation to expand production for short-term gains.

**Current Market Information:**
- Period: {current_round} of {config.number_of_rounds}
- Previous market prices: {history[-3:] if len(history) >= 3 else history}
- Your previous quantities: {player_history.get('quantities', [])[-3:]}
- Your previous period profits: {[round(p, 2) for p in player_history.get('profits', [])[-3:]]}

**Strategic Decision:** Choose your quantity for this period, balancing current profit maximization against the long-term health of the market and your relationships with competitors.

**Output Format:** {{"quantity": <number>, "reasoning": "<brief explanation of your strategic thinking>"}}"""
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Tuple[Dict[str, float], float]:
        constants = GameConstants(config)
        quantities = {pid: self._safe_get_numeric(action, 'quantity', 25.0)
                     for pid, action in actions.items()}
        
        total_quantity = sum(quantities.values())
        demand_shock = np.random.normal(0, constants.GP_DEMAND_SHOCK_STD)
        market_price = max(0, constants.GP_DEMAND_INTERCEPT - total_quantity + demand_shock)
        
        profits = {}
        for player_id, quantity in quantities.items():
            profits[player_id] = max(0, (market_price - constants.GP_MARGINAL_COST) * quantity)
            
        return profits, market_price
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         round_num: int) -> Dict:
        # Create a minimal config for constants - we only need number_of_players for scaling
        temp_config = GameConfig(number_of_players=len(actions))
        
        if 'price_history' not in game_state:
            game_state['price_history'] = []
        if 'player_histories' not in game_state:
            game_state['player_histories'] = {}
            
        # Calculate payoffs and market price
        profits, market_price = self.calculate_payoffs(actions, temp_config, game_state)
        game_state['price_history'].append(market_price)
        game_state['current_round'] = round_num
        
        # Update player histories
        for player_id, action in actions.items():
            if player_id not in game_state['player_histories']:
                game_state['player_histories'][player_id] = {'quantities': [], 'profits': []}
            
            quantity = self._safe_get_numeric(action, 'quantity', 25.0)
            profit = profits[player_id]
            
            game_state['player_histories'][player_id]['quantities'].append(quantity)
            game_state['player_histories'][player_id]['profits'].append(profit)
        
        return game_state
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default