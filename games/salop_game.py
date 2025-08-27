# games/salop_game.py
import numpy as np
from typing import Dict, Any, Optional
from .base_game import StaticGame
from config import GameConfig, GameConstants

class SalopGame(StaticGame):
    def __init__(self):
        super().__init__("Salop Spatial Competition")
    
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        neighbors = self._get_neighbors(player_id, config.number_of_players)
        
        return f"""**Context:** You are Firm {player_id} in a circular market with {config.number_of_players} competing firms. Each firm is located at a fixed position on a circle, and customers are distributed evenly around the circle. Customers will buy from the firm offering the lowest total cost (price + transportation cost to reach that firm).

**Market Dynamics:** Your market share depends critically on your price relative to your immediate neighbors on the circle. If you price too high, customers will switch to nearby competitors. If you price too low, you may capture more market share but sacrifice profit margins. The total market size is fixed at {GameConstants.SALOP_MARKET_SIZE} customers.

**Your Position:** You are located between Firm {neighbors[0]} and Firm {neighbors[1]}. Transportation costs are linear - customers pay ${GameConstants.SALOP_TRANSPORT_COST} per unit of distance to reach any firm.

**Economic Information:**
- Your marginal cost: ${GameConstants.SALOP_MARGINAL_COST} per unit
- Your fixed cost: ${GameConstants.SALOP_FIXED_COST} per period
- Profit calculation: (Price - ${GameConstants.SALOP_MARGINAL_COST}) × Quantity Sold - ${GameConstants.SALOP_FIXED_COST}
- Market demand is perfectly inelastic (customers always buy one unit)

**Strategic Considerations:** In this market, you face a fundamental trade-off between market share and profit margins. Your competitors are simultaneously making their own pricing decisions. Consider how your pricing strategy affects both your immediate neighbors and the overall market equilibrium.

**Current Market State:** You are entering a new period where all firms simultaneously choose prices. Based on typical market conditions, prices usually range between $10-$15.

**Your Task:** Choose your optimal price for this period, considering both your costs and the competitive dynamics.

**Output Format:** {{"price": <number>, "reasoning": "<brief explanation of your strategy>"}}"""
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig, 
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        prices = {pid: self._safe_get_numeric(action, 'price', 12.0) 
                 for pid, action in actions.items()}
        
        profits = {}
        n = config.number_of_players
        
        for player_id, price in prices.items():
            player_num = int(player_id)
            
            # Get neighbor prices
            left_neighbor = str((player_num - 2) % n + 1)
            right_neighbor = str(player_num % n + 1)
            
            left_price = prices.get(left_neighbor, price)
            right_price = prices.get(right_neighbor, price)
            
            # Calculate market boundaries (Salop model)
            if left_price == price:
                left_boundary = 0.5 / n
            else:
                left_boundary = (left_price - price) / (2 * GameConstants.SALOP_TRANSPORT_COST) + (1/n)/2
            
            if right_price == price:
                right_boundary = 0.5 / n
            else:
                right_boundary = (right_price - price) / (2 * GameConstants.SALOP_TRANSPORT_COST) + (1/n)/2
            
            left_boundary = max(0, min(1/n, left_boundary))
            right_boundary = max(0, min(1/n, right_boundary))
            
            market_share = left_boundary + right_boundary
            quantity_sold = market_share * GameConstants.SALOP_MARKET_SIZE
            
            profit = ((price - GameConstants.SALOP_MARGINAL_COST) * quantity_sold - 
                     GameConstants.SALOP_FIXED_COST)
            profits[player_id] = max(0, profit)
        
        return profits
    
    def _get_neighbors(self, player_id: str, total_players: int) -> tuple:
        player_num = int(player_id)
        prev_neighbor = str((player_num - 2) % total_players + 1)
        next_neighbor = str(player_num % total_players + 1)
        return prev_neighbor, next_neighbor
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default