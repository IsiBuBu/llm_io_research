# games/spulber_game.py
from typing import Dict, Any, Optional
from .base_game import StaticGame
from config import GameConfig, GameConstants

class SpulberGame(StaticGame):
    def __init__(self):
        super().__init__("Spulber Bertrand Competition")
    
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        return f"""**Context:** You are bidding in a winner-take-all auction against {config.number_of_players - 1} rival firms. The firm with the lowest bid wins the entire market worth ${GameConstants.SPULBER_MARKET_VALUE} in revenue. All losing firms earn zero profit.

**Information Structure:** Each firm knows its own marginal cost but not its rivals' costs. Your competitors' marginal costs are independently drawn from a normal distribution with mean = ${GameConstants.SPULBER_RIVAL_COST_MEAN} and standard deviation = ${GameConstants.SPULBER_RIVAL_COST_STD}. This creates uncertainty about how aggressively they will bid.

**Economic Information:**
- Your marginal cost: ${GameConstants.SPULBER_MARGINAL_COST} per unit
- Market value: ${GameConstants.SPULBER_MARKET_VALUE} (winner takes all)
- Your profit if you win: (${GameConstants.SPULBER_MARKET_VALUE} - Your Bid) (assuming unit production)
- Your profit if you lose: $0

**Strategic Considerations:** You face a classic trade-off under uncertainty. Bidding low increases your chance of winning but reduces your profit margin if you do win. Bidding high preserves profit margins but increases the risk that a competitor underbids you. Your optimal strategy should account for the probability distribution of rival bids.

**Key Strategic Questions:** 
- How much are you willing to sacrifice in profit margin to increase your winning probability?
- How should you adjust your bid given that you have a cost advantage (your cost of ${GameConstants.SPULBER_MARGINAL_COST} is below the expected competitor cost of ${GameConstants.SPULBER_RIVAL_COST_MEAN})?

**Your Task:** Choose your bid to maximize your expected profit, considering both the uncertainty about rival costs and the winner-take-all market structure.

**Output Format:** {{"price": <number>, "reasoning": "<brief explanation of your bidding strategy>"}}"""
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        prices = {pid: self._safe_get_numeric(action, 'price', 15.0)
                 for pid, action in actions.items()}
        
        profits = {pid: 0.0 for pid in prices.keys()}
        
        # Find winner(s) - lowest price
        min_price = min(prices.values())
        winners = [pid for pid, price in prices.items() if abs(price - min_price) < 0.01]
        
        # Winner(s) split the market
        if winners:
            market_value = GameConstants.SPULBER_MARKET_VALUE
            winner_revenue = market_value / len(winners)
            
            for winner_id in winners:
                profits[winner_id] = max(0, winner_revenue - min_price)
        
        return profits
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default