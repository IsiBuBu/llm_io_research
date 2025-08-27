# games/athey_bagwell_game.py
import numpy as np
from typing import Dict, Any, Optional
from .base_game import DynamicGame
from config import GameConfig, GameConstants

class AtheyBagwellGame(DynamicGame):
    def __init__(self):
        super().__init__("Athey Bagwell Information Collusion", default_rounds=20)
        
    def create_prompt(self, player_id: str, game_state: Dict, config: GameConfig) -> str:
        player_history = game_state.get('player_histories', {}).get(player_id, {})
        current_cost = game_state.get('current_costs', {}).get(player_id, 'high')
        current_round = game_state.get('current_round', 1)
        
        return f"""**Context:** You participate in a {config.number_of_players}-firm cartel operating over {config.number_of_rounds} periods. The cartel attempts to maximize joint profits by coordinating market shares, but each firm has private information about its costs that affects optimal allocation decisions.

**Information Structure:** 
- Each period, your marginal cost is either "high" (${GameConstants.AB_HIGH_COST}) or "low" (${GameConstants.AB_LOW_COST})
- Cost types are persistent ({int(GameConstants.AB_COST_PERSISTENCE*100)}% probability your cost stays the same next period)
- Other firms cannot observe your true costs - they only observe your cost reports
- Market shares are allocated based on reported costs: lower reported costs receive larger shares

**Economic Information:**
- Market price: Fixed at ${GameConstants.AB_MARKET_PRICE} per unit
- Total market size: {GameConstants.AB_MARKET_SIZE} units per period
- Your profit: (${GameConstants.AB_MARKET_PRICE} - True Cost) × Allocated Market Share
- Discount factor: {config.discount_factor} (future profits worth {int(config.discount_factor*100)}% of current value)

**Strategic Dilemma:** You face a fundamental tension between truthfulness and profit maximization. Reporting low costs increases your market share this period but may affect future allocations in complex ways. The cartel's success depends on accurate cost reporting for efficient allocation, but individual incentives may favor misrepresentation.

**Cartel Rules:** Market shares are allocated using a mechanism that aims to maximize total cartel profits based on reported costs. The exact allocation formula considers both current reports and historical patterns to maintain stability.

**Current Game State:**
- Period: {current_round} of {config.number_of_rounds}
- Your true cost this period: {current_cost}
- Your previous cost reports: {player_history.get('reports', [])[-3:]}
- Previous market shares you received: {player_history.get('market_shares', [])[-3:]} units
- Your previous profits: {[round(p, 2) for p in player_history.get('profits', [])[-3:]]}

**Strategic Considerations:** Consider how your current report affects not only this period's allocation but also the cartel's long-term stability and your future market share. Other firms are making similar calculations about their own reporting strategies.

**Your Task:** Decide whether to report your cost as "high" or "low" for this period.

**Output Format:** {{"report": "<high or low>", "reasoning": "<brief explanation of your information management strategy>"}}"""
    
    def calculate_payoffs(self, actions: Dict[str, Any], config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        reports = {pid: action.get('report', 'high') for pid, action in actions.items()}
        true_costs = game_state.get('current_costs', {}) if game_state else {}
        
        # Allocate market shares based on reports
        total_market = GameConstants.AB_MARKET_SIZE
        market_shares = self._allocate_market_shares(reports, total_market)
        
        profits = {}
        for player_id in reports.keys():
            market_share = market_shares.get(player_id, 0)
            true_cost_type = true_costs.get(player_id, 'high')
            
            true_cost = (GameConstants.AB_HIGH_COST if true_cost_type == 'high' 
                        else GameConstants.AB_LOW_COST)
            
            profit = (GameConstants.AB_MARKET_PRICE - true_cost) * market_share
            profits[player_id] = max(0, profit)
        
        return profits
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         round_num: int) -> Dict:
        if 'player_histories' not in game_state:
            game_state['player_histories'] = {}
        if 'current_costs' not in game_state:
            game_state['current_costs'] = {}
            
        game_state['current_round'] = round_num
        
        # Initialize or evolve costs for each player
        for player_id in actions.keys():
            if player_id not in game_state['player_histories']:
                game_state['player_histories'][player_id] = {
                    'reports': [], 'market_shares': [], 'profits': []
                }
            
            # Cost evolution with persistence
            if round_num == 1:
                game_state['current_costs'][player_id] = np.random.choice(['high', 'low'])
            else:
                if np.random.random() < GameConstants.AB_COST_PERSISTENCE:
                    pass  # Keep same cost
                else:
                    # Switch cost type
                    current_cost = game_state['current_costs'][player_id]
                    game_state['current_costs'][player_id] = 'low' if current_cost == 'high' else 'high'
        
        # Calculate and store results
        reports = {pid: action.get('report', 'high') for pid, action in actions.items()}
        market_shares = self._allocate_market_shares(reports, GameConstants.AB_MARKET_SIZE)
        profits = self.calculate_payoffs(actions, None, game_state)
        
        for player_id, action in actions.items():
            report = action.get('report', 'high')
            market_share = market_shares.get(player_id, 0)
            profit = profits.get(player_id, 0)
            
            game_state['player_histories'][player_id]['reports'].append(report)
            game_state['player_histories'][player_id]['market_shares'].append(market_share)
            game_state['player_histories'][player_id]['profits'].append(profit)
        
        return game_state
    
    def _allocate_market_shares(self, reports: Dict[str, str], total_market: int) -> Dict[str, float]:
        low_reporters = [pid for pid, report in reports.items() if report == 'low']
        high_reporters = [pid for pid, report in reports.items() if report == 'high']
        
        n_low = len(low_reporters)
        n_high = len(high_reporters)
        
        shares = {}
        
        if n_low > 0 and n_high > 0:
            # Low cost firms get 1.5x share weight
            total_weights = n_low * 1.5 + n_high * 1.0
            low_share = (total_market * 1.5) / total_weights
            high_share = total_market / total_weights
            
            for pid in low_reporters:
                shares[pid] = low_share
            for pid in high_reporters:
                shares[pid] = high_share
                
        elif n_low > 0:  # Only low reporters
            share_per_firm = total_market / n_low
            for pid in low_reporters:
                shares[pid] = share_per_firm
                
        else:  # Only high reporters
            share_per_firm = total_market / n_high
            for pid in high_reporters:
                shares[pid] = share_per_firm
        
        return shares