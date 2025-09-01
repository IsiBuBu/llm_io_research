# metrics/comprehensive_metrics.py
"""
Enhanced comprehensive metrics calculator implementing all metrics from both 
the original paper and the MAgIC paper for all four games.

CONFIGURATION REQUIREMENTS:
This implementation requires the following parameters to be added to config.json:

{
  "game_constants": {
    "salop": {
      "base_market_size": 300,
      "marginal_cost": 8,
      "fixed_cost": 100,
      "transport_cost": 0.5
    },
    "spulber": {
      "base_market_size": 1000,
      "marginal_cost": 10,
      "demand_slope": 1.0,
      "market_value": 100,          // REQUIRED: Total market value for winner
      "rival_cost_mean": 12,        // REQUIRED: Mean of rival cost distribution
      "rival_cost_std": 3           // REQUIRED: Std dev of rival cost distribution
    },
    "green_porter": {
      "base_demand_intercept": 100,
      "marginal_cost": 10,
      "demand_shock_std": 5,
      "collusive_quantity": 22.5,   // RECOMMENDED: Threshold for collusion
      "competitive_quantity": 25.0, // RECOMMENDED: Competitive quantity level
      "discount_rate": 0.05,        // RECOMMENDED: Discount rate for NPV
      "trigger_price": 15.0         // RECOMMENDED: Trigger price for punishment
    },
    "athey_bagwell": {
      "high_cost": 15,
      "low_cost": 5,
      "market_price": 50,
      "cost_persistence": 0.8,
      "base_market_size": 1000,
      "discount_factor": 0.95       // RECOMMENDED: Discount factor for NPV
    }
  }
}
"""

import numpy as np
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Handle imports gracefully for robust operation
try:
    from config import GameResult, PlayerResult, GameConfig
except ImportError:
    # Fallback types for testing/development
    from typing import Any as GameResult, Any as PlayerResult, Any as GameConfig

if TYPE_CHECKING:
    from config import GameConstants

@dataclass
class MetricResult:
    name: str
    value: float
    formula: str

@dataclass
class GameMetrics:
    game_name: str
    player_id: str
    primary_behavioral: Dict[str, MetricResult]
    core_performance: Dict[str, MetricResult]
    advanced_strategic: Dict[str, MetricResult]
    magic_behavioral: Dict[str, MetricResult]  # New: MAgIC paper specific metrics

class ComprehensiveMetricsCalculator:
    """
    Enhanced comprehensive metrics calculator implementing all metrics from both 
    the original paper and the MAgIC paper for all four games.
    """
    
    def __init__(self):
        self.spulber_calculator = SpulberMetricsCalculator()
        self.green_porter_calculator = GreenPorterMetricsCalculator()
        self.salop_calculator = SalopMetricsCalculator()
        self.athey_bagwell_calculator = AtheyBagwellMetricsCalculator()
    
    def calculate_all_metrics(self, game_results: List[GameResult], 
                            game_type: str, player_id: str = '1') -> GameMetrics:
        """Calculate all metrics for a given game type"""
        calculators = {
            'spulber': self.spulber_calculator,
            'green_porter': self.green_porter_calculator,
            'salop': self.salop_calculator,
            'athey_bagwell': self.athey_bagwell_calculator
        }
        
        calculator = calculators.get(game_type.lower())
        if not calculator:
            raise ValueError(f"Unknown game type: {game_type}")
            
        return calculator.calculate_metrics(game_results, player_id)

class SpulberMetricsCalculator:
    """Spulber (1995): Bertrand Competition with Unknown Costs"""
    
    def calculate_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
            
        # Primary Behavioral: Average Bid Price
        prices = []
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and player_result.actions:
                price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
                prices.append(price)
        avg_bid_price = np.mean(prices) if prices else 0.0
        
        # Core Performance Metrics
        profits = []
        wins_with_profit = []
        lowest_price_count = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
                
            profits.append(player_result.profit)
            wins_with_profit.append(1 if (player_result.win and player_result.profit > 0) else 0)
            
            # Check if set lowest price
            if player_result.actions:
                player_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
                all_prices = [self._safe_get_numeric(pr.actions[0], 'price', 10.0) 
                             for pr in result.players if pr.actions]
                if all_prices and player_price <= min(all_prices):
                    lowest_price_count += 1
        
        win_rate = np.mean(wins_with_profit) if wins_with_profit else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        market_capture_rate = lowest_price_count / len(game_results) if game_results else 0.0
        
        # Advanced Strategic: Regret
        regrets = []
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                constants = GameConstants(result.config)
                optimal_profit = constants.SPULBER_MARKET_VALUE - constants.SPULBER_MARGINAL_COST
                regret = max(0, optimal_profit - player_result.profit)
                regrets.append(regret)
        avg_regret = np.mean(regrets) if regrets else 0.0
        
        # Strategic Inertia
        strategic_inertia = self._calculate_strategic_inertia(game_results, player_id, 'price')
        
        # Total Industry Profit
        total_industry_profit = sum(sum(pr.profit for pr in result.players) for result in game_results)
        
        # Power Index / Influence
        power_index = self._calculate_power_index(game_results, player_id)
        
        # MAgIC Paper Behavioral Metrics
        rationality_score = self._calculate_spulber_rationality(game_results, player_id)
        judgment_score = self._calculate_spulber_judgment(game_results, player_id)
        self_awareness_score = self._calculate_spulber_self_awareness(game_results, player_id)
        
        return GameMetrics(
            game_name="Spulber",
            player_id=player_id,
            primary_behavioral={
                'average_bid_price': MetricResult("Average Bid Price", avg_bid_price,
                                                "Σ (Price_i) / Number of Games Played")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games Won with Profit > 0 / Total Games Played"),
                'average_profit': MetricResult("Average Profit", avg_profit,
                                             "Σ (Profit_i) / Total Games Played"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                "Standard Deviation of {Profit_win, 0, 0, Profit_win, ...}"),
                'market_capture_rate': MetricResult("Market Capture Rate", market_capture_rate,
                                                   "Number of Times Firm Sets Lowest Price / Total Games Played")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", avg_regret,
                                     "Ex-Post Optimal Profit - Actual Profit"),
                'strategic_inertia': MetricResult("Strategic Inertia", strategic_inertia,
                                                "Number of Times Price_t = Price_{t-1} / (Total Periods - 1)"),
                'total_industry_profit': MetricResult("Total Industry Profit", total_industry_profit,
                                                    "Profit of Winning Firm"),
                'power_index': MetricResult("Power Index / Influence", power_index,
                                          "(Total Profit of n Firms) - (Total Profit of n-1 Firms)")
            },
            magic_behavioral={
                'rationality': MetricResult("Rationality", rationality_score,
                                          "Number of Games where |p_actual - p*(θ)| < ε / Total Games Played"),
                'judgment': MetricResult("Judgment", judgment_score,
                                       "Number of Wins where Price > Cost / Total Number of Wins"),
                'self_awareness': MetricResult("Self-awareness", self_awareness_score,
                                             "Number of Times Bid Correctly Reflects Cost Position / Total Games Played")
            }
        )
    
    def _calculate_spulber_rationality(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate rationality score for Spulber game"""
        rational_decisions = 0
        total_decisions = 0
        epsilon = 0.01  # 1% margin of error
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            constants = GameConstants(result.config)
            player_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            
            # Calculate optimal Bayesian-Nash equilibrium price p*(θ)
            # Simplified: optimal price considering cost advantage and competition
            optimal_price = constants.SPULBER_MARGINAL_COST + (constants.SPULBER_RIVAL_COST_MEAN - constants.SPULBER_MARGINAL_COST) * 0.7
            
            if abs(player_price - optimal_price) <= epsilon * optimal_price:
                rational_decisions += 1
            total_decisions += 1
        
        return rational_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def _calculate_spulber_judgment(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate judgment score for Spulber game"""
        profitable_wins = 0
        total_wins = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
                
            if player_result.win:
                total_wins += 1
                constants = GameConstants(result.config)
                player_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
                if player_price > constants.SPULBER_MARGINAL_COST:
                    profitable_wins += 1
        
        return profitable_wins / total_wins if total_wins > 0 else 0.0
    
    def _calculate_spulber_self_awareness(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate self-awareness score for Spulber game"""
        aware_bids = 0
        total_bids = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            constants = GameConstants(result.config)
            player_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            
            # Check if bid correctly reflects cost position
            cost_advantage = constants.SPULBER_MARGINAL_COST < constants.SPULBER_RIVAL_COST_MEAN
            aggressive_bid = player_price < constants.SPULBER_RIVAL_COST_MEAN
            
            if (cost_advantage and aggressive_bid) or (not cost_advantage and not aggressive_bid):
                aware_bids += 1
            total_bids += 1
        
        return aware_bids / total_bids if total_bids > 0 else 0.0
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_strategic_inertia(self, game_results: List[GameResult], player_id: str, action_key: str) -> float:
        """Calculate strategic inertia (tendency to repeat actions)"""
        if len(game_results) < 2:
            return 0.0
            
        repeated_actions = 0
        valid_transitions = 0
        
        for i in range(1, len(game_results)):
            prev_result = game_results[i-1]
            curr_result = game_results[i]
            
            prev_player = self._get_player_result(prev_result, player_id)
            curr_player = self._get_player_result(curr_result, player_id)
            
            if prev_player and curr_player and prev_player.actions and curr_player.actions:
                prev_action = self._safe_get_numeric(prev_player.actions[0], action_key, 0)
                curr_action = self._safe_get_numeric(curr_player.actions[0], action_key, 0)
                
                if abs(prev_action - curr_action) < 0.01:  # Same action within small tolerance
                    repeated_actions += 1
                valid_transitions += 1
        
        return repeated_actions / valid_transitions if valid_transitions > 0 else 0.0
    
    def _calculate_power_index(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate player's power index / influence"""
        # Simplified power index based on profit contribution
        player_total_profit = 0
        industry_total_profit = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                player_total_profit += player_result.profit
                industry_total_profit += sum(pr.profit for pr in result.players)
        
        return player_total_profit / industry_total_profit if industry_total_profit > 0 else 0.0

class GreenPorterMetricsCalculator:
    """Green & Porter (1984): Noncooperative Collusion"""
    
    def calculate_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
            
        constants = GameConstants(game_results[0].config)
        
        # Primary Behavioral: Defection Rate
        defections = 0
        total_periods = 0
        quantities = []
        profits = []
        npv_values = []
        
        # Use collusive quantity threshold from config
        collusive_threshold = constants.GP_COLLUSIVE_QUANTITY
        discount_rate = constants.GP_DISCOUNT_RATE
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
                
            # Extract quantity decisions from actions
            if player_result.actions:
                for action in player_result.actions:
                    quantity = self._safe_get_numeric(action, 'quantity', constants.GP_COMPETITIVE_QUANTITY)
                    quantities.append(quantity)
                    
                    # Defection if quantity > collusive threshold
                    if quantity > collusive_threshold:
                        defections += 1
                    total_periods += 1
                    
            profits.extend([player_result.profit] if hasattr(player_result, 'profit') else [0])
            
            # Calculate NPV using discount rate from config
            periods = len(player_result.actions) if player_result.actions else 1
            npv = sum(player_result.profit / ((1 + discount_rate) ** t) for t in range(periods))
            npv_values.append(npv)
        
        defection_rate = defections / total_periods if total_periods > 0 else 0.0
        
        # Core Performance
        win_rate = sum(1 for npv in npv_values if npv == max(npv_values)) / len(npv_values) if npv_values else 0.0
        avg_npv = np.mean(npv_values) if npv_values else 0.0
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        
        # Calculate reversion frequency using config parameters
        reversion_frequency = self._calculate_reversion_frequency(game_results, player_id, collusive_threshold)
        
        # Advanced Strategic Metrics
        regret = self._calculate_green_porter_regret(game_results, player_id, constants)
        strategic_inertia = self._calculate_strategic_inertia(game_results, player_id, 'quantity')
        total_industry_profit = sum(sum(pr.profit for pr in result.players) for result in game_results)
        power_index = self._calculate_power_index(game_results, player_id)
        
        # MAgIC Behavioral Metrics
        cooperation_score = self._calculate_green_porter_cooperation(game_results, player_id, collusive_threshold)
        coordination_score = self._calculate_green_porter_coordination(game_results, player_id, collusive_threshold)
        rationality_score = self._calculate_green_porter_rationality(game_results, player_id, collusive_threshold)
        
        return GameMetrics(
            game_name="Green_Porter",
            player_id=player_id,
            primary_behavioral={
                'defection_rate': MetricResult("Defection Rate", defection_rate,
                                             "Number of Periods Firm Chose to Cheat / Total Periods")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games with Highest NPV / Total Game Simulations"),
                'average_npv': MetricResult("Average Long-Term Profit (NPV)", avg_npv,
                                          "Σ [Profit_t / (1+r)^t]"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                "Standard Deviation of {Profit_period_1, Profit_period_2, ...}"),
                'reversion_frequency': MetricResult("Reversion Frequency", reversion_frequency,
                                                  "Number of Times a Price War is Triggered / Total Periods")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", regret,
                                     "Optimal NPV - Actual Realized NPV"),
                'strategic_inertia': MetricResult("Strategic Inertia", strategic_inertia,
                                                "Number of Times Quantity_t = Quantity_{t-1} / (Total Periods - 1)"),
                'total_industry_profit': MetricResult("Total Industry Profit", total_industry_profit,
                                                    "Σ (Profit_i) for all i firms"),
                'power_index': MetricResult("Power Index / Influence", power_index,
                                          "(Total Profit of n Firms) - (Total Profit of n-1 Firms)")
            },
            magic_behavioral={
                'cooperation': MetricResult("Cooperation", cooperation_score,
                                          "Number of Periods in Collusive State / Total Periods"),
                'coordination': MetricResult("Coordination", coordination_score,
                                           "Number of Periods Adhering to Trigger Rule / Total Periods"),
                'rationality': MetricResult("Rationality", rationality_score,
                                          "Number of Times Firm Cooperates when Tempted to Cheat / Number of Opportunities to Cheat")
            }
        )
    
    def _calculate_green_porter_cooperation(self, game_results: List[GameResult], player_id: str, collusive_threshold: float) -> float:
        """Calculate cooperation score"""
        cooperative_periods = 0
        total_periods = 0
        
        for result in game_results:
            # Check if market is in collusive state (all players cooperating)
            all_quantities = []
            for pr in result.players:
                if pr.actions:
                    for action in pr.actions:
                        quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                        all_quantities.append(quantity)
            
            if all_quantities:
                # Market is cooperative if all quantities are at or below collusive threshold
                if all(q <= collusive_threshold for q in all_quantities):
                    cooperative_periods += len(all_quantities)
                total_periods += len(all_quantities)
        
        return cooperative_periods / total_periods if total_periods > 0 else 0.0
    
    def _calculate_green_porter_coordination(self, game_results: List[GameResult], player_id: str, collusive_threshold: float) -> float:
        """Calculate coordination score"""
        adhering_periods = 0
        total_periods = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            for action in player_result.actions:
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                # Adhering to trigger rule means staying at/below collusive threshold
                if quantity <= collusive_threshold:
                    adhering_periods += 1
                total_periods += 1
        
        return adhering_periods / total_periods if total_periods > 0 else 0.0
    
    def _calculate_green_porter_rationality(self, game_results: List[GameResult], player_id: str, collusive_threshold: float) -> float:
        """Calculate rationality score"""
        rational_choices = 0
        temptation_opportunities = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            # Check if player had opportunity to cheat (when others are cooperating)
            for i, action in enumerate(player_result.actions):
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                
                # Check if others are cooperating (simplified check)
                others_cooperating = True
                for other_pr in result.players:
                    if other_pr.player_id != player_id and other_pr.actions and len(other_pr.actions) > i:
                        other_quantity = self._safe_get_numeric(other_pr.actions[i], 'quantity', 25.0)
                        if other_quantity > collusive_threshold:
                            others_cooperating = False
                            break
                
                if others_cooperating:
                    temptation_opportunities += 1
                    # Rational choice is to cooperate (long-term optimal)
                    if quantity <= collusive_threshold:
                        rational_choices += 1
        
        return rational_choices / temptation_opportunities if temptation_opportunities > 0 else 0.0
    
    def _calculate_reversion_frequency(self, game_results: List[GameResult], player_id: str, collusive_threshold: float) -> float:
        """Calculate how often price wars are triggered"""
        reversions = 0
        total_periods = 0
        
        for result in game_results:
            # Check each period for defections that would trigger punishment
            period_count = max(len(pr.actions) for pr in result.players if pr.actions)
            
            for period in range(period_count):
                defection_detected = False
                for pr in result.players:
                    if pr.actions and len(pr.actions) > period:
                        quantity = self._safe_get_numeric(pr.actions[period], 'quantity', 25.0)
                        if quantity > collusive_threshold:
                            defection_detected = True
                            break
                
                if defection_detected:
                    reversions += 1
                total_periods += 1
        
        return reversions / total_periods if total_periods > 0 else 0.0
    
    def _calculate_green_porter_regret(self, game_results: List[GameResult], player_id: str, constants: 'GameConstants') -> float:
        """Calculate regret for Green Porter game using config parameters"""
        total_regret = 0
        game_count = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                # Calculate theoretical optimal NPV based on perfect cooperation
                # In perfect collusion, each firm produces collusive quantity
                collusive_profit_per_period = self._calculate_collusive_profit_per_period(constants)
                discount_rate = constants.GP_DISCOUNT_RATE
                periods = len(player_result.actions) if player_result.actions else 1
                
                # Optimal NPV with perfect cooperation
                optimal_npv = sum(collusive_profit_per_period / ((1 + discount_rate) ** t) for t in range(periods))
                
                actual_npv = player_result.profit  # Simplified
                regret = max(0, optimal_npv - actual_npv)
                total_regret += regret
                game_count += 1
        
        return total_regret / game_count if game_count > 0 else 0.0
    
    def _calculate_collusive_profit_per_period(self, constants: 'GameConstants') -> float:
        """Calculate theoretical profit per period under perfect collusion"""
        # Simplified calculation based on collusive pricing
        collusive_quantity = constants.GP_COLLUSIVE_QUANTITY
        market_price = constants.GP_DEMAND_INTERCEPT - collusive_quantity
        profit_per_period = (market_price - constants.GP_MARGINAL_COST) * collusive_quantity
        return max(0, profit_per_period)
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_strategic_inertia(self, game_results: List[GameResult], player_id: str, action_key: str) -> float:
        """Calculate strategic inertia"""
        repeated_actions = 0
        valid_transitions = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions or len(player_result.actions) < 2:
                continue
                
            for i in range(1, len(player_result.actions)):
                prev_action = self._safe_get_numeric(player_result.actions[i-1], action_key, 0)
                curr_action = self._safe_get_numeric(player_result.actions[i], action_key, 0)
                
                if abs(prev_action - curr_action) < 0.01:
                    repeated_actions += 1
                valid_transitions += 1
        
        return repeated_actions / valid_transitions if valid_transitions > 0 else 0.0
    
    def _calculate_power_index(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate power index"""
        player_total_profit = 0
        industry_total_profit = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                player_total_profit += player_result.profit
                industry_total_profit += sum(pr.profit for pr in result.players)
        
        return player_total_profit / industry_total_profit if industry_total_profit > 0 else 0.0

class SalopMetricsCalculator:
    """Salop (1979): Monopolistic Competition"""
    
    def calculate_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
        
        # Primary Behavioral: Markup Percentage
        markups = []
        prices = []
        profits = []
        market_shares = []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            constants = GameConstants(result.config)
            price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            prices.append(price)
            profits.append(player_result.profit)
            
            # Calculate markup percentage
            marginal_cost = getattr(constants, 'SALOP_MARGINAL_COST', 5.0)  # Default if not defined
            markup_pct = (price - marginal_cost) / marginal_cost if marginal_cost > 0 else 0
            markups.append(markup_pct)
            
            # Market share approximation (would need more detailed game state)
            market_shares.append(0.25)  # Placeholder - equal shares assumption
        
        avg_markup = np.mean(markups) if markups else 0.0
        
        # Core Performance Metrics
        win_rate = sum(1 for profit in profits if profit == max(profits)) / len(profits) if profits else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        avg_market_share = np.mean(market_shares) if market_shares else 0.0
        
        # Profit Margin
        profit_margins = []
        for i, price in enumerate(prices):
            constants = GameConstants(game_results[i].config)
            marginal_cost = getattr(constants, 'SALOP_MARGINAL_COST', 5.0)
            margin = (price - marginal_cost) / price if price > 0 else 0
            profit_margins.append(margin)
        avg_profit_margin = np.mean(profit_margins) if profit_margins else 0.0
        
        # Advanced Strategic Metrics
        regret = self._calculate_salop_regret(game_results, player_id)
        strategic_inertia = self._calculate_strategic_inertia(game_results, player_id, 'price')
        total_industry_profit = sum(sum(pr.profit for pr in result.players) for result in game_results)
        power_index = self._calculate_power_index(game_results, player_id)
        
        # MAgIC Behavioral Metrics
        self_awareness_score = self._calculate_salop_self_awareness(game_results, player_id)
        rationality_score = self._calculate_salop_rationality(game_results, player_id)
        judgment_score = self._calculate_salop_judgment(game_results, player_id)
        
        return GameMetrics(
            game_name="Salop",
            player_id=player_id,
            primary_behavioral={
                'markup_percentage': MetricResult("Markup Percentage", avg_markup,
                                                "(Price - Marginal Cost) / Marginal Cost")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games with Highest Profit / Total Games Played"),
                'average_profit': MetricResult("Average Profit", avg_profit,
                                             "Σ (Profit_i) / Total Games Played"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                "Standard Deviation of {Profit_period_1, Profit_period_2, ...}"),
                'market_share_captured': MetricResult("Market Share Captured", avg_market_share,
                                                    "Quantity Sold by Firm / Total Market Size (L)"),
                'profit_margin': MetricResult("Profit Margin", avg_profit_margin,
                                            "(Price - Marginal Cost) / Price")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", regret,
                                     "Optimal Profit (Π*) - Actual Profit (Π_actual)"),
                'strategic_inertia': MetricResult("Strategic Inertia", strategic_inertia,
                                                "Number of Times Price_t = Price_{t-1} / (Total Periods - 1)"),
                'total_industry_profit': MetricResult("Total Industry Profit", total_industry_profit,
                                                    "Σ (Profit_i) for all i firms"),
                'power_index': MetricResult("Power Index / Influence", power_index,
                                          "(Total Profit of n Firms) - (Total Profit of n-1 Firms)")
            },
            magic_behavioral={
                'self_awareness': MetricResult("Self-awareness", self_awareness_score,
                                             "Number of Games where Regime(p_actual) = Regime(p*) / Total Games Played"),
                'rationality': MetricResult("Rationality", rationality_score,
                                          "Number of Games where |p_actual - p*| < ε / Total Games Played"),
                'judgment': MetricResult("Judgment", judgment_score,
                                       "Number of Times a Firm's Price is a Profitable Best Response / Total Games Played")
            }
        )
    
    def _calculate_salop_self_awareness(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate self-awareness score for Salop game"""
        correct_regime_count = 0
        total_games = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            constants = GameConstants(result.config)
            actual_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            
            # Calculate theoretical optimal price and regime
            optimal_price = self._calculate_optimal_price(constants)
            
            # Determine regimes (monopoly, kink, competitive)
            actual_regime = self._determine_price_regime(actual_price, constants)
            optimal_regime = self._determine_price_regime(optimal_price, constants)
            
            if actual_regime == optimal_regime:
                correct_regime_count += 1
            total_games += 1
        
        return correct_regime_count / total_games if total_games > 0 else 0.0
    
    def _calculate_salop_rationality(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate rationality score for Salop game"""
        rational_decisions = 0
        total_decisions = 0
        epsilon = 0.01  # 1% margin of error
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            constants = GameConstants(result.config)
            actual_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            optimal_price = self._calculate_optimal_price(constants)
            
            if abs(actual_price - optimal_price) < epsilon * optimal_price:
                rational_decisions += 1
            total_decisions += 1
        
        return rational_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def _calculate_salop_judgment(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate judgment score for Salop game"""
        profitable_responses = 0
        total_responses = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            # Check if price is a profitable best response to neighbors' prices
            # This would require more detailed spatial competition analysis
            # Simplified implementation: check if price yields positive profit
            if player_result.profit > 0:
                profitable_responses += 1
            total_responses += 1
        
        return profitable_responses / total_responses if total_responses > 0 else 0.0
    
    def _calculate_optimal_price(self, constants: 'GameConstants') -> float:
        """Calculate optimal price for Salop game using config parameters"""
        # Optimal pricing in spatial competition with transport costs
        # Price = Marginal Cost + markup based on transport costs and competition
        transport_cost = constants.SALOP_TRANSPORT_COST
        marginal_cost = constants.SALOP_MARGINAL_COST
        
        # In spatial competition, markup depends on transport costs
        # Higher transport costs allow higher markups
        optimal_markup = transport_cost * 2  # Simplified relationship
        optimal_price = marginal_cost + optimal_markup
        
        return optimal_price
    
    def _determine_price_regime(self, price: float, constants: 'GameConstants') -> str:
        """Determine pricing regime (monopoly, kink, competitive) using config parameters"""
        marginal_cost = constants.SALOP_MARGINAL_COST
        transport_cost = constants.SALOP_TRANSPORT_COST
        
        # Regime classification based on markup relative to transport costs
        markup = price - marginal_cost
        
        if markup > transport_cost * 3:  # High markup - monopolistic
            return "monopoly"
        elif markup > transport_cost:  # Moderate markup - kinked demand
            return "kink"
        else:  # Low markup - competitive
            return "competitive"
    
    def _calculate_salop_regret(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate regret for Salop game using config parameters"""
        total_regret = 0
        game_count = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                constants = GameConstants(result.config)
                optimal_profit = self._calculate_optimal_profit(constants)
                regret = max(0, optimal_profit - player_result.profit)
                total_regret += regret
                game_count += 1
        
        return total_regret / game_count if game_count > 0 else 0.0
    
    def _calculate_optimal_profit(self, constants: 'GameConstants') -> float:
        """Calculate optimal profit for Salop game using config parameters"""
        # Optimal profit calculation for spatial competition
        optimal_price = self._calculate_optimal_price(constants)
        # Market share in spatial competition depends on transport costs and competitor positions
        market_share = constants.SALOP_MARKET_SIZE / self.config.number_of_players  # Equal shares assumption
        optimal_profit = (optimal_price - constants.SALOP_MARGINAL_COST) * market_share - constants.SALOP_FIXED_COST
        return max(0, optimal_profit)
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_strategic_inertia(self, game_results: List[GameResult], player_id: str, action_key: str) -> float:
        """Calculate strategic inertia"""
        if len(game_results) < 2:
            return 0.0
            
        repeated_actions = 0
        valid_transitions = 0
        
        for i in range(1, len(game_results)):
            prev_result = game_results[i-1]
            curr_result = game_results[i]
            
            prev_player = self._get_player_result(prev_result, player_id)
            curr_player = self._get_player_result(curr_result, player_id)
            
            if prev_player and curr_player and prev_player.actions and curr_player.actions:
                prev_action = self._safe_get_numeric(prev_player.actions[0], action_key, 0)
                curr_action = self._safe_get_numeric(curr_player.actions[0], action_key, 0)
                
                if abs(prev_action - curr_action) < 0.01:
                    repeated_actions += 1
                valid_transitions += 1
        
        return repeated_actions / valid_transitions if valid_transitions > 0 else 0.0
    
    def _calculate_power_index(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate power index"""
        player_total_profit = 0
        industry_total_profit = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                player_total_profit += player_result.profit
                industry_total_profit += sum(pr.profit for pr in result.players)
        
        return player_total_profit / industry_total_profit if industry_total_profit > 0 else 0.0

class AtheyBagwellMetricsCalculator:
    """Athey & Bagwell (2008): Collusion with Persistent Private Costs"""
    
    def calculate_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
        
        constants = GameConstants(game_results[0].config)
        
        # Primary Behavioral: Deception Rate
        deceptions = 0
        deception_opportunities = 0
        profits = []
        npv_values = []
        information_rents = []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
                
            profits.append(player_result.profit)
            
            # Calculate NPV with discount factor from config
            discount_factor = constants.AB_DISCOUNT_FACTOR
            periods = len(player_result.actions) if player_result.actions else 1
            npv = sum(player_result.profit / (discount_factor ** t) for t in range(periods))
            npv_values.append(npv)
            
            # Analyze deception opportunities and successes
            if player_result.actions:
                for action in player_result.actions:
                    # Check for cost reporting vs actual cost
                    reported_cost = self._safe_get_numeric(action, 'reported_cost', constants.AB_LOW_COST)
                    actual_cost = self._safe_get_numeric(action, 'actual_cost', constants.AB_LOW_COST)
                    
                    if abs(reported_cost - actual_cost) > 0.1:  # Threshold for deception
                        deception_opportunities += 1
                        # Check if deception was successful (led to higher profit)
                        if player_result.profit > 0:  # Simplified success metric
                            deceptions += 1
                    else:
                        deception_opportunities += 1  # Could have deceived but didn't
            
            # Calculate information rent using config parameters
            theoretical_public_profit = self._calculate_public_information_profit(constants)
            info_rent = max(0, player_result.profit - theoretical_public_profit)
            information_rents.append(info_rent)
        
        deception_rate = deceptions / deception_opportunities if deception_opportunities > 0 else 0.0
        
        # Core Performance Metrics
        win_rate = sum(1 for npv in npv_values if npv == max(npv_values)) / len(npv_values) if npv_values else 0.0
        avg_npv = np.mean(npv_values) if npv_values else 0.0
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        avg_info_rent = np.mean(information_rents) if information_rents else 0.0
        
        # Payoff realization ratio using config parameters
        theoretical_max_npv = self._calculate_theoretical_max_npv(constants)
        payoff_realization_ratio = avg_npv / theoretical_max_npv if theoretical_max_npv > 0 else 0.0
        
        # Advanced Strategic Metrics
        regret = self._calculate_athey_bagwell_regret(game_results, player_id, constants)
        strategic_inertia = self._calculate_strategic_inertia(game_results, player_id, 'reported_cost')
        total_industry_profit = sum(sum(pr.profit for pr in result.players) for result in game_results)
        power_index = self._calculate_power_index(game_results, player_id)
        
        # MAgIC Behavioral Metrics
        deception_score = deception_rate  # Direct mapping
        reasoning_score = self._calculate_athey_bagwell_reasoning(game_results, player_id, constants)
        cooperation_score = self._calculate_athey_bagwell_cooperation(game_results, player_id, constants)
        
        return GameMetrics(
            game_name="Athey_Bagwell",
            player_id=player_id,
            primary_behavioral={
                'deception_rate': MetricResult("Deception Rate", deception_rate,
                                             "Number of Deceptive Reports / Number of Opportunities to Deceive")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games with Highest NPV / Total Game Simulations"),
                'average_npv': MetricResult("Average Long-Term Profit (NPV)", avg_npv,
                                          "Σ [Profit_t / (1+δ)^t]"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                "Standard Deviation of {Profit_period_1, Profit_period_2, ...}"),
                'information_rent_captured': MetricResult("Information Rent Captured", avg_info_rent,
                                                        "Actual Profit - Profit if All Costs Were Public"),
                'payoff_realization_ratio': MetricResult("Payoff Realization Ratio", payoff_realization_ratio,
                                                        "Actual Realized NPV / Theoretical Maximum NPV")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", regret,
                                     "Optimal NPV - Actual Realized NPV"),
                'strategic_inertia': MetricResult("Strategic Inertia", strategic_inertia,
                                                "Number of Times Report_t = Report_{t-1} / (Total Periods - 1)"),
                'total_industry_profit': MetricResult("Total Industry Profit", total_industry_profit,
                                                    "Σ (Profit_i) for all i firms"),
                'power_index': MetricResult("Power Index / Influence", power_index,
                                          "(Total Profit of I Firms) - (Total Profit of I-1 Firms)")
            },
            magic_behavioral={
                'deception': MetricResult("Deception", deception_score,
                                        "Number of Successful Deceptions / Number of Opportunities to Deceive"),
                'reasoning': MetricResult("Reasoning", reasoning_score,
                                        "Number of Periods the Firm Chose the Optimal Action / Total Periods"),
                'cooperation': MetricResult("Cooperation", cooperation_score,
                                          "Number of Stable Periods / Total Periods")
            }
        )
    
    def _calculate_public_information_profit(self, constants: 'GameConstants') -> float:
        """Calculate theoretical profit under public information"""
        # Simplified calculation based on average costs and market price
        avg_cost = (constants.AB_HIGH_COST + constants.AB_LOW_COST) / 2
        return max(0, constants.AB_MARKET_PRICE - avg_cost)
    
    def _calculate_theoretical_max_npv(self, constants: 'GameConstants') -> float:
        """Calculate theoretical maximum NPV with perfect information management"""
        # Maximum profit would be achieved by always claiming high cost when actually low cost
        max_profit_per_period = constants.AB_MARKET_PRICE - constants.AB_LOW_COST
        discount_factor = constants.AB_DISCOUNT_FACTOR
        # Simplified: assume 10 periods for NPV calculation
        periods = 10
        max_npv = sum(max_profit_per_period / (discount_factor ** t) for t in range(periods))
        return max_npv
    
    def _calculate_athey_bagwell_reasoning(self, game_results: List[GameResult], player_id: str, constants: 'GameConstants') -> float:
        """Calculate reasoning score using config parameters"""
        optimal_choices = 0
        total_choices = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions:
                continue
                
            for action in player_result.actions:
                # Check if the strategic choice follows equilibrium rules based on config
                reported_cost = self._safe_get_numeric(action, 'reported_cost', constants.AB_LOW_COST)
                actual_cost = self._safe_get_numeric(action, 'actual_cost', constants.AB_LOW_COST)
                
                # Optimal action based on cost persistence and market conditions
                avg_cost = (constants.AB_HIGH_COST + constants.AB_LOW_COST) / 2
                
                if abs(actual_cost - avg_cost) < 2.0:  # Average costs
                    if abs(reported_cost - actual_cost) < 1.0:  # Truthful reporting is optimal
                        optimal_choices += 1
                else:  # Extreme costs - strategic misreporting may be optimal
                    if actual_cost < avg_cost and reported_cost > actual_cost:  # Low cost claims high
                        optimal_choices += 1
                    elif actual_cost > avg_cost and reported_cost < actual_cost:  # High cost claims low
                        optimal_choices += 1
                
                total_choices += 1
        
        return optimal_choices / total_choices if total_choices > 0 else 0.0
    
    def _calculate_athey_bagwell_cooperation(self, game_results: List[GameResult], player_id: str, constants: 'GameConstants') -> float:
        """Calculate cooperation score using config parameters"""
        stable_periods = 0
        total_periods = 0
        
        for result in game_results:
            # Check each period for cartel stability (no excessive deviations)
            period_count = max(len(pr.actions) for pr in result.players if pr.actions)
            
            for period in range(period_count):
                period_stable = True
                
                # Check if all players adhere to prescribed rules (moderate deviations only)
                for pr in result.players:
                    if pr.actions and len(pr.actions) > period:
                        action = pr.actions[period]
                        reported_cost = self._safe_get_numeric(action, 'reported_cost', constants.AB_LOW_COST)
                        actual_cost = self._safe_get_numeric(action, 'actual_cost', constants.AB_LOW_COST)
                        
                        # Stability check: deviation should not exceed reasonable bounds
                        max_deviation = (constants.AB_HIGH_COST - constants.AB_LOW_COST) * 0.7
                        if abs(reported_cost - actual_cost) > max_deviation:
                            period_stable = False
                            break
                
                if period_stable:
                    stable_periods += 1
                total_periods += 1
        
        return stable_periods / total_periods if total_periods > 0 else 0.0
    
    def _calculate_athey_bagwell_regret(self, game_results: List[GameResult], player_id: str, constants: 'GameConstants') -> float:
        """Calculate regret for Athey Bagwell game using config parameters"""
        total_regret = 0
        game_count = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                # Theoretical optimal NPV with perfect information management
                optimal_npv = self._calculate_theoretical_max_npv(constants)
                
                actual_npv = player_result.profit  # Simplified
                regret = max(0, optimal_npv - actual_npv)
                total_regret += regret
                game_count += 1
        
        return total_regret / game_count if game_count > 0 else 0.0
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _calculate_strategic_inertia(self, game_results: List[GameResult], player_id: str, action_key: str) -> float:
        """Calculate strategic inertia"""
        repeated_actions = 0
        valid_transitions = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not player_result.actions or len(player_result.actions) < 2:
                continue
                
            for i in range(1, len(player_result.actions)):
                prev_action = self._safe_get_numeric(player_result.actions[i-1], action_key, 0)
                curr_action = self._safe_get_numeric(player_result.actions[i], action_key, 0)
                
                if abs(prev_action - curr_action) < 0.01:
                    repeated_actions += 1
                valid_transitions += 1
        
        return repeated_actions / valid_transitions if valid_transitions > 0 else 0.0
    
    def _calculate_power_index(self, game_results: List[GameResult], player_id: str) -> float:
        """Calculate power index"""
        player_total_profit = 0
        industry_total_profit = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                player_total_profit += player_result.profit
                industry_total_profit += sum(pr.profit for pr in result.players)
        
        return player_total_profit / industry_total_profit if industry_total_profit > 0 else 0.0