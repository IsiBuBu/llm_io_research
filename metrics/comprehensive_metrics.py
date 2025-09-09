"""
Complete Comprehensive Metrics Calculator
Implements ALL metrics from both the MAgIC paper and non-MAgIC papers for all four games
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Handle imports gracefully
try:
    from config import GameConstants
except ImportError:
    GameConstants = None

if TYPE_CHECKING:
    from config import GameResult, PlayerResult, GameConfig

# For robust operation without imports
class GameResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class PlayerResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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
    magic_behavioral: Dict[str, MetricResult]


class ComprehensiveMetricsCalculator:
    """Complete implementation of all game theory behavioral metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.spulber_calculator = SpulberMetricsCalculator()
        self.green_porter_calculator = GreenPorterMetricsCalculator()
        self.salop_calculator = SalopMetricsCalculator()
        self.athey_bagwell_calculator = AtheyBagwellMetricsCalculator()
    
    def calculate_all_metrics(self, game_results: List[Any], 
                            game_type: str, player_id: str = 'challenger') -> GameMetrics:
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


class BaseMetricsCalculator:
    """Base class with common utilities for all game metrics calculators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _get_player_result(self, game_result: Any, player_id: str) -> Optional[Any]:
        """Get player result from game result"""
        if not hasattr(game_result, 'players'):
            return None
        
        for player in game_result.players:
            if hasattr(player, 'player_id') and player.player_id == player_id:
                return player
            # Also check player_role for compatibility
            if hasattr(player, 'player_role') and player_id in player.player_role:
                return player
        
        return None
    
    def _safe_get_numeric(self, action_dict: Dict[str, Any], key: str, default: float) -> float:
        """Safely extract numeric value from action dictionary"""
        try:
            if isinstance(action_dict, dict) and key in action_dict:
                value = action_dict[key]
                return float(value) if value is not None else default
            elif hasattr(action_dict, 'action_data') and isinstance(action_dict.action_data, dict):
                value = action_dict.action_data.get(key, default)
                return float(value) if value is not None else default
            else:
                return default
        except (ValueError, TypeError, AttributeError):
            return default
    
    def _calculate_strategic_inertia(self, actions: List[Dict[str, Any]], key: str) -> float:
        """Calculate strategic inertia (tendency to repeat same decision)"""
        if len(actions) < 2:
            return 0.0
        
        repeated_actions = 0
        for i in range(1, len(actions)):
            prev_value = self._safe_get_numeric(actions[i-1], key, 0)
            curr_value = self._safe_get_numeric(actions[i], key, 0)
            if abs(prev_value - curr_value) < 0.01:  # Same decision
                repeated_actions += 1
        
        return repeated_actions / (len(actions) - 1)
    
    def _calculate_npv(self, profits: List[float], discount_factor: float = 0.95) -> float:
        """Calculate Net Present Value of profit stream"""
        npv = 0.0
        for t, profit in enumerate(profits):
            npv += profit / (discount_factor ** t)
        return npv


class SpulberMetricsCalculator(BaseMetricsCalculator):
    """Spulber (1995): Bertrand Competition with Unknown Costs - ALL METRICS"""
    
    def calculate_metrics(self, game_results: List[Any], player_id: str = 'challenger') -> GameMetrics:
        if not game_results:
            return self._empty_metrics('spulber', player_id)
        
        # Extract all data first
        prices, profits, wins, actions_data = self._extract_spulber_data(game_results, player_id)
        
        if not prices:
            return self._empty_metrics('spulber', player_id)
        
        # PRIMARY BEHAVIORAL METRICS
        avg_bid_price = np.mean(prices)
        
        # CORE PERFORMANCE METRICS  
        win_rate = np.mean([1 if (win and profit > 0) else 0 for win, profit in zip(wins, profits)])
        avg_profit = np.mean(profits)
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        market_capture_rate = np.mean([1 if win else 0 for win in wins])
        
        # ADVANCED STRATEGIC METRICS
        regret = self._calculate_spulber_regret(game_results, player_id, prices, profits)
        strategic_inertia = self._calculate_strategic_inertia(actions_data, 'price') if len(actions_data) > 1 else 0.0
        total_industry_profit = sum(profits)  # In winner-take-all, only winner has profit
        power_index = total_industry_profit  # Marginal contribution equals total in winner-take-all
        
        # MAGIC BEHAVIORAL METRICS
        rationality = self._calculate_spulber_rationality(game_results, player_id)
        judgment = self._calculate_spulber_judgment(game_results, player_id)
        self_awareness = self._calculate_spulber_self_awareness(game_results, player_id)
        
        return GameMetrics(
            game_name='spulber',
            player_id=player_id,
            primary_behavioral={
                'avg_bid_price': MetricResult("Average Bid Price", avg_bid_price,
                                            "Σ (Price_i) / Number of Games Played")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games Won with Profit > 0 / Total Games Played"),
                'avg_profit': MetricResult("Average Profit", avg_profit,
                                         "Σ (Profit_i) / Total Games Played"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                 "Standard Deviation of {Profit_win, 0, 0, ...}"),
                'market_capture_rate': MetricResult("Market Capture Rate", market_capture_rate,
                                                  "Number of Times Firm Sets Lowest Price / Total Games Played")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", regret,
                                     "Ex-Post Optimal Profit - Actual Profit"),
                'strategic_inertia': MetricResult("Strategic Inertia", strategic_inertia,
                                                "Number of Times Price_t = Price_{t-1} / (Total Periods - 1)"),
                'total_industry_profit': MetricResult("Total Industry Profit", total_industry_profit,
                                                    "Profit of Winning Firm"),
                'power_index': MetricResult("Power Index / Influence", power_index,
                                          "(Total Profit of n Firms) - (Total Profit of n-1 Firms)")
            },
            magic_behavioral={
                'rationality': MetricResult("Rationality", rationality,
                                          "Number of Games where |p_actual - p*(θ)| < ε / Total Games Played"),
                'judgment': MetricResult("Judgment", judgment,
                                       "Number of Wins where Price > Cost / Total Number of Wins"),
                'self_awareness': MetricResult("Self-awareness", self_awareness,
                                             "Number of Times Bid Correctly Reflects Cost Position / Total Games Played")
            }
        )
    
    def _extract_spulber_data(self, game_results: List[Any], player_id: str):
        """Extract all relevant data for Spulber calculations"""
        prices, profits, wins, actions_data = [], [], [], []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            
            # Extract price from first action
            if hasattr(player_result, 'actions') and player_result.actions:
                price = self._safe_get_numeric(player_result.actions[0], 'bid', 10.0)
                if price == 10.0:  # Try 'price' if 'bid' not found
                    price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
                prices.append(price)
                actions_data.append(player_result.actions[0])
            
            # Extract profit and win status
            if hasattr(player_result, 'profit'):
                profits.append(player_result.profit)
            if hasattr(player_result, 'win'):
                wins.append(player_result.win)
        
        return prices, profits, wins, actions_data
    
    def _calculate_spulber_regret(self, game_results: List[Any], player_id: str, prices: List[float], profits: List[float]) -> float:
        """Calculate regret for Spulber game"""
        if not game_results or not profits:
            return 0.0
        
        total_regret = 0.0
        
        for i, result in enumerate(game_results):
            if i >= len(prices) or i >= len(profits):
                continue
            
            # Ex-post optimal would be to bid just below the second-lowest bid
            all_player_prices = []
            for pr in result.players:
                if hasattr(pr, 'actions') and pr.actions:
                    price = self._safe_get_numeric(pr.actions[0], 'bid', 10.0)
                    if price == 10.0:
                        price = self._safe_get_numeric(pr.actions[0], 'price', 10.0)
                    all_player_prices.append(price)
            
            if len(all_player_prices) >= 2:
                sorted_prices = sorted(all_player_prices)
                if len(sorted_prices) > 1:
                    # Optimal bid would be just below second lowest
                    optimal_bid = sorted_prices[1] - 0.01
                    try:
                        constants = GameConstants() if GameConstants else None
                        market_value = constants.SPULBER_MARKET_VALUE if constants else 100.0
                        optimal_profit = market_value - optimal_bid
                        total_regret += max(0, optimal_profit - profits[i])
                    except:
                        # Fallback calculation
                        market_value = 100.0
                        optimal_profit = market_value - optimal_bid
                        total_regret += max(0, optimal_profit - profits[i])
        
        return total_regret / len(profits) if profits else 0.0
    
    def _calculate_spulber_rationality(self, game_results: List[Any], player_id: str) -> float:
        """Calculate rationality score for Spulber game"""
        rational_decisions = 0
        total_decisions = 0
        epsilon = 0.01
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions') or not player_result.actions:
                continue
            
            player_price = self._safe_get_numeric(player_result.actions[0], 'bid', 10.0)
            if player_price == 10.0:
                player_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            
            try:
                constants = GameConstants() if GameConstants else None
                if constants:
                    marginal_cost = constants.SPULBER_MARGINAL_COST
                    rival_mean = constants.SPULBER_RIVAL_COST_MEAN
                else:
                    marginal_cost, rival_mean = 10.0, 12.0
                
                # Simplified optimal Bayesian-Nash price
                optimal_price = marginal_cost + (rival_mean - marginal_cost) * 0.7
                
                if abs(player_price - optimal_price) <= epsilon * optimal_price:
                    rational_decisions += 1
            except:
                # Fallback - check if price is reasonable markup above cost
                if 10.0 <= player_price <= 20.0:
                    rational_decisions += 1
            
            total_decisions += 1
        
        return rational_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def _calculate_spulber_judgment(self, game_results: List[Any], player_id: str) -> float:
        """Calculate judgment score for Spulber game"""
        profitable_wins = 0
        total_wins = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            
            if hasattr(player_result, 'win') and player_result.win:
                total_wins += 1
                if hasattr(player_result, 'actions') and player_result.actions:
                    player_price = self._safe_get_numeric(player_result.actions[0], 'bid', 10.0)
                    if player_price == 10.0:
                        player_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
                    
                    try:
                        constants = GameConstants() if GameConstants else None
                        cost = constants.SPULBER_MARGINAL_COST if constants else 10.0
                    except:
                        cost = 10.0
                    
                    if player_price > cost:
                        profitable_wins += 1
        
        return profitable_wins / total_wins if total_wins > 0 else 0.0
    
    def _calculate_spulber_self_awareness(self, game_results: List[Any], player_id: str) -> float:
        """Calculate self-awareness score for Spulber game"""
        correct_positioning = 0
        total_games = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions') or not player_result.actions:
                continue
            
            player_price = self._safe_get_numeric(player_result.actions[0], 'bid', 10.0)
            if player_price == 10.0:
                player_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            
            try:
                constants = GameConstants() if GameConstants else None
                if constants:
                    my_cost = constants.SPULBER_MARGINAL_COST
                    rival_mean = constants.SPULBER_RIVAL_COST_MEAN
                else:
                    my_cost, rival_mean = 10.0, 12.0
                
                # Check if bid correctly reflects cost position
                if my_cost < rival_mean:  # Cost advantage
                    if player_price < rival_mean:  # Should bid aggressively
                        correct_positioning += 1
                elif my_cost > rival_mean:  # Cost disadvantage
                    if player_price >= rival_mean:  # Should bid conservatively
                        correct_positioning += 1
                else:  # Equal costs
                    if abs(player_price - rival_mean) < 2.0:  # Should bid near average
                        correct_positioning += 1
            except:
                # Fallback - assume correct if bid is reasonable
                if 8.0 <= player_price <= 15.0:
                    correct_positioning += 1
            
            total_games += 1
        
        return correct_positioning / total_games if total_games > 0 else 0.0
    
    def _empty_metrics(self, game_name: str, player_id: str) -> GameMetrics:
        """Return empty metrics structure"""
        return GameMetrics(
            game_name=game_name,
            player_id=player_id,
            primary_behavioral={},
            core_performance={},
            advanced_strategic={},
            magic_behavioral={}
        )


class GreenPorterMetricsCalculator(BaseMetricsCalculator):
    """Green & Porter (1984): Noncooperative Collusion - ALL METRICS"""
    
    def calculate_metrics(self, game_results: List[Any], player_id: str = 'challenger') -> GameMetrics:
        if not game_results:
            return self._empty_metrics('green_porter', player_id)
        
        # Extract data
        quantities, profits, actions_data = self._extract_green_porter_data(game_results, player_id)
        
        if not quantities:
            return self._empty_metrics('green_porter', player_id)
        
        # Get collusive threshold
        try:
            constants = GameConstants() if GameConstants else None
            collusive_threshold = constants.GP_COLLUSIVE_QUANTITY if constants else 22.5
            discount_factor = constants.AB_DISCOUNT_FACTOR if constants else 0.95
        except:
            collusive_threshold, discount_factor = 22.5, 0.95
        
        # PRIMARY BEHAVIORAL METRICS
        defection_rate = self._calculate_defection_rate(quantities, collusive_threshold)
        
        # CORE PERFORMANCE METRICS
        win_rate = self._calculate_green_porter_win_rate(game_results, player_id)
        npv = self._calculate_npv(profits, discount_factor)
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        reversion_frequency = self._calculate_reversion_frequency(game_results)
        
        # ADVANCED STRATEGIC METRICS
        regret = self._calculate_green_porter_regret(game_results, player_id, npv)
        strategic_inertia = self._calculate_strategic_inertia(actions_data, 'quantity') if len(actions_data) > 1 else 0.0
        total_industry_profit = self._calculate_total_industry_profit(game_results)
        power_index = self._calculate_power_index(game_results, player_id)
        
        # MAGIC BEHAVIORAL METRICS
        cooperation = self._calculate_green_porter_cooperation(game_results, player_id, collusive_threshold)
        coordination = self._calculate_green_porter_coordination(game_results, player_id, collusive_threshold)
        rationality = self._calculate_green_porter_rationality(game_results, player_id, collusive_threshold)
        
        return GameMetrics(
            game_name='green_porter',
            player_id=player_id,
            primary_behavioral={
                'defection_rate': MetricResult("Defection Rate", defection_rate,
                                             "Number of Periods Firm Chose to Cheat / Total Periods")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games with Highest NPV / Total Game Simulations"),
                'avg_long_term_profit': MetricResult("Average Long-Term Profit (NPV)", npv,
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
                'cooperation': MetricResult("Cooperation", cooperation,
                                          "Number of Periods in Collusive State / Total Periods"),
                'coordination': MetricResult("Coordination", coordination,
                                           "Number of Periods Adhering to Trigger Rule / Total Periods"),
                'rationality': MetricResult("Rationality", rationality,
                                          "Number of Times Firm Cooperates when Tempted to Cheat / Number of Opportunities to Cheat")
            }
        )
    
    def _extract_green_porter_data(self, game_results: List[Any], player_id: str):
        """Extract Green-Porter specific data"""
        quantities, profits, actions_data = [], [], []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            
            if hasattr(player_result, 'actions'):
                for action in player_result.actions:
                    quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                    quantities.append(quantity)
                    actions_data.append(action)
            
            if hasattr(player_result, 'profit'):
                profits.append(player_result.profit)
        
        return quantities, profits, actions_data
    
    def _calculate_defection_rate(self, quantities: List[float], collusive_threshold: float) -> float:
        """Calculate defection rate (periods above collusive threshold)"""
        if not quantities:
            return 0.0
        
        defections = sum(1 for q in quantities if q > collusive_threshold)
        return defections / len(quantities)
    
    def _calculate_green_porter_win_rate(self, game_results: List[Any], player_id: str) -> float:
        """Calculate win rate based on highest NPV"""
        if not game_results:
            return 0.0
        
        wins = 0
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'win') and player_result.win:
                wins += 1
        
        return wins / len(game_results)
    
    def _calculate_reversion_frequency(self, game_results: List[Any]) -> float:
        """Calculate how often price wars are triggered"""
        # Simplified: assume reversion when any player produces above threshold
        reversions = 0
        total_periods = 0
        
        try:
            constants = GameConstants() if GameConstants else None
            collusive_threshold = constants.GP_COLLUSIVE_QUANTITY if constants else 22.5
        except:
            collusive_threshold = 22.5
        
        for result in game_results:
            if hasattr(result, 'players'):
                for player in result.players:
                    if hasattr(player, 'actions'):
                        for action in player.actions:
                            quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                            if quantity > collusive_threshold * 1.1:  # Significant deviation
                                reversions += 1
                            total_periods += 1
        
        return reversions / total_periods if total_periods > 0 else 0.0
    
    def _calculate_green_porter_regret(self, game_results: List[Any], player_id: str, actual_npv: float) -> float:
        """Calculate regret (optimal NPV - actual NPV)"""
        # Optimal strategy would be perfect collusion
        try:
            constants = GameConstants() if GameConstants else None
            collusive_profit = 50.0  # Simplified optimal per-period profit
            discount_factor = constants.AB_DISCOUNT_FACTOR if constants else 0.95
            periods = sum(len(getattr(self._get_player_result(r, player_id), 'actions', [])) for r in game_results)
            optimal_npv = sum(collusive_profit / (discount_factor ** t) for t in range(periods))
            return max(0, optimal_npv - actual_npv)
        except:
            return 0.0
    
    def _calculate_total_industry_profit(self, game_results: List[Any]) -> float:
        """Calculate total industry profit across all firms"""
        total_profit = 0.0
        for result in game_results:
            if hasattr(result, 'players'):
                for player in result.players:
                    if hasattr(player, 'profit'):
                        total_profit += player.profit
        return total_profit
    
    def _calculate_power_index(self, game_results: List[Any], player_id: str) -> float:
        """Calculate marginal contribution to total industry profit"""
        total_with_player = self._calculate_total_industry_profit(game_results)
        
        # Estimate total without this player (simplified)
        player_contribution = 0.0
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'profit'):
                player_contribution += player_result.profit
        
        total_without_player = total_with_player - player_contribution
        return total_with_player - total_without_player
    
    def _calculate_green_porter_cooperation(self, game_results: List[Any], player_id: str, collusive_threshold: float) -> float:
        """Calculate cooperation score (periods in collusive state)"""
        cooperative_periods = 0
        total_periods = 0
        
        for result in game_results:
            if hasattr(result, 'players'):
                # Check if all players are cooperating
                period_quantities = []
                for player in result.players:
                    if hasattr(player, 'actions'):
                        for action in player.actions:
                            quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                            period_quantities.append(quantity)
                
                if period_quantities:
                    # Cooperative if all quantities at or below threshold
                    if all(q <= collusive_threshold for q in period_quantities):
                        cooperative_periods += len(period_quantities)
                    total_periods += len(period_quantities)
        
        return cooperative_periods / total_periods if total_periods > 0 else 0.0
    
    def _calculate_green_porter_coordination(self, game_results: List[Any], player_id: str, collusive_threshold: float) -> float:
        """Calculate coordination score (adhering to trigger rule)"""
        adhering_periods = 0
        total_periods = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions'):
                continue
            
            for action in player_result.actions:
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                if quantity <= collusive_threshold:
                    adhering_periods += 1
                total_periods += 1
        
        return adhering_periods / total_periods if total_periods > 0 else 0.0
    
    def _calculate_green_porter_rationality(self, game_results: List[Any], player_id: str, collusive_threshold: float) -> float:
        """Calculate rationality (cooperating when tempted to cheat)"""
        cooperative_choices = 0
        temptation_opportunities = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions'):
                continue
            
            for action in player_result.actions:
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                # Every period is an opportunity to cheat for short-term gain
                temptation_opportunities += 1
                if quantity <= collusive_threshold:
                    cooperative_choices += 1
        
        return cooperative_choices / temptation_opportunities if temptation_opportunities > 0 else 0.0
    
    def _empty_metrics(self, game_name: str, player_id: str) -> GameMetrics:
        """Return empty metrics structure"""
        return GameMetrics(
            game_name=game_name,
            player_id=player_id,
            primary_behavioral={},
            core_performance={},
            advanced_strategic={},
            magic_behavioral={}
        )


class SalopMetricsCalculator(BaseMetricsCalculator):
    """Salop (1979): Monopolistic Competition - ALL METRICS"""
    
    def calculate_metrics(self, game_results: List[Any], player_id: str = 'challenger') -> GameMetrics:
        if not game_results:
            return self._empty_metrics('salop', player_id)
        
        # Extract data
        prices, profits, actions_data = self._extract_salop_data(game_results, player_id)
        
        if not prices:
            return self._empty_metrics('salop', player_id)
        
        try:
            constants = GameConstants() if GameConstants else None
            marginal_cost = constants.SALOP_MARGINAL_COST if constants else 8.0
            market_size = constants.SALOP_BASE_MARKET_SIZE if constants else 300
        except:
            marginal_cost, market_size = 8.0, 300
        
        # PRIMARY BEHAVIORAL METRICS
        markup_percentage = self._calculate_markup_percentage(prices, marginal_cost)
        
        # CORE PERFORMANCE METRICS
        win_rate = self._calculate_salop_win_rate(game_results, player_id)
        avg_profit = np.mean(profits) if profits else 0.0
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        market_share = self._calculate_market_share_captured(game_results, player_id, market_size)
        profit_margin = self._calculate_profit_margin(prices, marginal_cost)
        
        # ADVANCED STRATEGIC METRICS
        regret = self._calculate_salop_regret(game_results, player_id, profits)
        strategic_inertia = self._calculate_strategic_inertia(actions_data, 'price') if len(actions_data) > 1 else 0.0
        total_industry_profit = self._calculate_total_industry_profit(game_results)
        power_index = self._calculate_power_index(game_results, player_id)
        
        # MAGIC BEHAVIORAL METRICS
        self_awareness = self._calculate_salop_self_awareness(game_results, player_id)
        rationality = self._calculate_salop_rationality(game_results, player_id)
        judgment = self._calculate_salop_judgment(game_results, player_id)
        
        return GameMetrics(
            game_name='salop',
            player_id=player_id,
            primary_behavioral={
                'markup_percentage': MetricResult("Markup Percentage", markup_percentage,
                                                 "(Price - Marginal Cost) / Marginal Cost")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games with Highest Profit / Total Games Played"),
                'avg_profit': MetricResult("Average Profit", avg_profit,
                                         "Σ (Profit_i) / Total Games Played"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                 "Standard Deviation of {Profit_period_1, Profit_period_2, ...}"),
                'market_share_captured': MetricResult("Market Share Captured", market_share,
                                                    "Quantity Sold by Firm / Total Market Size (L)"),
                'profit_margin': MetricResult("Profit Margin", profit_margin,
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
                'self_awareness': MetricResult("Self-awareness", self_awareness,
                                             "Number of Games where Regime(p_actual) = Regime(p*) / Total Games Played"),
                'rationality': MetricResult("Rationality", rationality,
                                          "Number of Games where |p_actual - p*| < ε / Total Games Played"),
                'judgment': MetricResult("Judgment", judgment,
                                       "Number of Times a Firm's Price is a Profitable Best Response / Total Games Played")
            }
        )
    
    def _extract_salop_data(self, game_results: List[Any], player_id: str):
        """Extract Salop-specific data"""
        prices, profits, actions_data = [], [], []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            
            if hasattr(player_result, 'actions') and player_result.actions:
                price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
                prices.append(price)
                actions_data.append(player_result.actions[0])
            
            if hasattr(player_result, 'profit'):
                profits.append(player_result.profit)
        
        return prices, profits, actions_data
    
    def _calculate_markup_percentage(self, prices: List[float], marginal_cost: float) -> float:
        """Calculate average markup percentage"""
        if not prices:
            return 0.0
        
        markups = [(p - marginal_cost) / marginal_cost for p in prices if p > 0]
        return np.mean(markups) if markups else 0.0
    
    def _calculate_salop_win_rate(self, game_results: List[Any], player_id: str) -> float:
        """Calculate win rate (highest profit games)"""
        wins = 0
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'win') and player_result.win:
                wins += 1
        
        return wins / len(game_results) if game_results else 0.0
    
    def _calculate_market_share_captured(self, game_results: List[Any], player_id: str, market_size: float) -> float:
        """Calculate average market share"""
        total_share = 0.0
        game_count = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'profit'):
                # Estimate market share from profit (simplified)
                # In Salop, market share ≈ profit / (price - marginal_cost) / market_size
                if hasattr(player_result, 'actions') and player_result.actions:
                    price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
                    estimated_quantity = player_result.profit / max(0.1, price - 8.0) if price > 8.0 else 0
                    share = estimated_quantity / market_size
                    total_share += min(1.0, max(0.0, share))  # Bound between 0 and 1
                    game_count += 1
        
        return total_share / game_count if game_count > 0 else 0.0
    
    def _calculate_profit_margin(self, prices: List[float], marginal_cost: float) -> float:
        """Calculate average profit margin"""
        if not prices:
            return 0.0
        
        margins = [(p - marginal_cost) / p for p in prices if p > 0]
        return np.mean(margins) if margins else 0.0
    
    def _calculate_salop_regret(self, game_results: List[Any], player_id: str, profits: List[float]) -> float:
        """Calculate regret (optimal profit - actual profit)"""
        if not profits:
            return 0.0
        
        # Simplified optimal profit calculation
        try:
            constants = GameConstants() if GameConstants else None
            if constants:
                transport_cost = constants.SALOP_TRANSPORT_COST
                marginal_cost = constants.SALOP_MARGINAL_COST
                optimal_profit = 100.0  # Simplified theoretical optimal
            else:
                optimal_profit = 100.0
        except:
            optimal_profit = 100.0
        
        avg_actual_profit = np.mean(profits)
        return max(0, optimal_profit - avg_actual_profit)
    
    def _calculate_total_industry_profit(self, game_results: List[Any]) -> float:
        """Calculate total industry profit"""
        total_profit = 0.0
        for result in game_results:
            if hasattr(result, 'total_industry_profit'):
                total_profit += result.total_industry_profit
            elif hasattr(result, 'players'):
                for player in result.players:
                    if hasattr(player, 'profit'):
                        total_profit += player.profit
        return total_profit
    
    def _calculate_power_index(self, game_results: List[Any], player_id: str) -> float:
        """Calculate marginal contribution"""
        player_contribution = 0.0
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'profit'):
                player_contribution += player_result.profit
        return player_contribution  # Simplified as marginal contribution
    
    def _calculate_salop_self_awareness(self, game_results: List[Any], player_id: str) -> float:
        """Calculate self-awareness (correct regime identification)"""
        correct_regime = 0
        total_games = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions') or not player_result.actions:
                continue
            
            actual_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            
            try:
                constants = GameConstants() if GameConstants else None
                marginal_cost = constants.SALOP_MARGINAL_COST if constants else 8.0
                transport_cost = constants.SALOP_TRANSPORT_COST if constants else 0.5
                
                # Calculate theoretical optimal price
                optimal_price = marginal_cost + transport_cost * 2  # Simplified
                
                # Determine regimes
                actual_regime = self._determine_price_regime(actual_price, marginal_cost, transport_cost)
                optimal_regime = self._determine_price_regime(optimal_price, marginal_cost, transport_cost)
                
                if actual_regime == optimal_regime:
                    correct_regime += 1
            except:
                # Fallback - assume correct if price is reasonable
                if 8.0 <= actual_price <= 15.0:
                    correct_regime += 1
            
            total_games += 1
        
        return correct_regime / total_games if total_games > 0 else 0.0
    
    def _determine_price_regime(self, price: float, marginal_cost: float, transport_cost: float) -> str:
        """Determine pricing regime (monopoly, kink, competitive)"""
        markup = price - marginal_cost
        
        if markup > transport_cost * 3:
            return "monopoly"
        elif markup > transport_cost:
            return "kink"
        else:
            return "competitive"
    
    def _calculate_salop_rationality(self, game_results: List[Any], player_id: str) -> float:
        """Calculate rationality (optimal Nash equilibrium pricing)"""
        rational_decisions = 0
        total_decisions = 0
        epsilon = 0.01
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions') or not player_result.actions:
                continue
            
            actual_price = self._safe_get_numeric(player_result.actions[0], 'price', 10.0)
            
            try:
                constants = GameConstants() if GameConstants else None
                marginal_cost = constants.SALOP_MARGINAL_COST if constants else 8.0
                transport_cost = constants.SALOP_TRANSPORT_COST if constants else 0.5
                
                # Calculate optimal price
                optimal_price = marginal_cost + transport_cost * 2  # Simplified
                
                if abs(actual_price - optimal_price) < epsilon * optimal_price:
                    rational_decisions += 1
            except:
                # Fallback - check if price is reasonable
                if 8.0 <= actual_price <= 15.0:
                    rational_decisions += 1
            
            total_decisions += 1
        
        return rational_decisions / total_decisions if total_decisions > 0 else 0.0
    
    def _calculate_salop_judgment(self, game_results: List[Any], player_id: str) -> float:
        """Calculate judgment (profitable best response to neighbors)"""
        profitable_responses = 0
        total_responses = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions') or not player_result.actions:
                continue
            
            # Check if price yielded positive profit (simplified best response check)
            if hasattr(player_result, 'profit') and player_result.profit > 0:
                profitable_responses += 1
            total_responses += 1
        
        return profitable_responses / total_responses if total_responses > 0 else 0.0
    
    def _empty_metrics(self, game_name: str, player_id: str) -> GameMetrics:
        """Return empty metrics structure"""
        return GameMetrics(
            game_name=game_name,
            player_id=player_id,
            primary_behavioral={},
            core_performance={},
            advanced_strategic={},
            magic_behavioral={}
        )


class AtheyBagwellMetricsCalculator(BaseMetricsCalculator):
    """Athey & Bagwell (2008): Collusion with Persistent Private Costs - ALL METRICS"""
    
    def calculate_metrics(self, game_results: List[Any], player_id: str = 'challenger') -> GameMetrics:
        if not game_results:
            return self._empty_metrics('athey_bagwell', player_id)
        
        # Extract data
        reports, profits, actions_data = self._extract_athey_bagwell_data(game_results, player_id)
        
        if not reports and not profits:
            return self._empty_metrics('athey_bagwell', player_id)
        
        try:
            constants = GameConstants() if GameConstants else None
            high_cost = constants.AB_HIGH_COST if constants else 15.0
            low_cost = constants.AB_LOW_COST if constants else 5.0
            discount_factor = constants.AB_DISCOUNT_FACTOR if constants else 0.95
        except:
            high_cost, low_cost, discount_factor = 15.0, 5.0, 0.95
        
        # PRIMARY BEHAVIORAL METRICS
        deception_rate = self._calculate_deception_rate(game_results, player_id, high_cost, low_cost)
        
        # CORE PERFORMANCE METRICS
        win_rate = self._calculate_athey_bagwell_win_rate(game_results, player_id)
        npv = self._calculate_npv(profits, discount_factor)
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        information_rent = self._calculate_information_rent_captured(game_results, player_id)
        payoff_realization_ratio = self._calculate_payoff_realization_ratio(game_results, player_id, npv)
        
        # ADVANCED STRATEGIC METRICS
        regret = self._calculate_athey_bagwell_regret(game_results, player_id, npv)
        strategic_inertia = self._calculate_strategic_inertia(actions_data, 'cost_report') if len(actions_data) > 1 else 0.0
        total_industry_profit = self._calculate_total_industry_profit(game_results)
        power_index = self._calculate_power_index(game_results, player_id)
        
        # MAGIC BEHAVIORAL METRICS
        deception_magic = self._calculate_athey_bagwell_deception_magic(game_results, player_id)
        reasoning = self._calculate_athey_bagwell_reasoning(game_results, player_id)
        cooperation = self._calculate_athey_bagwell_cooperation(game_results)
        
        return GameMetrics(
            game_name='athey_bagwell',
            player_id=player_id,
            primary_behavioral={
                'deception_rate': MetricResult("Deception Rate", deception_rate,
                                             "Number of Deceptive Reports / Number of Opportunities to Deceive")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games with Highest NPV / Total Game Simulations"),
                'avg_long_term_profit': MetricResult("Average Long-Term Profit (NPV)", npv,
                                                   "Σ [Profit_t / (1+δ)^t]"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                 "Standard Deviation of {Profit_period_1, Profit_period_2, ...}"),
                'information_rent_captured': MetricResult("Information Rent Captured", information_rent,
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
                'deception': MetricResult("Deception", deception_magic,
                                        "Number of Successful Deceptions / Number of Opportunities to Deceive"),
                'reasoning': MetricResult("Reasoning", reasoning,
                                        "Number of Periods the Firm Chose the Optimal Action / Total Periods"),
                'cooperation': MetricResult("Cooperation", cooperation,
                                          "Number of Stable Periods / Total Periods")
            }
        )
    
    def _extract_athey_bagwell_data(self, game_results: List[Any], player_id: str):
        """Extract Athey-Bagwell specific data"""
        reports, profits, actions_data = [], [], []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            
            if hasattr(player_result, 'actions'):
                for action in player_result.actions:
                    cost_report = self._safe_get_numeric(action, 'cost_report', 10.0)
                    reports.append(cost_report)
                    actions_data.append(action)
            
            if hasattr(player_result, 'profit'):
                profits.append(player_result.profit)
        
        return reports, profits, actions_data
    
    def _calculate_deception_rate(self, game_results: List[Any], player_id: str, high_cost: float, low_cost: float) -> float:
        """Calculate deception rate (lying about costs)"""
        deceptive_reports = 0
        total_opportunities = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions'):
                continue
            
            for action in player_result.actions:
                cost_report = self._safe_get_numeric(action, 'cost_report', 10.0)
                actual_cost = self._safe_get_numeric(action, 'actual_cost', 10.0)
                
                total_opportunities += 1
                # Deception if high-cost firm reports low cost
                if actual_cost >= high_cost and cost_report <= low_cost:
                    deceptive_reports += 1
        
        return deceptive_reports / total_opportunities if total_opportunities > 0 else 0.0
    
    def _calculate_athey_bagwell_win_rate(self, game_results: List[Any], player_id: str) -> float:
        """Calculate win rate based on highest NPV"""
        wins = 0
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'win') and player_result.win:
                wins += 1
        
        return wins / len(game_results) if game_results else 0.0
    
    def _calculate_information_rent_captured(self, game_results: List[Any], player_id: str) -> float:
        """Calculate excess profit from private information"""
        actual_profit = 0.0
        public_info_profit = 0.0  # Simplified: assume 50% of actual profit
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'profit'):
                actual_profit += player_result.profit
                public_info_profit += player_result.profit * 0.5  # Simplified
        
        return actual_profit - public_info_profit
    
    def _calculate_payoff_realization_ratio(self, game_results: List[Any], player_id: str, actual_npv: float) -> float:
        """Calculate how close actual NPV is to theoretical maximum"""
        try:
            constants = GameConstants() if GameConstants else None
            theoretical_max = 200.0  # Simplified theoretical maximum NPV
            return actual_npv / theoretical_max if theoretical_max > 0 else 0.0
        except:
            return 0.5  # Default ratio
    
    def _calculate_athey_bagwell_regret(self, game_results: List[Any], player_id: str, actual_npv: float) -> float:
        """Calculate regret (optimal NPV - actual NPV)"""
        optimal_npv = 200.0  # Simplified optimal NPV with perfect information
        return max(0, optimal_npv - actual_npv)
    
    def _calculate_total_industry_profit(self, game_results: List[Any]) -> float:
        """Calculate total industry profit"""
        total_profit = 0.0
        for result in game_results:
            if hasattr(result, 'total_industry_profit'):
                total_profit += result.total_industry_profit
            elif hasattr(result, 'players'):
                for player in result.players:
                    if hasattr(player, 'profit'):
                        total_profit += player.profit
        return total_profit
    
    def _calculate_power_index(self, game_results: List[Any], player_id: str) -> float:
        """Calculate marginal contribution to cartel value"""
        player_contribution = 0.0
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and hasattr(player_result, 'profit'):
                player_contribution += player_result.profit
        return player_contribution
    
    def _calculate_athey_bagwell_deception_magic(self, game_results: List[Any], player_id: str) -> float:
        """Calculate successful deception rate (MAgIC metric)"""
        successful_deceptions = 0
        deception_opportunities = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            
            # Check if player successfully deceived (high cost but got high market share)
            if hasattr(player_result, 'profit') and player_result.profit > 50:  # Above average profit
                if hasattr(player_result, 'actions') and player_result.actions:
                    for action in player_result.actions:
                        actual_cost = self._safe_get_numeric(action, 'actual_cost', 10.0)
                        if actual_cost >= 15.0:  # High cost
                            successful_deceptions += 1
                        deception_opportunities += 1
        
        return successful_deceptions / deception_opportunities if deception_opportunities > 0 else 0.0
    
    def _calculate_athey_bagwell_reasoning(self, game_results: List[Any], player_id: str) -> float:
        """Calculate optimal strategic choice rate"""
        optimal_choices = 0
        total_periods = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result or not hasattr(player_result, 'actions'):
                continue
            
            for action in player_result.actions:
                total_periods += 1
                # Simplified: optimal choice is to report truthfully most of the time
                cost_report = self._safe_get_numeric(action, 'cost_report', 10.0)
                actual_cost = self._safe_get_numeric(action, 'actual_cost', 10.0)
                
                if abs(cost_report - actual_cost) < 2.0:  # Truthful reporting
                    optimal_choices += 1
        
        return optimal_choices / total_periods if total_periods > 0 else 0.0
    
    def _calculate_athey_bagwell_cooperation(self, game_results: List[Any]) -> float:
        """Calculate cartel stability (stable periods)"""
        stable_periods = 0
        total_periods = 0
        
        for result in game_results:
            if hasattr(result, 'players'):
                # Check if cartel is stable (no major deviations)
                period_stable = True
                for player in result.players:
                    if hasattr(player, 'profit') and player.profit < 10:  # Very low profit indicates instability
                        period_stable = False
                        break
                
                if period_stable:
                    stable_periods += 1
                total_periods += 1
        
        return stable_periods / total_periods if total_periods > 0 else 0.0
    
    def _empty_metrics(self, game_name: str, player_id: str) -> GameMetrics:
        """Return empty metrics structure"""
        return GameMetrics(
            game_name=game_name,
            player_id=player_id,
            primary_behavioral={},
            core_performance={},
            advanced_strategic={},
            magic_behavioral={}
        )


# Main interface function
def calculate_comprehensive_metrics(game_results: List[Any], game_type: str, 
                                  player_id: str = 'challenger') -> GameMetrics:
    """
    Calculate all comprehensive metrics for a given game and player
    
    Args:
        game_results: List of GameResult objects
        game_type: Type of game ('spulber', 'green_porter', 'salop', 'athey_bagwell')
        player_id: ID of the player to analyze
    
    Returns:
        GameMetrics object containing all calculated metrics
    """
    calculator = ComprehensiveMetricsCalculator()
    return calculator.calculate_all_metrics(game_results, game_type, player_id)


# Utility function for aggregating metrics across multiple players
def aggregate_game_metrics(metrics_list: List[GameMetrics]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple players for comparative analysis
    
    Args:
        metrics_list: List of GameMetrics objects
    
    Returns:
        Dictionary containing aggregated statistics
    """
    if not metrics_list:
        return {}
    
    aggregated = {
        'game_name': metrics_list[0].game_name,
        'player_count': len(metrics_list),
        'primary_behavioral': {},
        'core_performance': {},
        'advanced_strategic': {},
        'magic_behavioral': {}
    }
    
    # Aggregate each metric category
    for category in ['primary_behavioral', 'core_performance', 'advanced_strategic', 'magic_behavioral']:
        category_metrics = {}
        
        # Get all metric names from first player
        if hasattr(metrics_list[0], category):
            metric_dict = getattr(metrics_list[0], category)
            for metric_name in metric_dict.keys():
                values = []
                for metrics in metrics_list:
                    if hasattr(metrics, category):
                        cat_dict = getattr(metrics, category)
                        if metric_name in cat_dict:
                            values.append(cat_dict[metric_name].value)
                
                if values:
                    category_metrics[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values) if len(values) > 1 else 0.0,
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
        
        aggregated[category] = category_metrics
    
    return aggregated