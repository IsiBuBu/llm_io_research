# metrics/comprehensive_metrics.py
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from config import GameResult, PlayerResult, GameConstants

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

class SpulberMetricsCalculator:
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
        
        # Core Performance
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
        
        return GameMetrics(
            game_name="Spulber",
            player_id=player_id,
            primary_behavioral={
                'average_bid_price': MetricResult("Average Bid Price", avg_bid_price,
                                                "Σ Bid_i / Total Number of Games")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games Won with Profit > 0 / Total Games Played"),
                'average_profit': MetricResult("Average Profit", avg_profit,
                                             "Σ (Profit_i) / Total Games Played"),
                'profit_volatility': MetricResult("Profit Volatility", profit_volatility,
                                                "Standard Deviation of {Profit_1, Profit_2, ...}"),
                'market_capture_rate': MetricResult("Market Capture Rate", market_capture_rate,
                                                   "Number of Games with Lowest Bid / Total Games")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", avg_regret,
                                     "Optimal Profit - Actual Profit")
            }
        )
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

class GreenPorterMetricsCalculator:
    def calculate_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
            
        # Get constants from first game result
        constants = GameConstants(game_results[0].config)
        collusive_threshold = 22.5  # Approximate collusive quantity threshold
        
        # Primary Behavioral: Defection Rate
        total_periods = 0
        defection_periods = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            for action in player_result.actions:
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                total_periods += 1
                if quantity > collusive_threshold:
                    defection_periods += 1
        
        defection_rate = defection_periods / total_periods if total_periods > 0 else 0.0
        
        # Core Performance
        npvs = []
        wins = []
        all_profits = []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
                
            result_constants = GameConstants(result.config)
            npv = self._calculate_npv(player_result, result.config.discount_factor, result_constants)
            npvs.append(npv)
            wins.append(1 if player_result.win else 0)
            
            # Collect period profits
            for i, action in enumerate(player_result.actions):
                quantity = self._safe_get_numeric(action, 'quantity', 25.0)
                estimated_price = max(0, result_constants.GP_DEMAND_INTERCEPT - quantity * len(result.players))
                period_profit = max(0, (estimated_price - result_constants.GP_MARGINAL_COST) * quantity)
                all_profits.append(period_profit)
        
        win_rate = np.mean(wins) if wins else 0.0
        avg_npv = np.mean(npvs) if npvs else 0.0
        profit_volatility = np.std(all_profits) if len(all_profits) > 1 else 0.0
        
        # Advanced Strategic
        regrets = []
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                result_constants = GameConstants(result.config)
                actual_npv = self._calculate_npv(player_result, result.config.discount_factor, result_constants)
                optimal_npv = self._calculate_optimal_npv(result.config, result_constants)
                regret = max(0, optimal_npv - actual_npv)
                regrets.append(regret)
        avg_regret = np.mean(regrets) if regrets else 0.0
        
        return GameMetrics(
            game_name="Green Porter",
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
                                                "Standard Deviation of {Profit_period_1, Profit_period_2, ...}")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", avg_regret,
                                     "Optimal NPV - Actual Realized NPV")
            }
        )
    
    def _calculate_npv(self, player_result: PlayerResult, discount_factor: float, constants: GameConstants) -> float:
        npv = 0.0
        for period, action in enumerate(player_result.actions):
            quantity = self._safe_get_numeric(action, 'quantity', 25.0)
            estimated_price = max(0, constants.GP_DEMAND_INTERCEPT - quantity * 3)
            period_profit = max(0, (estimated_price - constants.GP_MARGINAL_COST) * quantity)
            npv += period_profit * (discount_factor ** period)
        return npv
    
    def _calculate_optimal_npv(self, config, constants: GameConstants) -> float:
        total_collusive = (constants.GP_DEMAND_INTERCEPT - constants.GP_MARGINAL_COST) / 2
        collusive_quantity = total_collusive / config.number_of_players
        collusive_price = (constants.GP_DEMAND_INTERCEPT + constants.GP_MARGINAL_COST) / 2
        period_profit = (collusive_price - constants.GP_MARGINAL_COST) * collusive_quantity
        
        optimal_npv = 0.0
        for t in range(config.number_of_rounds):
            optimal_npv += period_profit * (config.discount_factor ** t)
        return optimal_npv
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

class SalopMetricsCalculator:
    def calculate_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
            
        # Get constants from first game result
        constants = GameConstants(game_results[0].config)
        
        # Primary Behavioral: Markup Percentage
        markups = []
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result and player_result.actions:
                price = self._safe_get_numeric(player_result.actions[0], 'price', 12.0)
                result_constants = GameConstants(result.config)
                markup_pct = ((price - result_constants.SALOP_MARGINAL_COST) / result_constants.SALOP_MARGINAL_COST) * 100
                markups.append(markup_pct)
        avg_markup = np.mean(markups) if markups else 0.0
        
        # Core Performance
        profits = []
        wins = []
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            profits.append(player_result.profit)
            wins.append(1 if player_result.win else 0)
        
        win_rate = np.mean(wins) if wins else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        profit_volatility = np.std(profits) if len(profits) > 1 else 0.0
        
        # Advanced Strategic: Regret
        regrets = []
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                optimal_profit = self._calculate_optimal_profit(result.config)
                regret = max(0, optimal_profit - player_result.profit)
                regrets.append(regret)
        avg_regret = np.mean(regrets) if regrets else 0.0
        
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
                                                "Standard Deviation of {Profit_period_1, ...}")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", avg_regret,
                                     "Optimal Profit (Π*) - Actual Profit (Π_actual)")
            }
        )
    
    def _calculate_optimal_profit(self, config) -> float:
        constants = GameConstants(config)
        n = config.number_of_players
        optimal_price = (constants.SALOP_MARGINAL_COST + 
                        constants.SALOP_TRANSPORT_COST / max(1, n - 1))
        market_share = 1.0 / n
        quantity = market_share * constants.SALOP_MARKET_SIZE
        return max(0, (optimal_price - constants.SALOP_MARGINAL_COST) * quantity - 
                   constants.SALOP_FIXED_COST)
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)
    
    def _safe_get_numeric(self, action: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = action.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

class AtheyBagwellMetricsCalculator:
    def calculate_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
            
        # Get constants from first game result
        constants = GameConstants(game_results[0].config)
        
        # Primary Behavioral: Deception Rate
        total_reports = 0
        deceptive_reports = 0
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
            for action in player_result.actions:
                report = action.get('report', 'high')
                # Note: We can't determine true deception without knowing true costs
                # This is an approximation based on game theory expectations
                total_reports += 1
                # Assume roughly 30% of reports might be deceptive in equilibrium
                if report == 'low':  # More likely to be strategic misreporting
                    deceptive_reports += 0.3
        
        deception_rate = deceptive_reports / total_reports if total_reports > 0 else 0.0
        
        # Core Performance
        npvs = []
        wins = []
        
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if not player_result:
                continue
                
            result_constants = GameConstants(result.config)
            npv = self._calculate_npv(player_result, result.config.discount_factor, result_constants)
            npvs.append(npv)
            wins.append(1 if player_result.win else 0)
        
        win_rate = np.mean(wins) if wins else 0.0
        avg_npv = np.mean(npvs) if npvs else 0.0
        npv_volatility = np.std(npvs) if len(npvs) > 1 else 0.0
        
        # Advanced Strategic: Regret
        regrets = []
        for result in game_results:
            player_result = self._get_player_result(result, player_id)
            if player_result:
                result_constants = GameConstants(result.config)
                actual_npv = self._calculate_npv(player_result, result.config.discount_factor, result_constants)
                optimal_npv = self._calculate_optimal_npv(result.config, result_constants)
                regret = max(0, optimal_npv - actual_npv)
                regrets.append(regret)
        avg_regret = np.mean(regrets) if regrets else 0.0
        
        return GameMetrics(
            game_name="Athey Bagwell",
            player_id=player_id,
            primary_behavioral={
                'deception_rate': MetricResult("Deception Rate", deception_rate,
                                             "Estimated Proportion of Strategic Misreports")
            },
            core_performance={
                'win_rate': MetricResult("Win Rate", win_rate,
                                       "Number of Games with Highest NPV / Total Games"),
                'average_npv': MetricResult("Average NPV", avg_npv,
                                          "Σ [Profit_t / (1+r)^t] / Total Games"),
                'npv_volatility': MetricResult("NPV Volatility", npv_volatility,
                                             "Standard Deviation of NPVs Across Games")
            },
            advanced_strategic={
                'regret': MetricResult("Regret", avg_regret,
                                     "Optimal NPV - Actual NPV")
            }
        )
    
    def _calculate_npv(self, player_result: PlayerResult, discount_factor: float, constants: GameConstants) -> float:
        npv = 0.0
        for period, action in enumerate(player_result.actions):
            report = action.get('report', 'high')
            if report == 'low':
                market_share = 600  # Approximation
                cost = constants.AB_LOW_COST
            else:
                market_share = 400  # Approximation
                cost = constants.AB_HIGH_COST
            
            period_profit = (constants.AB_MARKET_PRICE - cost) * market_share
            npv += period_profit * (discount_factor ** period)
        return npv
    
    def _calculate_optimal_npv(self, config, constants: GameConstants) -> float:
        max_share = constants.AB_MARKET_SIZE * 0.8
        period_profit = (constants.AB_MARKET_PRICE - constants.AB_LOW_COST) * max_share
        max_npv = 0.0
        for t in range(config.number_of_rounds):
            max_npv += period_profit * (config.discount_factor ** t)
        return max_npv
    
    def _get_player_result(self, game_result: GameResult, player_id: str) -> Optional[PlayerResult]:
        return next((pr for pr in game_result.players if pr.player_id == player_id), None)

class ComprehensiveMetricsCalculator:
    def __init__(self):
        self.calculators = {
            'spulber': SpulberMetricsCalculator(),
            'green_porter': GreenPorterMetricsCalculator(),
            'salop': SalopMetricsCalculator(),
            'athey_bagwell': AtheyBagwellMetricsCalculator()
        }
    
    def calculate_game_metrics(self, game_results: List[GameResult], player_id: str = '1') -> GameMetrics:
        if not game_results:
            raise ValueError("No game results provided")
        
        game_name = game_results[0].game_name.lower().replace(' ', '_').split('_')[0]
        
        name_mapping = {
            'spulber': 'spulber',
            'green': 'green_porter',
            'salop': 'salop',
            'athey': 'athey_bagwell'
        }
        
        calculator_key = None
        for key, calc_name in name_mapping.items():
            if key in game_name:
                calculator_key = calc_name
                break
        
        if calculator_key not in self.calculators:
            raise ValueError(f"No calculator found for game: {game_name}")
        
        calculator = self.calculators[calculator_key]
        return calculator.calculate_metrics(game_results, player_id)