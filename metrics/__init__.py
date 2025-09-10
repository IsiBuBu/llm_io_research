# metrics/__init__.py
from .metric_utils import (
    MetricResult,
    GameResult,
    PlayerMetrics,
    ExperimentResults,
    MetricStorage,
    create_game_result,
    create_metric_result
)
from .performance_metrics import PerformanceMetricsCalculator
from .magic_metrics import MAgICMetricsCalculator

# Alias for backward compatibility
GameMetrics = PlayerMetrics
ComprehensiveMetricsCalculator = PerformanceMetricsCalculator

__all__ = [
    'MetricResult',
    'GameResult', 
    'PlayerMetrics',
    'ExperimentResults',
    'MetricStorage',
    'GameMetrics',
    'ComprehensiveMetricsCalculator',
    'PerformanceMetricsCalculator',
    'MAgICMetricsCalculator',
    'create_game_result',
    'create_metric_result'
]