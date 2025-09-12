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
from .dynamic_game_metrics import DynamicGameMetricsCalculator

# A list defining the public API of the 'metrics' package.
# When a user does 'from metrics import *', only these names will be imported.
__all__ = [
    'MetricResult',
    'GameResult',
    'PlayerMetrics',
    'ExperimentResults',
    'MetricStorage',
    'PerformanceMetricsCalculator',
    'MAgICMetricsCalculator',
    'DynamicGameMetricsCalculator',
    'create_game_result',
    'create_metric_result'
]