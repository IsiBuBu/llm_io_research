# metrics/metric_utils.py

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# --- Core Data Structures for Metrics and Results ---

@dataclass
class MetricResult:
    """Standardized container for the result of a single metric calculation."""
    name: str
    value: float
    description: str
    metric_type: str  # e.g., 'performance', 'magic_behavioral'
    game_name: str
    experiment_type: str
    condition_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the MetricResult to a dictionary."""
        return asdict(self)

@dataclass
class GameResult:
    """
    Standardized container for the complete output of a single game simulation.
    This structure aligns with the desired JSON output format for results.
    """
    simulation_id: int
    game_name: str
    experiment_type: str
    condition_name: str
    challenger_model: str
    players: List[str]
    actions: Dict[str, Any]
    payoffs: Dict[str, float]
    game_data: Dict[str, Any] # For game-specific outcomes and logged data

@dataclass
class PlayerMetrics:
    """Aggregates all calculated metrics for a single player model under a specific condition."""
    player_id: str
    game_name: str
    experiment_type: str
    condition_name: str
    performance_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    magic_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    dynamic_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentResults:
    """Top-level container for all results from a single game's experiments."""
    game_name: str
    challenger_models: List[str]
    defender_model: str
    # Nested dict structure: {challenger_model: {condition_name: PlayerMetrics}}
    results: Dict[str, Dict[str, PlayerMetrics]] = field(default_factory=dict)

# --- MISSING CLASS DEFINITION ---
@dataclass
class MetricStorage:
    """Utilities for saving and loading metrics results."""
    
    @staticmethod
    def save_player_metrics(metrics: PlayerMetrics, filepath: str):
        """Save player metrics to a JSON file."""
        # Implementation for saving metrics would go here
        pass

    @staticmethod
    def load_player_metrics(filepath: str) -> PlayerMetrics:
        """Load player metrics from a JSON file."""
        # Implementation for loading metrics would go here
        pass

# --- Base Class for Metric Calculators ---

class MetricCalculator:
    """Base class providing shared utility functions for all metric calculators."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def safe_divide(self, num: float, den: float, default: float = 0.0) -> float:
        """Safely divides, returning a default value if the denominator is zero."""
        return num / den if den != 0 else default

    def safe_mean(self, values: List[float]) -> float:
        """Safely calculates the mean, returning 0.0 for empty lists."""
        return np.mean(values) if values else 0.0
    
    def safe_std(self, values: List[float]) -> float:
        """Safely calculates standard deviation, returning 0.0 if not possible."""
        return np.std(values, ddof=1) if len(values) > 1 else 0.0

    def calculate_npv(self, profit_stream: List[float], discount_factor: float) -> float:
        """Calculates the Net Present Value of a stream of profits."""
        return sum(profit * (discount_factor ** t) for t, profit in enumerate(profit_stream))

# --- Factory Functions for Creating Data Objects ---

def create_metric_result(name: str, value: float, description: str, metric_type: str, game_name: str, experiment_type: str, condition_name: str) -> MetricResult:
    """Convenience factory for creating MetricResult objects."""
    return MetricResult(
        name=name, value=value, description=description, metric_type=metric_type,
        game_name=game_name, experiment_type=experiment_type, 
        condition_name=condition_name
    )

def create_game_result(simulation_id: int, game_name: str, experiment_type: str, condition_name: str, challenger_model: str, players: List[str], actions: Dict[str, Any], payoffs: Dict[str, float], game_data: Dict[str, Any]) -> GameResult:
    """Convenience factory for creating GameResult objects."""
    return GameResult(
        simulation_id=simulation_id, game_name=game_name, experiment_type=experiment_type,
        condition_name=condition_name, challenger_model=challenger_model, players=players,
        actions=actions, payoffs=payoffs, game_data=game_data
    )