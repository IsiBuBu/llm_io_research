"""
Metrics Utilities - Shared data structures and utilities for comprehensive metrics analysis
Supports both performance metrics and MAgIC behavioral metrics from t.txt
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class MetricResult:
    """Container for individual metric calculation results"""
    name: str
    value: float
    description: str
    metric_type: str  # 'performance', 'magic_behavioral'
    game_name: str
    experiment_type: str
    condition_name: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'value': self.value,
            'description': self.description,
            'metric_type': self.metric_type,
            'game_name': self.game_name,
            'experiment_type': self.experiment_type,
            'condition_name': self.condition_name,
            'raw_data': self.raw_data
        }


@dataclass
class GameResult:
    """Container for single game simulation result"""
    simulation_id: int
    game_name: str
    experiment_type: str
    condition_name: str
    players: List[str]
    actions: Dict[str, Any]
    payoffs: Dict[str, float]
    game_data: Dict[str, Any]
    round_data: List[Dict[str, Any]] = field(default_factory=list)  # For dynamic games
    
    def get_challenger_data(self, challenger_id: str = 'challenger') -> Dict[str, Any]:
        """Extract challenger-specific data for metrics calculation"""
        return {
            'simulation_id': self.simulation_id,
            'actions': self.actions.get(challenger_id, {}),
            'payoff': self.payoffs.get(challenger_id, 0.0),
            'game_data': self.game_data,
            'round_data': self.round_data
        }


@dataclass
class PlayerMetrics:
    """Container for all metrics calculated for a single player"""
    player_id: str
    game_name: str
    experiment_type: str
    condition_name: str
    
    # Performance metrics
    performance_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    
    # MAgIC behavioral metrics  
    magic_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    
    # Raw simulation data
    simulations_data: List[GameResult] = field(default_factory=list)
    
    def add_metric(self, metric: MetricResult):
        """Add metric to appropriate category"""
        if metric.metric_type == 'performance':
            self.performance_metrics[metric.name] = metric
        elif metric.metric_type == 'magic_behavioral':
            self.magic_metrics[metric.name] = metric
    
    def get_all_metrics(self) -> Dict[str, MetricResult]:
        """Get all metrics as single dictionary"""
        all_metrics = {}
        all_metrics.update(self.performance_metrics)
        all_metrics.update(self.magic_metrics)
        return all_metrics


@dataclass
class ExperimentResults:
    """Container for complete experiment results across all conditions"""
    challenger_models: List[str]
    defender_model: str
    game_name: str
    
    # Results by challenger model and condition
    results: Dict[str, Dict[str, PlayerMetrics]] = field(default_factory=dict)
    
    def add_player_metrics(self, challenger_model: str, condition: str, metrics: PlayerMetrics):
        """Add metrics for specific challenger and condition"""
        if challenger_model not in self.results:
            self.results[challenger_model] = {}
        self.results[challenger_model][condition] = metrics
    
    def get_challenger_metrics(self, challenger_model: str) -> Dict[str, PlayerMetrics]:
        """Get all condition results for specific challenger"""
        return self.results.get(challenger_model, {})


class MetricCalculator:
    """Base class for metric calculation with shared utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value"""
        if denominator == 0:
            return default
        return numerator / denominator
    
    def safe_mean(self, values: List[float]) -> float:
        """Safe mean calculation"""
        if not values:
            return 0.0
        return np.mean(values)
    
    def safe_std(self, values: List[float]) -> float:
        """Safe standard deviation calculation"""
        if len(values) < 2:
            return 0.0
        return np.std(values, ddof=1)
    
    def indicator_function(self, condition: bool) -> int:
        """Mathematical indicator function I(condition)"""
        return 1 if condition else 0
    
    def calculate_npv(self, profit_stream: List[float], discount_factor: float) -> float:
        """Calculate Net Present Value from profit stream"""
        npv = 0.0
        for t, profit in enumerate(profit_stream):
            npv += (discount_factor ** t) * profit
        return npv
    
    def extract_challenger_data(self, game_results: List[GameResult], 
                              challenger_id: str = 'challenger') -> List[Dict[str, Any]]:
        """Extract challenger data from game results"""
        challenger_data = []
        for result in game_results:
            challenger_data.append(result.get_challenger_data(challenger_id))
        return challenger_data


class DataValidator:
    """Utilities for validating game data and metrics inputs"""
    
    @staticmethod
    def validate_game_result(result: GameResult) -> Tuple[bool, List[str]]:
        """Validate game result data structure"""
        errors = []
        
        # Check required fields
        if not result.simulation_id:
            errors.append("Missing simulation_id")
        if not result.game_name:
            errors.append("Missing game_name")
        if not result.players:
            errors.append("Missing players list")
        if not result.actions:
            errors.append("Missing actions dictionary")
        if not result.payoffs:
            errors.append("Missing payoffs dictionary")
        
        # Check data consistency
        if set(result.actions.keys()) != set(result.payoffs.keys()):
            errors.append("Mismatch between action and payoff player IDs")
        
        if set(result.players) != set(result.actions.keys()):
            errors.append("Mismatch between players list and action keys")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_metric_inputs(data: List[Any], metric_name: str) -> bool:
        """Validate inputs for metric calculation"""
        if not data:
            logging.warning(f"Empty data for metric: {metric_name}")
            return False
        
        if not isinstance(data, list):
            logging.error(f"Invalid data type for metric {metric_name}: expected list")
            return False
        
        return True


class MetricFormatter:
    """Utilities for formatting and displaying metrics"""
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 1) -> str:
        """Format value as percentage"""
        return f"{value * 100:.{decimal_places}f}%"
    
    @staticmethod
    def format_currency(value: float, decimal_places: int = 2) -> str:
        """Format value as currency"""
        return f"${value:.{decimal_places}f}"
    
    @staticmethod
    def format_metric_summary(metrics: Dict[str, MetricResult]) -> str:
        """Format metrics dictionary as readable summary"""
        lines = []
        
        # Group by metric type
        performance = {k: v for k, v in metrics.items() if v.metric_type == 'performance'}
        magic = {k: v for k, v in metrics.items() if v.metric_type == 'magic_behavioral'}
        
        if performance:
            lines.append("Performance Metrics:")
            for name, metric in performance.items():
                lines.append(f"  {name}: {metric.value:.3f}")
        
        if magic:
            lines.append("MAgIC Behavioral Metrics:")
            for name, metric in magic.items():
                lines.append(f"  {name}: {metric.value:.3f}")
        
        return "\n".join(lines)


class MetricStorage:
    """Utilities for saving and loading metrics results"""
    
    @staticmethod
    def save_player_metrics(metrics: PlayerMetrics, filepath: Union[str, Path]):
        """Save player metrics to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            'player_id': metrics.player_id,
            'game_name': metrics.game_name,
            'experiment_type': metrics.experiment_type,
            'condition_name': metrics.condition_name,
            'performance_metrics': {k: v.to_dict() for k, v in metrics.performance_metrics.items()},
            'magic_metrics': {k: v.to_dict() for k, v in metrics.magic_metrics.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_player_metrics(filepath: Union[str, Path]) -> PlayerMetrics:
        """Load player metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metrics = PlayerMetrics(
            player_id=data['player_id'],
            game_name=data['game_name'],
            experiment_type=data['experiment_type'],
            condition_name=data['condition_name']
        )
        
        # Reconstruct MetricResult objects
        for name, metric_data in data['performance_metrics'].items():
            metric = MetricResult(**metric_data)
            metrics.performance_metrics[name] = metric
        
        for name, metric_data in data['magic_metrics'].items():
            metric = MetricResult(**metric_data)
            metrics.magic_metrics[name] = metric
        
        return metrics
    
    @staticmethod
    def save_experiment_results(results: ExperimentResults, dirpath: Union[str, Path]):
        """Save complete experiment results to directory"""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'challenger_models': results.challenger_models,
            'defender_model': results.defender_model,
            'game_name': results.game_name
        }
        
        with open(dirpath / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual player metrics
        for challenger_model, conditions in results.results.items():
            model_dir = dirpath / challenger_model
            model_dir.mkdir(exist_ok=True)
            
            for condition, metrics in conditions.items():
                filename = f"{condition}_metrics.json"
                MetricStorage.save_player_metrics(metrics, model_dir / filename)


# Factory functions for common metric calculations
def create_metric_result(name: str, value: float, description: str, 
                        metric_type: str, game_name: str, experiment_type: str,
                        condition_name: str, **raw_data) -> MetricResult:
    """Factory function for creating MetricResult objects"""
    return MetricResult(
        name=name,
        value=value,
        description=description,
        metric_type=metric_type,
        game_name=game_name,
        experiment_type=experiment_type,
        condition_name=condition_name,
        raw_data=raw_data
    )


def create_game_result(simulation_id: int, game_name: str, experiment_type: str,
                      condition_name: str, players: List[str], actions: Dict[str, Any],
                      payoffs: Dict[str, float], **game_data) -> GameResult:
    """Factory function for creating GameResult objects"""
    return GameResult(
        simulation_id=simulation_id,
        game_name=game_name,
        experiment_type=experiment_type,
        condition_name=condition_name,
        players=players,
        actions=actions,
        payoffs=payoffs,
        game_data=game_data
    )