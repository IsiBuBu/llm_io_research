# games/__init__.py

from .salop import SalopGame
from .green_porter import GreenPorterGame
from .spulber import SpulberGame
from .athey_bagwell import AtheyBagwellGame

# Game registry for easy, centralized instantiation
GAMES = {
    'salop': SalopGame,
    'green_porter': GreenPorterGame,
    'spulber': SpulberGame,
    'athey_bagwell': AtheyBagwellGame
}

def create_game(game_name: str):
    """
    Factory function to create a game engine instance by name.
    
    Args:
        game_name: The name of the game to create (e.g., 'salop').

    Returns:
        An instance of the corresponding game class.
        
    Raises:
        ValueError: If the game_name is not found in the registry.
    """
    game_class = GAMES.get(game_name)
    if not game_class:
        raise ValueError(f"Unknown game: '{game_name}'. Available games are: {list(GAMES.keys())}")
    return game_class()

def get_available_games() -> list:
    """Returns a list of all available game names."""
    return list(GAMES.keys())

# Make classes and functions available at the package level
__all__ = [
    'SalopGame',
    'GreenPorterGame',
    'SpulberGame',
    'AtheyBagwellGame',
    'create_game',
    'get_available_games',
    'GAMES'
]