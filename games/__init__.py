"""
Minimal game initialization for LLM game theory experiments
"""

from games.salop_game import SalopGame
from games.spulber_game import SpulberGame
from games.green_porter_game import GreenPorterGame
from games.athey_bagwell_game import AtheyBagwellGame

# Game registry for easy instantiation
GAMES = {
    'salop': SalopGame,
    'spulber': SpulberGame, 
    'green_porter': GreenPorterGame,
    'athey_bagwell': AtheyBagwellGame
}

def create_game(game_name: str):
    """Create game instance by name"""
    if game_name not in GAMES:
        raise ValueError(f"Unknown game: {game_name}. Available: {list(GAMES.keys())}")
    return GAMES[game_name]()

def get_available_games():
    """Get list of available games"""
    return list(GAMES.keys())

# Make classes available at package level
__all__ = [
    'SalopGame',
    'SpulberGame', 
    'GreenPorterGame',
    'AtheyBagwellGame',
    'create_game',
    'get_available_games',
    'GAMES'
]