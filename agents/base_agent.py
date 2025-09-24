# agents/base_agent.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Import GameConfig for type hinting
from config.config import GameConfig

@dataclass
class AgentResponse:
    """
    A standardized container for the response from any agent.
    This includes the raw content, success status, and performance metadata.
    """
    content: str
    model: str
    success: bool
    error: Optional[str] = None
    tokens_used: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    response_time: float = 0.0

class BaseLLMAgent(ABC):
    """
    An abstract base class that defines the common interface for all agents
    in the simulation, whether they are LLM-based or baseline models like RandomAgent.
    """

    def __init__(self, model_name: str, player_id: str):
        """
        Initializes the agent with a model name and a player ID.

        Args:
            model_name: The identifier for the model (e.g., "gemini-2.5-flash").
            player_id: The unique identifier for the player in the game (e.g., "challenger").
        """
        self.model_name = model_name
        self.player_id = player_id
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{player_id}")

    @abstractmethod
    async def get_response(self, prompt: str, call_id: str, game_config: GameConfig, seed: Optional[int] = None) -> AgentResponse:
        """
        The main method for an agent to generate a response. It takes a prompt and
        returns a structured AgentResponse object.

        Args:
            prompt: The full text of the prompt to be sent to the agent.
            call_id: A unique identifier for the API call, used for logging.
            game_config: The complete configuration object for the current game,
                         which can be used by baseline agents to determine their action.
            seed: An optional random seed for reproducibility.

        Returns:
            An AgentResponse object containing the agent's action and metadata.
        """
        pass