# agents/base_agent.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# --- Standardized Agent Response Structure ---

@dataclass
class AgentResponse:
    """
    A standardized data structure for returning the output of any agent.
    This ensures that real, mock, and baseline agents all provide results
    in a consistent format for the simulation engine and logging.
    """
    content: str
    model: str
    success: bool
    error: Optional[str] = None
    thoughts: Optional[str] = None
    tokens_used: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    response_time: float = 0.0
    temperature: float = 0.0
    max_tokens: int = 0

# --- Abstract Base Class for All Agents ---

class BaseLLMAgent(ABC):
    """
    Abstract base class defining the essential interface for any agent in the
    game theory experiments. All agents, including LLM-based, mock, and
    programmatic baselines, must inherit from this class.
    """

    def __init__(self, model_name: str, player_id: str):
        """Initializes the agent with its model name and player identifier."""
        self.model_name = model_name
        self.player_id = player_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def get_action(self, prompt: str, call_id: str) -> str:
        """
        The core logic of the agent, returning a raw string response.
        Subclasses must implement this method.
        """
        pass

    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """
        A wrapper method that calls get_action and packages the result
        into a standardized AgentResponse object, including error handling.
        """
        import time
        start_time = time.time()
        try:
            content = await self.get_action(prompt, call_id)
            return AgentResponse(
                content=content,
                model=self.model_name,
                success=True,
                response_time=time.time() - start_time
            )
        except Exception as e:
            self.logger.error(f"[{call_id}] Agent '{self.player_id}' failed to get response: {e}")
            return AgentResponse(
                content="",
                model=self.model_name,
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )