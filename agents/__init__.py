# agents/__init__.py

import logging
from typing import Dict, Any

# Import the base and specific agent classes from the other files in this directory
from .base_agent import BaseLLMAgent, AgentResponse
from .llm_agent import GeminiAgent
from .random_agent import RandomAgent

def create_agent(model_name: str, player_id: str, mock_mode: bool = False, **kwargs) -> BaseLLMAgent:
    """
    Factory function to create an appropriate agent based on the model name and mode.

    This function acts as a single point of entry for creating any agent,
    whether it's a real LLM agent, a mock agent for testing, or a simple
    baseline like the RandomAgent.

    Args:
        model_name: The name of the model or agent type to create.
        player_id: The unique identifier for this agent in a game.
        mock_mode: If True, returns a mock agent for fast, offline testing.
        **kwargs: Additional arguments to be passed to the agent's constructor.

    Returns:
        An instance of a class that inherits from BaseLLMAgent.
    """
    logger = logging.getLogger(__name__)

    # In mock mode, always return a fast, non-API-calling agent
    if mock_mode:
        # The RandomAgent can serve as a simple mock agent for testing workflows.
        logger.info(f"🎭 MOCK MODE: Creating RandomAgent for {model_name} as {player_id}")
        return RandomAgent(model_name="random_mock", player_id=player_id)

    # Logic for creating real agents based on their provider/name
    if 'gemini' in model_name.lower():
        logger.info(f"🤖 Creating real GeminiAgent for {model_name} as {player_id}")
        return GeminiAgent(model_name, player_id, **kwargs)
    
    # Add other agent types here if needed in the future (e.g., OpenAI, Anthropic)
    # elif 'gpt' in model_name.lower():
    #     raise NotImplementedError("OpenAI agents are not yet implemented.")

    else:
        # Fallback or error for unknown agent types
        raise ValueError(f"Unknown model type specified: '{model_name}'")

# Define the public API for the 'agents' package
__all__ = [
    "BaseLLMAgent",
    "AgentResponse",
    "GeminiAgent",
    "RandomAgent",
    "create_agent"
]