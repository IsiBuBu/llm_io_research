# agents/__init__.py

import logging
from typing import Dict, Any

from .base_agent import BaseLLMAgent, AgentResponse
from .experiment_agent import ExperimentAgent
from .judge_agent import JudgeAgent
from .random_agent import RandomAgent

def create_agent(model_name: str, player_id: str, agent_type: str = 'experiment', mock_mode: bool = False, **kwargs) -> BaseLLMAgent:
    """
    Factory function to create an appropriate agent based on the model name and type.
    """
    logger = logging.getLogger(__name__)

    if mock_mode:
        logger.info(f"🎭 MOCK MODE: Creating RandomAgent for {model_name} as {player_id}")
        return RandomAgent(model_name="random_mock", player_id=player_id)
        
    if agent_type == 'judge':
        logger.info(f"⚖️ Creating JudgeAgent for {model_name}")
        return JudgeAgent(model_name, player_id, **kwargs)

    # Default to experiment agent
    if model_name == 'random_agent':
        logger.info(f"🎲 Creating RandomAgent baseline for {model_name} as {player_id}")
        return RandomAgent(model_name=model_name, player_id=player_id)
    
    elif 'gemini' in model_name.lower():
        logger.info(f"🤖 Creating ExperimentAgent for {model_name} as {player_id}")
        return ExperimentAgent(model_name, player_id, **kwargs)
    
    else:
        raise ValueError(f"Unknown model or agent type specified: '{model_name}'")

__all__ = [
    "BaseLLMAgent",
    "AgentResponse",
    "ExperimentAgent",
    "JudgeAgent",
    "RandomAgent",
    "create_agent"
]