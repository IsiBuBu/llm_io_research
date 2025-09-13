# agents/random_agent.py

import random
import json
import logging
from typing import Dict, Any

# CORRECTED: Use a relative import within the same package
from .base_agent import BaseLLMAgent, AgentResponse
from config.config import GameConfig

class RandomAgent(BaseLLMAgent):
    """
    A non-strategic baseline agent that selects valid actions uniformly at random.
    """

    def __init__(self, model_name: str, player_id: str, seed: int = None):
        super().__init__(model_name, player_id)
        self.logger = logging.getLogger(f"{__name__}.RandomAgent")
        if seed is not None:
            random.seed(seed)

    async def get_action(self, prompt: str, call_id: str) -> str:
        """
        Determines the game from the prompt and returns a random, valid action as a JSON string.
        """
        prompt_lower = prompt.lower()
        action = {}

        # CORRECTED: Use more specific and unique keywords for game detection
        if "athey & bagwell" in prompt_lower or "market shares are allocated based on reports" in prompt_lower:
            report = random.choice(["high", "low"])
            action = {"report": report}
        elif "green & porter" in prompt_lower or "drops below the `trigger_price`" in prompt_lower:
            quantity = random.choice([17, 25])
            action = {"quantity": quantity}
        elif "salop" in prompt_lower or "circular market" in prompt_lower:
            price = random.uniform(8, 30)
            action = {"price": round(price, 2)}
        elif "spulber" in prompt_lower or "winner-take-all price auction" in prompt_lower:
            price = random.uniform(8, 100)
            action = {"price": round(price, 2)}
        else:
            self.logger.warning(f"[{call_id}] Could not determine game for RandomAgent. Defaulting.")
            action = {"price": random.uniform(10, 50)}
            
        return json.dumps(action)

    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """Wraps get_action to provide a standardized AgentResponse object."""
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
            self.logger.error(f"[{call_id}] RandomAgent failed: {e}")
            return AgentResponse(
                content="",
                model=self.model_name,
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )