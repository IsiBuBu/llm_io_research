# agents/random_agent.py

import random
import json
import logging
from typing import Dict, Any, Optional

from .base_agent import BaseLLMAgent, AgentResponse
from config.config import GameConfig

class RandomAgent(BaseLLMAgent):
    """
    A non-strategic baseline agent that selects valid actions uniformly at random.
    It uses the game_config object for deterministic game identification.
    """

    def __init__(self, model_name: str, player_id: str, seed: int = None):
        super().__init__(model_name, player_id)
        if seed is not None:
            random.seed(seed)

    def get_action(self, game_config: GameConfig) -> str:
        """
        Determines the game from the game_config and returns a random, valid action.
        """
        action = {}
        game_name = game_config.game_name
        constants = game_config.constants

        if game_name == "athey_bagwell":
            report = random.choice(["high", "low"])
            action = {"report": report}
        elif game_name == "green_porter":
            # UPDATED LOGIC: Choose with 50% probability from the two strategically relevant actions.
            chosen_action = random.choice(["Cooperate", "Defect"])
            action = {"action": chosen_action}
        elif game_name == "salop":
            # Chooses a random price between its marginal cost and the max willingness to pay
            price = random.uniform(constants.get('marginal_cost', 8), constants.get('reservation_price', 30))
            action = {"price": round(price, 2)}
        elif game_name == "spulber":
            # Chooses a random price between its own cost and the market demand intercept
            price = random.uniform(constants.get('your_cost', 8), constants.get('demand_intercept', 100))
            action = {"price": round(price, 2)}
        else:
            self.logger.warning(f"Unknown game '{game_name}' for RandomAgent. Defaulting to a generic price action.")
            action = {"price": random.uniform(10, 50)}
            
        return json.dumps(action)

    async def get_response(self, prompt: str, call_id: str, game_config: GameConfig, seed: Optional[int] = None) -> AgentResponse:
        """
        Wraps get_action to provide a standardized AgentResponse object.
        """
        import time
        start_time = time.time()
        try:
            content = self.get_action(game_config)
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
