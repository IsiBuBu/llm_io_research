"""
Minimal Mock Agents for Testing Game Workflow and Metrics
Simple, fast responses to verify game logic without API calls
"""

import asyncio
import random
import logging
from typing import Dict, Any

from agents import BaseLLMAgent, AgentResponse


class MockLLMAgent(BaseLLMAgent):
    """
    Minimal mock agent for testing game workflow and metric calculations.
    Returns basic valid responses to allow full pipeline testing.
    """
    
    def __init__(self, model_name: str, player_id: str):
        super().__init__(model_name, player_id)
        self.logger = logging.getLogger(f"{__name__}.MockLLMAgent")
        self.call_count = 0

    async def get_action(self, prompt: str, call_id: str) -> str:
        """Generate minimal valid response based on game type"""
        self.call_count += 1
        
        # Minimal delay to simulate response time
        await asyncio.sleep(0.01)
        
        # Detect game type and return basic valid response
        prompt_lower = prompt.lower()
        
        if 'salop' in prompt_lower or ('price' in prompt_lower and 'bid' not in prompt_lower):
            # Salop price response
            price = round(random.uniform(0.5, 1.5), 2)
            return f'{{"price": {price}}}'
            
        elif 'green' in prompt_lower or 'porter' in prompt_lower or 'quantity' in prompt_lower:
            # Green-Porter quantity response
            quantity = round(random.uniform(2.0, 8.0), 1)
            return f'{{"quantity": {quantity}}}'
            
        elif 'spulber' in prompt_lower or 'bid' in prompt_lower:
            # Spulber bid response
            bid = round(random.uniform(1.0, 2.5), 2)
            return f'{{"bid": {bid}}}'
            
        elif 'athey' in prompt_lower or 'bagwell' in prompt_lower or 'report' in prompt_lower:
            # Athey-Bagwell cost report
            report = random.choice(['low', 'high'])
            return f'{{"report": "{report}"}}'
            
        else:
            # Default fallback
            return f'{{"price": {round(random.uniform(0.5, 1.5), 2)}}}'

    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """Alternative interface returning AgentResponse object"""
        try:
            content = await self.get_action(prompt, call_id)
            return AgentResponse(
                content=content,
                model=self.model_name,
                success=True,
                tokens_used=50,  # Mock token count
                thinking_tokens=0,
                response_time=0.01
            )
        except Exception as e:
            return AgentResponse(
                content="",
                model=self.model_name,
                success=False,
                error=str(e),
                tokens_used=0,
                thinking_tokens=0,
                response_time=0.0
            )