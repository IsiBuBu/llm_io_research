"""
Fixed Mock Agents for Testing Game Workflow and Metrics
Enhanced game detection to ensure correct response formats for each game type
"""

import asyncio
import random
import logging
from typing import Dict, Any

from agents import BaseLLMAgent, AgentResponse


class MockLLMAgent(BaseLLMAgent):
    """
    Enhanced mock agent with robust game detection for testing workflow and metric calculations.
    Returns appropriate valid responses for each game type.
    """
    
    def __init__(self, model_name: str, player_id: str):
        super().__init__(model_name, player_id)
        self.logger = logging.getLogger(f"{__name__}.MockLLMAgent")
        self.call_count = 0

    async def get_action(self, prompt: str, call_id: str) -> str:
        """Generate appropriate valid response based on game type with enhanced detection"""
        self.call_count += 1
        
        # Minimal delay to simulate response time
        await asyncio.sleep(0.01)
        
        # Enhanced game detection with multiple keywords and fallbacks
        prompt_lower = prompt.lower()
        
        # Debug: log what we're detecting
        self.logger.debug(f"[{call_id}] Mock agent detecting game type from prompt keywords")
        
        # ATHEY-BAGWELL: Cost reporting game (most specific first)
        athey_keywords = [
            'athey', 'bagwell', 'cost_report', 'report', 'cost type', 'private cost', 
            'high cost', 'low cost', 'cost information', 'cartel', 'collusion'
        ]
        if any(keyword in prompt_lower for keyword in athey_keywords):
            report = random.choice(['low', 'high'])
            self.logger.debug(f"[{call_id}] Detected Athey-Bagwell game, returning report: {report}")
            return f'{{"report": "{report}"}}'
        
        # SPULBER: Bidding game  
        spulber_keywords = [
            'spulber', 'bid', 'auction', 'winner', 'bertrand', 'lowest bid'
        ]
        if any(keyword in prompt_lower for keyword in spulber_keywords):
            bid = round(random.uniform(1.0, 2.5), 2)
            self.logger.debug(f"[{call_id}] Detected Spulber game, returning bid: {bid}")
            return f'{{"bid": {bid}}}'
        
        # GREEN-PORTER: Quantity game
        green_porter_keywords = [
            'green', 'porter', 'quantity', 'produce', 'output', 'cournot', 
            'oligopoly', 'market state', 'collusive', 'punishment'
        ]
        if any(keyword in prompt_lower for keyword in green_porter_keywords):
            quantity = round(random.uniform(2.0, 8.0), 1)
            self.logger.debug(f"[{call_id}] Detected Green-Porter game, returning quantity: {quantity}")
            return f'{{"quantity": {quantity}}}'
        
        # SALOP: Pricing game (spatial competition)
        salop_keywords = [
            'salop', 'spatial', 'circular', 'transport cost', 'location', 'market share'
        ]
        if any(keyword in prompt_lower for keyword in salop_keywords):
            price = round(random.uniform(0.5, 1.5), 2)
            self.logger.debug(f"[{call_id}] Detected Salop game, returning price: {price}")
            return f'{{"price": {price}}}'
        
        # FALLBACK: Check for general action types
        if 'price' in prompt_lower and 'bid' not in prompt_lower:
            # General pricing game
            price = round(random.uniform(0.5, 1.5), 2)
            self.logger.debug(f"[{call_id}] Detected general pricing game, returning price: {price}")
            return f'{{"price": {price}}}'
        elif 'quantity' in prompt_lower:
            # General quantity game
            quantity = round(random.uniform(2.0, 8.0), 1)
            self.logger.debug(f"[{call_id}] Detected general quantity game, returning quantity: {quantity}")
            return f'{{"quantity": {quantity}}}'
        elif 'bid' in prompt_lower:
            # General bidding game
            bid = round(random.uniform(1.0, 2.5), 2)
            self.logger.debug(f"[{call_id}] Detected general bidding game, returning bid: {bid}")
            return f'{{"bid": {bid}}}'
        
        # ULTIMATE FALLBACK: Look at call_id for game name
        if 'athey_bagwell' in call_id:
            report = random.choice(['low', 'high'])
            self.logger.debug(f"[{call_id}] Detected Athey-Bagwell from call_id, returning report: {report}")
            return f'{{"report": "{report}"}}'
        elif 'spulber' in call_id:
            bid = round(random.uniform(1.0, 2.5), 2)
            self.logger.debug(f"[{call_id}] Detected Spulber from call_id, returning bid: {bid}")
            return f'{{"bid": {bid}}}'
        elif 'green_porter' in call_id:
            quantity = round(random.uniform(2.0, 8.0), 1)
            self.logger.debug(f"[{call_id}] Detected Green-Porter from call_id, returning quantity: {quantity}")
            return f'{{"quantity": {quantity}}}'
        elif 'salop' in call_id:
            price = round(random.uniform(0.5, 1.5), 2)
            self.logger.debug(f"[{call_id}] Detected Salop from call_id, returning price: {price}")
            return f'{{"price": {price}}}'
        
        # FINAL FALLBACK: Default to price
        price = round(random.uniform(0.5, 1.5), 2)
        self.logger.warning(f"[{call_id}] Could not detect game type, using default price: {price}")
        self.logger.warning(f"[{call_id}] Prompt preview: {prompt_lower[:200]}...")
        return f'{{"price": {price}}}'

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
            self.logger.error(f"[{call_id}] Mock agent failed: {e}")
            return AgentResponse(
                content="",
                model=self.model_name,
                success=False,
                error=str(e),
                tokens_used=0,
                thinking_tokens=0,
                response_time=0.0
            )