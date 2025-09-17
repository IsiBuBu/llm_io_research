# agents/judge_agent.py

import asyncio
import logging
import time
import os
from typing import Dict, Any, Optional

from google import genai
from google.genai import types

from .base_agent import BaseLLMAgent, AgentResponse
from config.config import load_config

class JudgeAgent(BaseLLMAgent):
    """
    An agent that uses the Google Gemini API to evaluate thought summaries.
    It uses specific judge configurations, including temperature and seeding for reliability.
    """

    def __init__(self, model_name: str, player_id: str = 'judge'):
        super().__init__(model_name, player_id)
        
        self.judge_config = load_config().get('judge_config', {})
        self.api_config = load_config().get('api_config', {})
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initializes and returns the Gemini API client."""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            return genai.Client(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    async def get_response(self, prompt: str, call_id: str, seed: Optional[int] = None) -> AgentResponse:
        """Gets a structured evaluation from the Gemini model, accepting a seed."""
        start_time = time.time()
        max_retries = self.api_config.get('max_retries', 3)
        delay = self.api_config.get('rate_limit_delay', 1.0)
        
        gen_config = types.GenerateContentConfig(
            temperature=self.judge_config.get('judge_temperature', 0.1),
            response_mime_type="application/json",
            seed=seed
        )

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=gen_config
                )
                
                if not response.candidates or not response.candidates[0].content.parts:
                    return AgentResponse(content="", model=self.model_name, success=False, error="Response contained no valid parts")

                answer_text = response.candidates[0].content.parts[0].text
                return AgentResponse(
                    content=answer_text.strip(),
                    model=self.model_name,
                    success=True,
                    response_time=time.time() - start_time
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
                else:
                    return AgentResponse(content="", model=self.model_name, success=False, error=str(e))
        return AgentResponse(content="", model=self.model_name, success=False, error="Max retries exceeded")