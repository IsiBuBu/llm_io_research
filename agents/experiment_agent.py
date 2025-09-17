# agents/experiment_agent.py

import asyncio
import logging
import time
import os
from typing import Dict, Any, Optional

from google import genai
from google.genai import types

from .base_agent import BaseLLMAgent, AgentResponse
from config.config import load_config, get_model_config, get_thinking_config

class ExperimentAgent(BaseLLMAgent):
    """
    An agent that uses the Google Gemini API to make decisions in game theory experiments.
    It handles thinking budgets, temperature, and other experiment-specific configurations.
    """

    def __init__(self, model_name: str, player_id: str):
        super().__init__(model_name, player_id)
        
        self.model_config = get_model_config(model_name)
        self.thinking_config = get_thinking_config(model_name)
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

    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """Gets a structured response from the Gemini model for an experiment."""
        start_time = time.time()
        max_retries = self.api_config.get('max_retries', 3)
        delay = self.api_config.get('rate_limit_delay', 1.0)
        
        thinking_conf = None
        if self.thinking_config and self.thinking_config.get('thinking_budget', 0) > 0:
            thinking_conf = types.ThinkingConfig(
                thinking_budget=self.thinking_config['thinking_budget'],
                include_thoughts=self.thinking_config.get('include_thoughts', True)
            )

        gen_config = types.GenerateContentConfig(
            temperature=self.model_config.get('temperature', 0.7),
            thinking_config=thinking_conf,
            response_mime_type="application/json"
        )

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_config['model_name'],
                    contents=prompt,
                    config=gen_config
                )
                
                if not response.candidates or not response.candidates[0].content.parts:
                    return AgentResponse(content="", model=self.model_name, success=False, error="Response contained no valid parts")

                usage = response.usage_metadata
                thought_summary = ""
                answer_text = ""
                for part in response.candidates[0].content.parts:
                    if part.text:
                        if hasattr(part, 'thought') and part.thought:
                            thought_summary = part.text
                        else:
                            answer_text = part.text

                return AgentResponse(
                    content=answer_text.strip(),
                    model=self.model_name,
                    success=True,
                    thoughts=thought_summary if thought_summary else None,
                    tokens_used=usage.total_token_count if usage else 0,
                    output_tokens=usage.candidates_token_count if usage else 0,
                    thinking_tokens=getattr(usage, 'thoughts_token_count', 0),
                    response_time=time.time() - start_time
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
                else:
                    return AgentResponse(content="", model=self.model_name, success=False, error=str(e))
        return AgentResponse(content="", model=self.model_name, success=False, error="Max retries exceeded")