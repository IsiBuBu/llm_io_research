# agents/llm_agent.py

import asyncio
import logging
import time
import os
from typing import Dict, Any

# Third-party imports - CORRECTED
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Local application imports
from .base_agent import BaseLLMAgent, AgentResponse
from config.config import load_config, get_model_config, get_thinking_config

class GeminiAgent(BaseLLMAgent):
    """
    An agent that uses the Google Gemini API to make decisions. It handles API
    authentication, rate limiting, retries, and "thinking" configurations as
    specified in the main config files.
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
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model_config['model_name'])
        except ImportError:
            self.logger.error("google-generativeai is not installed. Please run 'pip install google-generativeai'.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    async def get_action(self, prompt: str, call_id: str) -> str:
        """
        Gets an action from the Gemini model with retry and timeout logic.
        This method returns only the raw string content of the response.
        """
        response = await self.get_response(prompt, call_id)
        if not response.success:
            raise RuntimeError(f"Agent action failed: {response.error}")
        return response.content

    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """
        Gets a structured response from the Gemini model, including metadata.
        """
        start_time = time.time()
        max_retries = self.api_config.get('max_retries', 3)
        delay = self.api_config.get('rate_limit_delay', 1.0)

        # CORRECTED: ThinkingConfig is now passed as a dictionary within GenerationConfig
        thinking_conf_dict = {}
        if self.thinking_config and self.thinking_config.get('thinking_budget', 0) > 0:
            thinking_conf_dict = {
                "thinking_budget": self.thinking_config['thinking_budget'],
                "include_thoughts": self.thinking_config.get('include_thoughts', True)
            }

        gen_config = GenerationConfig(
            temperature=self.model_config.get('temperature', 0.0),
            max_output_tokens=self.model_config.get('max_tokens', 1024),
            **thinking_conf_dict # Unpack the thinking config here
        )

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.generate_content,
                    contents=prompt,
                    generation_config=gen_config
                )
                
                usage = response.usage_metadata
                return AgentResponse(
                    content=response.text,
                    model=self.model_name,
                    success=True,
                    tokens_used=usage.total_token_count,
                    thinking_tokens=getattr(usage, 'thoughts_token_count', 0),
                    response_time=time.time() - start_time
                )
            except Exception as e:
                self.logger.warning(f"[{call_id}] Attempt {attempt + 1}/{max_retries} for {self.player_id} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))
                else:
                    return AgentResponse(
                        content="",
                        model=self.model_name,
                        success=False,
                        error=str(e),
                        response_time=time.time() - start_time
                    )
        return AgentResponse(content="", model=self.model_name, success=False, error="Max retries exceeded")