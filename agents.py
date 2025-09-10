"""
Gemini Agents - Google AI integration for LLM game theory experiments
Handles thinking configurations, authentication, retries, and rate limiting
"""

import asyncio
import logging
import time
import concurrent.futures
from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from config import load_config_file, get_model_config, get_thinking_config


@dataclass
class AgentResponse:
    """Standardized response from any LLM agent"""
    content: str
    model: str
    success: bool
    error: Optional[str] = None
    tokens_used: int = 0
    thinking_tokens: int = 0
    response_time: float = 0.0


class BaseLLMAgent(ABC):
    """Base class for all LLM agents"""
    
    def __init__(self, model_name: str, player_id: str):
        self.model_name = model_name
        self.player_id = player_id
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        
        # Load model config
        config = load_config_file()
        self.model_config = get_model_config(model_name)
        self.api_config = config.get('api', {})
        
    @abstractmethod
    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """Get response from LLM API"""
        pass
    
    async def get_action(self, prompt: str, call_id: str) -> str:
        """Get action with standardized interface for competition.py"""
        response = await self.get_response(prompt, call_id)
        return response.content if response.success else f'{{"error": "{response.error}"}}'


class GeminiAgent(BaseLLMAgent):
    """Google Gemini API agent with thinking support"""
    
    def __init__(self, model_name: str, player_id: str):
        super().__init__(model_name, player_id)
        self.client = None
        self.actual_model_name = self.model_config.get('model_name', model_name)
        self.thinking_config = get_thinking_config(model_name)
        self._setup_client()
    
    def _setup_client(self):
        """Setup Google Gemini client"""
        try:
            # Import the newer Google Genai SDK
            from google import genai
            
            # Get API key from environment
            import os
            api_key_env = self.api_config.get('google', {}).get('api_key_env', 'GEMINI_API_KEY')
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} environment variable not set")
            
            # Create client with the newer SDK
            self.client = genai.Client(api_key=api_key)
            
            # Log setup info
            thinking_info = ""
            if self.thinking_config:
                budget = self.thinking_config.get('thinking_budget', 0)
                if budget == -1:
                    thinking_info = " (Dynamic Thinking)"
                elif budget > 0:
                    thinking_info = f" (Thinking Budget: {budget})"
                else:
                    thinking_info = " (Thinking Off)"
            else:
                thinking_info = " (No Thinking)"
            
            self.logger.info(f"Initialized {self.actual_model_name}{thinking_info}")
            
        except ImportError as e:
            self.logger.error(f"Google AI package not installed. Error: {e}")
            self.logger.error("Run: pip install google-genai")
            raise
        except Exception as e:
            self.logger.error(f"Failed to setup Gemini client: {e}")
            raise
    
    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """Get response from Gemini API with thinking support"""
        start_time = time.time()
        max_retries = self.api_config.get('google', {}).get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                # Create generation function based on thinking availability
                def sync_generate():
                    try:
                        from google.genai import types
                        
                        # Check if thinking is enabled and available
                        if (self.thinking_config and 
                            self.model_config.get('thinking_available', False) and
                            self.thinking_config.get('thinking_budget', 0) != 0):
                            
                            # Use thinking-enabled generation
                            thinking_budget = self.thinking_config.get('thinking_budget', -1)
                            include_thoughts = self.thinking_config.get('include_thoughts', False)
                            
                            self.logger.debug(f"[{call_id}] Using thinking mode: budget={thinking_budget}, include_thoughts={include_thoughts}")
                            
                            response = self.client.models.generate_content(
                                model=self.actual_model_name,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=self.model_config.get('temperature', 0.1),
                                    thinking_config=types.ThinkingConfig(
                                        thinking_budget=thinking_budget,
                                        include_thoughts=include_thoughts
                                    )
                                )
                            )
                        else:
                            # Standard generation without thinking
                            self.logger.debug(f"[{call_id}] Using standard mode")
                            
                            response = self.client.models.generate_content(
                                model=self.actual_model_name,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=self.model_config.get('temperature', 0.1)
                                )
                            )
                        
                        return response
                        
                    except Exception as e:
                        # Fallback to standard generation if thinking fails
                        self.logger.warning(f"[{call_id}] Thinking generation failed, using standard: {e}")
                        
                        try:
                            from google.genai import types
                            response = self.client.models.generate_content(
                                model=self.actual_model_name,
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=self.model_config.get('temperature', 0.1)
                                )
                            )
                            return response
                        except Exception as fallback_e:
                            self.logger.error(f"[{call_id}] Fallback generation also failed: {fallback_e}")
                            raise fallback_e
                
                # Run in executor since Google API is synchronous
                loop = asyncio.get_event_loop()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(executor, sync_generate)
                
                # Extract content from the response
                content = ""
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                content += part.text
                
                if not content:
                    content = str(response)
                
                response_time = time.time() - start_time
                
                # Get token counts if available
                tokens_used = 0
                thinking_tokens = 0
                
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    tokens_used = getattr(usage, 'total_token_count', 0) or 0
                    thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
                
                # Ensure token counts are integers
                tokens_used = int(tokens_used) if tokens_used is not None else 0
                thinking_tokens = int(thinking_tokens) if thinking_tokens is not None else 0
                
                # Log token usage for thinking models
                if thinking_tokens > 0:
                    self.logger.debug(f"[{call_id}] Tokens - Output: {tokens_used}, Thinking: {thinking_tokens}")
                
                return AgentResponse(
                    content=content,
                    model=self.model_name,
                    success=True,
                    tokens_used=tokens_used,
                    thinking_tokens=thinking_tokens,
                    response_time=response_time
                )
                
            except Exception as e:
                self.logger.warning(f"[{call_id}] Gemini attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                else:
                    response_time = time.time() - start_time
                    return AgentResponse(
                        content="",
                        model=self.model_name,
                        success=False,
                        error=str(e),
                        response_time=response_time
                    )


def create_agent(model_name: str, player_id: str) -> BaseLLMAgent:
    """Factory function to create appropriate agent based on model name"""
    return GeminiAgent(model_name, player_id)


def get_agent_info(model_name: str) -> Dict[str, Any]:
    """Get information about an agent configuration"""
    try:
        model_config = get_model_config(model_name)
        thinking_config = get_thinking_config(model_name)
        
        return {
            'model_name': model_name,
            'display_name': model_config.get('display_name', model_name),
            'actual_model_name': model_config.get('model_name', model_name),
            'thinking_available': model_config.get('thinking_available', False),
            'thinking_enabled': thinking_config is not None and thinking_config.get('thinking_budget', 0) != 0,
            'thinking_budget': thinking_config.get('thinking_budget', 0) if thinking_config else None,
            'temperature': model_config.get('temperature', 0.1),
            'timeout': model_config.get('timeout', 30)
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'error': str(e)
        }


def validate_agent_setup(model_name: str) -> bool:
    """Validate that an agent can be properly created"""
    try:
        # Check if model config exists
        model_config = get_model_config(model_name)
        
        # Check if API key is available
        config = load_config_file()
        api_config = config.get('api', {}).get('google', {})
        api_key_env = api_config.get('api_key_env', 'GEMINI_API_KEY')
        
        import os
        api_key = os.getenv(api_key_env)
        if not api_key:
            logging.getLogger(__name__).error(f"Missing API key: {api_key_env}")
            return False
        
        # Check if required packages are available
        try:
            from google import genai
        except ImportError:
            logging.getLogger(__name__).error("google-genai package not installed")
            return False
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent setup validation failed: {e}")
        return False


async def test_agent(model_name: str) -> Dict[str, Any]:
    """Test an agent with a simple prompt"""
    test_prompt = "What is 2 + 2?"
    
    try:
        agent = create_agent(model_name, "test")
        response = await agent.get_response(test_prompt, "test_call")
        
        return {
            'model_name': model_name,
            'success': response.success,
            'response_time': response.response_time,
            'content': response.content[:100] + "..." if len(response.content) > 100 else response.content,
            'tokens_used': response.tokens_used,
            'thinking_tokens': response.thinking_tokens,
            'error': response.error
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }