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
            import google.generativeai as genai
            
            # Get API key from environment
            import os
            api_key_env = self.api_config.get('google', {}).get('api_key_env', 'GEMINI_API_KEY')
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} environment variable not set")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.actual_model_name)
            
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
            
        except ImportError:
            self.logger.error("Google AI package not installed. Run: pip install google-generativeai")
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
                # Prepare generation config
                generation_config = {
                    'temperature': self.model_config.get('temperature', 0.1)
                }
                
                # Create generation function based on thinking availability
                if self.thinking_config and self.model_config.get('thinking_available', False):
                    # Use thinking-enabled generation
                    def sync_generate():
                        try:
                            from google.generativeai import types
                            
                            return self.client.generate_content(
                                prompt,
                                generation_config=generation_config,
                                thinking_config=types.ThinkingConfig(
                                    thinking_budget=self.thinking_config.get('thinking_budget', -1),
                                    include_thoughts=self.thinking_config.get('include_thoughts', False)
                                )
                            )
                        except Exception as e:
                            # Fallback to standard generation if thinking fails
                            self.logger.warning(f"[{call_id}] Thinking generation failed, using standard: {e}")
                            return self.client.generate_content(
                                prompt,
                                generation_config=generation_config
                            )
                else:
                    # Standard generation without thinking
                    def sync_generate():
                        return self.client.generate_content(
                            prompt,
                            generation_config=generation_config
                        )
                
                # Run in executor since Google API is synchronous
                loop = asyncio.get_event_loop()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(executor, sync_generate)
                
                # Extract content
                content = response.text if hasattr(response, 'text') else str(response)
                response_time = time.time() - start_time
                
                # Get token counts if available
                tokens_used = 0
                thinking_tokens = 0
                
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    tokens_used = getattr(usage, 'total_token_count', 0)
                    thinking_tokens = getattr(usage, 'thoughts_token_count', 0)
                
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
                    # Exponential backoff
                    await asyncio.sleep((2 ** attempt) + (0.1 * attempt))
                else:
                    return AgentResponse(
                        content="",
                        model=self.model_name,
                        success=False,
                        error=str(e),
                        response_time=time.time() - start_time
                    )


class RandomAgent(BaseLLMAgent):
    """Random baseline agent for testing and debugging"""
    
    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """Generate random responses for testing"""
        import random
        
        # Simple random responses based on prompt type
        if "price" in prompt.lower():
            price = round(random.uniform(8, 30), 2)
            content = f'{{"price": {price}, "reasoning": "Random pricing strategy"}}'
        elif "quantity" in prompt.lower():
            quantity = random.randint(15, 30)
            content = f'{{"quantity": {quantity}, "reasoning": "Random quantity selection"}}'
        elif "report" in prompt.lower():
            report = random.choice(["high", "low"])
            content = f'{{"report": "{report}", "reasoning": "Random cost report"}}'
        else:
            content = f'{{"action": "default", "reasoning": "Random default action"}}'
        
        # Simulate API delay
        await asyncio.sleep(0.05)
        
        return AgentResponse(
            content=content,
            model="random",
            success=True,
            tokens_used=50,
            response_time=0.05
        )


def create_agent(model_name: str, player_id: str) -> BaseLLMAgent:
    """Factory function to create appropriate agent based on model name"""
    
    if model_name == "random":
        return RandomAgent(model_name, player_id)
    
    # Validate model exists in config
    try:
        model_config = get_model_config(model_name)
    except ValueError as e:
        raise ValueError(f"Model {model_name} not found in config.json: {e}")
    
    # All non-random models are now Gemini models
    return GeminiAgent(model_name, player_id)


# Rate limiting utilities
class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        async with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            # Check if we need to wait
            if len(self.calls) >= self.calls_per_minute:
                wait_time = 60 - (now - self.calls[0]) + 0.1  # Small buffer
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Clean up old calls again after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            # Record this call
            self.calls.append(now)


# Global rate limiter for Gemini API
GEMINI_RATE_LIMITER = RateLimiter(calls_per_minute=60)


async def get_rate_limited_response(agent: BaseLLMAgent, prompt: str, call_id: str) -> AgentResponse:
    """Get response with rate limiting applied"""
    
    # Apply rate limiting for Gemini agents
    if isinstance(agent, GeminiAgent):
        await GEMINI_RATE_LIMITER.wait_if_needed()
    
    return await agent.get_response(prompt, call_id)


# Utility functions for agent management
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
            import google.generativeai
        except ImportError:
            logging.getLogger(__name__).error("google-generativeai package not installed")
            return False
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent setup validation failed: {e}")
        return False


async def test_agent(model_name: str) -> Dict[str, Any]:
    """Test an agent with a simple prompt"""
    test_prompt = "What is 2 + 2? Respond with a JSON format: {\"answer\": your_answer}"
    
    try:
        agent = create_agent(model_name, "test_player")
        
        start_time = time.time()
        response = await agent.get_response(test_prompt, "test_call")
        end_time = time.time()
        
        return {
            'model_name': model_name,
            'success': response.success,
            'response_time': end_time - start_time,
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


# Batch testing function
async def test_all_agents() -> Dict[str, Any]:
    """Test all configured agents"""
    config = load_config_file()
    challenger_models = config.get('models', {}).get('challenger_models', [])
    defender_model = config.get('models', {}).get('defender_model')
    
    all_models = challenger_models + ([defender_model] if defender_model else [])
    
    results = {}
    
    for model in all_models:
        print(f"Testing {model}...")
        result = await test_agent(model)
        results[model] = result
        
        if result['success']:
            print(f"  ✅ Success in {result['response_time']:.2f}s")
        else:
            print(f"  ❌ Failed: {result['error']}")
    
    return results