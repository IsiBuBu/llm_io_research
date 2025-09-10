"""
Gemini Agents - Google AI integration for LLM game theory experiments
Handles thinking configurations, authentication, retries, and rate limiting
NOW SUPPORTS MOCK MODE for testing workflow and metrics
"""

import asyncio
import logging
import time
import concurrent.futures
import os
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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def get_action(self, prompt: str, call_id: str) -> str:
        """Get action from LLM - returns raw string response"""
        pass
    
    async def get_response(self, prompt: str, call_id: str) -> AgentResponse:
        """Get structured response from LLM"""
        try:
            start_time = time.time()
            content = await self.get_action(prompt, call_id)
            response_time = time.time() - start_time
            
            return AgentResponse(
                content=content,
                model=self.model_name,
                success=True,
                response_time=response_time
            )
        except Exception as e:
            return AgentResponse(
                content="",
                model=self.model_name,
                success=False,
                error=str(e),
                response_time=0.0
            )


class GeminiAgent(BaseLLMAgent):
    """
    Google Gemini agent with thinking support and robust error handling
    """
    
    def __init__(self, model_name: str, player_id: str, **kwargs):
        super().__init__(model_name, player_id)
        
        self.model_config = get_model_config(model_name)
        self.thinking_config = get_thinking_config(model_name)
        
        # Initialize Gemini client
        self.client = self._initialize_client()
        
        # Thinking configuration
        self.thinking_enabled = self.thinking_config is not None
        if self.thinking_enabled:
            self.thinking_budget = self.thinking_config.get('thinking_budget', -1)
            self.thinking_used = 0
            
        # Rate limiting and retry configuration
        self.max_retries = self.model_config.get('max_retries', 3)
        self.retry_delay = self.model_config.get('retry_delay', 1.0)
        self.timeout = self.model_config.get('timeout', 30.0)
        
        # Performance tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.total_tokens = 0
        self.total_thinking_tokens = 0
        
        self.logger.info(f"Initialized Gemini agent: {model_name} for {player_id}")
        if self.thinking_enabled:
            self.logger.info(f"Thinking enabled with budget: {self.thinking_budget}")
    
    def _initialize_client(self):
        """Initialize Gemini client with proper authentication"""
        try:
            # Get API configuration
            config = load_config_file()
            api_config = config.get('api', {}).get('google', {})
            api_key_env = api_config.get('api_key_env', 'GEMINI_API_KEY')
            
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"Gemini API key not found in configuration")
            
            # Import and configure Gemini
            from google import genai
            genai.configure(api_key=api_key)
            
            # Create client
            client = genai.GenerativeModel(self.model_config.get('model_name', 'gemini-pro'))
            
            return client
            
        except ImportError:
            self.logger.error("google-genai package not installed. Run: pip install google-genai")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def _prepare_request(self, prompt: str) -> Dict[str, Any]:
        """Prepare request parameters with thinking configuration"""
        request_config = {
            'contents': prompt,
            'generation_config': {
                'temperature': self.model_config.get('temperature', 0.0),
                'max_output_tokens': self.model_config.get('max_tokens', 1024),
            }
        }
        
        # Add thinking configuration if enabled
        if self.thinking_enabled and self.thinking_budget != 0:
            if self.thinking_budget > 0 and self.thinking_used >= self.thinking_budget:
                self.logger.warning(f"Thinking budget ({self.thinking_budget}) exceeded, disabling thinking")
            else:
                request_config['generation_config']['thinking'] = True
                
        return request_config
    
    async def get_action(self, prompt: str, call_id: str) -> str:
        """Get action from Gemini with retries and error handling"""
        self.total_calls += 1
        
        for attempt in range(self.max_retries):
            try:
                # Prepare request
                request_config = self._prepare_request(prompt)
                
                # Make API call with timeout
                start_time = time.time()
                
                # Use executor for timeout handling
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self.client.generate_content,
                        **request_config
                    )
                    
                    try:
                        response = await asyncio.wait_for(
                            asyncio.wrap_future(future),
                            timeout=self.timeout
                        )
                    except asyncio.TimeoutError:
                        future.cancel()
                        raise TimeoutError(f"Request timed out after {self.timeout}s")
                
                # Extract content and update metrics
                content = self._extract_content(response)
                self.successful_calls += 1
                
                # Update token usage
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    self.total_tokens += getattr(usage, 'total_token_count', 0)
                    if hasattr(usage, 'thinking_token_count'):
                        thinking_tokens = getattr(usage, 'thinking_token_count', 0)
                        self.total_thinking_tokens += thinking_tokens
                        if thinking_tokens > 0:
                            self.thinking_used += 1
                
                response_time = time.time() - start_time
                self.logger.debug(f"[{call_id}] Successful response in {response_time:.2f}s")
                
                return content
                
            except Exception as e:
                self.logger.warning(f"[{call_id}] Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(f"[{call_id}] All {self.max_retries} attempts failed")
                    raise
    
    def _extract_content(self, response) -> str:
        """Extract text content from Gemini response"""
        try:
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    return ''.join(part.text for part in parts if hasattr(part, 'text'))
            
            # Fallback
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Failed to extract content from response: {e}")
            raise ValueError(f"Could not extract content from Gemini response: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent"""
        success_rate = self.successful_calls / self.total_calls if self.total_calls > 0 else 0
        
        return {
            'model_name': self.model_name,
            'player_id': self.player_id,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'success_rate': success_rate,
            'total_tokens': self.total_tokens,
            'total_thinking_tokens': self.total_thinking_tokens,
            'thinking_enabled': self.thinking_enabled,
            'thinking_used': self.thinking_used if self.thinking_enabled else 0,
            'thinking_budget': self.thinking_budget if self.thinking_enabled else 0
        }


def create_agent(model_name: str, player_id: str, mock_mode: bool = False, **kwargs) -> BaseLLMAgent:
    """
    Create appropriate LLM agent based on model name and configuration
    Supports both real and mock agents for testing
    
    Args:
        model_name: Name of the model to create
        player_id: ID for this player/agent
        mock_mode: If True, create mock agent instead of real one
        **kwargs: Additional arguments for real agents
    """
    logger = logging.getLogger(f"{__name__}.create_agent")
    
    # Use mock agent if mock mode is enabled
    if mock_mode:
        from mock_agents import MockLLMAgent
        logger.info(f"🎭 Creating mock agent for {model_name} as {player_id}")
        return MockLLMAgent(model_name, player_id)
    
    # Original logic for real agents
    if model_name.startswith('gemini'):
        logger.info(f"🤖 Creating Gemini agent for {model_name} as {player_id}")
        return GeminiAgent(model_name, player_id, **kwargs)
    elif model_name.startswith('gpt'):
        # Add OpenAI agent creation here when implemented
        raise NotImplementedError(f"Real OpenAI agents not yet implemented for {model_name}")
    elif model_name.startswith('claude'):
        # Add Anthropic agent creation here when implemented  
        raise NotImplementedError(f"Real Anthropic agents not yet implemented for {model_name}")
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def validate_agent_setup(model_name: str) -> bool:
    """Validate that an agent can be properly configured"""
    try:
        model_config = get_model_config(model_name)
        
        # Check required configuration
        if model_name.startswith('gemini'):
            # Check API key availability
            config = load_config_file()
            api_config = config.get('api', {}).get('google', {})
            api_key_env = api_config.get('api_key_env', 'GEMINI_API_KEY')
            
            api_key = os.getenv(api_key_env)
            if not api_key:
                return False
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent validation failed for {model_name}: {e}")
        return False


async def test_agent_connectivity(model_name: str, player_id: str = "test", mock_mode: bool = False) -> Dict[str, Any]:
    """Test agent connectivity and basic functionality"""
    logger = logging.getLogger(__name__)
    
    try:
        agent = create_agent(model_name, player_id, mock_mode=mock_mode)
        
        test_prompt = "Respond with a simple JSON: {\"status\": \"ok\", \"model\": \"" + model_name + "\"}"
        
        start_time = time.time()
        response = await agent.get_response(test_prompt, "connectivity_test")
        end_time = time.time()
        
        return {
            'model_name': model_name,
            'success': response.success,
            'response_time': end_time - start_time,
            'error': response.error,
            'content_preview': response.content[:100] if response.content else None
        }
        
    except Exception as e:
        logger.error(f"Connectivity test failed for {model_name}: {e}")
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e),
            'response_time': 0.0,
            'content_preview': None
        }


def get_agent_performance_summary(agents: list) -> Dict[str, Any]:
    """Get performance summary across multiple agents"""
    summary = {
        'total_agents': len(agents),
        'total_calls': 0,
        'total_successful_calls': 0,
        'total_tokens': 0,
        'total_thinking_tokens': 0,
        'agents_with_thinking': 0,
        'agent_details': []
    }
    
    for agent in agents:
        if hasattr(agent, 'get_performance_stats'):
            stats = agent.get_performance_stats()
            summary['total_calls'] += stats['total_calls']
            summary['total_successful_calls'] += stats['successful_calls']
            summary['total_tokens'] += stats['total_tokens']
            summary['total_thinking_tokens'] += stats['total_thinking_tokens']
            
            if stats['thinking_enabled']:
                summary['agents_with_thinking'] += 1
            
            summary['agent_details'].append(stats)
    
    # Calculate overall success rate
    if summary['total_calls'] > 0:
        summary['overall_success_rate'] = summary['total_successful_calls'] / summary['total_calls']
    else:
        summary['overall_success_rate'] = 0.0
    
    return summary


# Backward compatibility
def create_gemini_agent(model_name: str, player_id: str, **kwargs) -> GeminiAgent:
    """Backward compatibility function for creating Gemini agents"""
    return GeminiAgent(model_name, player_id, **kwargs)


async def batch_test_agents(model_names: list, test_prompt: str = None, mock_mode: bool = False) -> Dict[str, Any]:
    """Test multiple agents in parallel"""
    if test_prompt is None:
        test_prompt = "Respond with JSON: {\"test\": \"success\"}"
    
    tasks = []
    for model_name in model_names:
        task = test_agent_connectivity(model_name, mock_mode=mock_mode)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    test_summary = {
        'total_tested': len(model_names),
        'successful': 0,
        'failed': 0,
        'results': {}
    }
    
    for i, result in enumerate(results):
        model_name = model_names[i]
        
        if isinstance(result, Exception):
            test_summary['results'][model_name] = {
                'success': False,
                'error': str(result)
            }
            test_summary['failed'] += 1
        else:
            test_summary['results'][model_name] = result
            if result.get('success', False):
                test_summary['successful'] += 1
            else:
                test_summary['failed'] += 1
    
    return test_summary