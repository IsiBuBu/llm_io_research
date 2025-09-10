"""
Gemini Agents - Google AI integration for LLM game theory experiments
Handles thinking configurations, authentication, retries, and rate limiting
NOW SUPPORTS MOCK MODE for testing workflow and metrics
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
            import google.generativeai as genai
            
            # Get API configuration
            api_config = self.model_config.get('api_config', {})
            api_key = api_config.get('api_key')
            
            if not api_key:
                raise ValueError("Gemini API key not found in configuration")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Create model instance
            generation_config = {
                'temperature': self.model_config.get('temperature', 0.7),
                'top_p': self.model_config.get('top_p', 0.9),
                'top_k': self.model_config.get('top_k', 40),
                'max_output_tokens': self.model_config.get('max_output_tokens', 1024),
            }
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    async def get_action(self, prompt: str, call_id: str) -> str:
        """Get action from Gemini with thinking support and error handling"""
        self.total_calls += 1
        
        for attempt in range(self.max_retries):
            try:
                # Check thinking budget
                if self.thinking_enabled and self.thinking_budget > 0:
                    if self.thinking_used >= self.thinking_budget:
                        self.logger.warning(f"[{call_id}] Thinking budget exhausted, disabling thinking")
                        self.thinking_enabled = False
                
                # Prepare the prompt
                final_prompt = self._prepare_prompt(prompt, call_id)
                
                # Make API call
                response = await self._make_api_call(final_prompt, call_id)
                
                # Process response
                content = self._extract_content(response)
                
                # Update thinking usage if applicable
                if self.thinking_enabled and hasattr(response, 'usage_metadata'):
                    thinking_tokens = getattr(response.usage_metadata, 'thinking_tokens', 0)
                    self.thinking_used += thinking_tokens
                    self.total_thinking_tokens += thinking_tokens
                
                # Update success metrics
                self.successful_calls += 1
                if hasattr(response, 'usage_metadata'):
                    self.total_tokens += getattr(response.usage_metadata, 'total_token_count', 0)
                
                self.logger.debug(f"[{call_id}] Successful response from {self.model_name}")
                return content
                
            except Exception as e:
                self.logger.warning(f"[{call_id}] Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    self.logger.error(f"[{call_id}] All {self.max_retries} attempts failed")
                    raise
        
        # This should never be reached due to the raise above
        raise RuntimeError(f"[{call_id}] Unexpected error in get_action")
    
    def _prepare_prompt(self, prompt: str, call_id: str) -> str:
        """Prepare prompt for Gemini API call"""
        return prompt
    
    async def _make_api_call(self, prompt: str, call_id: str) -> Any:
        """Make actual API call to Gemini"""
        try:
            # Create a timeout for the API call
            response = await asyncio.wait_for(
                self._call_gemini_api(prompt),
                timeout=self.timeout
            )
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"API call timed out after {self.timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")
    
    async def _call_gemini_api(self, prompt: str):
        """Actual Gemini API call in executor to avoid blocking"""
        loop = asyncio.get_event_loop()
        
        def sync_call():
            return self.client.generate_content(prompt)
        
        # Run in thread pool to avoid blocking the event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(executor, sync_call)
            return response
    
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


def create_agent(model_name: str, player_id: str, **kwargs) -> BaseLLMAgent:
    """
    Create appropriate LLM agent based on model name and configuration
    Supports both real and mock agents for testing
    """
    logger = logging.getLogger(f"{__name__}.create_agent")
    
    # Import here to get current value of MOCK_MODE
    import runner
    mock_mode = getattr(runner, 'MOCK_MODE', False)
    
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
            api_config = model_config.get('api_config', {})
            if not api_config.get('api_key'):
                return False
        
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent validation failed for {model_name}: {e}")
        return False


async def test_agent_connectivity(model_name: str, player_id: str = "test") -> Dict[str, Any]:
    """Test agent connectivity and basic functionality"""
    logger = logging.getLogger(__name__)
    
    try:
        agent = create_agent(model_name, player_id)
        
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


async def batch_test_agents(model_names: list, test_prompt: str = None) -> Dict[str, Any]:
    """Test multiple agents in parallel"""
    if test_prompt is None:
        test_prompt = "Respond with JSON: {\"test\": \"success\"}"
    
    tasks = []
    for model_name in model_names:
        task = test_agent_connectivity(model_name)
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