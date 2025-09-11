"""
Base Game Classes - Fixed version with proper parameter validation and enhanced parsing
Provides abstract base classes for economic games with robust error handling
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from config import GameConfig


def extract_numeric_value(text: str, field_name: str) -> Optional[float]:
    """Extract numeric value for a specific field from text"""
    # Try field-specific patterns first
    field_patterns = [
        rf'{field_name}["\']?\s*[:=]\s*([0-9]+\.?[0-9]*)',
        rf'"{field_name}"["\']?\s*[:=]\s*([0-9]+\.?[0-9]*)',
        rf"'{field_name}'[\"']?\s*[:=]\s*([0-9]+\.?[0-9]*)"
    ]
    
    for pattern in field_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None


def validate_action_bounds(action: Dict[str, Any], bounds: Dict[str, tuple]) -> Dict[str, Any]:
    """Validate and clamp action values to specified bounds"""
    validated = action.copy()
    
    for field, (min_val, max_val) in bounds.items():
        if field in validated:
            value = validated[field]
            if isinstance(value, (int, float)):
                validated[field] = max(min_val, min(max_val, float(value)))
    
    return validated


def validate_game_params(game_state, game_config, method_name=""):
    """Validate that parameters are correct types"""
    if game_state is not None and not isinstance(game_state, dict):
        raise TypeError(
            f"ERROR in {method_name}: game_state must be Dict, got {type(game_state)}. "
            f"This suggests GameConfig was passed instead of game_state. "
            f"Check parameter order!"
        )
    
    if game_config is not None and not isinstance(game_config, GameConfig):
        raise TypeError(
            f"ERROR in {method_name}: game_config must be GameConfig, got {type(game_config)}. "
            f"Check parameter order!"
        )


class PriceParsingMixin:
    """Mixin for games that need price parsing (Salop, Spulber)"""
    
    def parse_price_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse price/bid decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.robust_json_parse(response)
        if json_action:
            for price_field in ['price', 'bid']:
                if price_field in json_action:
                    price = json_action[price_field]
                    if isinstance(price, (int, float)) and price >= 0:
                        return {price_field: float(price), 'raw_response': response, 'parsing_method': 'json'}
        
        # Try numeric extraction
        for field in ['price', 'bid']:
            price = extract_numeric_value(response, field)
            if price is not None and price > 0:
                return {field: price, 'parsing_method': 'regex', 'raw_response': response}
        
        # Try simple number extraction - first reasonable number found
        number_patterns = [r'\$?(\d+\.?\d*)', r'(\d+\.\d+)']
        
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    price = float(matches[0])
                    if price > 0:
                        return {'price': price, 'parsing_method': 'number_extraction', 'raw_response': response}
                except ValueError:
                    continue
        
        return None


class QuantityParsingMixin:
    """Mixin for games that need quantity parsing (Green-Porter)"""
    
    def parse_quantity_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse quantity decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.robust_json_parse(response)
        if json_action:
            if 'quantity' in json_action:
                quantity = json_action['quantity']
                if isinstance(quantity, (int, float)) and quantity >= 0:
                    return {'quantity': float(quantity), 'raw_response': response, 'parsing_method': 'json'}
        
        # Try numeric extraction
        quantity = extract_numeric_value(response, 'quantity')
        if quantity is not None and quantity >= 0:
            return {'quantity': quantity, 'parsing_method': 'regex', 'raw_response': response}
        
        # Try simple number extraction
        number_patterns = [r'(\d+\.?\d*)', r'(\d+\.\d+)']
        
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    quantity = float(matches[0])
                    if quantity >= 0:
                        return {'quantity': quantity, 'parsing_method': 'number_extraction', 'raw_response': response}
                except ValueError:
                    continue
        
        return None


class ReportParsingMixin:
    """Enhanced mixin for games that need report parsing (Athey-Bagwell)"""
    
    def parse_report_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse cost report decision from LLM response with enhanced robustness"""
        
        if not response or not isinstance(response, str):
            return None
        
        # Log the response for debugging
        self.logger.debug(f"[{call_id}] Parsing response from {player_id}: {response[:200]}...")
        
        # Clean and normalize the response
        text_lower = response.lower().strip()
        
        # Strategy 1: Try JSON parsing first
        json_action = self.robust_json_parse(response)
        if json_action:
            for report_field in ['report', 'cost_report', 'decision', 'action']:
                if report_field in json_action:
                    report = json_action[report_field]
                    if isinstance(report, str) and report.lower() in ['high', 'low']:
                        return {'report': report.lower(), 'raw_response': response, 'parsing_method': 'json'}
        
        # Strategy 2: Look for explicit report patterns with various formats
        report_patterns = [
            # JSON-like patterns
            r'"?report"?\s*[:\s=]\s*"?(high|low)"?',
            r'"?cost_report"?\s*[:\s=]\s*"?(high|low)"?',
            r'"?decision"?\s*[:\s=]\s*"?(high|low)"?',
            r'"?action"?\s*[:\s=]\s*"?(high|low)"?',
            
            # Natural language patterns
            r'report(?:ing)?\s+(?:a\s+)?(high|low)\s+cost',
            r'(?:my\s+)?cost\s+(?:is\s+)?(high|low)',
            r'(?:i\s+(?:will\s+)?)?report\s+(high|low)',
            r'(?:i\s+(?:am\s+)?)?report(?:ing)?\s+(high|low)',
            r'(?:i\s+)?choose\s+(?:to\s+report\s+)?(high|low)',
            r'(?:i\s+)?select\s+(high|low)',
            r'(?:my\s+)?decision\s+(?:is\s+)?(high|low)',
            
            # Answer patterns
            r'answer\s*[:\s]\s*(high|low)',
            r'response\s*[:\s]\s*(high|low)',
            r'final\s+(?:answer|decision)\s*[:\s]\s*(high|low)',
            
            # Direct declaration patterns
            r'(high|low)\s+cost(?:\s+report)?',
            r'report\s*[:\s]\s*(high|low)',
        ]
        
        for pattern in report_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                report_value = match.group(1).lower()
                self.logger.debug(f"[{call_id}] Found report '{report_value}' using pattern: {pattern}")
                return {'report': report_value, 'parsing_method': 'regex', 'raw_response': response}
        
        # Strategy 3: Simple word detection with context validation
        high_words = ['high']
        low_words = ['low']
        
        # Count occurrences in relevant context
        high_count = 0
        low_count = 0
        
        # Look for words in decision-making context
        decision_context_patterns = [
            r'(?:report|decision|choose|select|answer|cost)\s+\w*\s*(high|low)',
            r'(high|low)\s+(?:cost|report|decision)',
        ]
        
        for pattern in decision_context_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match == 'high':
                    high_count += 1
                elif match == 'low':
                    low_count += 1
        
        # If no context matches, do simple word counting
        if high_count == 0 and low_count == 0:
            high_count = len(re.findall(r'\bhigh\b', text_lower))
            low_count = len(re.findall(r'\blow\b', text_lower))
        
        # Decide based on counts
        if high_count > low_count and high_count > 0:
            self.logger.debug(f"[{call_id}] Found 'high' by word detection (high: {high_count}, low: {low_count})")
            return {'report': 'high', 'parsing_method': 'word_detection', 'raw_response': response}
        elif low_count > high_count and low_count > 0:
            self.logger.debug(f"[{call_id}] Found 'low' by word detection (high: {high_count}, low: {low_count})")
            return {'report': 'low', 'parsing_method': 'word_detection', 'raw_response': response}
        
        # Strategy 4: Fallback patterns for common LLM response formats
        fallback_patterns = [
            r'based\s+on\s+.*?(high|low)',
            r'therefore\s+.*?(high|low)',
            r'so\s+(?:i\s+)?(?:will\s+)?.*?(high|low)',
            r'thus\s+.*?(high|low)',
            r'consequently\s+.*?(high|low)',
            r'(?:my|the)\s+choice\s+(?:is\s+)?(high|low)',
            r'(?:i\s+)?(?:would\s+)?(?:like\s+to\s+)?(?:choose\s+)?(high|low)',
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, text_lower)
            if match:
                report_value = match.group(1).lower()
                self.logger.debug(f"[{call_id}] Found report '{report_value}' using fallback pattern: {pattern}")
                return {'report': report_value, 'parsing_method': 'fallback_regex', 'raw_response': response}
        
        # Strategy 5: Last resort - look for any occurrence of "high" or "low"
        if 'low' in text_lower and 'high' not in text_lower:
            self.logger.debug(f"[{call_id}] Last resort: found only 'low' in response")
            return {'report': 'low', 'parsing_method': 'last_resort', 'raw_response': response}
        elif 'high' in text_lower and 'low' not in text_lower:
            self.logger.debug(f"[{call_id}] Last resort: found only 'high' in response")
            return {'report': 'high', 'parsing_method': 'last_resort', 'raw_response': response}
        
        # If all strategies fail, log the response for debugging
        self.logger.warning(f"[{call_id}] FAILED to parse report from {player_id}. Response: {response[:500]}")
        return None


class EconomicGame(ABC):
    """
    Abstract base class for all economic games
    Provides common functionality for prompt generation, response parsing, and game management
    """
    
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.logger = logging.getLogger(f"{__name__}.{game_name}")
        
        # Load template with improved error handling
        self.prompt_template = self._load_prompt_template(game_name)

    def _load_prompt_template(self, game_name: str) -> str:
        """Load prompt template for the game with multiple fallback strategies"""
        
        # Try multiple possible template locations
        template_paths = [
            Path(f"prompts/{game_name}_prompt.md"),
            Path(f"prompts/{game_name}.md"),
            Path(f"templates/{game_name}_prompt.md"),
            Path(f"templates/{game_name}.md"),
            Path(f"{game_name}_prompt.md"),
            Path(f"{game_name}.md")
        ]
        
        for prompt_path in template_paths:
            if prompt_path.exists():
                try:
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # Check if content is wrapped in code blocks and extract
                    template_match = re.search(r'```(?:markdown)?\n(.*?)\n```', content, re.DOTALL)
                    if template_match:
                        self.logger.info(f"Loaded template from {prompt_path}")
                        return template_match.group(1)
                    else:
                        # If no code blocks, use entire content
                        self.logger.info(f"Using entire file content from {prompt_path}")
                        return content
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load template from {prompt_path}: {e}")
                    continue
        
        # If no template found, create a basic one
        self.logger.error(f"No template found for {game_name}. Creating basic template.")
        basic_template = f"""## Economic Game: {game_name.title()}

### Context
You are a player in an economic game. Make your decision to maximize your profit.

### Your Task  
Choose your optimal action for this period.

### Output Format
Respond with valid JSON only:
{{"action": <your_decision>}}"""
        
        return basic_template

    def robust_json_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """Robust JSON parsing with multiple fallback strategies"""
        
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from code blocks
        json_patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(\{.*?\})\s*\n```',
            r'```\s*(\{.*?\})\s*```'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Find JSON-like structures
        json_like_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_like_pattern, response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Strategy 4: Extract key-value pairs
        kv_patterns = [
            r'"?(\w+)"?\s*:\s*"?([^",\}]+)"?',
            r'"?(\w+)"?\s*=\s*"?([^",\}]+)"?'
        ]
        
        for pattern in kv_patterns:
            matches = re.findall(pattern, response)
            if matches:
                result = {}
                for key, value in matches:
                    # Try to convert to number
                    try:
                        if '.' in value:
                            result[key] = float(value)
                        else:
                            result[key] = int(value)
                    except ValueError:
                        result[key] = value.strip('"\'')
                
                if result:
                    return result
        
        return None

    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def generate_player_prompt(self, player_id: str, game_state: Dict, 
                             game_config: GameConfig) -> str:
        """Generate prompt for player using config system"""
        pass
    
    @abstractmethod
    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into action dictionary with multiple fallback strategies"""
        pass
    
    @abstractmethod
    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig,
                         game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate payoffs for all players"""
        pass
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Update game state after round - default implementation with validation"""
        
        # CRITICAL FIX: Add type validation to prevent GameConfig being passed as game_state
        if not isinstance(game_state, dict):
            raise TypeError(f"game_state must be a Dict, got {type(game_state)}. "
                          f"This suggests a parameter order issue.")
        
        if not isinstance(game_config, GameConfig):
            raise TypeError(f"game_config must be a GameConfig, got {type(game_config)}")
        
        # FIXED: Use safe dictionary access
        current_round = game_state.get('current_round', 1)
        game_state['current_round'] = current_round + 1
        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float],
                                game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Get structured data for metrics calculation and logging"""
        return {
            'game_name': game_config.game_name,
            'experiment_type': game_config.experiment_type,
            'condition_name': game_config.condition_name,
            'actions': actions,
            'payoffs': payoffs,
            'constants': game_config.constants,
            'game_state': game_state
        }


class StaticGame(EconomicGame):
    """
    Base class for static (single-round) games like Salop and Spulber.
    No state updates needed between rounds.
    """
    
    def __init__(self, game_name: str):
        super().__init__(game_name)
        # Load template with improved error handling
        self.prompt_template = self._load_prompt_template(game_name)


class DynamicGame(EconomicGame):
    """
    Base class for dynamic (multi-round) games like Green-Porter and Athey-Bagwell.
    Includes state management and history tracking.
    """
    
    def __init__(self, game_name: str):
        super().__init__(game_name)
        self.prompt_template = self._load_prompt_template(game_name)
    
    @abstractmethod
    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initialize game state for dynamic games"""
        pass
    
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], 
                         game_config: GameConfig) -> Dict:
        """Update game state after each round - subclasses should override"""
        
        # CRITICAL FIX: Add type validation
        if not isinstance(game_state, dict):
            raise TypeError(f"game_state must be a Dict, got {type(game_state)}. "
                          f"This suggests a parameter order issue where GameConfig was passed as game_state.")
        
        if not isinstance(game_config, GameConfig):
            raise TypeError(f"game_config must be a GameConfig, got {type(game_config)}")
        
        # FIXED: Use safe dictionary access instead of direct .get() that was failing
        current_round = game_state.get('current_round', 1)
        game_state['current_round'] = current_round + 1
        return game_state


# Utility functions for game implementations

def calculate_market_shares_salop(prices: Dict[str, float], transport_cost: float, 
                                 market_size: float) -> Dict[str, float]:
    """Calculate market shares for Salop spatial competition"""
    
    if not prices:
        return {}
    
    # Convert to list for processing
    player_ids = list(prices.keys())
    price_values = list(prices.values())
    n = len(player_ids)
    
    if n == 1:
        return {player_ids[0]: market_size}
    
    market_shares = {}
    
    for i, player_id in enumerate(player_ids):
        # Calculate demand from each neighbor
        left_neighbor = price_values[(i - 1) % n]
        right_neighbor = price_values[(i + 1) % n]
        
        # Market share calculation
        left_share = (left_neighbor - price_values[i] + transport_cost) / (2 * transport_cost)
        right_share = (right_neighbor - price_values[i] + transport_cost) / (2 * transport_cost)
        
        # Clamp to valid range
        left_share = max(0, min(0.5, left_share))
        right_share = max(0, min(0.5, right_share))
        
        total_share = left_share + right_share
        market_shares[player_id] = total_share * market_size
    
    return market_shares


def calculate_cournot_quantities(num_players: int, marginal_cost: float, 
                               demand_intercept: float) -> float:
    """Calculate Cournot equilibrium quantities"""
    return (demand_intercept - marginal_cost) / (num_players + 1)


def calculate_collusive_quantities(num_players: int, marginal_cost: float, 
                                 demand_intercept: float) -> float:
    """Calculate collusive (monopoly) quantities split among players"""
    monopoly_quantity = (demand_intercept - marginal_cost) / 2
    return monopoly_quantity / num_players


# Game state validation utilities

def validate_game_state(game_state: Any, required_fields: List[str]) -> None:
    """Validate that game_state is a dict with required fields"""
    
    if not isinstance(game_state, dict):
        raise TypeError(f"game_state must be a dictionary, got {type(game_state)}")
    
    missing_fields = [field for field in required_fields if field not in game_state]
    if missing_fields:
        raise ValueError(f"game_state missing required fields: {missing_fields}")


def validate_game_config(game_config: Any) -> None:
    """Validate that game_config is a GameConfig object"""
    
    if not isinstance(game_config, GameConfig):
        raise TypeError(f"game_config must be a GameConfig object, got {type(game_config)}")


# Error handling decorators

def validate_parameters(func):
    """Decorator to validate game method parameters"""
    def wrapper(self, *args, **kwargs):
        # Get method signature to validate parameters
        if len(args) >= 2:
            game_state = args[0] if 'game_state' in func.__code__.co_varnames else None
            game_config = args[1] if 'game_config' in func.__code__.co_varnames else None
            
            if game_state is not None and not isinstance(game_state, dict):
                raise TypeError(f"Parameter validation failed in {func.__name__}: "
                              f"game_state must be Dict, got {type(game_state)}")
            
            if game_config is not None and not isinstance(game_config, GameConfig):
                raise TypeError(f"Parameter validation failed in {func.__name__}: "
                              f"game_config must be GameConfig, got {type(game_config)}")
        
        return func(self, *args, **kwargs)
    return wrapper


# Debugging utilities

def debug_game_state(game_state: Any, context: str = "") -> None:
    """Debug utility to log game state information"""
    logger = logging.getLogger(__name__)
    
    prefix = f"[{context}] " if context else ""
    logger.debug(f"{prefix}game_state type: {type(game_state)}")
    
    if isinstance(game_state, dict):
        logger.debug(f"{prefix}game_state keys: {list(game_state.keys())}")
        if 'current_period' in game_state:
            logger.debug(f"{prefix}current_period: {game_state['current_period']}")
        if 'current_round' in game_state:
            logger.debug(f"{prefix}current_round: {game_state['current_round']}")
    else:
        logger.debug(f"{prefix}game_state value: {game_state}")


def debug_parameters(player_id: str, game_state: Any, game_config: Any, context: str = "") -> None:
    """Debug utility to log all parameters"""
    logger = logging.getLogger(__name__)
    
    prefix = f"[{context}] " if context else ""
    logger.debug(f"{prefix}player_id: {player_id}")
    logger.debug(f"{prefix}game_state type: {type(game_state)}")
    logger.debug(f"{prefix}game_config type: {type(game_config)}")
    
    if hasattr(game_config, 'game_name'):
        logger.debug(f"{prefix}game_config.game_name: {game_config.game_name}")