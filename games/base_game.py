"""
Base game classes for LLM game theory experiments.
Updated to work with new config system and robust response parsing.
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
from config import GameConfig


def extract_numeric_value(response: str, field_name: str) -> float:
    """Extract numeric value from response text for a given field"""
    if not response:
        return -1.0
    
    # Clean response
    response = response.lower().strip()
    
    # Pattern to match field: value patterns
    patterns = [
        rf'"{field_name}"\s*:\s*(\d+\.?\d*)',  # "field": value
        rf'"{field_name}":\s*(\d+\.?\d*)',     # "field":value  
        rf'{field_name}\s*:\s*(\d+\.?\d*)',     # field: value
        rf'{field_name}:\s*(\d+\.?\d*)',        # field:value
        rf'{field_name}\s*=\s*(\d+\.?\d*)',     # field = value
        rf'{field_name}\s+is\s+(\d+\.?\d*)',    # field is value
        rf'my\s+{field_name}\s*:\s*(\d+\.?\d*)', # my field: value
        rf'choose\s+{field_name}\s*:\s*(\d+\.?\d*)', # choose field: value
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                continue
    
    return -1.0


def validate_action_bounds(action: Dict[str, Any], bounds: Dict[str, tuple]) -> Dict[str, Any]:
    """Validate and clamp action values to specified bounds"""
    validated_action = action.copy()
    
    for field, (min_val, max_val) in bounds.items():
        if field in validated_action:
            value = validated_action[field]
            if isinstance(value, (int, float)):
                # Clamp to bounds
                validated_action[field] = max(min_val, min(max_val, float(value)))
    
    return validated_action


class EconomicGame(ABC):
    """Base class for all economic games"""
    
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def robust_json_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """Robust JSON parsing from LLM response with multiple fallback strategies"""
        if not response:
            return None
            
        try:
            # Clean response
            response = response.strip()
            
            # Strategy 1: Direct JSON parse
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Strategy 2: Extract JSON from code blocks
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON in json code blocks
                r'```\s*(\{.*?\})\s*```',     # JSON in any code blocks  
                r'(\{[^{}]*"[^"]*"[^{}]*:[^{}]*\})',  # Simple JSON objects
                r'(\{.*?\})'  # Any braces content (greedy)
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.DOTALL)
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue
            
            # Strategy 3: Manual JSON construction for common patterns
            json_like_patterns = [
                r'(\w+)\s*:\s*(["\']?)([^,}\n]+)\2',  # field: value patterns
                r'["\'](\w+)["\']\s*:\s*(["\']?)([^,}\n]+)\2',  # "field": value patterns
            ]
            
            constructed_json = {}
            for pattern in json_like_patterns:
                matches = re.findall(pattern, response)
                for match in matches:
                    if len(match) == 3:  # field: value pattern
                        key = match[0]
                        value_str = match[2].strip()
                        # Try to convert to appropriate type
                        try:
                            if '.' in value_str:
                                constructed_json[key] = float(value_str)
                            else:
                                constructed_json[key] = int(value_str)
                        except ValueError:
                            constructed_json[key] = value_str.strip('"\'')
            
            if constructed_json:
                return constructed_json
            
            return None
            
        except Exception as e:
            self.logger.debug(f"JSON parsing failed: {e}")
            return None

    def _load_prompt_template(self, game_name: str) -> str:
        """Load prompt template with better error handling and path resolution"""
        # Try multiple possible locations
        possible_paths = [
            Path(f"prompts/{game_name}.md"),
            Path(f"templates/{game_name}.md"),  
            Path(f"games/{game_name}.md"),
            Path(f"{game_name}.md"),
            Path(f"prompts/{game_name}_prompt.md"),
            Path(f"templates/{game_name}_template.md")
        ]
        
        for prompt_path in possible_paths:
            if prompt_path.exists():
                try:
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract template between ``` blocks
                    template_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
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
        """Update game state after round - default implementation"""
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

    def parse_llm_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Base implementation with comprehensive parsing strategies.
        Subclasses should override for game-specific parsing.
        """
        
        # Strategy 1: Try robust JSON parsing first
        json_result = self.robust_json_parse(response)
        if json_result:
            # Add metadata
            json_result['raw_response'] = response
            json_result['parsing_method'] = 'json'
            return json_result
        
        # Strategy 2: Text-based extraction (subclasses implement specific logic)
        # This is just the base - each game overrides with specific field extraction
        self.logger.warning(f"[{call_id}] Base parser failed for {player_id}, subclass should override")
        return None


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
        # Basic implementation - increment round
        game_state['current_round'] = game_state.get('current_round', 1) + 1
        return game_state


# Response Parsing Examples for Different Game Types

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
            if price > 0:
                return {field: price, 'parsing_method': 'regex', 'raw_response': response}
        
        # Try simple number extraction - first reasonable number found
        number_patterns = [r'\$?(\d+\.?\d*)', r'(\d+\.?\d*)', r'(\d+\.?\d+)']
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    price = float(matches[0])
                    if price >= 0:
                        return {'price': price, 'parsing_method': 'number', 'raw_response': response}
                except ValueError:
                    continue
        
        return None


class QuantityParsingMixin:
    """Mixin for games that need quantity parsing (Green-Porter)"""
    
    def parse_quantity_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse quantity decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.robust_json_parse(response)
        if json_action and 'quantity' in json_action:
            quantity = json_action['quantity']
            if isinstance(quantity, (int, float)) and quantity >= 0:
                return {'quantity': float(quantity), 'raw_response': response, 'parsing_method': 'json'}
        
        # Try numeric extraction
        quantity = extract_numeric_value(response, 'quantity')
        if quantity >= 0:
            return {'quantity': quantity, 'parsing_method': 'regex', 'raw_response': response}
        
        # Try simple number extraction
        number_patterns = [r'(\d+\.?\d*)', r'(\d+\.?\d+)', r'(\d+)']
        for pattern in number_patterns:
            matches = re.findall(pattern, response)
            if matches:
                try:
                    quantity = float(matches[0])
                    if quantity >= 0:
                        return {'quantity': quantity, 'parsing_method': 'number', 'raw_response': response}
                except ValueError:
                    continue
        
        return None


class ReportParsingMixin:
    """Mixin for games that need report parsing (Athey-Bagwell)"""
    
    def parse_report_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Parse cost report decision from LLM response"""
        
        # Try JSON parsing first
        json_action = self.robust_json_parse(response)
        if json_action and 'report' in json_action:
            report = json_action['report'].lower().strip()
            if report in ['high', 'low']:
                return {'report': report, 'raw_response': response, 'parsing_method': 'json'}
        
        # Try direct text extraction
        response_lower = response.lower()
        
        # Look for explicit report statements
        if '"high"' in response_lower or "'high'" in response_lower or 'report "high"' in response_lower:
            return {'report': 'high', 'parsing_method': 'text', 'raw_response': response}
        elif '"low"' in response_lower or "'low'" in response_lower or 'report "low"' in response_lower:
            return {'report': 'low', 'parsing_method': 'text', 'raw_response': response}
        
        # Look for decision keywords
        if 'high cost' in response_lower or 'report high' in response_lower:
            return {'report': 'high', 'parsing_method': 'keyword', 'raw_response': response}
        elif 'low cost' in response_lower or 'report low' in response_lower:
            return {'report': 'low', 'parsing_method': 'keyword', 'raw_response': response}
        
        return None