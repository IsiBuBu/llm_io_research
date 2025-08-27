# agents.py
import os
import json
import time
import numpy as np
from typing import Dict, Any, Optional
import google.generativeai as genai

class LLMAgent:
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: Optional[str] = None):
        self.model_name = model_name
        
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv('GEMINI_API_KEY'):
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        else:
            raise ValueError("Gemini API key required")
        
        self.model = genai.GenerativeModel(model_name)
    
    def make_decision(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                time.sleep(1)  # Rate limiting
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=1000
                    )
                )
                
                # Parse JSON from response
                text = response.text.strip()
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx]
                    return json.loads(json_str)
                
                return {"error": "No valid JSON found"}
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": str(e)}
                time.sleep(2 ** attempt)
        
        return {"error": "Max retries exceeded"}

class RandomAgent:
    def __init__(self):
        pass
    
    def make_decision(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        if 'price' in prompt.lower():
            return {'price': np.random.uniform(8, 20), 'reasoning': 'Random price'}
        elif 'quantity' in prompt.lower():
            return {'quantity': np.random.uniform(10, 40), 'reasoning': 'Random quantity'}
        elif 'report' in prompt.lower():
            return {'report': np.random.choice(['high', 'low']), 'reasoning': 'Random report'}
        return {'error': 'Unknown decision type'}

class OptimalAgent:
    def __init__(self):
        pass
    
    def make_decision(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        if 'salop' in prompt.lower() or ('price' in prompt.lower() and 'circular' in prompt.lower()):
            return {'price': 10.0, 'reasoning': 'Nash equilibrium price'}
        elif 'spulber' in prompt.lower() or 'auction' in prompt.lower():
            return {'price': 9.5, 'reasoning': 'Bayesian Nash bid'}
        elif 'quantity' in prompt.lower():
            return {'quantity': 20.0, 'reasoning': 'Collusive quantity'}
        elif 'report' in prompt.lower():
            cost = 'high' if 'high' in prompt.lower() else 'low'
            return {'report': cost, 'reasoning': 'Truthful reporting'}
        return {'error': 'Unknown strategy'}