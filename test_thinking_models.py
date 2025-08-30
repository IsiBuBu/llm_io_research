#!/usr/bin/env python3

import sys
import os
import json
import time
from config import get_model_config, get_available_models
from competition import GeminiLLMAgent

def test_thinking_model(model_key):
    """Test a specific thinking model configuration"""
    print(f"\n--- Testing {model_key} ---")
    
    try:
        # Get available models
        available_models = get_available_models()
        
        # Get model config
        if model_key not in available_models:
            print(f"❌ Model {model_key} not found in config")
            return False
            
        model_config = get_model_config(model_key)
        print(f"✅ Model config loaded: {model_config.display_name}")
        print(f"   - Base model: {model_config.model_name}")
        print(f"   - Thinking available: {model_config.thinking_available}")
        print(f"   - Thinking enabled: {model_config.thinking_enabled}")
        
        # Create agent
        agent = GeminiLLMAgent(model_config=model_config)
        print(f"✅ Agent created")
        
        # Test prompt
        test_prompt = "You are playing an economic game. Calculate the optimal price for a product with marginal cost $10 and market demand of 100 units. Show your reasoning."
        
        start_time = time.time()
        response = agent.generate_response(test_prompt, include_thinking=True)
        end_time = time.time()
        
        print(f"✅ Response received in {end_time - start_time:.2f}s")
        
        # Check if we got thinking tokens
        if hasattr(response, 'thinking_tokens') and response.thinking_tokens:
            print(f"💭 Thinking tokens: {response.thinking_tokens}")
        else:
            print("💭 No thinking tokens detected")
            
        # Show response preview
        if response and response.final_response:
            preview = response.final_response[:100] + "..." if len(response.final_response) > 100 else response.final_response
            print(f"✅ Response preview: {preview}")
            return True
        else:
            print(f"❌ No response text received")
            if response and response.error:
                print(f"   Error: {response.error}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {model_key}: {str(e)}")
        return False

def main():
    models_to_test = [
        "gemini-2.0-flash-thinking",
        "gemini-2.0-flash",  # For comparison
        "gemini-2.0-flash-exp",  # Test exp version
        "gemini-2.0-flash-exp-thinking",  # Test exp with thinking
    ]
    
    print("Testing Gemini 2.0 Flash thinking capabilities...")
    print("=" * 60)
    
    results = {}
    for model_key in models_to_test:
        results[model_key] = test_thinking_model(model_key)
        time.sleep(2)  # Rate limiting
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model_key, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{status}: {model_key}")

if __name__ == "__main__":
    main()
