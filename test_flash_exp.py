#!/usr/bin/env python3
"""Test the new gemini-2.0-flash-exp models individually"""

import os
import logging
from runner import ExperimentRunner

def test_flash_exp_models():
    """Test just the gemini-2.0-flash-exp models"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== TESTING GEMINI 2.0 FLASH EXP MODELS ===")
    
    try:
        # Initialize runner
        runner = ExperimentRunner(debug=True)
        
        # Test prompt
        test_prompt = "You are in a 3-player economic game. Market demand is Q = 100 - P. Your marginal cost is 10. What quantity should you produce to maximize profit? Show your reasoning."
        
        # Models to test
        test_models = [
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-exp-thinking"
        ]
        
        results = {}
        
        for model_key in test_models:
            logger.info(f"Testing {model_key}...")
            
            try:
                from competition import GeminiLLMAgent
                model_config = runner.available_models[model_key]
                agent = GeminiLLMAgent(model_config, debug=True)
                response = agent.generate_response(test_prompt, include_thinking=True)
                
                results[model_key] = {
                    'success': response.error is None,
                    'error': response.error,
                    'response_time': response.response_time,
                    'final_response_length': len(response.final_response) if response.final_response else 0,
                    'thinking_length': len(response.thinking_output) if response.thinking_output else 0
                }
                
                if response.error:
                    logger.warning(f"  ❌ FAILED: {response.error}")
                else:
                    logger.info(f"  ✅ SUCCESS: {response.response_time:.2f}s, output: {len(response.final_response)} chars")
                    if response.thinking_output:
                        logger.info(f"    Thinking: {len(response.thinking_output)} chars")
                
            except Exception as e:
                error_msg = f"Exception testing {model_key}: {str(e)}"
                logger.error(f"  ❌ EXCEPTION: {error_msg}")
                results[model_key] = {
                    'success': False,
                    'error': error_msg,
                    'response_time': 0,
                    'final_response_length': 0,
                    'thinking_length': 0
                }
        
        # Summary
        logger.info("=== TEST SUMMARY ===")
        successful = [k for k, v in results.items() if v['success']]
        failed = [k for k, v in results.items() if not v['success']]
        
        logger.info(f"Successful: {len(successful)}/{len(test_models)}")
        if successful:
            logger.info(f"  Working models: {', '.join(successful)}")
        if failed:
            logger.warning(f"  Failed models: {', '.join(failed)}")
            for model in failed:
                logger.warning(f"    {model}: {results[model]['error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {}

if __name__ == "__main__":
    results = test_flash_exp_models()
    success = len([r for r in results.values() if r['success']]) > 0
    exit(0 if success else 1)
