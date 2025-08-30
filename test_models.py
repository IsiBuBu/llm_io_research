#!/usr/bin/env python3
"""
Test which Gemini models are actually available and working
"""

import os
import json
import google.generativeai as genai

def test_models():
    """Test which models are available and working"""
    
    # Configure API
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No GEMINI_API_KEY found")
        return
    
    genai.configure(api_key=api_key)
    
    # Models to test - including the problematic one
    models_to_test = [
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'gemini-2.0-flash-lite',
        'gemini-2.5-flash',
        'gemini-2.5-pro',  # This is the problematic one
        'gemini-pro',
        'gemini-flash'
    ]
    
    simple_prompt = "Respond with: {\"test\": \"ok\"}"
    
    working_models = []
    failed_models = []
    
    print("Testing Gemini models availability...")
    print("=" * 50)
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            # Try to create the model
            model = genai.GenerativeModel(model_name)
            print(f"✅ Model {model_name} created")
            
            # Try to generate content
            response = model.generate_content(
                simple_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=100,
                    top_k=1,
                    top_p=0.95
                )
            )
            
            print(f"✅ Response received")
            
            # Check if we got content
            if response and response.candidates and response.candidates[0].content:
                content = response.candidates[0].content
                if content.parts and content.parts[0].text:
                    text = content.parts[0].text.strip()
                    print(f"✅ Got response: {text[:50]}...")
                    working_models.append(model_name)
                else:
                    print(f"❌ No text in response")
                    failed_models.append((model_name, "No text in response"))
            else:
                print(f"❌ No content in response")
                failed_models.append((model_name, "No content in response"))
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error with {model_name}: {error_msg}")
            failed_models.append((model_name, error_msg))
    
    print(f"\n" + "=" * 50)
    print("MODEL AVAILABILITY SUMMARY")
    print("=" * 50)
    
    print(f"\n✅ WORKING MODELS ({len(working_models)}):")
    for model in working_models:
        print(f"  - {model}")
    
    print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
    for model, error in failed_models:
        print(f"  - {model}: {error}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if 'gemini-2.5-pro' in [m for m, _ in failed_models]:
        print("  - gemini-2.5-pro is not working - replace with working alternative")
        if 'gemini-1.5-pro' in working_models:
            print("  - Use gemini-1.5-pro instead of gemini-2.5-pro")
        if 'gemini-2.5-flash' in working_models:
            print("  - Use gemini-2.5-flash instead of gemini-2.5-pro")
    
    if working_models:
        print(f"  - Recommended stable models: {', '.join(working_models[:3])}")

if __name__ == "__main__":
    test_models()
