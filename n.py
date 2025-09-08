#!/usr/bin/env python3
"""
Script to test different thinking configurations in Gemini 2.5 Flash
Tests: thinking summary, thinking off, dynamic thinking, and thinking budget with normal budget
"""

from google import genai
from google.genai import types
import os

# Setup API key
API_KEY = "AIzaSyANLSAbDwH2UjLEjoeW7vZt-uYAeg9enZ0"  # Replace with your actual API key
MODEL_ID = "gemini-2.5-flash"

# Configure API
client = genai.Client(api_key=API_KEY)

# Test prompt that benefits from reasoning
test_prompt = "how are you?"

# Complex reasoning prompt for better testing
complex_prompt = """
You need to solve this puzzle: A farmer has 17 sheep, and all but 9 die. How many sheep are left?
Additionally, if each remaining sheep produces 3 pounds of wool per month, and wool sells for $8 per pound,
how much money does the farmer make from wool in 6 months? Show your reasoning clearly.
"""

def test_thinking_configurations():
    print("🧠 Comparing Dynamic vs Fixed Budget Thinking with Same Prompt\n")
    print("=" * 80)
    
    # Test prompt that benefits from reasoning
    test_prompt = """
    A chess tournament has 16 players. In the first round, players are paired randomly. 
    If a player wins their first game, they advance to the second round. If they lose, they're eliminated.
    In the second round, the remaining players are again paired randomly.
    
    Question: What is the maximum possible number of players who could advance to a potential third round,
    and what specific conditions would need to be met for this to happen? Show your reasoning step by step.
    """
    
    print(f"🎯 Test Prompt:")
    print(f"{test_prompt}")
    print("\n" + "=" * 80)
    
    # Test 1: Dynamic Thinking Mode with Summary
    print("\n1️⃣ DYNAMIC THINKING MODE (Model decides budget)")
    print("-" * 60)
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=test_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,  # Dynamic thinking
                    include_thoughts=True
                )
            )
        )
        
        print("📊 DYNAMIC THINKING RESULTS:")
        thinking_summary = ""
        answer_text = ""
        
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking_summary = part.text
            else:
                answer_text = part.text
        
        # Display thinking summary
        if thinking_summary:
            print("\n🧠 THINKING SUMMARY:")
            print("-" * 40)
            print(thinking_summary)
        else:
            print("\n� No thinking summary available")
        
        # Display answer
        print("\n📝 FINAL ANSWER:")
        print("-" * 40)
        print(answer_text)
        
        # Display token usage
        if hasattr(response, 'usage_metadata'):
            total_tokens = response.usage_metadata.total_token_count
            thinking_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            print(f"\n💾 TOKEN USAGE (Dynamic):")
            print(f"   • Total tokens: {total_tokens}")
            print(f"   • Thinking tokens: {thinking_tokens}")
            print(f"   • Output tokens: {output_tokens}")
                
    except Exception as e:
        print(f"❌ Error in Dynamic Thinking: {e}")
    
    print("\n" + "=" * 80)
    
    # Test 2: Fixed Budget Thinking Mode with Summary
    print("\n2️⃣ FIXED BUDGET THINKING MODE (2048 tokens)")
    print("-" * 60)
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=test_prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=2048,  # Fixed thinking budget
                    include_thoughts=True
                )
            )
        )
        
        print("📊 FIXED BUDGET THINKING RESULTS:")
        thinking_summary = ""
        answer_text = ""
        
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking_summary = part.text
            else:
                answer_text = part.text
        
        # Display thinking summary
        if thinking_summary:
            print("\n🧠 THINKING SUMMARY:")
            print("-" * 40)
            print(thinking_summary)
        else:
            print("\n🚫 No thinking summary available")
        
        # Display answer
        print("\n📝 FINAL ANSWER:")
        print("-" * 40)
        print(answer_text)
        
        # Display token usage
        if hasattr(response, 'usage_metadata'):
            total_tokens = response.usage_metadata.total_token_count
            thinking_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            print(f"\n💾 TOKEN USAGE (Fixed Budget 2048):")
            print(f"   • Total tokens: {total_tokens}")
            print(f"   • Thinking tokens: {thinking_tokens} / 2048 (budget)")
            print(f"   • Output tokens: {output_tokens}")
            print(f"   • Budget utilization: {(thinking_tokens/2048)*100:.1f}%")
                
    except Exception as e:
        print(f"❌ Error in Fixed Budget Thinking: {e}")
    
    print("\n" + "=" * 80)
    
    # Comparison Summary
    print("\n📈 COMPARISON SUMMARY")
    print("-" * 60)
    print("Both modes tested with identical complex chess tournament prompt.")
    print("Key differences:")
    print("• Dynamic: Model chooses optimal thinking token allocation")
    print("• Fixed Budget: Constrained to exactly 2048 thinking tokens")
    print("• Both modes provide thinking summaries for transparency")
    print("\n🎯 This comparison shows how budget constraints affect reasoning depth.")

if __name__ == "__main__":
    test_thinking_configurations()
    print("\n✅ Thinking comparison test complete!")
    print("📊 Results show thinking token usage and summaries for both dynamic and fixed budget modes")