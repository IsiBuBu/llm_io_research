import sys
import re
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from config.config import get_all_game_configs, load_config, get_prompt_variables
from games import create_game

# --- Part 1: Functions to Display Prompt Examples ---

def show_game_prompt_examples():
    """Generates and prints a sample filled prompt for one condition of each game."""
    print("=" * 80)
    print("📜 DISPLAYING FILLED GAME PROMPT EXAMPLES 📜")
    print("=" * 80)
    
    example_configs = {
        "salop": "few_players",
        "green_porter": "more_players",
        "spulber": "few_players",
        "athey_bagwell": "few_players"
    }

    for game_name, condition_name in example_configs.items():
        print(f"\n--- Example for Game: {game_name.title()} (Condition: {condition_name}) ---")
        try:
            all_configs = get_all_game_configs(game_name)
            target_config = next((c for c in all_configs if c.condition_name == condition_name), None)

            if not target_config:
                print(f"❌ Error: Could not find configuration.")
                continue

            game = create_game(game_name)
            game_state = game.initialize_game_state(target_config, simulation_id=0)

            if game_name in ["green_porter", "athey_bagwell"]:
                time_horizon = target_config.constants.get('time_horizon', 3)
                history_length = time_horizon - 1
                game_state['current_period'] = time_horizon
                
                if game_name == "green_porter":
                    game_state['market_state'] = 'Reversionary'
                    game_state['price_history'] = [65.3, 62.1, 58.9, 54.2][:history_length]

                elif game_name == "athey_bagwell":
                    player_ids = list(game_state['report_history'].keys())
                    sample_reports = {
                        'challenger': ['low', 'low'], 'defender_1': ['low', 'high'], 'defender_2': ['high', 'low']
                    }
                    for pid in player_ids:
                        game_state['report_history'][pid] = sample_reports.get(pid, ['high', 'high'])[:history_length]

            prompt = game.generate_player_prompt("challenger", game_state, target_config)
            
            print(prompt)
            print("--- End of Example ---\n")

        except Exception as e:
            print(f"An error occurred while generating example for {game_name}: {e}")


def show_judge_prompt_examples():
    """Generates and prints a sample filled judge prompt for each game."""
    print("\n" + "=" * 80)
    print("🕵️  DISPLAYING FILLED JUDGE PROMPT EXAMPLES 🕵️")
    print("=" * 80)
    
    config = load_config()
    all_game_names = config.get('game_configs', {}).keys()

    for game_name in all_game_names:
        print(f"\n--- Example for Judge: {game_name.title()} ---")
        try:
            game_configs = get_all_game_configs(game_name)
            target_config = game_configs[0]
            game = create_game(game_name)
            game_state = game.initialize_game_state(target_config, simulation_id=0)

            # --- FIXED LOGIC: Add detailed sample history for dynamic games ---
            if game_name in ["green_porter", "athey_bagwell"]:
                time_horizon = target_config.constants.get('time_horizon', 3)
                history_length = time_horizon - 1
                game_state['current_period'] = time_horizon
                if game_name == "green_porter":
                    game_state['market_state'] = 'Reversionary'
                    game_state['price_history'] = [65.3, 62.1][:history_length]
                elif game_name == "athey_bagwell":
                    player_ids = list(game_state['report_history'].keys())
                    sample_reports = {'challenger': ['low'], 'defender_1':['high']}
                    for pid in player_ids:
                         game_state['report_history'][pid] = sample_reports.get(pid, ['high'])[:history_length]


            initial_llm_prompt = game.generate_player_prompt("challenger", game_state, target_config)

            thought_summary_text = (
                f"Analyzing the {game_name.title()} game. The core tension seems to be between short-term gains and long-term stability. "
                "My objective is to maximize profit, considering the actions of my competitors. I need to evaluate the payoffs associated with different choices. "
                "Based on the rules and current state, I believe the best course of action is to balance aggression with caution to secure a favorable outcome."
            )

            prompts_dir = Path("prompts")
            judge_prompt_path = prompts_dir / "judge_prompts" / f"judge_{game_name}.md"
            with open(judge_prompt_path, 'r') as f:
                prompt_template = f.read()
            
            final_judge_prompt = prompt_template.format(
                initial_llm_prompt=initial_llm_prompt,
                thought_summary_text=thought_summary_text
            )
            print(final_judge_prompt)
            print("--- End of Example ---\n")

        except KeyError as e:
            print(f"❌ An error occurred: A required key was missing for the prompt template: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while generating judge example for {game_name}: {e}")


# --- Part 2: Functions to Verify Prompts ---
def find_placeholders(template_text: str) -> set:
    """Finds all {placeholder} variables in a string."""
    return set(re.findall(r'\{(\w+)\}', template_text))

def test_game_prompts():
    """Tests all game prompts to ensure all variables from config.json are correctly parsed and inserted."""
    print("\n" + "=" * 80)
    print("🔬 RUNNING VERIFICATION FOR ALL GAME PROMPTS 🔬")
    print("=" * 80)
    all_passed = True
    config = load_config()
    all_game_names = config.get('game_configs', {}).keys()

    for game_name in all_game_names:
        print(f"\n--- Verifying Game: {game_name.title()} ---")
        game = create_game(game_name)
        all_configs = get_all_game_configs(game_name)
        for game_config in all_configs:
            print(f"  - Condition: {game_config.condition_name:<30}", end="")
            game_state = game.initialize_game_state(game_config, 0)
            
            dynamic_kwargs = {}
            if game_name in ["green_porter", "athey_bagwell"]:
                time_horizon = game_config.constants.get('time_horizon', 3)
                history_length = time_horizon - 1
                game_state['current_period'] = time_horizon
                if game_name == "green_porter":
                    game_state['market_state'] = 'Reversionary'
                    game_state['price_history'] = [65.3, 62.1][:history_length]
                    dynamic_kwargs.update({
                        'current_round': game_state['current_period'],
                        'current_market_state': game_state['market_state'],
                        'price_history': game_state['price_history']
                    })
                elif game_name == "athey_bagwell":
                    player_ids = list(game_state['report_history'].keys())
                    sample_reports = { 'challenger': ['low', 'low'], 'defender_1': ['low', 'high'], 'defender_2': ['high', 'low'] }
                    for pid in player_ids:
                        game_state['report_history'][pid] = sample_reports.get(pid, ['high', 'high'])[:history_length]
                    your_history = game_state['report_history'].get('challenger', [])
                    other_history = {pid: r for pid, r in game_state['report_history'].items() if pid != 'challenger'}
                    your_history_str = ", ".join(your_history) or "N/A"
                    other_history_lines = [f"Period {i+1}: " + ", ".join([f"{pid}: {reports[i]}" for pid, reports in other_history.items() if i < len(reports)]) for i in range(history_length)]
                    other_history_str = "; ".join(other_history_lines) or "No other player reports yet."
                    dynamic_kwargs.update({
                        'current_round': game_state['current_period'],
                        'your_cost_type': game_state['cost_sequences']['challenger'][history_length],
                        'your_reports_history_detailed': your_history_str,
                        'all_other_reports_history_detailed': other_history_str
                    })

            all_available_vars = get_prompt_variables(game_config, "challenger", **dynamic_kwargs)
            prompt = game.generate_player_prompt("challenger", game_state, game_config)
            placeholders_in_template = find_placeholders(game.prompt_template)
            
            missing_vars = []
            for placeholder in placeholders_in_template:
                if placeholder in all_available_vars:
                    value_to_check = all_available_vars[placeholder]
                    if str(value_to_check) not in prompt:
                        missing_vars.append(f"Value for {{{placeholder}}}")
                else:
                    missing_vars.append(f"Variable {{{placeholder}}} not found")
            if not missing_vars:
                print("✅ PASS")
            else:
                print(f"❌ FAIL - Missing: {', '.join(missing_vars)}")
                all_passed = False
    return all_passed

def test_judge_prompts():
    """Tests all judge prompts to ensure placeholders are present."""
    print("\n" + "=" * 80)
    print("🕵️  RUNNING VERIFICATION FOR ALL JUDGE PROMPTS 🕵️")
    print("=" * 80)
    all_passed = True
    prompts_dir = Path("prompts") / "judge_prompts"
    for prompt_file in prompts_dir.glob("judge_*.md"):
        game_name = prompt_file.stem.replace("judge_", "")
        print(f"  - Verifying Judge Prompt: {game_name.title():<20}", end="")
        with open(prompt_file, 'r') as f:
            template = f.read()
        placeholders = find_placeholders(template)
        expected_placeholders = {'initial_llm_prompt', 'thought_summary_text'}
        if placeholders == expected_placeholders:
            print("✅ PASS")
        else:
            missing = expected_placeholders - placeholders
            extra = placeholders - expected_placeholders
            print(f"❌ FAIL - Missing: {missing or 'None'}, Extra: {extra or 'None'}")
            all_passed = False
    return all_passed

if __name__ == '__main__':
    # Show examples first
    show_game_prompt_examples()
    show_judge_prompt_examples()
    
    # Then run the verification tests
    games_passed = test_game_prompts()
    judges_passed = test_judge_prompts()
    
    print("\n" + "=" * 80)
    if games_passed and judges_passed:
        print("🎉 All prompts verified successfully! 🎉")
    else:
        print("🔥 Verification failed. Please check the errors above. 🔥")
    print("=" * 80)