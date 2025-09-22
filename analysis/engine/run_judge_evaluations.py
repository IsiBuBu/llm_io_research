
# analysis/run_judge_evaluations.py

import pandas as pd
import numpy as np
import logging
import json
import asyncio
import random
from pathlib import Path
import sys

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import load_config
from agents import create_agent, BaseLLMAgent

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
logger = logging.getLogger("JudgeEvaluator")

CONFIG = load_config()
JUDGE_CONFIG = CONFIG.get('judge_config', {})
JUDGE_MODEL_NAME = JUDGE_CONFIG.get('judge_model', 'gemini-2.5-flash')
SAMPLES_PER_VARIATION = JUDGE_CONFIG.get('samples_per_variation_per_game', 50)
REPLICATIONS = JUDGE_CONFIG.get('reliability_replications', 10)
BASE_SEED = JUDGE_CONFIG.get('judge_random_seed', 123)

RESULTS_DIR = Path(CONFIG.get('output', {}).get('results_dir', 'results'))
ANALYSIS_DIR = Path(CONFIG.get('output', {}).get('analysis_dir', 'analysis'))
PROMPTS_DIR = Path("prompts")

# --- Helper Functions ---

def load_all_thought_summaries():
    """
    Loads one representative thought summary per simulation from the raw experiment results.
    This version correctly filters for thinking models and randomly samples one strategic
    round from each dynamic game simulation.
    """
    records = []
    # Set seed for reproducibility of the random choice
    random.seed(BASE_SEED)

    for file_path in RESULTS_DIR.glob("*/*/*_competition_result*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for sim in data.get('simulation_results', []):
                is_dynamic = 'rounds' in sim.get('game_data', {})
                challenger_model = sim['challenger_model']

                if 'thinking_medium' not in challenger_model:
                    continue

                if is_dynamic:
                    # --- UPDATED LOGIC: Randomly sample one thinking round per simulation ---
                    # 1. Find all rounds where a strategic decision with thoughts was made
                    thinking_rounds = []
                    for round_data in sim['game_data']['rounds']:
                        challenger_meta = round_data.get('llm_metadata', {}).get('challenger', {})
                        if challenger_meta and challenger_meta.get('thoughts'):
                            thinking_rounds.append(round_data)
                    
                    # 2. If any such rounds exist, randomly select one
                    if thinking_rounds:
                        selected_round = random.choice(thinking_rounds)
                        records.append({
                            'game': sim['game_name'],
                            'model': challenger_model,
                            'condition': sim['condition_name'],
                            'simulation_id': sim['simulation_id'],
                            'round': selected_round.get('period'),
                            'thought_summary': selected_round.get('llm_metadata', {}).get('challenger', {}).get('thoughts'),
                            'initial_prompt': selected_round.get('initial_prompt_for_challenger', '')
                        })

                else: # Static game logic remains the same
                    challenger_meta = sim.get('game_data', {}).get('llm_metadata', {}).get('challenger', {})
                    if challenger_meta.get('thoughts'):
                        records.append({
                            'game': sim['game_name'],
                            'model': challenger_model,
                            'condition': sim['condition_name'],
                            'simulation_id': sim['simulation_id'],
                            'round': 1,
                            'thought_summary': challenger_meta['thoughts'],
                            'initial_prompt': sim['game_data'].get('initial_prompt_for_challenger', '')
                        })

    return pd.DataFrame(records)


def sample_summaries(df: pd.DataFrame):
    """Samples a balanced set of summaries for each game."""
    samples = []
    for game in df['game'].unique():
        game_df = df[df['game'] == game].copy()
        game_df['structural_variation'] = np.where(game_df['condition'].str.contains('5_players|more_players', regex=True), '5-Player', '3-Player')
        
        for variation in ['3-Player', '5-Player']:
            stratum_df = game_df[game_df['structural_variation'] == variation]
            sample_size = min(SAMPLES_PER_VARIATION, len(stratum_df))
            if sample_size > 0:
                # Use a different seed for the final sampling to ensure it's independent of the round choice
                samples.append(stratum_df.sample(n=sample_size, random_state=BASE_SEED + 1))
    
    if not samples:
        return pd.DataFrame()
    return pd.concat(samples, ignore_index=True)

async def get_judgement(judge_agent: BaseLLMAgent, judge_prompt: str, replication_id: int):
    """Calls the judge agent with a unique seed and safely parses the JSON response."""
    unique_seed = BASE_SEED + replication_id
    try:
        response = await judge_agent.get_response(judge_prompt, call_id=f"judge-rep-{replication_id}", seed=unique_seed)
        if response.success:
            content = response.content.strip().replace('```json', '').replace('```', '')
            return json.loads(content)
        else:
            logger.error(f"Judge API call failed: {response.error}")
            return None
    except Exception as e:
        logger.error(f"Failed to get or parse judgement for replication {replication_id}: {e}")
        return None

async def main():
    """Main script to run the LLM as Judge evaluation pipeline."""
    logger.info("Starting LLM as Judge evaluation pipeline...")
    
    all_summaries = load_all_thought_summaries()
    if all_summaries.empty:
        logger.error("No thought summaries found. Ensure experiments were run with 'thinking_medium' models.")
        return
        
    sampled_df = sample_summaries(all_summaries)
    logger.info(f"Sampled {len(sampled_df)} thought summaries for evaluation.")

    judge_agent = create_agent(JUDGE_MODEL_NAME, player_id='judge', agent_type='judge')

    all_evaluations = []
    for index, row in sampled_df.iterrows():
        game_name = row['game']
        
        try:
            with open(PROMPTS_DIR / "judge_prompts" / f"judge_{game_name}.md", 'r') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            logger.error(f"Judge prompt for {game_name} not found. Skipping.")
            continue
            
        formatted_prompt = prompt_template.format(
            initial_llm_prompt=row['initial_prompt'],
            thought_summary_text=row['thought_summary']
        )
        
        logger.info(f"Evaluating summary {index + 1}/{len(sampled_df)} for game {game_name}...")
        
        tasks = [get_judgement(judge_agent, formatted_prompt, i) for i in range(REPLICATIONS)]
        judgements = await asyncio.gather(*tasks)
        
        for rep_id, result in enumerate(judgements):
            if result and 'evaluations' in result:
                for evaluation in result['evaluations']:
                    all_evaluations.append({
                        'sample_id': f"{game_name}_{index}",
                        'replication_id': rep_id,
                        'game': game_name,
                        'model': row['model'],
                        'condition': row['condition'],
                        'metric': evaluation.get('metric'),
                        'alignment_score': evaluation.get('alignment_score'),
                        'justification': evaluation.get('justification')
                    })

    output_df = pd.DataFrame(all_evaluations)
    output_path = ANALYSIS_DIR / "judge_evaluations.csv"
    output_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(output_df)} judgements to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())