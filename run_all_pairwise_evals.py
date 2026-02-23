#!/usr/bin/env python3
"""
Run pairwise evaluations for ALL model pairs (not just vs mi102l300b).

Creates unique pairs (A vs B, not B vs A) and groups by model folder.
Uses adaptive batching with timeout handling.
"""

import argparse
import asyncio
import json
import itertools
import os
import random
import re
from pathlib import Path
from asyncio import TimeoutError as AsyncTimeoutError

try:
    import anthropic
    from anthropic import APITimeoutError as AnthropicTimeoutError
except ImportError:
    anthropic = None
    AnthropicTimeoutError = Exception

from openai import AsyncOpenAI
from openai import APITimeoutError as OpenAITimeoutError
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# All models to compare (excluding mi102l300b)
ALL_MODELS = [
    # '70b', '405b', 'claude4opus', 'claude37sonnet', 'claude45haiku', 
    # 'claude45sonnet', 'gemini25flash', 'gemini25pro', 'gpt4o', 
    # 'gpt5', 'gpt41mini', 'gpto3', 'kimik2', 'maverick', 'vanilla'
    'gemini3pro', 'mi102l300b_completion'
]

# Map model names to their completion field names in the JSON
MODEL_FIELD_MAP = {
    '70b': '70b_completion',
    '405b': '405b_completion',
    'claude4opus': 'claude4opus',
    'claude37sonnet': 'claude37sonnet',
    'claude45haiku': 'claude45haiku',
    'claude45sonnet': 'claude45sonnet',
    'gemini25flash': 'gemini25flash',
    'gemini25pro': 'gemini25pro',
    'gpt4o': 'gpt4o',
    'gpt5': 'gpt5',
    'gpt41mini': 'gpt41mini',
    'gpto3': 'gpto3',
    'kimik2': 'kimik2',
    'maverick': 'maverick',
    'vanilla': 'vanilla_completion',
    'gemini3pro': 'gemini3pro',
    'mi102l300b': 'mi102l300b_completion'
}

BASE_DIR = Path(__file__).resolve().parent
COMPLETIONS_FILE = BASE_DIR / "all_model_completions_filled_without_gemini_errors.json"
OUTPUT_FOLDER = BASE_DIR / "pairwise-evals"

# Batch size configuration
INITIAL_BATCH_SIZE = 50
MIN_BATCH_SIZE = 5
BATCH_REDUCTION_FACTOR = 0.5

# Timeout configuration (in seconds)
BASE_TIMEOUT_CLAUDE = 300  # 5 minutes
BASE_TIMEOUT_OPENAI = 600  # 10 minutes (o1 can be slow)
BASE_TIMEOUT_GEMINI = 300  # 5 minutes
MAX_TIMEOUT = 900  # 15 minutes maximum
TIMEOUT_INCREASE_FACTOR = 1.5  # Increase timeout by 50% on retry

def load_jsonl(filepath):
    """Load a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    with open(filepath, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def get_evaluation_system_message():
    """Get the system message for evaluations."""
    return (
        "You are an expert evaluator of emotional support conversations. "
        "You will compare two conversations across multiple criteria and provide "
        "detailed evaluations. Always pick a winner for each criterion and rate "
        "the magnitude of the win using the specified scale."
    )


def build_evaluation_prompt(dialogue_history, final_turn_a, final_turn_b, conv_a_model, conv_b_model):
    """Build the evaluation prompt and set a fresh randomized presentation order."""
    conversation_parts = []
    for i, turn in enumerate(dialogue_history):
        if i % 2 == 0:
            conversation_parts.append(f"Seeker: {turn}")
        else:
            conversation_parts.append(f"Supporter: {turn}")
    
    conversation_a = "\n".join(conversation_parts) + f"\nFinal Response: {final_turn_a}"
    conversation_b = "\n".join(conversation_parts) + f"\nFinal Response: {final_turn_b}"

    if random.choice([True, False]):
        conversation_history_A = conversation_a
        conversation_history_B = conversation_b
        final_a = final_turn_a
        final_b = final_turn_b
        presentation_mapping = {"A0493": conv_a_model, "A0488": conv_b_model}
    else:
        conversation_history_A = conversation_b
        conversation_history_B = conversation_a
        final_a = final_turn_b
        final_b = final_turn_a
        presentation_mapping = {"A0493": conv_b_model, "A0488": conv_a_model}

    scenario_notes = "Final responses from each model:\n"
    scenario_notes += f"A0493 Final Turn: {final_a}\n"
    scenario_notes += f"A0488 Final Turn: {final_b}"
    
    prompt = f"""[RESPONDENT A0493]
{conversation_history_A}

[RESPONDENT A0488]
{conversation_history_B}

You will compare the relative ability of each respondent for these evaluation criteria.
Categories and subcategories to help guide your judgment:

Empathic Responsiveness
• Empathy / Warmth / Affirmation
• Acknowledgement & Validation
• Perspective-taking
• Empowering language (respect for autonomy)
• Gentle Challenge

Emotional Handling & Insight
• Emotional & social reasoning
• Tone and frustration de-escalation
• Handling of complex or mixed emotions

Personalization & Contextual Adaptation
• Message tailoring (content + tone)
• Contextual awareness & memory retention
• Emotion-informed action planning
• Self-disclosure
• Likeability & Engagement

Conversational Fluency & Naturalness
• Human-like Conversation
• Turn-taking, timing, interactive flow
• Response variation & sentence structure
• Groundedness, clarity, conversational fillers when appropriate

Instruction Following & Safety
• Instruction following
• Boundary setting
• Giving Information

Notes on the scenario to assist judging:
{scenario_notes}

Judging Instructions:

For each criterion, you must always pick a winner (A0493 or A0488; never a tie).
For each, rate the magnitude of the win using a plus-based scale: "+" / "++" / "+++" / "++++" / "+++++" Example: "A0493++" = A0493 is somewhat stronger; "A0488++++" = A0488 is far stronger.
Judgments are based only on assistant messages in the dialogue above—ignore the user and any "debrief" content.
Good emotional intelligence is context-sensitive and doesn't always mean being analytical or verbose.
When comparing two safe responses at the same turn, adjust your preference by conversation stage: in early turns (first 1–2 user messages), prefer responses that name and validate feelings and invite more sharing; in later turns (after initial validation), prefer responses that briefly validate and then offer tailored, concrete next steps. Do not let strengths in one criterion (e.g., personalization or action planning) influence scores for a different criterion.
THINK about all the sub categories in each criterion and then pick a winner for the overall criterion. Eg : Empathic responsiveness has sub categories like empathy, acknowledgement & validation, perspective-taking, empowering language (respect for autonomy), gentle challenge. NOTE: it does not include personalization which is a separate criterion etc.
The output must be valid JSON, structured exactly as below (escape all strings correctly):
{{
  "chain_of_thought_reasoning": "detailed chain of thought reasoning about the coming scoring decisions. Include exactly one line of reasoning for each of the 6 categories: (1) empathic_responsiveness, (2) emotional_handling_and_insight, (3) personalization_and_contextual_adaptation, (4) conversational_fluency_and_naturalness, (5) instruction_following_and_safety, and (6) overall_eq",
  "empathic_responsiveness": "winner & disparity rating",
  "emotional_handling_and_insight": "winner & disparity rating",
  "personalization_and_contextual_adaptation": "winner & disparity rating",
  "conversational_fluency_and_naturalness": "winner & disparity rating",
  "instruction_following_and_safety": "winner & disparity rating",
  "overall_eq": "winner & disparity rating"
}}
Judge solely on what is present in the assistant dialogues for each respondent, referencing the above criteria and subcategories. 
DONOT MIX ONE CRITERION WITH ANOTHER. For example personalisation for empathic responsiveness or concrete suggestions for empathetic responsiveness etc."""

    return prompt, presentation_mapping


def parse_evaluation_result(evaluation_text):
    """Parse the evaluation result into a structured format."""
    evaluation_text = evaluation_text.strip()
    
    try:
        result = json.loads(evaluation_text)
        if isinstance(result, dict) and 'overall_eq' in result:
            return result
    except json.JSONDecodeError:
        pass
    
    json_start = evaluation_text.find('{')
    json_end = evaluation_text.rfind('}')
    
    if json_start != -1 and json_end != -1:
        try:
            json_text = evaluation_text[json_start:json_end+1]
            result = json.loads(json_text)
            if isinstance(result, dict) and 'overall_eq' in result:
                return result
        except json.JSONDecodeError:
            pass
    
    return {"error": "Could not parse evaluation", "raw": evaluation_text[:500]}


def replace_model_names_in_evaluation(evaluation_result, presentation_mapping):
    """Replace A0493/A0488 with actual model names in evaluation results."""
    updated_eval = {}
    for key, value in evaluation_result.items():
        if isinstance(value, str):
            updated_value = value
            for identifier, model_name in presentation_mapping.items():
                pattern = f"{identifier}(\\+*)"
                replacement = f"{model_name}\\1"
                updated_value = re.sub(pattern, replacement, updated_value)
            updated_eval[key] = updated_value
        else:
            updated_eval[key] = value
    
    return updated_eval


def get_all_model_pairs(models):
    """Get all unique model pairs (order doesn't matter)."""
    # Use combinations to get unique pairs
    return list(itertools.combinations(sorted(models), 2))


def create_pairwise_file(completions_data, model_a, model_b, output_path):
    """Create a pairwise JSONL file for model_a vs model_b."""
    field_a = MODEL_FIELD_MAP.get(model_a)
    field_b = MODEL_FIELD_MAP.get(model_b)
    
    if not field_a or not field_b:
        print(f"    Warning: Missing field mapping for {model_a} or {model_b}")
        return False
    
    pairwise_data = []
    
    for idx, entry in enumerate(completions_data):
        completion_a = entry.get(field_a, '')
        completion_b = entry.get(field_b, '')
        
        if not completion_a or not completion_b:
            continue
        
        # Skip error entries
        if str(completion_a).startswith('ERROR') or str(completion_b).startswith('ERROR'):
            continue
        
        # Randomize presentation order
        # if random.choice([True, False]):
        #     presentation_order = {"A0493": model_a, "A0488": model_b}
        #     conv_a_model = model_a
        #     conv_a_response = completion_a
        #     conv_b_model = model_b
        #     conv_b_response = completion_b
        # else:
        #     presentation_order = {"A0493": model_b, "A0488": model_a}
        #     conv_a_model = model_b
        #     conv_a_response = completion_b
        #     conv_b_model = model_a
        #     conv_b_response = completion_a
        
        pairwise_entry = {
            "row_index": idx,
            "dialogue_history": entry.get("dialogue_history", []),
            "conversation_a_metadata": {
                "emotion_type": entry.get('emotion_type', ''),
                "problem_type": entry.get('problem_type', ''),
                "final_turn": completion_a,
                "turn": entry.get('turn', 1),
                "model_name": model_a
            },
            "conversation_b_metadata": {
                "emotion_type": entry.get('emotion_type', ''),
                "problem_type": entry.get('problem_type', ''),
                "final_turn": completion_b,
                "turn": entry.get('turn', 1),
                "model_name": model_b
            },
            "evaluation": {}
        }
        pairwise_data.append(pairwise_entry)
    
    save_jsonl(pairwise_data, output_path)
    return len(pairwise_data)


async def evaluate_entry_claude(client, entry, idx, timeout=None):
    """Evaluate a single entry using async Claude API."""
    if timeout is None:
        timeout = BASE_TIMEOUT_CLAUDE
    is_timeout = False
    try:
        existing_eval = entry.get('evaluation', {})
        # Skip if already successfully evaluated
        if existing_eval and not existing_eval.get('error') and existing_eval.get('overall_eq'):
            return (entry, idx, False, False)
        
        dialogue_history = entry.get('dialogue_history', [])
        final_turn_a = entry['conversation_a_metadata']['final_turn']
        final_turn_b = entry['conversation_b_metadata']['final_turn']
        model_a = entry['conversation_a_metadata']['model_name']
        model_b = entry['conversation_b_metadata']['model_name']
        
        eval_prompt, presentation_mapping = build_evaluation_prompt(
            dialogue_history, final_turn_a, final_turn_b, model_a, model_b
        )
        eval_system_msg = get_evaluation_system_message()
        
        response = await asyncio.wait_for(
            client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0,
                system=eval_system_msg,
                messages=[{"role": "user", "content": eval_prompt}]
            ),
            timeout=timeout
        )
        
        evaluation_text = response.content[0].text.strip()
        evaluation_dict = parse_evaluation_result(evaluation_text)
        evaluation_dict = replace_model_names_in_evaluation(evaluation_dict, presentation_mapping)
        
        entry['presentation_order'] = presentation_mapping
        entry['evaluation'] = evaluation_dict
        
        return (entry, idx, True, False)
        
    except (AsyncTimeoutError, AnthropicTimeoutError) as e:
        entry['evaluation'] = {"error": f"timeout: {str(e)}"}
        return (entry, idx, True, True)
        
    except Exception as e:
        error_str = str(e).lower()
        is_timeout = 'timeout' in error_str or 'timed out' in error_str
        entry['evaluation'] = {"error": str(e)}
        return (entry, idx, True, is_timeout)


async def evaluate_entry_openai(client, entry, idx, timeout=None):
    """Evaluate a single entry using async OpenAI o1 API."""
    if timeout is None:
        timeout = BASE_TIMEOUT_OPENAI
    is_timeout = False
    try:
        existing_eval = entry.get('evaluation', {})
        # Skip if already successfully evaluated
        if existing_eval and not existing_eval.get('error') and existing_eval.get('overall_eq'):
            return (entry, idx, False, False)
        
        dialogue_history = entry.get('dialogue_history', [])
        final_turn_a = entry['conversation_a_metadata']['final_turn']
        final_turn_b = entry['conversation_b_metadata']['final_turn']
        model_a = entry['conversation_a_metadata']['model_name']
        model_b = entry['conversation_b_metadata']['model_name']
        
        eval_prompt, presentation_mapping = build_evaluation_prompt(
            dialogue_history, final_turn_a, final_turn_b, model_a, model_b
        )
        eval_system_msg = get_evaluation_system_message()
        
        full_prompt = f"{eval_system_msg}\n\n{eval_prompt}"
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="o1",
                # temperature=0,
                messages=[{"role": "user", "content": full_prompt}]
            ),
            timeout=timeout
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        evaluation_dict = parse_evaluation_result(evaluation_text)
        evaluation_dict = replace_model_names_in_evaluation(evaluation_dict, presentation_mapping)
        
        entry['presentation_order'] = presentation_mapping
        entry['evaluation'] = evaluation_dict
        
        return (entry, idx, True, False)
        
    except (AsyncTimeoutError, OpenAITimeoutError) as e:
        entry['evaluation'] = {"error": f"timeout: {str(e)}"}
        return (entry, idx, True, True)
        
    except Exception as e:
        error_str = str(e).lower()
        is_timeout = 'timeout' in error_str or 'timed out' in error_str
        entry['evaluation'] = {"error": str(e)}
        return (entry, idx, True, is_timeout)


async def evaluate_entry_gemini(client, entry, idx, timeout=None):
    """Evaluate a single entry using Gemini 2.5 Flash."""
    if timeout is None:
        timeout = BASE_TIMEOUT_GEMINI
    is_timeout = False
    try:
        existing_eval = entry.get('evaluation', {})
        # Skip if already successfully evaluated
        if existing_eval and not existing_eval.get('error') and existing_eval.get('overall_eq'):
            return (entry, idx, False, False)

        if genai is None:
            raise ImportError("google-generativeai package is required for Gemini evaluations")

        dialogue_history = entry.get('dialogue_history', [])
        final_turn_a = entry['conversation_a_metadata']['final_turn']
        final_turn_b = entry['conversation_b_metadata']['final_turn']
        model_a = entry['conversation_a_metadata']['model_name']
        model_b = entry['conversation_b_metadata']['model_name']

        eval_prompt, presentation_mapping = build_evaluation_prompt(
            dialogue_history, final_turn_a, final_turn_b, model_a, model_b
        )
        eval_system_msg = get_evaluation_system_message()
        full_prompt = f"{eval_system_msg}\n\n{eval_prompt}"

        def _call_gemini():
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required for Gemini evaluations")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                )
            )
            if hasattr(response, "text") and response.text:
                return response.text
            if response.candidates:
                for candidate in response.candidates:
                    content = getattr(candidate, "content", None)
                    if content and getattr(content, "parts", None):
                        text = content.parts[0].text
                        if text:
                            return text
            raise ValueError("Gemini response missing text")

        evaluation_text = await asyncio.wait_for(
            asyncio.to_thread(_call_gemini),
            timeout=timeout
        )
        evaluation_dict = parse_evaluation_result(evaluation_text.strip())
        evaluation_dict = replace_model_names_in_evaluation(evaluation_dict, presentation_mapping)

        entry['presentation_order'] = presentation_mapping
        entry['evaluation'] = evaluation_dict

        return (entry, idx, True, False)

    except Exception as e:
        error_str = str(e).lower()
        is_timeout = 'timeout' in error_str or 'timed out' in error_str
        entry['evaluation'] = {"error": str(e)}
        return (entry, idx, True, is_timeout)


async def process_batch_with_retry(client, eval_func, data, indices, batch_size, filepath):
    """Process a batch with adaptive retry on timeouts."""
    current_batch_size = batch_size
    remaining_indices = list(indices)
    total_evaluated = 0
    timeout_count = 0
    
    # Track timeout retry counts per entry (for progressive timeout increases)
    timeout_retry_counts = {idx: 0 for idx in indices}
    
    # Determine base timeout based on evaluator type
    if 'claude' in str(eval_func.__name__).lower():
        base_timeout = BASE_TIMEOUT_CLAUDE
    elif 'openai' in str(eval_func.__name__).lower():
        base_timeout = BASE_TIMEOUT_OPENAI
    elif 'gemini' in str(eval_func.__name__).lower():
        base_timeout = BASE_TIMEOUT_GEMINI
    else:
        base_timeout = BASE_TIMEOUT_CLAUDE  # default
    
    while remaining_indices and current_batch_size >= MIN_BATCH_SIZE:
        batch_indices = remaining_indices[:current_batch_size]
        batch_entries = [(data[i], i) for i in batch_indices]
        
        # Calculate timeout for each entry (progressive increase for retries)
        tasks = []
        for entry, idx in batch_entries:
            retry_count = timeout_retry_counts.get(idx, 0)
            # Increase timeout progressively, but cap at MAX_TIMEOUT
            entry_timeout = min(
                int(base_timeout * (TIMEOUT_INCREASE_FACTOR ** retry_count)),
                MAX_TIMEOUT
            )
            tasks.append(eval_func(client, entry, idx, timeout=entry_timeout))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_indices = []
        batch_timeouts = 0
        
        for i, result in enumerate(results):
            idx = batch_indices[i]
            
            if isinstance(result, Exception):
                data[idx]['evaluation'] = {"error": str(result)}
                if 'timeout' in str(result).lower():
                    batch_timeouts += 1
                    timeout_retry_counts[idx] = timeout_retry_counts.get(idx, 0) + 1
            else:
                updated_entry, entry_idx, was_evaluated, was_timeout = result
                data[entry_idx] = updated_entry
                
                if was_timeout:
                    batch_timeouts += 1
                    timeout_retry_counts[idx] = timeout_retry_counts.get(idx, 0) + 1
                elif was_evaluated:
                    total_evaluated += 1
                    successful_indices.append(idx)
                    # Clear retry count on success
                    timeout_retry_counts.pop(idx, None)
                else:
                    successful_indices.append(idx)
                    # Clear retry count on success
                    timeout_retry_counts.pop(idx, None)
        
        # Keep indices that weren't successful (including timeouts for retry)
        remaining_indices = [i for i in remaining_indices if i not in successful_indices]
        
        # Remove non-timeout failures from retry list
        failed_non_timeout = []
        for idx in batch_indices:
            if idx not in successful_indices:
                eval_result = data[idx].get('evaluation', {})
                if eval_result.get('error') and 'timeout' not in eval_result.get('error', '').lower():
                    failed_non_timeout.append(idx)
        remaining_indices = [i for i in remaining_indices if i not in failed_non_timeout]
        
        save_jsonl(data, filepath)
        
        if batch_timeouts > 0:
            timeout_count += batch_timeouts
            old_size = current_batch_size
            current_batch_size = max(MIN_BATCH_SIZE, int(current_batch_size * BATCH_REDUCTION_FACTOR))
            avg_retry = sum(timeout_retry_counts.values()) / len(timeout_retry_counts) if timeout_retry_counts else 0
            print(f"      ⚡ {batch_timeouts} timeouts! Batch size: {old_size} → {current_batch_size} (avg retries: {avg_retry:.1f})")
            await asyncio.sleep(2)
    
    return total_evaluated, timeout_count, current_batch_size


async def process_file(filepath, batch_size):
    """Process a single pairwise file."""
    filepath = Path(filepath)
    data = load_jsonl(filepath)
    total_entries = len(data)
    
    needs_eval = [i for i, e in enumerate(data) 
                  if not e.get('evaluation') or e.get('evaluation', {}).get('error') or not e.get('evaluation', {}).get('overall_eq')]
    
    if not needs_eval:
        return 0, 0, len(data), total_entries, batch_size  # Return current batch size
    
    if '-claude' in filepath.name.lower():
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        eval_func = evaluate_entry_claude
    elif '-gemini' in filepath.name.lower():
        client = None
        eval_func = evaluate_entry_gemini
    else:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        eval_func = evaluate_entry_openai
    
    total_evaluated, timeout_count, final_batch_size = await process_batch_with_retry(
        client, eval_func, data, needs_eval, batch_size, filepath
    )
    
    save_jsonl(data, filepath)
    
    final_complete = sum(1 for e in data if e.get('evaluation') and not e.get('evaluation', {}).get('error') and e.get('evaluation', {}).get('overall_eq'))
    
    return total_evaluated, timeout_count, final_complete, total_entries, final_batch_size


async def main_async(args):
    """Async main function."""
    output_folder = Path(args.output_folder)
    completions_file = Path(args.completions_file)

    print("=" * 80)
    print("All Model Pairwise Evaluations")
    print(f"Output folder: {output_folder}")
    print(f"Initial batch size: {args.batch_size}")
    print("=" * 80)
    
    # Load completions data
    print(f"\nLoading completions from {completions_file}...")
    with open(completions_file, 'r') as f:
        completions_data = json.load(f)
    print(f"Loaded {len(completions_data)} entries")
    
    # Get all model pairs
    all_pairs = get_all_model_pairs(ALL_MODELS)
    print(f"\nTotal unique model pairs: {len(all_pairs)}")
    print(f"With 2 evaluators each: {len(all_pairs) * 2} files to create/process")
    
    # Create folder structure and pairwise files
    print(f"\nCreating pairwise files...")
    
    files_to_process = []
    
    for model_a, model_b in all_pairs:
        # Create folder for model_a (first model alphabetically)
        model_folder = output_folder / model_a
        model_folder.mkdir(parents=True, exist_ok=True)
        
        # Create files for both evaluators
        for evaluator in ['claude', 'openai', 'gemini']:
            filename = f"{model_a}-{model_b}-{evaluator}.json"
            filepath = model_folder / filename
            
            if not filepath.exists():
                count = create_pairwise_file(completions_data, model_a, model_b, filepath)
                if count:
                    print(f"  Created {filename} ({count} entries)")
            
            files_to_process.append(filepath)
    
    print(f"\nTotal files to process: {len(files_to_process)}")
    
    # Separate by evaluator
    claude_files = [Path(f) for f in files_to_process if '-claude.json' in str(f)]
    openai_files = [Path(f) for f in files_to_process if '-openai.json' in str(f)]
    gemini_files = [Path(f) for f in files_to_process if '-gemini.json' in str(f)]
    
    print(f"  Claude files: {len(claude_files)}")
    print(f"  OpenAI files: {len(openai_files)}")
    print(f"  Gemini files: {len(gemini_files)}")
    
    # Process files
    total_evaluated = 0
    total_timeouts = 0
    
    async def process_files(files, batch_size, evaluator_name):
        nonlocal total_evaluated, total_timeouts
        current_batch_size = batch_size

        for i, filepath in enumerate(files):
            filepath = Path(filepath)
            filename = filepath.name
            parent = filepath.parent.name
            print(f"\n  [{evaluator_name}] ({i+1}/{len(files)}) {parent}/{filename}")
            
            evaluated, timeouts, complete, total_entries, final_batch_size = await process_file(filepath, batch_size)
            total_evaluated += evaluated
            total_timeouts += timeouts
            current_batch_size = final_batch_size  # Update batch size for next file
            
            if evaluated > 0:
                print(f"      ✓ Evaluated {evaluated}, complete: {complete}/{total_entries}")
            else:
                print(f"      ✅ Already complete ({complete}/{total_entries})")
    
    # Run both evaluators concurrently
    await asyncio.gather(
        process_files(claude_files, args.batch_size, "Claude"),
        process_files(openai_files, args.batch_size, "OpenAI"),
        process_files(gemini_files, args.batch_size, "Gemini")
    )
    
    print(f"\n{'='*80}")
    print(f"✅ All evaluations complete!")
    print(f"   Total evaluated: {total_evaluated}")
    print(f"   Total timeouts: {total_timeouts}")
    print(f"   Output folder: {output_folder}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Run all model pairwise evaluations')
    parser.add_argument('--batch-size', type=int, default=INITIAL_BATCH_SIZE, 
                        help=f'Initial batch size (default: {INITIAL_BATCH_SIZE})')
    parser.add_argument('--models', nargs='+', help='Specific models to include (default: all)')
    parser.add_argument('--completions-file', type=Path, default=COMPLETIONS_FILE,
                        help=f'Path to completions JSON (default: {COMPLETIONS_FILE})')
    parser.add_argument('--output-folder', type=Path, default=OUTPUT_FOLDER,
                        help=f'Folder to write pairwise evals (default: {OUTPUT_FOLDER})')
    args = parser.parse_args()
    
    if args.models:
        global ALL_MODELS
        ALL_MODELS = [m for m in ALL_MODELS if m in args.models]
        print(f"Using subset of models: {ALL_MODELS}")
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
