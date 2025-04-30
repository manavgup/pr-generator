import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import os

def safe_extract_json(raw_lines: List[str]) -> dict:
    """Extract first valid JSON object from list of lines."""
    raw_text = "\n".join(raw_lines)
    json_start = raw_text.find('{')
    if json_start == -1:
        return {}
    try:
        return json.loads(raw_text[json_start:])
    except json.JSONDecodeError:
        return {}

def slugify_filename(text: str) -> str:
    """Create a filesystem-safe filename from task hint."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text)[:80]

def parse_log_file_v2(log_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse CrewAI log file into structured LLM call blocks and token usage."""
    lines = Path(log_path).read_text().splitlines()
    blocks = []
    token_usage_info = []
    current_block = None
    inside_request = inside_response = False

    lite_cost_buffer = []

    for idx, line in enumerate(lines):
        if "Request to litellm:" in line:
            if current_block:
                blocks.append(current_block)
            current_block = {
                'task_hint': '',
                'litellm_request': '',
                'raw_response': '',
                'thought': '',
                'action': '',
                'final_answer': '',
                'parsed_usage': {},
                'parsing_error': False,
                'api_errors': []  # New field to capture API errors
            }
            inside_request = True
            inside_response = False
            current_block['litellm_request'] += line + '\n'
        elif "RAW RESPONSE:" in line and current_block:
            inside_response = True
            inside_request = False
            current_block['raw_response'] += line + '\n'
        elif "APIStatusError" in line and current_block:  # Capture API errors
            current_block['api_errors'].append(line.strip())
        else:
            if inside_request and current_block:
                current_block['litellm_request'] += line + '\n'
            if inside_response and current_block:
                current_block['raw_response'] += line + '\n'

    if current_block:
        blocks.append(current_block)

    # Extract information from each block
    for idx, block in enumerate(blocks):
        task_match = re.search(r"Current Task: (.*?)\n", block['litellm_request'], re.DOTALL)
        if task_match:
            block['task_hint'] = task_match.group(1).strip()
        else:
            block['task_hint'] = 'Unknown Task'

        raw = block.get('raw_response', '')
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                usage = parsed.get('usage', {})
                if usage:
                    block['parsed_usage'] = usage
                content = parsed.get('choices', [{}])[0].get('message', {}).get('content', '')
                if content:
                    thought_match = re.search(r"Thought:(.*?)\n", content, re.DOTALL)
                    action_match = re.search(r"Action:(.*?)\n", content, re.DOTALL)
                    final_answer_match = re.search(r"Final Answer:(.*?)$", content, re.DOTALL)

                    if thought_match:
                        block['thought'] = thought_match.group(1).strip()
                    if action_match:
                        block['action'] = action_match.group(1).strip()
                    if final_answer_match:
                        block['final_answer'] = final_answer_match.group(1).strip()
        except Exception:
            block['parsing_error'] = True

        # Build token usage info
        usage = block['parsed_usage']
        if usage:
            token_usage_info.append({
                'task_hint': block['task_hint'],
                'model': parsed.get('model', 'unknown'),
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'cost_usd': (usage.get('prompt_tokens', 0) * 1.5e-07) + (usage.get('completion_tokens', 0) * 6e-07)
            })
        elif idx < len(lite_cost_buffer):
            fallback = lite_cost_buffer[idx]
            token_usage_info.append({
                'task_hint': block['task_hint'],
                'model': fallback['model'],
                'prompt_tokens': None,
                'completion_tokens': None,
                'total_tokens': None,
                'cost_usd': fallback['cost_usd']
            })

    return blocks, token_usage_info

def extract_token_usage_v2(token_usage_info: List[Dict]) -> pd.DataFrame:
    """Extract token usage and cost from token usage information."""
    df = pd.DataFrame(token_usage_info)
    return df

def save_analysis(blocks: List[Dict], output_dir: str):
    """Save each LLM call's input and output to a file."""
    os.makedirs(output_dir, exist_ok=True)
    for idx, block in enumerate(blocks):
        safe_task_hint = slugify_filename(block['task_hint'])
        file_prefix = f"{idx+1:03d}_{safe_task_hint}"
        input_path = Path(output_dir) / f"{file_prefix}_input.txt"
        output_path = Path(output_dir) / f"{file_prefix}_output.txt"
        input_path.write_text(block['litellm_request'])
        output_path.write_text(block['raw_response'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse CrewAI log file for LLM analysis.")
    parser.add_argument("log_path", type=str, help="Path to the log file.")
    parser.add_argument("output_dir", type=str, help="Directory to save parsed analysis files.")
    args = parser.parse_args()

    log_file = args.log_path
    analysis_dir = args.output_dir

    print(f"Got {len(Path(log_file).read_text().splitlines())} lines from {log_file}")

    parsed_blocks, token_usage_info = parse_log_file_v2(log_file)
    print(f"Total LLM calls detected: {len(parsed_blocks)}")

    token_df = extract_token_usage_v2(token_usage_info)

    parsing_errors = [i for i, block in enumerate(parsed_blocks) if block.get('parsing_error')]

    print("\n--- Token Usage Summary ---")
    if not token_df.empty:
        print(token_df)
    else:
        print("No valid token usage could be parsed.")

    print("\n--- Parsing Error Summary ---")
    print(f"Total parsing errors: {len(parsing_errors)} out of {len(parsed_blocks)} calls.")
    if parsing_errors:
        print(f"Parsing errors detected in calls: {parsing_errors}")

    total_with_final_answer = sum(1 for block in parsed_blocks if block['final_answer'])
    print(f"\n--- Final Answer Detection ---")
    print(f"Final Answers detected in {total_with_final_answer} out of {len(parsed_blocks)} calls.")

    save_analysis(parsed_blocks, analysis_dir)
    print(f"\nSaved detailed LLM input/output to folder: {analysis_dir}")
