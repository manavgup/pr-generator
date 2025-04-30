# parse_crew_log_enhanced.py
import re
import json
import pandas as pd
from pathlib import Path
import argparse
import os
import uuid
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

class LogParser:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.lines = Path(log_path).read_text().splitlines()
        self.blocks = []
        self.token_usage_info = []
        self.response_times = []
        self.tool_invocations = []
        
    def parse_log_file(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Parse CrewAI log file into structured blocks with enhanced metrics."""
        current_block = None
        inside_raw = False
        current_raw_lines = []
        request_timestamp = None

        for idx, line in enumerate(self.lines):
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            current_timestamp = timestamp_match.group(1) if timestamp_match else None
            
            if "Request to litellm:" in line:
                if current_block:
                    self.blocks.append(current_block)
                current_block = self._create_empty_block()
                current_block['litellm_request'] += line + '\n'
                inside_raw = False
                request_timestamp = self._parse_timestamp(line) if timestamp_match else None
                
            elif "RAW RESPONSE:" in line and current_block:
                inside_raw = True
                current_raw_lines = []
                response_timestamp = self._parse_timestamp(line) if timestamp_match else None
                
                # Calculate response time if both timestamps are available
                if request_timestamp and response_timestamp:
                    response_time = (response_timestamp - request_timestamp).total_seconds()
                    current_block['response_time_seconds'] = response_time
                    self.response_times.append({
                        'task_hint': current_block.get('task_hint', 'Unknown_Task'),
                        'response_time_seconds': response_time
                    })
                
            elif inside_raw:
                current_raw_lines.append(line)
                if line.strip().endswith('}') or line.strip() == '}':
                    parsed_json = self._safe_extract_json(current_raw_lines)
                    if parsed_json:
                        self._process_raw_response(current_block, parsed_json)
                    inside_raw = False
                    
            elif "Action:" in line and current_block:
                tool_match = re.search(r"Action: (.*?)(?:\n|$)", line)
                if tool_match:
                    tool_name = tool_match.group(1).strip()
                    current_block['tools_used'].append(tool_name)
                    self.tool_invocations.append({
                        'task_hint': current_block.get('task_hint', 'Unknown_Task'),
                        'tool_name': tool_name,
                        'timestamp': current_timestamp
                    })
                    
            else:
                if current_block and not inside_raw:
                    current_block['litellm_request'] += line + '\n'

        if current_block:
            self.blocks.append(current_block)

        # Extract task hints
        for block in self.blocks:
            if not block['task_hint']:
                match = re.search(r"Current Task: (.*?)\\n", block['litellm_request'], re.DOTALL)
                if match:
                    block['task_hint'] = match.group(1).strip()
                else:
                    block['task_hint'] = 'Unknown_Task'

        # Process token usage
        for block in self.blocks:
            usage = block.get('parsed_usage', {})
            if usage:
                self.token_usage_info.append({
                    'task_hint': block['task_hint'],
                    'model': block['model'],
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'cost_usd': (usage.get('prompt_tokens', 0) * 1.5e-07) + (usage.get('completion_tokens', 0) * 6e-07),
                    'response_time_seconds': block.get('response_time_seconds', 0)
                })

        return self.blocks, self.token_usage_info, self.response_times, self.tool_invocations

    def _create_empty_block(self) -> Dict:
        """Create an empty block with default values."""
        return {
            'task_hint': '',
            'litellm_request': '',
            'raw_response': '',
            'parsed_json': {},
            'parsed_usage': {},
            'model': 'unknown',
            'thought': '',
            'action': '',
            'final_answer': '',
            'parsing_error': False,
            'tools_used': [],
            'response_time_seconds': 0,
            'request_id': str(uuid.uuid4())
        }
        
    def _parse_timestamp(self, line: str) -> Optional[datetime]:
        """Parse timestamp from log line."""
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            try:
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return None
        return None
        
    def _safe_extract_json(self, raw_lines: List[str]) -> dict:
        """Extract first valid JSON object from list of lines."""
        raw_text = "\n".join(raw_lines)
        json_start = raw_text.find('{')
        if json_start == -1:
            return {}
        try:
            return json.loads(raw_text[json_start:])
        except json.JSONDecodeError:
            return {}
            
    def _process_raw_response(self, block: Dict, parsed_json: Dict) -> None:
        """Process raw response JSON into block structure."""
        block['parsed_json'] = parsed_json
        usage = parsed_json.get('usage', {})
        if usage:
            block['parsed_usage'] = usage
        block['model'] = parsed_json.get('model', 'unknown')
        
        content = parsed_json.get('choices', [{}])[0].get('message', {}).get('content', '')
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

    def extract_token_usage(self) -> pd.DataFrame:
        """Extract token usage and cost from token usage information."""
        return pd.DataFrame(self.token_usage_info)
        
    def extract_response_times(self) -> pd.DataFrame:
        """Extract response time metrics."""
        return pd.DataFrame(self.response_times)
        
    def extract_tool_usage(self) -> pd.DataFrame:
        """Extract tool usage patterns."""
        return pd.DataFrame(self.tool_invocations)
        
    def save_analysis(self, output_dir: str):
        """Save each LLM call's input and output to a file."""
        os.makedirs(output_dir, exist_ok=True)
        for idx, block in enumerate(self.blocks):
            safe_task_hint = self._slugify_filename(block.get('task_hint', 'Unknown_Task'))
            file_prefix = f"{idx+1:03d}_{safe_task_hint}"
            (Path(output_dir) / f"{file_prefix}_input.txt").write_text(block['litellm_request'])
            (Path(output_dir) / f"{file_prefix}_output.txt").write_text(json.dumps(block.get('parsed_json', {}), indent=2))
            
    def _slugify_filename(self, text: str) -> str:
        """Create a filesystem-safe filename from task hint."""
        return re.sub(r'[^a-zA-Z0-9_-]', '_', text)[:80]


def main():
    parser = argparse.ArgumentParser(description="Enhanced CrewAI log file parser for LLM analysis.")
    parser.add_argument("log_path", type=str, help="Path to the log file.")
    parser.add_argument("output_dir", type=str, help="Directory to save parsed analysis files.")
    args = parser.parse_args()

    log_file = args.log_path
    analysis_dir = args.output_dir

    print(f"Got {len(Path(log_file).read_text().splitlines())} lines from {log_file}")

    log_parser = LogParser(log_file)
    parsed_blocks, token_usage_info, response_times, tool_invocations = log_parser.parse_log_file()
    
    print(f"Total LLM calls detected: {len(parsed_blocks)}")

    token_df = log_parser.extract_token_usage()
    response_time_df = log_parser.extract_response_times()
    tool_usage_df = log_parser.extract_tool_usage()

    parsing_errors = [i for i, block in enumerate(parsed_blocks) if block.get('parsing_error')]

    print("\n--- Token Usage Summary ---")
    if not token_df.empty:
        print(token_df[['task_hint', 'model', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd']].head())
        print(f"Total cost: ${token_df['cost_usd'].sum():.6f}")
    else:
        print("No valid token usage could be parsed.")
        
    print("\n--- Response Time Summary ---")
    if not response_time_df.empty:
        print(f"Average response time: {response_time_df['response_time_seconds'].mean():.2f} seconds")
        print(f"Max response time: {response_time_df['response_time_seconds'].max():.2f} seconds")
    else:
        print("No valid response times could be parsed.")
        
    print("\n--- Tool Usage Summary ---")
    if not tool_usage_df.empty:
        tool_counts = tool_usage_df['tool_name'].value_counts()
        print(tool_counts)
    else:
        print("No valid tool usage could be parsed.")

    print("\n--- Parsing Error Summary ---")
    print(f"Total parsing errors: {len(parsing_errors)} out of {len(parsed_blocks)} calls.")
    if parsing_errors:
        print(f"Parsing errors detected in calls: {parsing_errors}")

    total_with_final_answer = sum(1 for block in parsed_blocks if block['final_answer'])
    print(f"\n--- Final Answer Detection ---")
    print(f"Final Answers detected in {total_with_final_answer} out of {len(parsed_blocks)} calls.")

    # Save detailed analysis
    log_parser.save_analysis(analysis_dir)
    
    # Save dataframes
    os.makedirs(os.path.join(analysis_dir, "data"), exist_ok=True)
    token_df.to_csv(os.path.join(analysis_dir, "data", "token_usage.csv"), index=False)
    response_time_df.to_csv(os.path.join(analysis_dir, "data", "response_times.csv"), index=False)
    tool_usage_df.to_csv(os.path.join(analysis_dir, "data", "tool_usage.csv"), index=False)
    
    print(f"\nSaved detailed LLM input/output to folder: {analysis_dir}")
    print(f"Saved analysis data to folder: {os.path.join(analysis_dir, 'data')}")


if __name__ == "__main__":
    main()