#!/usr/bin/env python
"""
Simple utility to test the JSON cleaning functionality.
This can be run directly to verify that the JSON cleaning logic works correctly.
"""
import re
import json
import argparse

def clean_json_string(json_str):
    """Clean a JSON string by removing potential code block formatting and trailing backticks."""
    if not isinstance(json_str, str):
        return json_str
    
    # Remove opening code block
    json_str = re.sub(r'```json\s*', '', json_str)
    
    # Remove closing code block
    json_str = re.sub(r'```\s*$', '', json_str)
    
    # Ensure the JSON string ends properly with a closing brace or bracket
    json_str = json_str.strip()
    if json_str.endswith('`'):
        # Count the number of trailing backticks and remove them
        count = 0
        for char in reversed(json_str):
            if char == '`':
                count += 1
            else:
                break
        if count > 0:
            json_str = json_str[:-count]
    
    # Try to find proper JSON boundaries by bracket matching
    try:
        start_index = json_str.find('{')
        if start_index >= 0:
            # Count braces to find matching end
            open_count = 0
            end_index = -1
            for i, char in enumerate(json_str[start_index:], start_index):
                if char == '{':
                    open_count += 1
                elif char == '}':
                    open_count -= 1
                    if open_count == 0:
                        end_index = i + 1
                        break
                        
            if end_index > start_index:
                # Extract just the valid JSON object
                json_str = json_str[start_index:end_index]
    except Exception:
        # If extraction fails, just return the cleaned string
        pass
    
    return json_str

def main():
    parser = argparse.ArgumentParser(description='Test JSON cleaning function')
    parser.add_argument('--json-file', type=str, help='Path to a JSON file to clean')
    parser.add_argument('--test-cases', action='store_true', help='Run built-in test cases')
    args = parser.parse_args()
    
    if args.test_cases:
        test_cases = [
            {"input": '{"key": "value"}```', "expected": '{"key": "value"}'},
            {"input": '```json\n{"key": "value"}\n```', "expected": '{"key": "value"}'},
            {"input": '{"key": "value"} Some extra text', "expected": '{"key": "value"}'},
            {"input": '{"key": {"nested": "value"}}```', "expected": '{"key": {"nested": "value"}}'},
        ]
        
        for i, test in enumerate(test_cases):
            result = clean_json_string(test["input"])
            try:
                # Verify the result can be parsed
                json.loads(result)
                is_valid = True
            except Exception as e:
                is_valid = False
                
            print(f"Test {i+1}: " + ("✅ PASS" if is_valid else "❌ FAIL"))
            print(f"  Input: {test['input']}")
            print(f"  Result: {result}")
            print(f"  Expected: {test['expected']}")
            print(f"  Parseable: {is_valid}")
    
    if args.json_file:
        try:
            with open(args.json_file, 'r') as f:
                content = f.read()
            
            print(f"Original: {content[:100]}... ({len(content)} chars)")
            cleaned = clean_json_string(content)
            print(f"Cleaned: {cleaned[:100]}... ({len(cleaned)} chars)")
            
            # Test if parseable
            try:
                json.loads(cleaned)
                print("✅ Cleaned JSON is valid and parseable")
            except json.JSONDecodeError as e:
                print(f"❌ Cleaned JSON still has issues: {e}")
                
        except Exception as e:
            print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()