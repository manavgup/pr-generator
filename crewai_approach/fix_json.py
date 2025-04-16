#!/usr/bin/env python
"""
Quick script to fix the JSON file for the PR Generator tool.
Removes trailing backticks from the 'raw' field in the JSON.
"""

import json
import re
import os

# Path to the problematic file
file_path = 'outputs/step_1_initial_analysis.json'
output_path = 'outputs/step_1_initial_analysis.json'

print(f"Reading file: {file_path}")

# Read the file
with open(file_path, 'r') as f:
    content = f.read()

print("Parsing JSON...")
try:
    # Parse the JSON to extract the 'raw' field
    data = json.loads(content)
    if 'raw' in data and isinstance(data['raw'], str):
        raw_content = data['raw']
        
        # Check if there are trailing backticks and remove them
        cleaned_content = re.sub(r'```\s*$', '', raw_content)
        
        # Update the raw field
        data['raw'] = cleaned_content
        
        # Convert back to JSON
        fixed_json = json.dumps(data, indent=2)
        
        # Write the fixed content
        with open(output_path, 'w') as f:
            f.write(fixed_json)
        
        # Validate by trying to parse the raw field
        try:
            fixed_raw = json.loads(data['raw'])
            print(f"✅ Success! Fixed JSON saved to {output_path}")
            print("The raw field is now valid JSON and can be parsed correctly.")
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: The raw field still contains invalid JSON: {e}")
            print("Manual inspection may be required.")
    else:
        print("❌ Error: Could not find 'raw' field in the JSON data.")
except json.JSONDecodeError as e:
    print(f"❌ Error: Could not parse the JSON file: {e}")
    
print("\nHow to use the fixed file:")
print(f"PYTHONPATH=. python crewai_approach/test_toolchain.py --start-from {output_path}")