import os
import sys
import json
from textwrap import dedent
from typing import List, Dict, Optional, Any, Set, Type

# --- Configuration ---
REPO_PATH = "/Users/mg/mg-work/manav/work/ai-experiments/rag_modulo"
AGENT_LLM = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o") # Or your preferred model

# Add diagnostics for Python environment
print(f"Python Version: {sys.version}")
print(f"Using LLM: {AGENT_LLM}")
print(f"REPO_PATH: {REPO_PATH}")

if not os.path.isdir(REPO_PATH):
    print(f"ERROR: REPO_PATH '{REPO_PATH}' does not exist or is not a directory.")
    exit(1)

# --- CrewAI Imports ---
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError
# Import LLM provider if needed
# from langchain_openai import ChatOpenAI

# --- Tool Imports ---
try:
    from tools.base_tool import BaseRepoTool
    print("Successfully imported BaseRepoTool")
    # Ensure you are importing the REFACTORED FileGrouperTool
    from tools.file_grouper_tool import FileGrouperTool
    from tools.group_validator_tool import GroupValidatorTool
    from tools.group_refiner_tool import GroupRefinerTool
except ImportError as e:
    print(f"Failed to import tools: {e}")
    print("Ensure tools are in the 'tools/' directory and PYTHONPATH is set.")
    exit(1)

# -- Model imports ---
try:
    from shared.models.base_models import FileType, FileStatusType
    from shared.models.git_models import FileChange, LineChanges
    from shared.models.directory_models import DirectorySummary
    from shared.models.analysis_models import RepositoryAnalysis
    from models.agent_models import (
        GroupingStrategyType, PatternAnalysisResult, PRGroupingStrategy,
        GroupingStrategyDecision, PRGroup, PRValidationResult, GroupValidationIssue
    )
    print("Successfully imported all models")
except ImportError as e:
    print(f"Failed to import models: {e}")
    # Define basic placeholders if imports fail
    class BaseModel: pass
    class FileChange(BaseModel): pass
    class GroupingStrategyDecision(BaseModel): pass
    class RepositoryAnalysis(BaseModel): pass
    class PatternAnalysisResult(BaseModel): pass
    class PRGroupingStrategy(BaseModel): pass
    class PRValidationResult(BaseModel): pass
    class PRGroup(BaseModel): pass
    class GroupValidationIssue(BaseModel): pass
    class GroupingStrategyType(str): MIXED="mixed"

# Modified task description approach
# Instead of asking the agent to access variables by name, provide the values directly

def create_task_with_embedded_inputs(batch_files, strategy_value, repo_analysis_json, pattern_analysis):
    """Creates a task description with the inputs directly embedded in the instructions"""
    
    # Convert inputs to string representations for inclusion in task
    batch_files_str = str(batch_files)
    strategy_value_str = str(strategy_value)
    repo_analysis_escaped = repo_analysis_json.replace("{", "{{").replace("}", "}}")

    task_desc = f"""
    Process the assigned batch of files using the File Grouper Tool, Group Validator Tool, and Group Refiner Tool.
    
    You have the following EXACT input values which you MUST use without any modification:
    
    1. batch_file_paths = {batch_files_str}
    
    2. strategy_type_value = "{strategy_value_str}"
    
    3. repository_analysis_json = ```
{repo_analysis_escaped}
```
    
    4. pattern_analysis_json = {pattern_analysis if pattern_analysis else "null"}
    
    **CRITICAL: You must use these EXACT values. DO NOT modify them in any way.**
    
    **Your Steps:**
    1. Call the File Grouper Tool with the EXACT values above:
       - batch_file_paths: Use the exact list from #1 above
       - strategy_type_value: Use the exact string from #2 above
       - repository_analysis_json: Use the exact string from #3 above (everything between the triple backticks)
       - pattern_analysis_json: Use the value from #4 above
    
    2. Take the JSON string output (PRGroupingStrategy) from the File Grouper Tool. 
       If it indicates an error, stop and return that error JSON string as your Final Answer.
    
    3. Use the Group Validator Tool, providing:
       - pr_grouping_strategy_json: The exact output string from step 2
       - is_final_validation: false
    
    4. Take the JSON string output (PRValidationResult) from the Group Validator Tool.
    
    5. Use the Group Refiner Tool, providing:
       - pr_grouping_strategy_json: The exact output string from step 2
       - pr_validation_result_json: The exact output string from step 4
       - original_repository_analysis_json: null
    
    6. Your final answer is the JSON string output from the Group Refiner Tool.
    """
    
    return task_desc

# --- Tool Instantiation ---
print(f"Initializing tools for repo: {REPO_PATH}")
try:
    # Inspect the BaseRepoTool class - this is for diagnostic purposes
    print(f"BaseRepoTool is in: {BaseRepoTool.__module__}")
    
    file_grouper = FileGrouperTool(repo_path=REPO_PATH)
    # Check the internal state of file_grouper
    print(f"FileGrouperTool instance created with _repo_path: {file_grouper._repo_path}")
    print(f"FileGrouperTool has git_ops: {hasattr(file_grouper, '_git_ops')}")
    
    group_validator = GroupValidatorTool(repo_path=REPO_PATH)
    group_refiner = GroupRefinerTool(repo_path=REPO_PATH)
    worker_tools = [file_grouper, group_validator, group_refiner]
    print("All tools initialized successfully")
except Exception as e:
    print(f"Error initializing tools: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
print("Tools initialized.")

# --- Agent Definition (Worker Only) ---
print("Defining Batch Processor Agent...")
try:
    batch_processor_agent = Agent(
        role="PR Batch Grouping Specialist",
        goal=dedent("""\
            Process a specific batch of file changes assigned by the Manager.
            Apply the globally chosen grouping strategy (provided in context) to the assigned files.
            Generate, validate, and refine PR groups specifically for this batch, returning a
            well-formed result for the assigned files."""),
        backstory=dedent("""\
            You are a focused developer tasked with analyzing and grouping a specific subset of code changes.
            You receive a batch of files and the overall grouping strategy from your manager.
            You meticulously apply the assigned strategy to the files in your batch using the provided tools.
            When using tools, you pass the EXACT data you were given without any modifications or reformatting."""),
        tools=worker_tools,
        allow_delegation=False,
        verbose=True,
        # llm=ChatOpenAI(model_name=AGENT_LLM, temperature=0.1) # Set LLM explicitly if needed
    )
    print("Agent defined.")
except Exception as e:
    print(f"Error defining agent: {e}")
    exit(1)


# --- Construct Input Data ---
print("Constructing Task Input Data...")

# 1. Batch file paths (Example batch)
batch_file_paths = [
    ".github/scripts/fix_issue.py", ".github/workflows/ci.yml",
    ".github/workflows/test-and-issue.yml", ".github/workflows/watsonx-benchmarks.yml",
    ".github/workflows/publish.yml", ".gitignore", "backend/Dockerfile.backend",
    "backend/Dockerfile.backend.dockerignore", "backend/Dockerfile.test",
    "backend/Dockerfile.test.dockerignore"
]

# 2. GroupingStrategyDecision (as JSON string)
grouping_strategy_decision_str = r"""
{
  "strategy_type": "mixed",
  "recommendations": [{
    "strategy_type": "mixed",
    "confidence": 0.7,
    "rationale": "No single strategy strongly indicated by metrics. A mixed approach is suggested.",
    "estimated_pr_count": 2
  }],
  "repository_metrics": {
    "total_files_changed": 10,
    "total_lines_changed": 630,
    "directory_count": 0,
    "max_files_in_directory": 0,
    "directory_concentration": 0.0,
    "file_type_count": 0,
    "is_distributed": false
  },
  "explanation": "Selected 'mixed' as the primary strategy with 0.70 confidence. Rationale: No single strategy strongly indicated by metrics. A mixed approach is suggested."
}
"""
# Extract the strategy type value needed by the refactored tool
try:
    strategy_decision_data = json.loads(grouping_strategy_decision_str)
    strategy_type_value = strategy_decision_data.get("strategy_type", "mixed") # Default to mixed if missing
    # Validate it's a known type (optional but good)
    try:
        _ = GroupingStrategyType(strategy_type_value)
    except ValueError:
        print(f"Warning: Strategy type '{strategy_type_value}' from decision JSON is not a valid GroupingStrategyType enum member. Defaulting to 'mixed'.")
        strategy_type_value = "mixed"
except json.JSONDecodeError as e:
    print(f"ERROR: Could not parse GroupingStrategyDecision JSON: {e}")
    exit(1)

# 3. RepositoryAnalysis (as JSON string)
#    Using the valid JSON from previous logs
repository_analysis_str = r"""
{
  "repo_path": "/Users/mg/mg-work/manav/work/ai-experiments/rag_modulo",
  "file_changes": [
    { "file_id": "bd83e4c3-66a1-42cc-a0b8-ba757ef5693a", "path": ".github/scripts/fix_issue.py", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 97, "deleted": 0 }, "content_hash": "9eda9230553f90c18c00bf30fbc7432e", "token_estimate": 0, "directory": ".github/scripts", "extension": ".py", "filename": "fix_issue.py" },
    { "file_id": "dd02dd91-c732-4f1a-9921-c36361f9bb31", "path": ".github/workflows/ci.yml", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 110, "deleted": 0 }, "content_hash": "194d7d1c3d2711bbee1f9023228fc84f", "token_estimate": 0, "directory": ".github/workflows", "extension": ".yml", "filename": "ci.yml" },
    { "file_id": "8d834255-5921-405e-be9d-e735094e1f11", "path": ".github/workflows/publish.yml", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 35, "deleted": 0 }, "content_hash": "323b15280f7960268c4e11eea5dc3e65", "token_estimate": 0, "directory": ".github/workflows", "extension": ".yml", "filename": "publish.yml" },
    { "file_id": "a7c317b0-142a-4906-88cf-a73a4e82348e", "path": ".github/workflows/test-and-issue.yml", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 100, "deleted": 0 }, "content_hash": "d595aec28f3c8db0300f8f2118cb6aa9", "token_estimate": 0, "directory": ".github/workflows", "extension": ".yml", "filename": "test-and-issue.yml" },
    { "file_id": "c2044fd8-f79c-4542-b52d-284d44ed0bd8", "path": ".github/workflows/watsonx-benchmarks.yml", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 161, "deleted": 0 }, "content_hash": "119e40e2925ff6344035aa3121433f6d", "token_estimate": 0, "directory": ".github/workflows", "extension": ".yml", "filename": "watsonx-benchmarks.yml" },
    { "file_id": "48cbcd3b-dc89-4b84-9e7a-1c9135efe642", "path": ".gitignore", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 24, "deleted": 0 }, "content_hash": "12bb35dc41fc4916ce63a991a5585bec", "token_estimate": 0, "directory": "(root)", "extension": null, "filename": ".gitignore" },
    { "file_id": "870b5ef6-5db2-4121-af43-8db807370a9c", "path": "backend/Dockerfile.backend", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 39, "deleted": 0 }, "content_hash": "63b935a5bc9dd5a99e2c189a53a5f40f", "token_estimate": 0, "directory": "backend", "extension": ".backend", "filename": "Dockerfile.backend" },
    { "file_id": "5807c5ce-4570-4c3c-8a24-3c63a36d0d6d", "path": "backend/Dockerfile.backend.dockerignore", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 15, "deleted": 0 }, "content_hash": "8d09187c3538045ffad1561a4948ce4c", "token_estimate": 0, "directory": "backend", "extension": ".dockerignore", "filename": "Dockerfile.backend.dockerignore" },
    { "file_id": "745c16c2-0a2b-497d-8476-bbd267e8216d", "path": "backend/Dockerfile.test", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 35, "deleted": 0 }, "content_hash": "1d5f203d00cc34fef282be7d0346a1c9", "token_estimate": 0, "directory": "backend", "extension": ".test", "filename": "Dockerfile.test" },
    { "file_id": "b4e68ebb-4dab-452d-a057-901e695ceec7", "path": "backend/Dockerfile.test.dockerignore", "staged_status": "M", "unstaged_status": " ", "original_path": null, "file_type": "text", "changes": { "added": 14, "deleted": 0 }, "content_hash": "d1c4b6e904f5ec75b9cc90a68291b98d", "token_estimate": 0, "directory": "backend", "extension": ".dockerignore", "filename": "Dockerfile.test.dockerignore" }
  ],
  "directory_summaries": [
    { "path": ".github/scripts", "file_count": 1, "files": [".github/scripts/fix_issue.py"], "total_changes": 97, "extensions": { ".py": 1 }, "is_root": false, "depth": 2, "parent_directory": ".github" },
    { "path": ".github/workflows", "file_count": 4, "files": [".github/workflows/ci.yml", ".github/workflows/publish.yml", ".github/workflows/test-and-issue.yml", ".github/workflows/watsonx-benchmarks.yml"], "total_changes": 406, "extensions": { ".yml": 4 }, "is_root": false, "depth": 2, "parent_directory": ".github" },
    { "path": "(root)", "file_count": 1, "files": [".gitignore"], "total_changes": 24, "extensions": { "none": 1 }, "is_root": true, "depth": 0, "parent_directory": null },
    { "path": "backend", "file_count": 4, "files": ["backend/Dockerfile.backend", "backend/Dockerfile.backend.dockerignore", "backend/Dockerfile.test", "backend/Dockerfile.test.dockerignore"], "total_changes": 103, "extensions": { ".backend": 1, ".dockerignore": 2, ".test": 1 }, "is_root": false, "depth": 1, "parent_directory": "(root)" }
  ],
  "total_files_changed": 10,
  "total_lines_changed": 630,
  "timestamp": 1744404765.238562,
  "error": null,
  "extensions_summary": { ".py": 1, ".yml": 4, "none": 1, ".backend": 1, ".dockerignore": 2, ".test": 1 },
  "directories": [ "(root)", ".github/scripts", ".github/workflows", "backend" ]
}
"""

# Verify repository analysis JSON is valid
try:
    json.loads(repository_analysis_str)
    print("Repository analysis JSON is valid.")
except json.JSONDecodeError as e:
    print(f"ERROR: Repository analysis JSON is invalid: {e}")
    exit(1)

# 4. PatternAnalysisResult (as JSON string or None)
# Let's use None for this test
pattern_analysis_json_string = None
# If you had pattern results:
# pattern_analysis_json_string = r"""{"naming_patterns": [], ... }"""

# --- Task Definition ---
# --- UPDATED Task Description with Emphasis on EXACT Input Passing ---
process_single_batch_task_desc = create_task_with_embedded_inputs(
    batch_files=batch_file_paths,
    strategy_value=strategy_type_value,
    repo_analysis_json=repository_analysis_str.strip(),
    pattern_analysis=pattern_analysis_json_string
)

# --- END UPDATED Task Description ---

process_single_batch_task_expected_output = dedent("""\
    A JSON string serialization of the refined PRGroupingStrategy object specific to the processed batch.
    """)

# Create the Task object
process_single_batch_task = Task(
    description=process_single_batch_task_desc,
    expected_output=process_single_batch_task_expected_output,
    agent=batch_processor_agent
)
print("Task defined.")

# --- Create Crew ---
print("Creating Crew...")
crew = Crew(
    agents=[batch_processor_agent],
    tasks=[process_single_batch_task],
    verbose=True,
    process=Process.sequential  # Ensure sequential processing
)
print("Crew created.")

# --- Kickoff Crew ---
print("Kicking off Crew with separate context items...")
# Prepare inputs with special processing to ensure JSON strings are passed correctly
inputs = {
    "BATCH_FILES": batch_file_paths, # The list of paths
    "STRATEGY_VALUE": strategy_type_value, # The strategy string ('mixed')
    "REPO_ANALYSIS_JSON": repository_analysis_str.strip(), # The full repo analysis JSON string, stripped
    "PATTERN_ANALYSIS_JSON": pattern_analysis_json_string # None in this case
}

# Print the exact input values being provided to help with debugging
print(f"Input BATCH_FILES (first 3): {inputs['BATCH_FILES'][:3]}")
print(f"Input STRATEGY_VALUE: {inputs['STRATEGY_VALUE']}")
print(f"Input REPO_ANALYSIS_JSON (first 100 chars): {inputs['REPO_ANALYSIS_JSON'][:100]}...")
print(f"Input PATTERN_ANALYSIS_JSON: {inputs['PATTERN_ANALYSIS_JSON']}")

try:
    result = crew.kickoff(inputs=inputs)
    print("\n--- Crew Run Result ---")
    print(result)
    print("-" * 30)

    # Try to parse the result as JSON to check validity
    parsed_result_json = None
    try:
        raw_output = result
        if hasattr(result, 'raw'): raw_output = result.raw
        elif hasattr(result, 'raw_output'): raw_output = result.raw_output

        if isinstance(raw_output, str):
            # Handle potential triple backticks that might be in the output
            if raw_output.strip().startswith('```json'):
                raw_output = raw_output.strip().split('```json')[1].split('```')[0].strip()
            elif raw_output.strip().startswith('```'):
                raw_output = raw_output.strip().split('```')[1].strip()
                
            parsed_result_json = json.loads(raw_output)
            print("\n--- Parsed Result JSON ---")
            print(json.dumps(parsed_result_json, indent=2))
            print("-" * 30)
        else:
            print("WARNING: Crew result was not a string, cannot parse as JSON.")
            print(f"Result type: {type(raw_output)}")

    except (json.JSONDecodeError, TypeError) as json_err:
        print(f"ERROR: Crew result could not be parsed as JSON. Raw result was:\n{raw_output}\nError: {json_err}")
    except Exception as e:
        print(f"Error processing result: {e}")

    # Check if groups were created
    if parsed_result_json and isinstance(parsed_result_json.get("groups"), list) and len(parsed_result_json["groups"]) > 0:
        print(f"SUCCESS: Worker agent produced {len(parsed_result_json['groups'])} group(s).")
    elif parsed_result_json and parsed_result_json.get("explanation", "").startswith("Error"):
        print(f"ERROR: Worker agent likely failed in FileGrouperTool. Explanation: {parsed_result_json.get('explanation')}")
    else:
        print("WARNING: Worker agent finished but produced an empty or invalid 'groups' list.")
        print("Check agent logs and tool outputs for potential errors during execution.")

except Exception as e:
    print(f"\n--- Crew Run Failed ---")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("-" * 30)

# --- Testing Direct Tool Invocation (Using the NEW tool schema) ---
print("\n--- Testing Direct Tool Invocation ---")
try:
    print("1. Calling file_grouper_tool directly with separate args...")
    # Create a new instance to ensure a clean state
    file_grouper = FileGrouperTool(repo_path=REPO_PATH)
    
    # Call with keyword arguments matching the new schema
    # Use the exact same inputs as provided to the crew
    grouping_result = file_grouper.run(
        batch_file_paths=batch_file_paths,
        strategy_type_value=strategy_type_value,
        repository_analysis_json=repository_analysis_str.strip(),
        pattern_analysis_json=pattern_analysis_json_string
    )

    # Analyze the result
    try:
        grouping_result_obj = json.loads(grouping_result)
        if "error" in grouping_result_obj or (grouping_result_obj.get("explanation", "").startswith("Error")):
            print(f"File grouper returned an error: {grouping_result_obj.get('explanation')}")
            print(f"Full result: {grouping_result}")
            raise ValueError("Stopping direct test after file_grouper error")
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse file_grouper result as JSON: {grouping_result}")
        raise

    print(f"File grouper result (first 100 chars): {grouping_result[:100]}...")

    # 2. Call group_validator
    print("2. Calling group_validator_tool...")
    group_validator = GroupValidatorTool(repo_path=REPO_PATH)
    validation_result = group_validator.run(
        pr_grouping_strategy_json=grouping_result,
        is_final_validation=False
    )
    print(f"Validation result (first 100 chars): {validation_result[:100]}...")

    # 3. Call group_refiner
    print("3. Calling group_refiner_tool...")
    group_refiner = GroupRefinerTool(repo_path=REPO_PATH)
    refinement_result = group_refiner.run(
        pr_grouping_strategy_json=grouping_result,
        pr_validation_result_json=validation_result
        # No original_repository_analysis_json for batch refinement
    )
    print(f"Final result (first 100 chars): {refinement_result[:100]}...")

    # 4. Parse the result to check validity
    print("\n--- Direct Tool Invocation Result ---")
    result_json = json.loads(refinement_result)
    print(json.dumps(result_json, indent=2))
    print("-" * 30)
    if result_json and result_json.get("groups"):
        print(f"DIRECT TEST SUCCESS: Tools produced {len(result_json['groups'])} group(s).")
    else:
        print("DIRECT TEST WARNING: Tools produced empty or invalid 'groups' list.")

except Exception as e:
    print(f"Error during direct tool invocation: {e}")
    import traceback
    traceback.print_exc()

print("Test finished.")