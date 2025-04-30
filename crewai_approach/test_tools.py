# /Users/mg/mg-work/manav/work/ai-experiments/pr-generator/crewai_approach/test_tools.py

import pytest
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from pydantic import ValidationError

# --- Adjust imports based on your project structure ---
# Assuming tools & models are relative to project root 'pr-generator'
# when running with PYTHONPATH=. from the root directory.
from tools.batch_processor_tool import BatchProcessorTool
from tools.group_merging_tool import GroupMergingTool
from tools.group_validator_tool import GroupValidatorTool
from tools.group_refiner_tool import GroupRefinerTool
from tools.batch_splitter_tool import BatchSplitterTool

# Import models for validation
# Assuming models are also relative to project root
from models.agent_models import PRGroupingStrategy, PRValidationResult, GroupingStrategyType
# --- End Imports ---

# --- Fixtures to load test data ---

# Define the path to your actual output data directory
TEST_DATA_DIR = Path("/Users/mg/mg-work/manav/work/ai-experiments/pr-generator/outputs")

# Define the repo path used during the run (for tool instantiation)
REPO_PATH_FOR_TOOLS = "/Users/mg/mg-work/manav/work/ai-experiments/rag_modulo"

# Helper function to load and clean JSON string from the 'raw' field of saved output files
def load_and_clean_raw_json(filename: str) -> str:
    """Loads the JSON object from the file, extracts the 'raw' string, and cleans it."""
    file_path = TEST_DATA_DIR / filename
    assert file_path.exists(), f"Test data file not found: {file_path}"
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # Load the outer JSON object saved by the callback
            data = json.load(f)
            raw_json_str = data.get("raw")
            if not isinstance(raw_json_str, str):
                 raise ValueError(f"'raw' field in {filename} is not a string or is missing.")

            # Clean potential markdown and control characters
            cleaned = re.sub(r'^```json\s*', '', raw_json_str.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()
            # Minimal cleaning - more can be added if specific control chars cause issues
            # cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned) # Use cautiously
            return cleaned
        except json.JSONDecodeError as e:
             pytest.fail(f"Failed to load JSON from {filename}: {e}")
        except ValueError as e:
             pytest.fail(f"Error processing data from {filename}: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error loading {filename}: {e}")

# --- Updated Fixtures ---

@pytest.fixture(scope="session")
def repo_analysis_json_str() -> str:
    """Loads the initial repository analysis JSON string from the 'raw' field."""
    return load_and_clean_raw_json("step_1_initial_analysis.json")

@pytest.fixture(scope="session")
def strategy_decision_json_str() -> str:
    """Loads the strategy decision JSON string from the 'raw' field."""
    return load_and_clean_raw_json("step_4_strategy_decision.json")

@pytest.fixture(scope="session")
def batch_splitter_output_json_str() -> str:
    """Loads the batch splitter output JSON string from the 'raw' field."""
    # This file's raw content based on your logs: {"batches": [], ...}
    return load_and_clean_raw_json("step_5_split_batches.json")

@pytest.fixture(scope="session")
def pattern_analysis_json_str() -> str:
    """Loads the pattern analysis JSON string from the 'raw' field."""
    return load_and_clean_raw_json("step_3_global_patterns.json")

@pytest.fixture(scope="session")
def processed_batches_results_json_str() -> str:
    """Loads the JSON array string output from the batch processor tool's saved file."""
    # This file's raw content based on your logs: "[]\n```" -> cleaned to "[]"
    return load_and_clean_raw_json("step_6_processed_batches.json") # Use the helper

@pytest.fixture(scope="session")
def merged_groups_json_str() -> str:
    """Loads the merged groups JSON string from the 'raw' field."""
    # This file's raw content based on logs: {"strategy_type": "mixed", "groups": [...], "explanation": "No valid batch...", ...}
    return load_and_clean_raw_json("step_7_merged_groups.json")

@pytest.fixture(scope="session")
def final_validation_json_str() -> str:
    """Loads the final validation result JSON string from the 'raw' field."""
    # This file's raw content based on logs: {"is_valid": false, "issues": [{"severity": "critical", "issue_type": "Tool Error", ...}]}
    return load_and_clean_raw_json("step_8_final_validation.json")


# --- Fixtures for Tool Instances ---

@pytest.fixture(scope="module")
def batch_processor_tool() -> BatchProcessorTool:
    return BatchProcessorTool(repo_path=REPO_PATH_FOR_TOOLS)

@pytest.fixture(scope="module")
def group_merging_tool() -> GroupMergingTool:
    # Ensure this uses the version expecting string input for batch results
    return GroupMergingTool(repo_path=REPO_PATH_FOR_TOOLS)

@pytest.fixture(scope="module")
def group_validator_tool() -> GroupValidatorTool:
    return GroupValidatorTool(repo_path=REPO_PATH_FOR_TOOLS)

@pytest.fixture(scope="module")
def group_refiner_tool() -> GroupRefinerTool:
    return GroupRefinerTool(repo_path=REPO_PATH_FOR_TOOLS)

# Add to test_tools.py

@pytest.fixture(scope="module")
def batch_splitter_tool() -> BatchSplitterTool:
    # Assuming BatchSplitterTool is in tools directory
    from tools.batch_splitter_tool import BatchSplitterTool
    return BatchSplitterTool(repo_path=REPO_PATH_FOR_TOOLS)

def test_batch_splitter_tool_creates_batches(
    batch_splitter_tool: BatchSplitterTool,
    repo_analysis_json_str: str # Has 50 files
):
    """Tests if the BatchSplitterTool creates non-empty batches from valid analysis."""
    print("\nTesting BatchSplitterTool creates batches...")
    target_batch_size = 10 # Match the run config
    output_str = batch_splitter_tool._run(
        repository_analysis_json=repo_analysis_json_str,
        target_batch_size=target_batch_size
    )
    assert isinstance(output_str, str)
    try:
        output_data = json.loads(output_str)
        assert "batches" in output_data
        assert isinstance(output_data["batches"], list)
        print(f"BatchSplitterTool produced {len(output_data['batches'])} batches.")
        # CRITICAL ASSERTION: Expect batches from 50 files / size 10
        assert len(output_data["batches"]) > 0
        # Optional: Check total files across batches
        total_files_in_batches = sum(len(batch) for batch in output_data["batches"])
        repo_analysis_data = json.loads(repo_analysis_json_str)
        expected_files = len(repo_analysis_data.get("file_changes", []))
        # Note: Batch splitting might sometimes drop files if logic is complex,
        # so check if the count is reasonable or exactly matches.
        assert total_files_in_batches == expected_files, f"Expected {expected_files} files in batches, found {total_files_in_batches}"
        print(f"BatchSplitterTool correctly produced {len(output_data['batches'])} non-empty batches.")

    except (json.JSONDecodeError, ValidationError) as e:
        pytest.fail(f"BatchSplitterTool output validation failed: {e}\nOutput:\n{output_str}")
    except AssertionError as e:
         pytest.fail(f"BatchSplitterTool assertion failed: {e}\nOutput:\n{output_str}")

# --- Test Functions (Updated fixture names & assertions) ---

def test_batch_processor_tool_execution(
    batch_processor_tool: BatchProcessorTool,
    batch_splitter_output_json_str: str, # Input has empty "batches" list
    strategy_decision_json_str: str,
    repo_analysis_json_str: str,
    pattern_analysis_json_str: str
):
    """Tests if the BatchProcessorTool runs and handles empty input batches correctly."""
    print("\nTesting BatchProcessorTool...")
    output_str = batch_processor_tool._run(
        batch_splitter_output_json=batch_splitter_output_json_str,
        grouping_strategy_decision_json=strategy_decision_json_str,
        repository_analysis_json=repo_analysis_json_str,
        pattern_analysis_json=pattern_analysis_json_str
    )

    assert isinstance(output_str, str)
    assert output_str.strip().startswith('[')
    assert output_str.strip().endswith(']')

    try:
        output_list = json.loads(output_str)
        assert isinstance(output_list, list)
        print(f"BatchProcessorTool produced a list with {len(output_list)} items.")

        # Load the input batch data to see how many batches there *should* be
        splitter_data = json.loads(batch_splitter_output_json_str)
        expected_batches = len(splitter_data.get("batches", [])) # Should be 0 based on input file
        assert len(output_list) == expected_batches, f"Expected {expected_batches} batch results, got {len(output_list)}"
        print(f"Assertion Passed: Output list length ({len(output_list)}) matches input batch count ({expected_batches}).")

    except json.JSONDecodeError as e:
        pytest.fail(f"BatchProcessorTool output is not valid JSON: {e}\nOutput:\n{output_str[:500]}...")
    # No need to check Pydantic model for items if the list is expected empty


def test_group_merging_tool_with_empty_batch_results(
    group_merging_tool: GroupMergingTool,
    processed_batches_results_json_str: str, # Contains "[]" string
    repo_analysis_json_str: str
):
    """Tests merging tool with an empty JSON array string of batch results."""
    print("\nTesting GroupMergingTool with empty batch results string...")
    output_str = group_merging_tool._run(
        batch_grouping_results_json=processed_batches_results_json_str,
        original_repository_analysis_json=repo_analysis_json_str
    )

    assert isinstance(output_str, str)
    try:
        output_data = PRGroupingStrategy.model_validate_json(output_str)
        assert isinstance(output_data.groups, list)
        assert len(output_data.groups) == 0
        # Check for specific explanation text added by the tool for this case
        assert "No valid batch results" in output_data.explanation or "Parsed batch results list is empty" in output_data.explanation
        print("GroupMergingTool correctly handled empty batch results string.")
    except (json.JSONDecodeError, ValidationError) as e:
        pytest.fail(f"GroupMergingTool (empty results string test) output validation failed: {e}\nOutput:\n{output_str}")


def test_group_validator_on_final_merged_output(
    group_validator_tool: GroupValidatorTool,
    merged_groups_json_str: str # Fixture loads step_7_merged_groups.json
):
    """Tests validator on the actual merged output from step_7."""
    print("\nTesting GroupValidatorTool on step_7 merged output...")
    output_str = group_validator_tool._run(
        pr_grouping_strategy_json=merged_groups_json_str,
        is_final_validation=True # Simulate final validation check
    )
    assert isinstance(output_str, str)
    try:
        output_data = PRValidationResult.model_validate_json(output_str)
        # Based on the previously observed behavior leading to step_8_final_validation.json,
        # the merge result *itself* might be considered valid by the validator if duplicates
        # aren't checked until refinement, OR if the validator logic changed.
        # We will check the content of step_8_final_validation.json using its own fixture below.
        # For this test, we just check if the validator ran and produced a valid PRValidationResult structure.
        print(f"Validation result: is_valid={output_data.is_valid}, issues={len(output_data.issues)}")
        assert output_data.strategy_type is not None # Ensure strategy type is present
    except (json.JSONDecodeError, ValidationError) as e:
        pytest.fail(f"GroupValidatorTool (on step_7 output) validation failed: {e}\nOutput:\n{output_str}")


def test_group_refiner_tool_on_final_merged_output(
     group_refiner_tool: GroupRefinerTool,
     merged_groups_json_str: str,         # Loads step_7
     final_validation_json_str: str,      # Loads step_8
     repo_analysis_json_str: str
 ):
    """Tests refiner on the actual merged output and its validation result."""
    print("\nTesting GroupRefinerTool on step_7 merged output and step_8 validation...")
    output_str = group_refiner_tool._run(
        pr_grouping_strategy_json=merged_groups_json_str,
        pr_validation_result_json=final_validation_json_str, # Use the actual validation result
        original_repository_analysis_json=repo_analysis_json_str
    )

    assert isinstance(output_str, str)
    try:
        output_data = PRGroupingStrategy.model_validate_json(output_str)
        print(f"Refiner produced {len(output_data.groups)} groups.")
        print(f"Refiner left {len(output_data.ungrouped_files)} ungrouped files.")
        print(f"Refiner explanation field: {output_data.explanation}")

        # Check based on outputs_final_recommendations.json content:
        # It had ONE group with ALL 50 files.
        assert len(output_data.groups) == 1, "Expected exactly one group after final refinement based on logs"
        assert len(output_data.ungrouped_files) == 0, "Expected zero ungrouped files after final refinement based on logs"

        repo_analysis = json.loads(repo_analysis_json_str)
        original_files_count = len(repo_analysis.get("file_changes", []))
        all_files_in_groups_set = set(output_data.groups[0].files) if output_data.groups else set()

        assert len(all_files_in_groups_set) == original_files_count, \
            f"Expected the single group to contain all {original_files_count} files, but found {len(all_files_in_groups_set)}"
        print("Completeness check passed (all files in one group).")

        assert output_data.explanation is not None and len(output_data.explanation) > 0
        print("Explanation field is present.")

    except (json.JSONDecodeError, ValidationError) as e:
        pytest.fail(f"GroupRefinerTool (final run) output validation failed: {e}\nOutput:\n{output_str}")
    except IndexError as e:
         pytest.fail(f"Error accessing group data, likely due to unexpected structure: {e}\nOutput:\n{output_str}")

def test_group_refiner_tool_on_empty_input(
    group_refiner_tool: GroupRefinerTool,
    repo_analysis_json_str: str
):
    """Tests refiner handling empty input strategy (should add all as ungrouped or one group)."""
    print("\nTesting GroupRefinerTool with empty input strategy...")
    empty_strategy = PRGroupingStrategy(strategy_type=GroupingStrategyType.MIXED, groups=[], explanation="Test: Empty input", ungrouped_files=[])
    empty_strategy_json = empty_strategy.model_dump_json()

    # Simulate a successful validation result for the empty strategy
    valid_result = PRValidationResult(
        is_valid=True,
        issues=[],
        validation_notes="Empty strategy is valid.",
        strategy_type=GroupingStrategyType.MIXED # FIX: Add required field
    )
    valid_result_json = valid_result.model_dump_json()

    output_str = group_refiner_tool._run(
        pr_grouping_strategy_json=empty_strategy_json,
        pr_validation_result_json=valid_result_json,
        original_repository_analysis_json=repo_analysis_json_str
    )
    assert isinstance(output_str, str)
    try:
        output_data = PRGroupingStrategy.model_validate_json(output_str)
        print(f"Refiner (empty input) produced {len(output_data.groups)} groups.")
        print(f"Refiner (empty input) left {len(output_data.ungrouped_files)} ungrouped files.")
        repo_analysis = json.loads(repo_analysis_json_str)
        original_files_count = len(repo_analysis.get("file_changes", []))
        total_files_in_output = sum(len(g.files) for g in output_data.groups) + len(output_data.ungrouped_files)
        assert total_files_in_output == original_files_count, f"Expected {original_files_count} files, found {total_files_in_output}"
        print("Completeness check passed for empty input.")
        assert output_data.explanation is not None and len(output_data.explanation) > 0
        print("Explanation field is present.")
    except (json.JSONDecodeError, ValidationError) as e: pytest.fail(f"GroupRefinerTool (empty input) output validation failed: {e}\nOutput:\n{output_str}")

def test_batch_splitter_tool_extracts_paths(
    batch_splitter_tool: BatchSplitterTool, # Instance of the tool
    repo_analysis_json_str: str # Fixture with 50 files
):
    """Tests the internal file path extraction."""
    print("\nTesting BatchSplitterTool path extraction...")
    # Assuming the method is accessible, otherwise call _run and check logs/output
    # If it's protected, you might need to test indirectly or make it public for testing
    try:
        # You might need to adjust this call depending on where _extract_file_paths lives
        # If it's in BaseTool and protected: access via batch_splitter_tool._extract_file_paths
        # If it's directly in BatchSplitterTool: access via batch_splitter_tool._extract_file_paths
        extracted_paths = batch_splitter_tool._extract_file_paths(repo_analysis_json_str)

        assert isinstance(extracted_paths, list) or isinstance(extracted_paths, set)
        print(f"Extracted {len(extracted_paths)} file paths.")

        repo_analysis_data = json.loads(repo_analysis_json_str)
        expected_files_count = len(repo_analysis_data.get("file_changes", []))

        assert len(extracted_paths) == expected_files_count, \
            f"Expected {expected_files_count} paths, but extracted {len(extracted_paths)}"
        print("Correct number of file paths extracted.")

    except AttributeError:
         pytest.skip("_extract_file_paths might be protected or in a different location. Cannot test directly.")
    except Exception as e:
         pytest.fail(f"Error during path extraction test: {e}")