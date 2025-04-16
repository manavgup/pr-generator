#!/usr/bin/env python
"""
Test script to verify the entire PR recommendation toolchain works correctly.
This script runs each tool in sequence, using the output of one tool as input to the next.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- Configuration ---
REPO_PATH = "/Users/mg/mg-work/manav/work/ai-experiments/rag_modulo"  # Replace with your repo path
OUTPUT_DIR = Path("outputs")

# --- Tool Imports ---
try:
    from tools.base_tool import BaseRepoTool
    from tools.repo_analyzer_tool import RepoAnalyzerTool
    from tools.repo_metrics_tool import RepositoryMetricsCalculator
    from tools.pattern_analyzer_tool import PatternAnalyzerTool
    from tools.directory_analyzer_tool import DirectoryAnalyzer
    from tools.grouping_strategy_selector_tool import GroupingStrategySelector
    from tools.batch_splitter_tool import BatchSplitterTool
    from tools.file_grouper_tool import FileGrouperTool
    from tools.group_validator_tool import GroupValidatorTool
    from tools.group_refiner_tool import GroupRefinerTool
    from tools.group_merging_tool import GroupMergingTool
except ImportError as e:
    print(f"Failed to import tools: {e}")
    print("Ensure tools are in the 'tools/' directory and PYTHONPATH is set.")
    sys.exit(1)

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "tool_test.log", mode='w')
    ]
)
logger = logging.getLogger("PR_TOOLS_TEST")


def ensure_output_dir():
    """Ensure the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")


def save_output(stage: str, output_data: str):
    """Save tool output to a JSON file."""
    output_path = OUTPUT_DIR / f"{stage}.json"
    with open(output_path, "w") as f:
        f.write(output_data)
    logger.info(f"Saved {stage} output to {output_path}")
    return output_path


def load_json_from_file(path: str) -> str:
    """Load JSON from a file as a string."""
    logger.info(f"Loading JSON from file: {path}")
    with open(path, "r") as f:
        file_content = f.read()
    
    # Try to parse and determine if it's a CrewAI task result with a 'raw' field
    try:
        data = json.loads(file_content)
        # Check if this is a CrewAI task result with the actual data in 'raw'
        if 'raw' in data and isinstance(data['raw'], str):
            logger.info("Detected CrewAI task result format. Extracting 'raw' field.")
            # The actual repository analysis is in the 'raw' field
            return data['raw']
        else:
            # Return the original content if it's already the right format
            return file_content
    except json.JSONDecodeError:
        logger.warning(f"File {path} does not contain valid JSON. Using as-is.")
        return file_content


def test_repo_analyzer(repo_path: str, max_files: Optional[int] = None):
    """Test the Repository Analyzer Tool."""
    logger.info("=== Testing Repository Analyzer Tool ===")
    
    tool = RepoAnalyzerTool(repo_path=repo_path)
    result = tool.run(max_files=max_files)
    
    # Save output
    output_path = save_output("01_repository_analysis", result)
    logger.info(f"Repository Analyzer completed successfully. Output: {output_path}")
    
    return result


def test_repo_metrics(repository_analysis_json: str):
    """Test the Repository Metrics Calculator Tool."""
    logger.info("=== Testing Repository Metrics Calculator Tool ===")
    
    tool = RepositoryMetricsCalculator(repo_path=REPO_PATH)
    result = tool.run(repository_analysis_json=repository_analysis_json)
    
    # Save output
    output_path = save_output("02_repository_metrics", result)
    logger.info(f"Repository Metrics Calculator completed successfully. Output: {output_path}")
    
    return result


def test_pattern_analyzer(repository_analysis_json: str):
    """Test the Pattern Analyzer Tool."""
    logger.info("=== Testing Pattern Analyzer Tool ===")
    
    # Extract file paths from repository analysis JSON
    file_paths = []
    try:
        cleaned_json = BaseRepoTool._clean_json_string(None, repository_analysis_json)
        repo_analysis = json.loads(cleaned_json)
        file_changes = repo_analysis.get("file_changes", [])
        file_paths = [fc.get("path") for fc in file_changes if "path" in fc]
    except Exception as e:
        logger.error(f"Error extracting file paths: {e}")
        return None
    
    tool = PatternAnalyzerTool(repo_path=REPO_PATH)
    result = tool.run(file_paths=file_paths)
    
    # Save output
    output_path = save_output("03_pattern_analysis", result)
    logger.info(f"Pattern Analyzer completed successfully. Output: {output_path}")
    
    return result


def test_directory_analyzer(repository_analysis_json: str):
    """Test the Directory Analyzer Tool."""
    logger.info("=== Testing Directory Analyzer Tool ===")
    
    tool = DirectoryAnalyzer(repo_path=REPO_PATH)
    result = tool.run(repository_analysis_json=repository_analysis_json)
    
    # Save output
    output_path = save_output("04_directory_analysis", result)
    logger.info(f"Directory Analyzer completed successfully. Output: {output_path}")
    
    return result


def test_strategy_selector(repository_analysis_json: str, repository_metrics_json: str = None, pattern_analysis_json: str = None):
    """Test the Grouping Strategy Selector Tool."""
    logger.info("=== Testing Grouping Strategy Selector Tool ===")
    
    tool = GroupingStrategySelector(repo_path=REPO_PATH)
    result = tool.run(
        repository_analysis_json=repository_analysis_json,
        repository_metrics_json=repository_metrics_json,
        pattern_analysis_json=pattern_analysis_json
    )
    
    # Save output
    output_path = save_output("05_strategy_decision", result)
    logger.info(f"Grouping Strategy Selector completed successfully. Output: {output_path}")
    
    return result


def test_batch_splitter(repository_analysis_json: str, pattern_analysis_json: str = None, target_batch_size: int = 10):
    """Test the Batch Splitter Tool."""
    logger.info("=== Testing Batch Splitter Tool ===")
    
    tool = BatchSplitterTool(repo_path=REPO_PATH)
    result = tool.run(
        repository_analysis_json=repository_analysis_json,
        pattern_analysis_json=pattern_analysis_json,
        target_batch_size=target_batch_size
    )
    
    # Save output
    output_path = save_output("06_batch_splitting", result)
    logger.info(f"Batch Splitter completed successfully. Output: {output_path}")
    
    # Parse the result to get the batches
    result_obj = json.loads(result)
    batches = result_obj.get("batches", [])
    logger.info(f"Split into {len(batches)} batches")
    
    return result, batches


def test_file_grouper(batch_files: List[str], strategy_type_value: str, repository_analysis_json: str, pattern_analysis_json: Optional[str] = None):
    """Test the File Grouper Tool."""
    logger.info(f"=== Testing File Grouper Tool (Batch of {len(batch_files)} files) ===")
    
    tool = FileGrouperTool(repo_path=REPO_PATH)
    result = tool.run(
        batch_file_paths=batch_files,
        strategy_type_value=strategy_type_value,
        repository_analysis_json=repository_analysis_json,
        pattern_analysis_json=pattern_analysis_json
    )
    
    # Save output
    batch_num = len(os.listdir(OUTPUT_DIR)) - 6  # Adjust based on existing files
    output_path = save_output(f"07_{batch_num}_file_grouping", result)
    logger.info(f"File Grouper (Batch {batch_num}) completed successfully. Output: {output_path}")
    
    return result


def test_group_validator(pr_grouping_strategy_json: str, is_final_validation: bool = False):
    """Test the Group Validator Tool."""
    logger.info(f"=== Testing Group Validator Tool ({'Final' if is_final_validation else 'Batch'}) ===")
    
    tool = GroupValidatorTool(repo_path=REPO_PATH)
    result = tool.run(
        pr_grouping_strategy_json=pr_grouping_strategy_json,
        is_final_validation=is_final_validation
    )
    
    # Save output
    stage = "09_final_validation" if is_final_validation else "08_batch_validation"
    output_path = save_output(stage, result)
    logger.info(f"Group Validator completed successfully. Output: {output_path}")
    
    return result


def test_group_refiner(pr_grouping_strategy_json: str, pr_validation_result_json: str, original_repository_analysis_json: Optional[str] = None):
    """Test the Group Refiner Tool."""
    logger.info("=== Testing Group Refiner Tool ===")
    
    tool = GroupRefinerTool(repo_path=REPO_PATH)
    result = tool.run(
        pr_grouping_strategy_json=pr_grouping_strategy_json,
        pr_validation_result_json=pr_validation_result_json,
        original_repository_analysis_json=original_repository_analysis_json
    )
    
    # Save output
    stage = "10_final_refinement" if original_repository_analysis_json else "08_batch_refinement"
    output_path = save_output(stage, result)
    logger.info(f"Group Refiner completed successfully. Output: {output_path}")
    
    return result


def test_group_merger(batch_grouping_results_json: List[str], original_repository_analysis_json: str, pattern_analysis_json: Optional[str] = None):
    """Test the Group Merger Tool."""
    logger.info("=== Testing Group Merger Tool ===")
    
    tool = GroupMergingTool(repo_path=REPO_PATH)
    result = tool.run(
        batch_grouping_results_json=batch_grouping_results_json,
        original_repository_analysis_json=original_repository_analysis_json,
        pattern_analysis_json=pattern_analysis_json
    )
    
    # Save output
    output_path = save_output("09_merged_groups", result)
    logger.info(f"Group Merger completed successfully. Output: {output_path}")
    
    return result


def run_full_test(repo_path: str, max_files: Optional[int] = None, start_from_existing: Optional[str] = None):
    """Run the full toolchain test."""
    ensure_output_dir()
    
    logger.info(f"Starting full PR tools test for repo: {repo_path}")
    logger.info(f"Max files: {max_files if max_files else 'No limit'}")
    
    try:
        # 1. Repository Analysis
        if start_from_existing and os.path.exists(start_from_existing):
            repository_analysis_json = load_json_from_file(start_from_existing)
            logger.info(f"Loaded repository analysis from: {start_from_existing}")
            
            # Save it again to our output directory for consistency
            save_output("01_repository_analysis", repository_analysis_json)
        else:
            repository_analysis_json = test_repo_analyzer(repo_path, max_files)
        
        # 2. Repository Metrics
        repository_metrics_json = test_repo_metrics(repository_analysis_json)
        
        # 3. Pattern Analysis
        pattern_analysis_json = test_pattern_analyzer(repository_analysis_json)
        
        # 4. Directory Analysis (optional but useful)
        directory_analysis_json = test_directory_analyzer(repository_analysis_json)
        
        # 5. Strategy Selection
        strategy_decision_json = test_strategy_selector(
            repository_analysis_json, 
            repository_metrics_json, 
            pattern_analysis_json
        )
        strategy_decision = json.loads(strategy_decision_json)
        strategy_type = strategy_decision.get("strategy_type", "mixed")
        
        # 6. Batch Splitting
        batch_splitting_json, batches = test_batch_splitter(
            repository_analysis_json,
            pattern_analysis_json,
            target_batch_size=10
        )
        
        # 7. Process each batch through File Grouper, Validator, and Refiner
        batch_results = []
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            
            # File Grouper
            grouping_result = test_file_grouper(
                batch, 
                strategy_type, 
                repository_analysis_json, 
                pattern_analysis_json
            )
            
            # Group Validator (batch mode)
            validation_result = test_group_validator(grouping_result, False)
            
            # Group Refiner (batch mode)
            refined_result = test_group_refiner(grouping_result, validation_result)
            
            # Store the refined result for merging
            batch_results.append(refined_result)
        
        # 8. Merge batch results
        merged_result = test_group_merger(batch_results, repository_analysis_json, pattern_analysis_json)
        
        # 9. Final Validation
        final_validation_result = test_group_validator(merged_result, True)
        
        # 10. Final Refinement
        final_result = test_group_refiner(
            merged_result, 
            final_validation_result, 
            repository_analysis_json
        )
        
        logger.info("===== FULL TEST COMPLETED SUCCESSFULLY =====")
        logger.info(f"Final output: {OUTPUT_DIR}/10_final_refinement.json")
        
        return final_result
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the PR recommendation toolchain')
    parser.add_argument('--repo-path', type=str, default=REPO_PATH, 
                        help='Path to the repository to analyze')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to analyze')
    parser.add_argument('--start-from', type=str, default=None,
                        help='Path to existing repository analysis JSON file to start from')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Directory to save outputs')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Update global variables based on args
    REPO_PATH = args.repo_path
    OUTPUT_DIR = Path(args.output_dir)
    
    # Ensure the output directory exists
    ensure_output_dir()
    
    # Run the full test
    result = run_full_test(REPO_PATH, args.max_files, args.start_from)
    
    if result:
        print("\n=== TEST SUCCEEDED ===")
        print(f"Final PR recommendations are in: {OUTPUT_DIR}/10_final_refinement.json")
        sys.exit(0)
    else:
        print("\n=== TEST FAILED ===")
        print("Check the logs for details")
        sys.exit(1)