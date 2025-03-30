#!/usr/bin/env python
"""
Test script to diagnose SummarizeChangesTool input issue.
"""
import json
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import the tool and related models
from shared.models.pr_models import GitAnalysisOutput, DirectorySummary, FileChange, LineChanges
from tools.pr_strategy_tools import SummarizeChangesTool, SummarizeChangesInput

def main():
    """Run tests to diagnose the issue."""
    logger.info("Starting diagnostic test")
    
    # Create tool instance
    tool = SummarizeChangesTool()
    
    # Test 1: Pass a dictionary with metadata only (simulating what appears to be happening)
    test_input_1 = {
        'description': 'Analysis result from GitAnalysisTool',
        'type': 'GitAnalysisOutput'
    }
    
    logger.info("TEST 1: Passing metadata dictionary")
    try:
        result_1 = tool._run(test_input_1)
        logger.info(f"Test 1 result: {result_1}")
    except Exception as e:
        logger.exception(f"Test 1 failed with error: {e}")
    
    # Test 2: Pass a proper GitAnalysisOutput object
    test_input_2 = GitAnalysisOutput(
        changes=[
            FileChange(
                file_path="test.py",
                changes=LineChanges(added=10, deleted=5),
                diff="--- test.py\n+++ test.py"
            )
        ],
        total_files_changed=1,
        repo_path="/path/to/repo",
        directory_summaries=[
            DirectorySummary(
                name="/",
                file_count=1,
                files=["test.py"]
            )
        ]
    )
    
    logger.info("TEST 2: Passing proper GitAnalysisOutput object")
    try:
        # Convert to dict for tool
        result_2 = tool._run(test_input_2.model_dump())
        logger.info(f"Test 2 result: {result_2}")
    except Exception as e:
        logger.exception(f"Test 2 failed with error: {e}")
    
    # Test 3: Simulate what might be passed from CrewAI
    # This is a guess at what might actually be happening
    test_input_3 = {
        'analysis_result': {
            'description': 'Analysis result from GitAnalysisTool',
            'type': 'GitAnalysisOutput'
        }
    }
    
    logger.info("TEST 3: Simulating nested data structure")
    try:
        result_3 = tool._run(test_input_3)
        logger.info(f"Test 3 result: {result_3}")
    except Exception as e:
        logger.exception(f"Test 3 failed with error: {e}")

if __name__ == "__main__":
    main()