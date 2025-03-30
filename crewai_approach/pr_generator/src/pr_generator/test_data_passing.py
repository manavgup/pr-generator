#!/usr/bin/env python
"""
Test script to diagnose data passing between agents.
"""
import json
import logging
import os
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_data_passing.log')
    ]
)
logger = logging.getLogger(__name__)

# Load agent output from file
def load_test_data(filename='code_analyzer_output.json'):
    """Load test data from file"""
    try:
        # First try to load from the file
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load from file: {e}")
        # Fall back to a sample if file not found
        return create_sample_data()

def create_sample_data():
    """Create a minimal sample dataset"""
    return {
        "changes": [
            {
                "file_path": "backend/auth/oidc.py",
                "status": "modified", 
                "changes": {"added": 28, "deleted": 4},
                "diff": "sample diff content"
            },
            {
                "file_path": "backend/core/authentication_middleware.py",
                "status": "modified",
                "changes": {"added": 40, "deleted": 27},
                "diff": "sample diff content"
            }
        ],
        "total_files_changed": 2,
        "repo_path": "/test/repo",
        "directory_summaries": [
            {
                "name": "backend/auth",
                "file_count": 1,
                "files": ["backend/auth/oidc.py"]
            },
            {
                "name": "backend/core",
                "file_count": 1,
                "files": ["backend/core/authentication_middleware.py"]
            }
        ]
    }

# Test the SummarizeChangesTool directly
def test_summarize_changes_tool(data):
    """Test the SummarizeChangesTool directly"""
    from tools.pr_strategy_tools import SummarizeChangesTool
    
    logger.info("Testing SummarizeChangesTool directly")
    tool = SummarizeChangesTool()
    
    # Test with different input formats
    
    # Test 1: Direct data
    logger.info("Test 1: Direct data")
    try:
        result = tool._run(data)
        logger.info(f"Test 1 result: {result[:100]}...")
    except Exception as e:
        logger.exception(f"Test 1 failed: {e}")
    
    # Test 2: Wrapped data
    logger.info("Test 2: Wrapped data")
    try:
        result = tool._run({"analysis_result": data})
        logger.info(f"Test 2 result: {result[:100]}...")
    except Exception as e:
        logger.exception(f"Test 2 failed: {e}")

def main():
    """Run diagnostic tests"""
    logger.info("Starting diagnostic tests")
    
    # Load test data
    data = load_test_data()
    logger.info(f"Loaded test data with {len(data.get('changes', []))} changes")
    
    # Test the SummarizeChangesTool
    test_summarize_changes_tool(data)
    
    logger.info("Diagnostic tests completed")

if __name__ == "__main__":
    main()