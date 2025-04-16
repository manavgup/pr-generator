#!/usr/bin/env python3
"""
Non-destructive test script for Git operations and models.
This script tests functionality with an existing git repository.
Enhanced with specific tests for max_files parameter.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import the modules to test - adjust these imports based on your actual structure
from shared.models.base_models import FileType, FileStatusType
from shared.models.git_models import LineChanges, FileChange
from shared.models.directory_models import DirectorySummary
from shared.models.analysis_models import RepositoryAnalysis
from shared.models.pr_suggestion_models import ChangeGroup, PullRequestGroup
from shared.tools.git_operations import GitOperations

def test_git_operations(repo_path: str):
    """Test GitOperations functionality with the specified repository."""
    print(f"\n=== Testing GitOperations on {repo_path} ===")
    
    # Initialize GitOperations
    git_ops = GitOperations(repo_path, verbose=True)
    
    # Test getting changed files list
    files = git_ops.get_changed_file_list()
    print(f"Changed files: {len(files)}")
    if files:
        print(f"Sample files: {files[:5]}")
        print(f"..." if len(files) > 5 else "")
    
    # Test getting file stats
    stats = git_ops.get_changed_files_stats()
    print(f"Files with stats: {len(stats)}")
    
    # Test file type detection if there are changed files
    if files and len(files) > 0:
        sample_file = files[0]
        file_type = git_ops.detect_file_type(sample_file)
        print(f"Detected file type for {sample_file}: {file_type}")
    
    # Test partial repository analysis (limit to 10 files to be safe)
    print("\nRunning partial repository analysis (limited to 10 files)...")
    analysis = git_ops.analyze_repository(max_files=10, use_summarization=True)
    print(f"Repository analysis: {analysis.total_files_changed} files (limited to 10), {analysis.total_lines_changed} lines changed")
    
    # Check directory summaries
    print("\nDirectory summaries:")
    for dir_summary in analysis.directory_summaries:
        print(f"- Directory {dir_summary.path}: {dir_summary.file_count} files")
    
    # Test token estimation for these files
    tokens = git_ops.estimate_repository_tokens()
    print(f"Estimated repository tokens: {tokens}")
    
    # Return analysis for further use if needed
    return analysis

def test_max_files_parameter(repo_path: str):
    """Specifically test the max_files parameter to verify it limits file processing."""
    print(f"\n=== Testing max_files Parameter on {repo_path} ===")
    
    # Initialize GitOperations
    git_ops = GitOperations(repo_path, verbose=True)
    
    # First, get total number of changed files for reference
    all_files = git_ops.get_changed_file_list()
    total_files = len(all_files)
    print(f"Total changed files in repository: {total_files}")
    
    if total_files < 2:
        print("Not enough changed files to test max_files parameter effectively.")
        return
    
    # Test with different max_files values
    max_files_values = [1, min(5, total_files), min(20, total_files)]
    
    for max_val in max_files_values:
        print(f"\nTesting with max_files={max_val}")
        
        # Time the operation
        start_time = time.time()
        
        # Run analysis with the specified max_files
        analysis = git_ops.analyze_repository(max_files=max_val, use_summarization=True)
        
        # Verify the number of files actually processed
        processed_files = len(analysis.file_changes)
        
        elapsed = time.time() - start_time
        print(f"Analysis completed in {elapsed:.2f} seconds")
        print(f"Files processed: {processed_files} (requested max: {max_val})")
        
        # Verify the parameter worked correctly
        assert processed_files <= max_val, f"Expected at most {max_val} files, but processed {processed_files}"
        
        if processed_files < max_val and processed_files < total_files:
            print(f"WARNING: Processed fewer files ({processed_files}) than requested ({max_val})")
        
        # Print the first few processed files
        print("Sample processed files:")
        for i, file_change in enumerate(analysis.file_changes[:3]):  # Show up to 3 files
            print(f"  {i+1}. {file_change.path}")
        
        if processed_files > 3:
            print(f"  ... and {processed_files - 3} more files")
    
    print("\nmax_files parameter test completed successfully!")

def test_compare_full_vs_limited(repo_path: str):
    """Compare full repository analysis versus limited analysis to verify differences."""
    print(f"\n=== Comparing Full vs Limited Analysis on {repo_path} ===")
    
    # Initialize GitOperations
    git_ops = GitOperations(repo_path, verbose=True)
    
    # Get total number of changed files
    all_files = git_ops.get_changed_file_list()
    total_files = len(all_files)
    
    if total_files < 5:
        print("Not enough changed files for meaningful comparison.")
        return
    
    # Choose a limit value less than the total
    limit = min(total_files // 2, 10)  # Half of total or 10, whichever is smaller
    
    print(f"Running full analysis (all {total_files} files)...")
    start_full = time.time()
    full_analysis = git_ops.analyze_repository(max_files=None)
    elapsed_full = time.time() - start_full
    
    print(f"Running limited analysis (max {limit} files)...")
    start_limited = time.time()
    limited_analysis = git_ops.analyze_repository(max_files=limit)
    elapsed_limited = time.time() - start_limited
    
    # Compare results
    print(f"\nFull analysis: {full_analysis.total_files_changed} files in {elapsed_full:.2f} seconds")
    print(f"Limited analysis: {limited_analysis.total_files_changed} files in {elapsed_limited:.2f} seconds")
    
    # Verify that limited analysis processed fewer files
    assert limited_analysis.total_files_changed <= limit, \
        f"Limited analysis should process at most {limit} files, but processed {limited_analysis.total_files_changed}"
    
    # Check if the first N files in both analyses match
    match_count = 0
    for i in range(min(limit, len(limited_analysis.file_changes))):
        if i < len(full_analysis.file_changes) and \
           str(limited_analysis.file_changes[i].path) == str(full_analysis.file_changes[i].path):
            match_count += 1
    
    print(f"Files matching between analyses: {match_count} of {len(limited_analysis.file_changes)}")
    
    # Performance comparison
    if elapsed_full > 0 and elapsed_limited > 0:
        speedup = elapsed_full / elapsed_limited
        print(f"Performance speedup with limited analysis: {speedup:.2f}x")
    
    print("\nComparison test completed!")

def test_models():
    """Test model functionality."""
    print("\n=== Testing Models ===")
    
    # Test LineChanges
    line_changes = LineChanges(added=10, deleted=5)
    print(f"Line changes - added: {line_changes.added}, deleted: {line_changes.deleted}, total: {line_changes.total}")
    assert line_changes.total == 15, "Expected total to be sum of added and deleted"
    
    # Test FileChange
    file_change = FileChange(
        path=Path("src/main.py"),
        staged_status=FileStatusType.MODIFIED,
        unstaged_status=FileStatusType.NONE,
        changes=line_changes
    )
    print(f"File change - path: {file_change.path}, directory: {file_change.directory}, extension: {file_change.extension}")
    
    # Test DirectorySummary
    dir_summary = DirectorySummary(
        path="src",
        file_count=2,
        files=["src/main.py", "src/utils.py"],
        total_changes=15
    )
    print(f"Directory summary - path: {dir_summary.path}, file count: {dir_summary.file_count}, depth: {dir_summary.depth}")
    
    # Test ChangeGroup
    change_group = ChangeGroup(
        name="Feature implementation",
        files=["src/main.py", "src/utils.py", "src/main.py"]  # Intentional duplicate
    )
    print(f"Change group - name: {change_group.name}, files: {change_group.files}")
    
    # Test PullRequestGroup
    pr_group = PullRequestGroup(
        title="Implement new feature",
        files=[Path("src/main.py"), Path("src/utils.py")],
        rationale="These files implement feature X",
        suggested_branch="feature/x-implementation"
    )
    print(f"PR group - title: {pr_group.title}, primary directory: {pr_group.primary_directory}")

def main():
    """Run all tests."""
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Test Git operations on an existing repository")
    parser.add_argument("repo_path", help="Path to the git repository")
    parser.add_argument("--max-files-test", action="store_true", help="Run specific tests for max_files parameter")
    parser.add_argument("--skip-basic", action="store_true", help="Skip basic tests")
    args = parser.parse_args()
    
    print(f"Starting tests on repository: {args.repo_path}")
    start_time = time.time()
    
    try:
        # Run basic tests unless skipped
        if not args.skip_basic:
            test_git_operations(args.repo_path)
            test_models()
        
        # Run max_files specific tests if requested or if no specific tests were selected
        if args.max_files_test or not args.skip_basic:
            test_max_files_parameter(args.repo_path)
            test_compare_full_vs_limited(args.repo_path)
        
        elapsed = time.time() - start_time
        print(f"\nAll tests completed successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()