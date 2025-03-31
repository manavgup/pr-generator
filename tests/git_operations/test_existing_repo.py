#!/usr/bin/env python3
"""
Non-destructive test script for Git operations and models.
This script tests functionality with an existing git repository.
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
    args = parser.parse_args()
    
    print(f"Starting tests on repository: {args.repo_path}")
    start_time = time.time()
    
    try:
        test_git_operations(args.repo_path)
        test_models()
        
        elapsed = time.time() - start_time
        print(f"\nAll tests completed successfully in {elapsed:.2f} seconds!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()