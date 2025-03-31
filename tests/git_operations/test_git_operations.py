#!/usr/bin/env python3
"""
Test script for Git operations and models without mocking.
This script tests the actual functionality with a real git repository.
"""

import os
import sys
import tempfile
import shutil
import subprocess
import time
from pathlib import Path

# Import the modules to test - adjust these imports based on your actual structure
from shared.models.base_models import FileType, FileStatusType
from shared.models.git_models import LineChanges, FileChange, DiffSummary
from shared.models.directory_models import DirectorySummary
from shared.models.analysis_models import RepositoryAnalysis
from shared.models.pr_suggestion_models import ChangeGroup, PullRequestGroup
from shared.tools.git_operations import GitOperations

def create_test_repo():
    """Create a temporary git repository for testing."""
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, 'test_repo')
    os.makedirs(repo_path)
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    
    # Create and commit initial files
    create_file(repo_path, "README.md", "# Test Repository\n\nThis is a test repository.\n")
    create_file(repo_path, "src/main.py", "def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()\n")
    create_file(repo_path, "src/utils.py", "def helper():\n    return 'Helper function'\n")
    create_file(repo_path, "docs/README.md", "# Documentation\n\nThis is documentation.\n")
    
    # Add files to git
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
    
    return repo_path, temp_dir

def create_file(repo_path, relative_path, content):
    """Create a file in the test repository."""
    file_path = os.path.join(repo_path, relative_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)

def make_changes(repo_path):
    """Make various types of changes to test different scenarios."""
    # Modify existing file
    create_file(repo_path, "src/main.py", "def main():\n    print('Hello, modified world!')\n\nif __name__ == '__main__':\n    main()\n")
    
    # Add new file
    create_file(repo_path, "src/new_module.py", "def new_function():\n    return 'New function'\n")
    
    # Add file in new directory
    create_file(repo_path, "tests/test_main.py", "def test_main():\n    assert True\n")
    
    # Add binary file
    with open(os.path.join(repo_path, "binary_file.bin"), 'wb') as f:
        f.write(bytes([0, 1, 2, 3, 4, 5, 0, 255]))
    
    # Delete a file
    os.remove(os.path.join(repo_path, "docs/README.md"))

def test_git_operations():
    """Test GitOperations functionality with a real repository."""
    print("\n=== Testing GitOperations ===")
    
    # Create a test repository
    repo_path, temp_dir = create_test_repo()
    try:
        # Initialize GitOperations
        git_ops = GitOperations(repo_path, verbose=True)
        
        # Test initial state
        files = git_ops.get_changed_file_list()
        print(f"Initial changed files: {len(files)}")
        assert len(files) == 0, "Expected no changes in initial state"
        
        # Make changes
        make_changes(repo_path)
        
        # Test get_changed_file_list
        files = git_ops.get_changed_file_list()
        print(f"Changed files after modifications: {len(files)}")
        assert len(files) > 0, "Expected changes after modifications"
        
        # Test get_changed_files_stats
        stats = git_ops.get_changed_files_stats()
        print(f"File stats: {stats}")
        
        # Test file type detection
        if "binary_file.bin" in files:
            file_type = git_ops.detect_file_type("binary_file.bin")
            print(f"Detected file type for binary_file.bin: {file_type}")
            assert file_type == FileType.BINARY, "Expected binary file type"
        
        if "src/main.py" in files:
            file_type = git_ops.detect_file_type("src/main.py")
            print(f"Detected file type for src/main.py: {file_type}")
            assert file_type == FileType.TEXT, "Expected text file type"
        
        # Test full repository analysis
        analysis = git_ops.analyze_repository(use_summarization=True)
        print(f"Repository analysis: {analysis.total_files_changed} files, {analysis.total_lines_changed} lines changed")
        
        # Check directory summaries
        for dir_summary in analysis.directory_summaries:
            print(f"Directory {dir_summary.path}: {dir_summary.file_count} files")
        
        # Test token estimation
        tokens = git_ops.estimate_repository_tokens()
        print(f"Estimated repository tokens: {tokens}")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print("Temporary repository removed")

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
    assert file_change.extension == ".py", "Expected .py extension"
    assert file_change.directory == Path("src"), "Expected src directory"
    
    # Test DirectorySummary
    dir_summary = DirectorySummary(
        path="src",
        file_count=2,
        files=["src/main.py", "src/utils.py"],
        total_changes=15
    )
    print(f"Directory summary - path: {dir_summary.path}, file count: {dir_summary.file_count}, depth: {dir_summary.depth}")
    assert dir_summary.depth == 1, "Expected depth of 1 for src directory"
    
    # Test ChangeGroup
    change_group = ChangeGroup(
        name="Feature implementation",
        files=["src/main.py", "src/utils.py", "src/main.py"]  # Intentional duplicate
    )
    print(f"Change group - name: {change_group.name}, files: {change_group.files}")
    assert len(change_group.files) == 2, "Expected duplicate files to be removed"
    
    # Test PullRequestGroup
    pr_group = PullRequestGroup(
        title="Implement new feature",
        files=[Path("src/main.py"), Path("src/utils.py")],
        rationale="These files implement feature X",
        suggested_branch="feature/x-implementation"
    )
    print(f"PR group - title: {pr_group.title}, primary directory: {pr_group.primary_directory}")
    assert pr_group.primary_directory == Path("src"), "Expected primary directory to be src"

def main():
    """Run all tests."""
    print("Starting tests...")
    start_time = time.time()
    
    try:
        test_git_operations()
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