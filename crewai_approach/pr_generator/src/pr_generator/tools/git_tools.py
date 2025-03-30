import os
import logging
import json
import time
import subprocess
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, ClassVar

from shared.models.pr_models import (
    PRSuggestion, 
    PullRequestGroup, 
    FileChange,
    GitAnalysisToolInput,
    DirectorySummary,
    GitAnalysisOutput,
    LineChanges
)
from shared.git_operations import (
    get_changed_files, 
    get_changed_file_list,
    get_changed_files_stats,
    GitOperations
)

logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = "tool_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GitAnalysisTool(BaseTool):
    """Tool for analyzing git changes."""
    name: str = "analyze_git_changes"
    description: str = "Analyze changes in a Git repository to understand what files have changed and how"
    args_schema: type = GitAnalysisToolInput
    
    # Class variable to store repo path (not part of pydantic validation)
    _repo_path: ClassVar[str] = ""
    
    # Store quick mode flag as a class variable
    _use_quick_mode: ClassVar[bool] = False
    
    def __init__(self, repo_path: str, use_quick_mode: bool = False):
        """
        Initialize the tool with repository path.
        
        Args:
            repo_path: Path to the git repository
            use_quick_mode: If True, uses faster methods with fewer details
        """
        super().__init__()
        # Store repo_path using a method instead of direct attribute assignment
        self.set_repo_path(repo_path)
        self.set_quick_mode(use_quick_mode)
    
    def set_repo_path(self, repo_path: str) -> None:
        """Set the repository path for this tool."""
        # Store in a class variable that won't be validated by Pydantic
        self.__class__._repo_path = repo_path
        # You could also modify the description to include the repo path
        self.description = f"Analyze changes in the git repository at {repo_path}"
    
    def get_repo_path(self) -> str:
        """Get the currently set repository path."""
        return getattr(self.__class__, '_repo_path', '')
    
    def set_quick_mode(self, use_quick_mode: bool) -> None:
        """Set the quick mode flag."""
        self.__class__._use_quick_mode = use_quick_mode
        
    def get_quick_mode(self) -> bool:
        """Get the quick mode flag."""
        return getattr(self.__class__, '_use_quick_mode', False)
    
    def verify_git_file_count(self, repo_path: str) -> int:
        """Directly verify the number of changed files using git command."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
                cwd=repo_path
            )
            if result.returncode != 0:
                logger.error(f"Error running git command: {result.stderr}")
                return 0
                
            files = [f for f in result.stdout.splitlines() if f.strip()]
            return len(files)
        except Exception as e:
            logger.error(f"Error verifying file count: {e}")
            return 0
    
    def _run(self, repo_path: Optional[str] = None, query: Optional[str] = None) -> GitAnalysisOutput:
        """Run the tool with smarter chunking."""
        # Use provided repo_path or default to self.repo_path
        actual_repo_path = repo_path or self.get_repo_path()
        
        logger.info(f"Analyzing git changes in {actual_repo_path}")
        
        # Add timing metrics
        start_time = time.time()
        
        try:
            # Verify actual file count first
            file_count_start = time.time()
            actual_file_count = self.verify_git_file_count(actual_repo_path)
            file_count_end = time.time()
            logger.info(f"Verified file count: {actual_file_count} files in {file_count_end - file_count_start:.2f}s")
            
            # Get all file stats first (lightweight operation)
            file_stats_start = time.time()
            all_file_stats = get_changed_files_stats(actual_repo_path)
            file_stats_end = time.time()
            logger.info(f"Retrieved stats for {len(all_file_stats)} files in {file_stats_end - file_stats_start:.2f}s")
            
            # Get a complete listing of all directories with changes
            all_directories = {}
            for stat in all_file_stats:
                file_path = stat['file_path']
                directory = os.path.dirname(file_path) or "root"
                if directory not in all_directories:
                    all_directories[directory] = []
                all_directories[directory].append(file_path)
            
            # Process files in a smarter way to avoid context window limitations
            MAX_FILES_PER_DIRECTORY = 3  # Limit files per directory to ensure representation
            MAX_TOTAL_FILES = 50         # Limit total files to stay under context window
            
            # Select representative files from each directory
            selected_files = []
            for directory, files in all_directories.items():
                # Take up to MAX_FILES_PER_DIRECTORY from each directory
                selected_files.extend(files[:MAX_FILES_PER_DIRECTORY])
                # Break if we've reached our total limit
                if len(selected_files) >= MAX_TOTAL_FILES:
                    break
            
            # Further trim if we still have too many
            selected_files = selected_files[:MAX_TOTAL_FILES]
            
            logger.info(f"Selected {len(selected_files)} representative files for detailed analysis")
            
            # Process only the selected files with full diffs
            file_processing_start = time.time()
            changes_data = []
            
            # Create a mapping of file path to stats for easier lookup
            file_stats_map = {item['file_path']: item for item in all_file_stats}
            
            # Process selected files with diffs in batches
            MAX_BATCH_SIZE = 10
            file_batches = [selected_files[i:i+MAX_BATCH_SIZE] 
                        for i in range(0, len(selected_files), MAX_BATCH_SIZE)]
            
            for batch in file_batches:
                batch_results = []
                for file_path in batch:
                    try:
                        # Get diff with limited context to reduce size
                        diff = subprocess.run(
                            ["git", "diff", "--unified=3", "--", file_path],
                            capture_output=True,
                            text=True,
                            cwd=actual_repo_path
                        ).stdout
                        
                        # Get stats for this file
                        stats = file_stats_map.get(file_path, {'added': 0, 'deleted': 0})
                        
                        # Create FileChange object
                        file_change = FileChange(
                            file_path=file_path,
                            changes=LineChanges(
                                added=stats['added'],
                                deleted=stats['deleted']
                            ),
                            # Truncate very large diffs
                            diff=diff[:1000] + "..." if len(diff) > 1000 else diff
                        )
                        changes_data.append(file_change)
                    except Exception as e:
                        logger.error(f"Error processing diff for {file_path}: {e}")
            
            file_processing_end = time.time()
            logger.info(f"Processed {len(changes_data)} files with diffs in {file_processing_end - file_processing_start:.2f}s")
            
            # Create complete directory summaries for ALL files, not just the ones with diffs
            dir_groups = {}
            for stat in all_file_stats:
                file_path = stat['file_path']
                directory = os.path.dirname(file_path) or "root"
                if directory not in dir_groups:
                    dir_groups[directory] = []
                dir_groups[directory].append(file_path)
            
            # Create directory summaries
            directory_summaries = []
            for dir_name, files in dir_groups.items():
                directory_summaries.append(DirectorySummary(
                    name=dir_name,
                    file_count=len(files),
                    files=files
                ))
            
            # Create analysis result with CORRECT total files count
            analysis = GitAnalysisOutput(
                changes=changes_data,
                total_files_changed=actual_file_count,  # Use verified count
                directory_summaries=directory_summaries,
                repo_path=actual_repo_path
            )
            
            end_time = time.time()
            logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
            
            return analysis
            
        except Exception as e:
            logger.exception(f"Error analyzing git changes: {e}")
            error_result = {"error": str(e)}
            return json.dumps(error_result)


class QuickGitAnalysisTool(BaseTool):
    """A faster version of GitAnalysisTool that skips full diffs."""
    name: str = "quick_git_analysis"
    description: str = "Quickly analyze changes in a Git repository (file paths and stats only, no diffs)"
    args_schema: type = GitAnalysisToolInput
    
    # Class variable to store repo path (not part of pydantic validation)
    _repo_path: ClassVar[str] = ""
    
    def __init__(self, repo_path: str):
        """Initialize with quick mode enabled."""
        super().__init__()
        self.set_repo_path(repo_path)
    
    def set_repo_path(self, repo_path: str) -> None:
        """Set the repository path for this tool."""
        self.__class__._repo_path = repo_path
        self.description = f"Quickly analyze changes in the git repository at {repo_path} (stats only, no diffs)"
    
    def get_repo_path(self) -> str:
        """Get the currently set repository path."""
        return getattr(self.__class__, '_repo_path', '')
    
    def verify_git_file_count(self, repo_path: str) -> int:
        """Directly verify the number of changed files using git command."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
                cwd=repo_path
            )
            if result.returncode != 0:
                logger.error(f"Error running git command: {result.stderr}")
                return 0
                
            files = [f for f in result.stdout.splitlines() if f.strip()]
            return len(files)
        except Exception as e:
            logger.error(f"Error verifying file count: {e}")
            return 0
    
    def _run(self, repo_path: Optional[str] = None, query: Optional[str] = None) -> GitAnalysisOutput:
        """Run the tool with quick mode."""
        # Use provided repo_path or default to self.repo_path
        actual_repo_path = repo_path or self.get_repo_path()
        
        logger.info(f"Quick analyzing git changes in {actual_repo_path}")
        
        # Add timing metrics
        start_time = time.time()
        
        # Use a different cache file for quick mode
        output_file = os.path.join(OUTPUT_DIR, "quick_git_analysis_output.json")
        if os.path.exists(output_file):
            logger.info(f"Using cached quick git analysis from {output_file}")
            try:
                with open(output_file, 'r') as f:
                    cached_content = f.read()
                    logger.info(f"Using cached quick result (from {time.time() - start_time:.2f}s)")
                    return cached_content
            except Exception as e:
                logger.error(f"Error reading cached quick git analysis: {e}")
        
        try:
            # Verify actual file count first
            actual_file_count = self.verify_git_file_count(actual_repo_path)
            logger.info(f"Actual git file count: {actual_file_count}")
            
            # Get just file stats (faster)
            logger.info("Using quick mode for file analysis")
            file_stats = get_changed_files_stats(actual_repo_path)
            
            # Convert stats to FileChange objects (without full diffs)
            changes_data = []
            for stat in file_stats:
                changes_data.append(FileChange(
                    file_path=stat['file_path'],
                    status="modified",
                    changes=LineChanges(
                        added=stat['added'],
                        deleted=stat['deleted']
                    ),
                    diff=""  # No diff in quick mode
                ))
            
            # Group by directory for additional information
            dir_groups = {}
            for change in changes_data:
                # Extract directory from file path
                directory = os.path.dirname(change.file_path) or "root"
                if directory not in dir_groups:
                    dir_groups[directory] = []
                dir_groups[directory].append(change)
            
            # Create directory summaries
            directory_summaries = []
            for dir_name, files in dir_groups.items():
                directory_summaries.append(DirectorySummary(
                    name=dir_name,
                    file_count=len(files),
                    files=[f.file_path for f in files]
                ))
            
            # Create analysis result
            analysis = GitAnalysisOutput(
                changes=changes_data,
                total_files_changed=actual_file_count,  # Use verified count
                directory_summaries=directory_summaries,
                repo_path=actual_repo_path
            )
            
            end_time = time.time()
            logger.info(f"Quick analysis completed in {end_time - start_time:.2f} seconds")
            
            # Save result to file for caching
            try:
                with open(output_file, 'w') as f:
                    json_content = analysis.model_dump_json(indent=2)
                    f.write(json_content)
                    logger.info(f"Saved quick analysis to {output_file}")
            except Exception as e:
                logger.error(f"Error saving quick analysis result: {e}")
            
            return analysis
            
        except Exception as e:
            logger.exception(f"Error in quick git analysis: {e}")
            error_result = {"error": str(e)}
            
            # Save error to file
            try:
                with open(output_file, 'w') as f:
                    json.dump(error_result, f, indent=2)
            except Exception as file_e:
                logger.error(f"Error saving quick error result: {file_e}")
                
            return json.dumps(error_result)