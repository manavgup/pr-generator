import re
import time
import subprocess
import concurrent.futures
import os
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

from shared.exceptions.pr_exceptions import GitOperationError, RepositoryNotFoundError, GitCommandError
from shared.utils.logging_utils import get_logger
from shared.models.base_models import FileType, FileStatusType
from shared.models.git_models import LineChanges, FileChange, DiffSummary
from shared.models.directory_models import DirectorySummary
from shared.models.analysis_models import RepositoryAnalysis
from shared.models.pr_suggestion_models import ChangeGroup
from shared.models.utility_models import ProgressReporter

logger = get_logger(__name__)


class GitOperations:
    """Enhanced class for Git operations with Pydantic models and performance optimizations."""
    
    def __init__(self, repo_path: str, verbose: bool = False):
        """
        Initialize GitOperations with repository path.
        
        Args:
            repo_path: Path to the git repository
            verbose: Whether to show verbose output
        """
        self.repo_path = os.path.abspath(repo_path)
        self.verbose = verbose
        
        # Verify this is a git repository
        if not os.path.exists(os.path.join(repo_path, '.git')):
            raise RepositoryNotFoundError(f"Not a git repository: {repo_path}")
        
        # Cache directory for storing analysis results
        self._cache_dir = os.path.join(self.repo_path, '.git', 'pr_generator_cache')
        os.makedirs(self._cache_dir, exist_ok=True)
        
        logger.info(f"Initialized GitOperations for repository: {self.repo_path}")
    
    def run_git_command(self, command: List[str], timeout: Optional[int] = None) -> str:
        """
        Run a git command in the repository directory with improved error handling.
        
        Args:
            command: Git command as a list of strings
            timeout: Command timeout in seconds
            
        Returns:
            Command output as a string
            
        Raises:
            GitCommandError: If the command fails
        """
        start_time = time.time()
        full_command = command.copy()
        
        if self.verbose:
            logger.debug(f"Running git command: {' '.join(full_command)}")
        
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=timeout
            )
            
            if result.returncode != 0:
                raise GitCommandError(
                    command=full_command,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr
                )
            
            if self.verbose:
                elapsed = time.time() - start_time
                logger.debug(f"Git command completed in {elapsed:.2f}s")
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out after {timeout}s: {' '.join(full_command)}")
            raise GitCommandError(
                command=full_command,
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s"
            )
        except subprocess.SubprocessError as e:
            logger.error(f"Git command failed: {' '.join(full_command)}\nError: {str(e)}")
            raise GitCommandError(
                command=full_command,
                returncode=-1,
                stdout="",
                stderr=str(e)
            )
    
    def list_unstaged_changes(self) -> List[Tuple[str, str]]:
        """
        List all unstaged files without getting full diffs.
        
        Returns:
            List of tuples containing status and file path
        """
        logger.info("Listing unstaged changes")
        output = self.run_git_command(["git", "status", "--porcelain"])
        changes = []
        
        for line in output.splitlines():
            if line.strip():
                status = line[:2]
                file_path = line[3:]
                changes.append((status, file_path))
        
        logger.info(f"Found {len(changes)} unstaged changes")
        return changes
    
    def get_changed_file_list(self) -> List[str]:
        """
        Get a quick list of all changed files without fetching diffs.
        Much faster than get_changed_files() when you only need file paths.
        
        Returns:
            List of changed file paths
        """
        logger.info("Getting quick list of changed files")
        
        try:
            # Get both staged and unstaged files
            output = self.run_git_command(["git", "diff", "--name-only", "HEAD"])
            changed_files = [path for path in output.splitlines() if path.strip()]
            
            # Also get untracked files
            untracked_output = self.run_git_command(["git", "ls-files", "--others", "--exclude-standard"])
            untracked_files = [path for path in untracked_output.splitlines() if path.strip()]
            
            all_files = list(set(changed_files + untracked_files))
            logger.info(f"Found {len(all_files)} changed files")
            return all_files
            
        except GitCommandError as e:
            # Handle special case for empty repositories
            if "ambiguous argument 'HEAD'" in e.stderr:
                logger.warning("Repository appears to be empty or without commits")
                # Get untracked files instead
                untracked_output = self.run_git_command(["git", "ls-files", "--others", "--exclude-standard"])
                untracked_files = [path for path in untracked_output.splitlines() if path.strip()]
                logger.info(f"Found {len(untracked_files)} untracked files")
                return untracked_files
            else:
                logger.error(f"Error getting changed file list: {e}")
                raise
    
    def get_changed_files_stats(self) -> List[Dict]:
        """
        Get all changed files with their stats including untracked files.
        """
        logger.info("Getting changed files with stats")
        
        # Get stats for tracked files with changes
        stat_output = self.run_git_command(["git", "diff", "--numstat"])
        
        # Parse stats for tracked files
        files_with_stats = []
        for line in stat_output.splitlines():
            if not line.strip():
                continue
                
            try:
                added, deleted, file_path = line.split('\t')
                files_with_stats.append({
                    'file_path': file_path,
                    'added': int(added) if added != '-' else 0,
                    'deleted': int(deleted) if deleted != '-' else 0
                })
            except ValueError as e:
                logger.warning(f"Couldn't parse stat line: {line} - {e}")
        
        # Get untracked files and add them with placeholder stats
        try:
            untracked_output = self.run_git_command(["git", "ls-files", "--others", "--exclude-standard"])
            untracked_files = [path for path in untracked_output.splitlines() if path.strip()]
            
            # Add untracked files (assuming they're all newly added)
            for file_path in untracked_files:
                # Get file size as a rough approximation of "added" lines
                try:
                    abs_path = os.path.join(self.repo_path, file_path)
                    if os.path.isfile(abs_path):
                        with open(abs_path, 'rb') as f:
                            content = f.read()
                            # Estimate lines as bytes divided by average line length (40 chars)
                            estimated_lines = len(content) // 40
                            files_with_stats.append({
                                'file_path': file_path,
                                'added': estimated_lines,
                                'deleted': 0  # New file, so no deletions
                            })
                except Exception as e:
                    logger.warning(f"Error estimating lines for {file_path}: {e}")
                    # Add with minimal info
                    files_with_stats.append({
                        'file_path': file_path,
                        'added': 1,  # At least one line
                        'deleted': 0
                    })
        except Exception as e:
            logger.error(f"Error getting untracked files: {e}")
        
        logger.info(f"Processed stats for {len(files_with_stats)} files")
        return files_with_stats
    
    def detect_file_type(self, file_path: str) -> FileType:
        """
        Detect whether a file is text or binary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileType enum value
        """
        try:
            # Use git's internal file type detection
            output = self.run_git_command(["git", "diff", "--no-index", "--numstat", "/dev/null", file_path])
            
            # If the output has "-" for line counts, it's a binary file
            if "-\t-\t" in output:
                return FileType.BINARY
            
            return FileType.TEXT
        except GitCommandError:
            # If git command fails, try a simple heuristic
            try:
                abs_path = os.path.join(self.repo_path, file_path)
                with open(abs_path, 'rb') as f:
                    chunk = f.read(4096)
                    # Check for null bytes which typically indicate binary content
                    if b'\x00' in chunk:
                        return FileType.BINARY
                    
                    # Try to decode as text
                    try:
                        chunk.decode('utf-8')
                        return FileType.TEXT
                    except UnicodeDecodeError:
                        return FileType.BINARY
            except Exception:
                return FileType.UNKNOWN
    
    def calculate_content_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate a hash of the file content for similarity detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content hash as a string, or None if the file cannot be read
        """
        try:
            # Get the current content
            output = self.run_git_command(["git", "show", f":{file_path}"], timeout=5)
            
            # Calculate hash of the content
            content_hash = hashlib.md5(output.encode('utf-8')).hexdigest()
            return content_hash
        except GitCommandError:
            # The file might be new and not in the index yet
            try:
                abs_path = os.path.join(self.repo_path, file_path)
                with open(abs_path, 'rb') as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()
                return content_hash
            except Exception as e:
                logger.warning(f"Could not calculate content hash for {file_path}: {e}")
                return None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text.
        This is a simple heuristic based on whitespace and punctuation.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation: 1 token per ~4 characters
        return len(text) // 4
    
    def summarize_diff(self, diff: str, max_size: int = 2000) -> DiffSummary:
        """
        Summarize a diff to fit within context window constraints.
        
        Args:
            diff: Original diff content
            max_size: Maximum size of the summarized diff
            
        Returns:
            DiffSummary object
        """
        original_size = len(diff)
        
        # If the diff is already small enough, return it as is
        if original_size <= max_size:
            return DiffSummary(
                original_size=original_size,
                summarized_size=original_size,
                key_sections=[diff],
                truncated=False
            )
        
        # Split the diff into chunks by file and extract key patterns
        chunks = re.split(r'(diff --git .*?\n)', diff)
        file_diffs = []
        
        # Recombine the chunks into file diffs
        for i in range(1, len(chunks) - 1, 2):
            if i + 1 < len(chunks):
                file_diffs.append(chunks[i] + chunks[i + 1])
        
        # If only one chunk and no file headers found, treat the whole diff as one file
        if not file_diffs and len(chunks) > 0:
            file_diffs = [diff]
        
        # Count modification patterns
        patterns: Dict[str, int] = {}
        pattern_re = re.compile(r'^\+.*?(\w+)\s*[=:]|^\+.*?("\w+"\s*[:,])', re.MULTILINE)
        for file_diff in file_diffs:
            for match in pattern_re.finditer(file_diff):
                pattern = match.group(1) or match.group(2)
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Extract important sections (function changes, imports, etc.)
        key_sections = []
        
        # First, extract file headers
        for file_diff in file_diffs:
            header_match = re.search(r'diff --git .*?\n(?:index .*?\n)?', file_diff)
            if header_match:
                key_sections.append(header_match.group(0))
        
        # Extract function/method changes
        func_pattern = re.compile(r'(@@ .*? @@.*?\n)(?:[\+\- ].*?\n)+?', re.MULTILINE)
        for file_diff in file_diffs:
            for match in func_pattern.finditer(file_diff):
                # Get the hunk header and a few lines of context
                hunk = match.group(0)
                # Limit each hunk to ~200 chars to ensure we get a variety
                if len(hunk) > 200:
                    hunk = hunk[:200] + "...[truncated]"
                key_sections.append(hunk)
        
        # Combine key sections up to max_size
        summarized_diff = ""
        for section in key_sections:
            if len(summarized_diff) + len(section) <= max_size:
                summarized_diff += section
            else:
                # If we can't fit more sections, truncate
                remaining = max_size - len(summarized_diff)
                if remaining > 50:  # Only add if we can include something meaningful
                    summarized_diff += section[:remaining] + "...[truncated]"
                break
        
        return DiffSummary(
            original_size=original_size,
            summarized_size=len(summarized_diff),
            key_sections=key_sections[:10],  # Limit to 10 key sections
            modification_patterns=patterns,
            truncated=True
        )
    
    def get_changed_files(self, 
                      max_files: Optional[int] = None,
                      use_summarization: bool = True,
                      max_diff_size: int = 2000) -> List[FileChange]:
        """
        Get all changed files with their diffs using an optimized approach.
        
        Args:
            max_files: Maximum number of files to process, or None for all
            use_summarization: Whether to summarize large diffs
            max_diff_size: Maximum size of diffs in characters
            
        Returns:
            List of FileChange objects
        """
        logger.info(f"Getting changed files with optimized method")
        
        # First, get the file stats (faster operation)
        file_stats_list = self.get_changed_files_stats()
        
        if not file_stats_list:
            logger.info("No files with changes found")
            return []
        
        # Create a mapping of file path to stats for easier lookup
        file_stats = {item['file_path']: {
            'added': item['added'],
            'deleted': item['deleted']
        } for item in file_stats_list}
        
        # Get all changed file paths
        changed_files = list(file_stats.keys())
        
        # Limit to max_files if specified
        if max_files is not None and max_files > 0:
            changed_files = changed_files[:max_files]
        
        # Set up progress reporting
        progress = ProgressReporter(len(changed_files), "Processing files")
        
        # Process files in batches to avoid command line length limitations
        MAX_BATCH_SIZE = 10
        all_changes: List[FileChange] = []
        
        def process_file_batch(file_batch: List[str]) -> List[FileChange]:
            batch_results = []
            for file_path in file_batch:
                try:
                    # Get diff with limited context to reduce size (for token estimation)
                    diff = self.run_git_command(["git", "diff", "--unified=3", "--", file_path])
                    
                    # Detect file type
                    file_type = self.detect_file_type(file_path)
                    
                    # Calculate content hash for similarity detection
                    content_hash = self.calculate_content_hash(file_path) if file_type == FileType.TEXT else None
                    
                    # Estimate tokens (even though we don't store the diff)
                    token_estimate = self.estimate_tokens(diff) if diff else 0
                    
                    # Create FileChange object without diff or diff_summary
                    file_change = FileChange(
                        path=Path(file_path),
                        staged_status=FileStatusType.MODIFIED,
                        unstaged_status=FileStatusType.NONE,
                        changes=LineChanges(
                            added=file_stats[file_path]['added'],
                            deleted=file_stats[file_path]['deleted']
                        ),
                        file_type=file_type,
                        content_hash=content_hash,
                        token_estimate=token_estimate
                    )
                    batch_results.append(file_change)
                    
                    # Store the diff separately if needed
                    # self._cache_diff(file_path, diff)
                    
                except Exception as e:
                    logger.error(f"Error processing diff for {file_path}: {e}")
                    # Add a minimal FileChange object to ensure the file is included
                    batch_results.append(FileChange(
                        path=Path(file_path),
                        staged_status=FileStatusType.UNTRACKED,
                        unstaged_status=FileStatusType.UNTRACKED,
                        changes=LineChanges(
                            added=file_stats.get(file_path, {}).get('added', 0),
                            deleted=file_stats.get(file_path, {}).get('deleted', 0)
                        )
                    ))
                progress.update()
            return batch_results
        
        # Split files into batches
        file_batches: List[List[str]] = []
        for i in range(0, len(changed_files), MAX_BATCH_SIZE):
            file_batches.append(changed_files[i:min(i + MAX_BATCH_SIZE, len(changed_files))])
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            results = list(executor.map(process_file_batch, file_batches))
            
        # Flatten results
        for batch_result in results:
            all_changes.extend(batch_result)
            
        logger.info(f"Processed {len(all_changes)} changed files")
        return all_changes
    
    def get_change_summary(self) -> List[FileChange]:
        """
        Retrieves a summary of all changes in the working directory.
        
        Returns:
            List of FileChange objects with minimal information
        """
        command = ["git", "diff", "--stat", "--numstat", "--no-color"]
        output = self.run_git_command(command)

        changes = []
        for line in output.split("\n"):
            parts = line.split("\t")
            if len(parts) == 3:
                added, deleted, file_path_str = parts
                changes.append(FileChange(
                    path=Path(file_path_str),
                    staged_status=FileStatusType.MODIFIED,  # Assuming modified for summary
                    unstaged_status=FileStatusType.NONE,
                    changes=LineChanges(
                        added=int(added) if added != '-' else 0,
                        deleted=int(deleted) if deleted != '-' else 0
                    )
                    # No diff or diff_summary fields
                ))

        return changes
    
    def get_diff(self, file_path: Union[str, Path]) -> str:
        """
        Get the diff for a file on demand.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Diff as a string
        """
        file_path_str = str(file_path) if isinstance(file_path, Path) else file_path
        return self.run_git_command(["git", "diff", "--unified=3", "--", file_path_str])

    def get_diff_summary(self, file_path: Union[str, Path], max_size: int = 2000) -> DiffSummary:
        """
        Get a summary of the diff for a file on demand.
        
        Args:
            file_path: Path to the file
            max_size: Maximum size of the summarized diff
            
        Returns:
            DiffSummary object
        """
        diff = self.get_diff(file_path)
        return self.summarize_diff(diff, max_size)

    def get_diffs_for_group(self, group: ChangeGroup) -> List[str]:
        """
        Fetches detailed diffs for a given ChangeGroup.
        
        Args:
            group: ChangeGroup object
            
        Returns:
            List of diff strings
        """
        MAX_BATCH_SIZE = 10
        all_diffs = []
        
        # Split files into batches
        file_batches = [group.files[i:i+MAX_BATCH_SIZE] 
                        for i in range(0, len(group.files), MAX_BATCH_SIZE)]
        
        # Process each batch
        for batch in file_batches:
            try:
                # Get diffs for all files in this batch with one command
                batch_command = ["git", "diff", "--unified=3", "--"] + batch
                batch_diff = self.run_git_command(batch_command)
                all_diffs.append(batch_diff)
            except GitCommandError as e:
                logger.error(f"Error getting diffs for batch: {e}")
                # Fall back to individual file processing if the batch command fails
                for file_path in batch:
                    try:
                        diff = self.run_git_command(["git", "diff", "--unified=3", "--", file_path])
                        all_diffs.append(diff)
                    except GitCommandError as inner_e:
                        logger.error(f"Error getting diff for {file_path}: {inner_e}")
        
        return all_diffs
    
    def analyze_repository(self, 
                          max_files: Optional[int] = None,
                          use_summarization: bool = True,
                          max_diff_size: int = 2000) -> RepositoryAnalysis:
        """
        Perform complete analysis of the repository.
        
        Args:
            max_files: Maximum number of files to process, or None for all
            use_summarization: Whether to summarize large diffs
            max_diff_size: Maximum size of diffs in characters
            
        Returns:
            RepositoryAnalysis object
        """
        logger.info(f"Starting full repository analysis for {self.repo_path}")
        
        # Get all file changes
        file_changes = self.get_changed_files(
            max_files=max_files,
            use_summarization=use_summarization,
            max_diff_size=max_diff_size
        )
        
        # Create directory summaries
        directories: Dict[str, DirectorySummary] = {}
        
        for change in file_changes:
            # Use the computed directory property from Path object
            directory_str = str(change.directory)
            
            if directory_str not in directories:
                directories[directory_str] = DirectorySummary(
                    path=directory_str,
                    file_count=0,
                    files=[],
                    total_changes=0,
                    extensions={}
                )
            
            # Update directory summary
            dir_summary = directories[directory_str]
            dir_summary.file_count += 1
            # Use str(change.path) to get file path as string
            dir_summary.files.append(str(change.path))
            dir_summary.total_changes += change.total_changes
            
            # Update extensions count - use computed property
            extension = change.extension or "none"
            dir_summary.extensions[extension] = dir_summary.extensions.get(extension, 0) + 1
        
        # Calculate total lines changed
        total_lines = sum(change.total_changes for change in file_changes)
        
        # Create repository analysis
        analysis = RepositoryAnalysis(
            repo_path=self.repo_path,
            file_changes=file_changes,
            directory_summaries=list(directories.values()),
            total_files_changed=len(file_changes),
            total_lines_changed=total_lines
        )
        
        logger.info(f"Completed repository analysis: {analysis.total_files_changed} files, "
                   f"{analysis.total_lines_changed} lines changed")
        
        return analysis
    
    def create_directory_tree(self) -> Dict[str, Any]:
        """
        Create a hierarchical directory tree of changed files.
        
        Returns:
            Nested dictionary representing directory structure
        """
        tree: Dict[str, Any] = {"name": "(root)", "children": {}, "files": []}
        
        # Get all changed files
        changed_files = self.get_changed_file_list()
        
        for file_path in changed_files:
            parts = file_path.split(os.path.sep)
            
            # Handle files in root directory
            if len(parts) == 1:
                tree["files"].append(file_path)
                continue
            
            # Navigate to the correct node in the tree
            current = tree
            for i, part in enumerate(parts[:-1]):  # All but the last part (filename)
                if part not in current["children"]:
                    current["children"][part] = {"name": part, "children": {}, "files": []}
                current = current["children"][part]
            
            # Add the file to the current directory
            current["files"].append(parts[-1])
        
        return tree
    
    def batch_process_large_repo(self, 
                                max_batch_size: int = 100,
                                use_summarization: bool = True) -> RepositoryAnalysis:
        """
        Process a large repository in batches to manage memory and LLM context size.
        
        Args:
            max_batch_size: Maximum number of files to process in each batch
            use_summarization: Whether to summarize large diffs
            
        Returns:
            Combined RepositoryAnalysis
        """
        logger.info(f"Batch processing large repository: {self.repo_path}")
        
        # Get all changed files without diffs
        all_files = self.get_changed_file_list()
        total_files = len(all_files)
        
        if total_files == 0:
            logger.info("No files found to process")
            return RepositoryAnalysis(repo_path=self.repo_path)
        
        logger.info(f"Found {total_files} files to process in batches of {max_batch_size}")
        
        # Process in batches
        all_changes: List[FileChange] = []
        batch_count = (total_files + max_batch_size - 1) // max_batch_size  # Ceiling division
        
        for batch_idx in range(batch_count):
            start_idx = batch_idx * max_batch_size
            end_idx = min(start_idx + max_batch_size, total_files)
            
            logger.info(f"Processing batch {batch_idx + 1}/{batch_count}: files {start_idx + 1}-{end_idx}")
            
            batch_files = all_files[start_idx:end_idx]
            
            # Process this batch of files
            for file_path in batch_files:
                try:
                    # Get full diff info
                    diff = self.run_git_command(["git", "diff", "--unified=3", "--", file_path])
                    
                    # Get stats
                    stats = self.get_changed_files_stats()
                    file_stat = next((s for s in stats if s['file_path'] == file_path), {'added': 0, 'deleted': 0})
                    
                    # Detect file type
                    file_type = self.detect_file_type(file_path)

                    # Estimate tokens (but don't store diff)
                    token_estimate = self.estimate_tokens(diff) if diff else 0

                    # Calculate content hash
                    content_hash = self.calculate_content_hash(file_path) if file_type == FileType.TEXT else None
                                      
                    # Create FileChange without diff
                    change = FileChange(
                        path=Path(file_path),
                        staged_status=FileStatusType.MODIFIED,
                        unstaged_status=FileStatusType.NONE,
                        changes=LineChanges(
                            added=file_stat['added'],
                            deleted=file_stat['deleted']
                        ),
                        diff=diff,
                        file_type=file_type,
                        token_estimate=token_estimate
                    )
                    all_changes.append(change)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    # Add minimal entry to ensure the file is included
                    all_changes.append(FileChange(
                        path=Path(file_path),
                        staged_status=FileStatusType.UNTRACKED,
                        unstaged_status=FileStatusType.UNTRACKED,
                        changes=LineChanges()
                    ))
        
        # Combine results into a single analysis
        all_dirs: Dict[str, DirectorySummary] = {}
        
        for change in all_changes:
            directory = change.directory
            if directory not in all_dirs:
                all_dirs[directory] = DirectorySummary(
                    path=directory,
                    file_count=0,
                    files=[],
                    total_changes=0,
                    extensions={}
                )
            
            # Update directory summary
            dir_summary = all_dirs[directory]
            dir_summary.file_count += 1
            dir_summary.files.append(change.file_path)
            dir_summary.total_changes += change.total_changes
            
            # Update extensions count
            extension = change.extension or "none"
            dir_summary.extensions[extension] = dir_summary.extensions.get(extension, 0) + 1
        
        # Calculate total lines changed
        total_lines = sum(change.total_changes for change in all_changes)
        
        # Create final repository analysis
        analysis = RepositoryAnalysis(
            repo_path=self.repo_path,
            file_changes=all_changes,
            directory_summaries=list(all_dirs.values()),
            total_files_changed=len(all_changes),
            total_lines_changed=total_lines
        )
        
        logger.info(f"Completed batch processing: {analysis.total_files_changed} files, "
                   f"{analysis.total_lines_changed} lines changed")
        
        return analysis
    
    def estimate_repository_tokens(self) -> int:
        """
        Estimate the total number of tokens needed to represent the repository changes.
        
        Returns:
            Estimated token count
        """
        # Get repository analysis with summarized diffs
        analysis = self.analyze_repository(use_summarization=True)
        
        # Sum token estimates
        total_tokens = 0
        for change in analysis.file_changes:
            if change.token_estimate:
                total_tokens += change.token_estimate
            else:
                # Fallback if token_estimate not available - fetch diff on demand
                diff = self.get_diff(change.path)
                total_tokens += len(diff) // 4  # Rough estimate
        
        # Add overhead for metadata
        metadata_tokens = sum(len(str(change.path)) // 4 for change in analysis.file_changes)
        overhead_tokens = 1000  # Baseline overhead
        
        logger.info(f"Estimated {total_tokens} tokens for diffs, "
                   f"{metadata_tokens} for metadata, {overhead_tokens} overhead")
        
        return total_tokens + metadata_tokens + overhead_tokens
    
    @lru_cache(maxsize=128)
    def check_complex_changes(self, file_path: str) -> bool:
        """
        Check if a file has complex changes like function moves or renames.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if complex changes detected
        """
        try:
            # Use git's -M option to detect renames and copies
            diff = self.run_git_command(["git", "diff", "-M", "-C", "--", file_path])
            
            # Check for similarity index, indicating rename or copy
            if re.search(r'similarity index \d+%', diff):
                return True
            
            # Check for large moved blocks
            if re.search(r'@@ -\d+,\d+ \+\d+,\d+ @@', diff):
                # Multiple hunks might indicate complex changes
                hunks = re.findall(r'@@ -\d+,\d+ \+\d+,\d+ @@', diff)
                if len(hunks) > 3:  # Arbitrary threshold, adjust as needed
                    return True
            
            return False
        except Exception:
            return False


# Standalone functions for easier importing

def get_changed_files(repo_path: str) -> List[FileChange]:
    """
    Standalone function to get changed files from a git repository.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        List of FileChange objects
    """
    git_ops = GitOperations(repo_path)
    return git_ops.get_changed_files()

def get_changed_file_list(repo_path: str) -> List[str]:
    """
    Quick function to get only the list of changed files.
    Much faster than the full get_changed_files.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        List of file paths
    """
    git_ops = GitOperations(repo_path)
    return git_ops.get_changed_file_list()

def get_changed_files_stats(repo_path: str) -> List[Dict]:
    """
    Get statistics about changed files without full diffs.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        List of dictionaries with file path and change statistics
    """
    git_ops = GitOperations(repo_path)
    return git_ops.get_changed_files_stats()

def analyze_repository(repo_path: str, 
                      max_files: Optional[int] = None,
                      use_summarization: bool = True) -> RepositoryAnalysis:
    """
    Perform a complete analysis of a git repository.
    
    Args:
        repo_path: Path to the git repository
        max_files: Maximum number of files to process
        use_summarization: Whether to summarize large diffs
        
    Returns:
        RepositoryAnalysis object
    """
    git_ops = GitOperations(repo_path)
    return git_ops.analyze_repository(max_files=max_files, use_summarization=use_summarization)