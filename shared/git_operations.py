import os
import subprocess
import logging
import concurrent.futures
from typing import List, Tuple, Dict

from shared.models.pr_models import FileChange, ChangeGroup, LineChanges

logger = logging.getLogger(__name__)


class GitOperations:
    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)
        if not os.path.exists(os.path.join(repo_path, '.git')):
            raise ValueError(f"Not a git repository: {repo_path}")
        logger.info(f"Initialized GitOperations for repository: {self.repo_path}")

    def run_git_command(self, command: List[str]) -> str:
        """Run a git command in the repository directory"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(command)}\nError: {e.stderr}")
            raise

    def list_unstaged_changes(self) -> List[Tuple[str, str]]:
        """List all unstaged files without getting full diffs"""
        output = self.run_git_command(["git", "status", "--porcelain"])
        changes = []
        for line in output.splitlines():
            if line.strip():
                status = line[:2]
                file_path = line[3:]
                changes.append((status, file_path))
        return changes
    
    def get_changed_file_list(self) -> List[str]:
        """
        Get a quick list of all changed files without fetching diffs.
        Much faster than get_changed_files() when you only need file paths.
        
        Returns:
            List[str]: List of changed file paths
        """
        logger.info("Getting quick list of changed files")
        
        # Get all changed file paths in one command
        try:
            # Try to get both staged and unstaged files
            output = self.run_git_command(["git", "diff", "--name-only", "HEAD"])
            return [path for path in output.splitlines() if path.strip()]
        except Exception as e:
            logger.error(f"Error getting changed file list: {e}")
            return []

    def get_changed_files_stats(self) -> List[Dict]:
        """
        Get all changed files with their stats (faster than full diffs).
        Returns basic file information and change statistics without the full diff content.
        
        Returns:
            List[Dict]: List of dictionaries containing file path and change statistics
        """
        logger.info("Getting changed files with stats only")
        
        # Get all stats in one command
        stat_output = self.run_git_command(["git", "diff", "--numstat"])
        
        # Parse stats
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
        
        return files_with_stats

    def get_changed_files(self) -> List[FileChange]:
        """
        Get all changed files with their diffs using an optimized approach:
        1. Get all stats in one command
        2. Get all file names in one command
        3. Process diffs in parallel batches
        """
        logger.info("Getting changed files with optimized method")
        
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
        
        # Process files in batches to avoid command line length limitations
        MAX_BATCH_SIZE = 10
        all_changes = []
        
        def process_file_batch(file_batch):
            batch_results = []
            for file_path in file_batch:
                try:
                    # Get diff with limited context to reduce size
                    diff = self.run_git_command(["git", "diff", "--unified=3", "--", file_path])
                    
                    # Create FileChange object
                    file_change = FileChange(
                        file_path=file_path,
                        changes=LineChanges(
                            added=file_stats[file_path]['added'],
                            deleted=file_stats[file_path]['deleted']
                        ),
                        # Truncate very large diffs
                        diff=diff[:2000] + "..." if len(diff) > 2000 else diff
                    )
                    batch_results.append(file_change)
                except Exception as e:
                    logger.error(f"Error processing diff for {file_path}: {e}")
            return batch_results
        
        # Split files into batches
        file_batches = [changed_files[i:i+MAX_BATCH_SIZE] 
                        for i in range(0, len(changed_files), MAX_BATCH_SIZE)]
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            results = list(executor.map(process_file_batch, file_batches))
            
        # Flatten results
        for batch_result in results:
            all_changes.extend(batch_result)
            
        logger.info(f"Processed {len(all_changes)} changed files")
        return all_changes
    
    def get_change_summary(self) -> List[FileChange]:
        """Retrieves a summary of all changes in the working directory."""
        command = ["git", "diff", "--stat", "--numstat", "--no-color"]
        output = self.run_git_command(command)

        changes = []
        for line in output.split("\n"):
            parts = line.split("\t")
            if len(parts) == 3:
                added, deleted, path = parts
                changes.append(FileChange(
                    file_path=path,
                    changes=LineChanges(
                        added=int(added) if added != '-' else 0,
                        deleted=int(deleted) if deleted != '-' else 0
                    ),
                    diff=""  # Diff not needed at this stage
                ))

        return changes
    
    def get_diffs_for_group(self, group: ChangeGroup) -> List[str]:
        """Fetches detailed diffs for a given ChangeGroup."""
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
            except Exception as e:
                logger.error(f"Error getting diffs for batch: {e}")
                # Fall back to individual file processing if the batch command fails
                for file_path in batch:
                    try:
                        diff = self.run_git_command(["git", "diff", "--unified=3", "--", file_path])
                        all_diffs.append(diff)
                    except Exception as inner_e:
                        logger.error(f"Error getting diff for {file_path}: {inner_e}")
        
        return all_diffs


# Standalone function for easier importing
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

# New standalone helper functions
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