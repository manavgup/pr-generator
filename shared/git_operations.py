import os
import subprocess
import logging
from typing import List, Tuple

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

    def get_changed_files(self) -> List[FileChange]:
        """Get all changed files with their diffs"""
        logger.info("Getting changed files and their diffs")
        
        # Get changed files with stats
        stat_output = self.run_git_command(["git", "diff", "--numstat"])
        
        changes = []
        for line in stat_output.splitlines():
            if not line.strip():
                continue
                
            try:
                added, deleted, file_path = line.split('\t')
                # Get short diff summary
                diff_summary = self.run_git_command(
                    ["git", "diff", "--shortstat", "--", file_path]
                ).strip()
                
                # Get actual diff
                full_diff = self.run_git_command(
                    ["git", "diff", "--", file_path]
                )
                
                # Create FileChange using the new structure
                added_lines = int(added) if added != '-' else 0
                deleted_lines = int(deleted) if deleted != '-' else 0
                
                changes.append(FileChange(
                    file_path=file_path,
                    changes=LineChanges(
                        added=added_lines,
                        deleted=deleted_lines
                    ),
                    diff=full_diff
                ))
                
            except ValueError as e:
                logger.warning(f"Couldn't parse stat line: {line} - {e}")
        
        return changes
    
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
                        added=int(added),
                        deleted=int(deleted)
                    ),
                    diff=""  # Diff not needed at this stage
                ))

        return changes
    
    def get_diffs_for_group(self, group: ChangeGroup) -> List[str]:
        """Fetches detailed diffs for a given ChangeGroup."""
        diffs = []
        for file_path in group.files:
            command = ["git", "diff", file_path]
            diffs.append(self.run_git_command(command))
        return diffs


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