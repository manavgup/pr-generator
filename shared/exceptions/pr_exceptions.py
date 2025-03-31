from pathlib import Path
from typing import List

class GitOperationError(Exception):
    """Base exception for Git operations failures."""
    pass

class RepositoryNotFoundError(GitOperationError):
    """Exception raised when a git repository is not found."""
    pass

class GitCommandError(GitOperationError):
    """Raised when a git command fails."""
    def __init__(self, command, returncode, stdout, stderr):
        self.command = command
        self.stderr = stderr
        self.return_code = returncode
        super().__init__(
            f"Git command '{' '.join(command)}' failed with exit code {returncode}.\n"
            f"Stderr:\n{stderr}"
        )