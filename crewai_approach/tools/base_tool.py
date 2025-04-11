"""
Base tool implementations for the PR Recommendation System.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import json
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict
from shared.utils.logging_utils import get_logger
from shared.tools.git_operations import GitOperations
from shared.exceptions.pr_exceptions import GitOperationError, RepositoryNotFoundError

logger = get_logger(__name__)


class BaseRepoTool(BaseTool, ABC):
    # Pydantic Config to allow custom types like GitOperations
    model_config = ConfigDict(arbitrary_types_allowed=True,
                              extra='allow')

    """Base class for repository analysis tools."""
    
    # Class-level cache across all instances
    _git_ops_cache = {}

    def __init__(self, repo_path: str, **kwargs) -> None:
        """
        Initialize the tool and the GitOperations instance.
        """
        # First initialize parent
        super().__init__(**kwargs)
        
        # Store repo_path locally
        self._repo_path = str(Path(repo_path).resolve())
        
        # Initialize GitOperations instance
        if self._repo_path not in self._git_ops_cache:
            logger.info(f"Creating new GitOperations instance for: {self._repo_path} (Tool: {self.name})")
            try:
                self._git_ops_cache[self._repo_path] = GitOperations(self._repo_path)
            except Exception as e:
                logger.error(f"Failed to initialize GitOperations: {e}")
                raise
        
        # Store git_ops in private attribute to avoid Pydantic validation
        self._git_ops = self._git_ops_cache[self._repo_path]
    
    # Property to access git_ops
    @property
    def git_ops(self) -> GitOperations:
        return self._git_ops

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool with logging and error handling."""
        # No need to call _before_run for git_ops initialization anymore
        start_time = time.time()
        logger.info(f"⏳ Starting execution: Tool '{self.name}'")
        logger.debug(f"Tool '{self.name}' called with kwargs: {kwargs.keys()}")

        try:
            # Filter kwargs based on args_schema before calling _run
            filtered_kwargs = {}
            if hasattr(self, 'args_schema') and self.args_schema:
                 schema_fields = self.args_schema.model_fields.keys()
                 for key, value in kwargs.items():
                      if key in schema_fields:
                           filtered_kwargs[key] = value
                 logger.debug(f"Filtered kwargs for {self.name}._run: {filtered_kwargs.keys()}")
            else:
                 filtered_kwargs = kwargs # Pass all if no schema

            # Check if git_ops was successfully initialized
            if not hasattr(self, 'git_ops') or not self.git_ops:
                 # This check should ideally not be needed if __init__ raises error on failure
                 logger.error(f"CRITICAL: git_ops not available in {self.name}.run(). Tool cannot operate.")
                 # Depending on the tool, either raise or return an error JSON
                 # Let's try returning error JSON consistent with other tools
                 error_json = json.dumps({"error": f"Tool {self.name} failed: GitOperations not available."})
                 return error_json # Or raise appropriate exception

            result = self._run(**filtered_kwargs) # Call _run with filtered args

            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ Finished execution: Tool '{self.name}' took {duration:.4f} seconds.")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"❌ Failed execution: Tool '{self.name}' after {duration:.4f} seconds. Error: {e}", exc_info=True)
            # Re-raise or return error JSON? Re-raising is often better for CrewAI to handle.
            raise
    
    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """Implement this method to define tool functionality."""
        pass