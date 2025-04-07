"""
Base tool implementations for the PR Recommendation System.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
from crewai.tools import BaseTool
from pydantic import BaseModel
from shared.utils.logging_utils import get_logger
from shared.tools.git_operations import GitOperations

logger = get_logger(__name__)


class BaseRepoTool(BaseTool, ABC):
    """Base class for repository analysis tools."""
    
    # Class-level cache across all instances
    _git_ops_cache: Dict[str, GitOperations] = {}
    
    def _get_git_ops(self, repo_path: str) -> GitOperations:
        """
        Get a GitOperations instance for the given repository path.
        Caches instances for better performance.
        """
        if repo_path not in self._git_ops_cache:
            self._git_ops_cache[repo_path] = GitOperations(repo_path)
        return self._git_ops_cache[repo_path]
    
    def _before_run(self, **kwargs) -> None:
        """Prepare GitOperations before running the tool."""
        # Try to get repo_path from kwargs
        repo_path = kwargs.get("repo_path")
        
        # If not directly in kwargs, look for it in input model
        if not repo_path:
            for value in kwargs.values():
                if isinstance(value, BaseModel) and hasattr(value, "repo_path"):
                    repo_path = value.repo_path
                    break
        
        # Initialize git_ops if we found a repo_path
        if repo_path:
            logger.info(f"Running {self.name} on repository: {repo_path}")
            self.git_ops = self._get_git_ops(repo_path)
    
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool with proper initialization and error handling."""
        self._before_run(**kwargs)
        start_time = time.time()
        # Use self.name for logging
        logger.info(f"⏳ Starting execution: Tool '{self.name}'")
        try:
            result = self._run(**kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ Finished execution: Tool '{self.name}' took {duration:.4f} seconds.")
            
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"❌ Failed execution: Tool '{self.name}' after {duration:.4f} seconds. Error: {e}", exc_info=True) 
            raise
    
    @abstractmethod
    def _run(self, **kwargs) -> Any:
        """Implement this method to define tool functionality."""
        pass