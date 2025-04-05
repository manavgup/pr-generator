"""
Repository analyzer tool that wraps GitOperations.
"""
from typing import Optional, Dict, Any
from pathlib import Path

from pydantic import BaseModel, Field

from shared.utils.logging_utils import get_logger
from shared.models.analysis_models import RepositoryAnalysis
from .base_tools import BaseRepoTool


logger = get_logger(__name__)


class RepoAnalyzerInput(BaseModel):
    """Input schema for Repository Analyzer Tool."""

    repo_path: str = Field(..., description="Path to the git repository")  # Required
    max_files: Optional[int] = Field(None, description="Maximum number of files to analyze")
    use_summarization: Optional[bool] = Field(True, description="Whether to summarize large diffs")
    max_diff_size: Optional[int] = Field(2000, description="Maximum size of diffs in characters")  # Added max_diff_size


class RepoAnalyzerTool(BaseRepoTool):
    """Tool for analyzing a git repository."""

    name: str = "Repository Analyzer"
    description: str = """
    Analyzes a git repository to extract metadata and changes.
    Identifies files that have been modified, their directories, and change statistics.
    Helps understand the structure and scale of changes in a repository.
    """
    args_schema: type[BaseModel] = RepoAnalyzerInput

    def _run(self, **kwargs) -> RepositoryAnalysis:
        """
        Analyze a git repository.

        Args:
            **kwargs: Keyword arguments containing the necessary parameters.

        Returns:
            RepositoryAnalysis object with raw repository data
        """
        logger.info(f"Analyzing repository with parameters: {kwargs}")

        # Extract parameters from kwargs (now that we know they're there)
        repo_path = kwargs.get("repo_path")
        max_files = kwargs.get("max_files")
        use_summarization = kwargs.get("use_summarization", True)  # Default to True if not provided
        max_diff_size = kwargs.get("max_diff_size", 2000)  # Default to 2000 if not provided

        if not repo_path:
            raise ValueError("repo_path is a required parameter.")

        try:
            # Get a GitOperations instance from the cache, use self.git_ops directly
            git_ops = self._get_git_ops(repo_path)

            # Use the existing analyze_repository function
            analysis = git_ops.analyze_repository(
                max_files=max_files,
                use_summarization=use_summarization,
                max_diff_size=max_diff_size,
            )

            return analysis  # Returns the raw analysis now

        except Exception as e:
            logger.error(f"Error analyzing repository: {e}")
            # Return a minimal valid result in case of error
            return RepositoryAnalysis(
                repo_path=repo_path,
                file_changes=[],
                directory_summaries=[],
                total_files_changed=0,
                total_lines_changed=0,
            )