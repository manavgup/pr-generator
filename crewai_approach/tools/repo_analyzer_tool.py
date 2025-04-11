# --- START OF FILE repo_analyzer.py ---
"""
Repository analyzer tool that wraps GitOperations.
Relies on BaseRepoTool to initialize GitOperations.
"""
import json
import time
from typing import Optional, Dict, Any, Type
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError # Added ValidationError

from shared.utils.logging_utils import get_logger
from shared.models.analysis_models import RepositoryAnalysis
from .base_tool import BaseRepoTool

logger = get_logger(__name__)


class RepoAnalyzerSchema(BaseModel):
    """Input schema for Repository Analyzer Tool."""
    max_files: Optional[int] = Field(None, description="Maximum number of files to analyze")
    use_summarization: Optional[bool] = Field(True, description="Whether to summarize large diffs")
    max_diff_size: Optional[int] = Field(2000, description="Maximum size of diffs in characters")


class RepoAnalyzerTool(BaseRepoTool):
    """Tool for analyzing a git repository using initialized GitOperations."""

    name: str = "Repository Analyzer"
    description: str = """
    Analyzes the git repository (path provided during setup) to extract metadata and changes.
    Identifies files that have been modified, their directories, and change statistics.
    Helps understand the structure and scale of changes in a repository.
    """
    args_schema: Type[BaseModel] = RepoAnalyzerSchema # Correct type hint

    # No need for __init__ here unless adding specific RepoAnalyzerTool state,
    # BaseRepoTool.__init__ handles repo_path and git_ops setup.

    def _run(
        self,
        max_files: Optional[int] = None,
        use_summarization: Optional[bool] = True,
        max_diff_size: Optional[int] = 2000
    ) -> str: # Return type is str
        """
        Analyze the git repository using the initialized GitOperations instance.

        Args:
            max_files: Maximum number of files to analyze.
            use_summarization: Whether to summarize large diffs.
            max_diff_size: Maximum size of diffs in characters.

        Returns:
            A JSON string serialization of the RepositoryAnalysis object.
        """

        # Check if git_ops was successfully initialized by BaseRepoTool.__init__
        if not hasattr(self, 'git_ops') or not self.git_ops:
             # This should ideally not happen if BaseRepoTool.__init__ raises errors
             logger.error(f"CRITICAL: GitOperations not available in {self.name}. Tool cannot operate.")
             error_data = {"error": f"Tool {self.name} failed: GitOperations not available."}
             # Try to construct a minimal error response
             return json.dumps(error_data) # Simple error JSON if we don't even know the repo_path

        # Use the repo_path from the initialized git_ops instance
        repo_path = self.git_ops.repo_path
        logger.info(f"Analyzing repository '{repo_path}' with max_files={max_files}, use_summarization={use_summarization}, max_diff_size={max_diff_size}")

        try:
            # Use the analyze_repository function from the initialized git_ops
            analysis: RepositoryAnalysis = self.git_ops.analyze_repository(
                max_files=max_files,
                use_summarization=use_summarization,
                max_diff_size=max_diff_size,
            )

            # Ensure the repo_path in the result matches the one used
            if analysis.repo_path != repo_path:
                 logger.warning(f"Repo path mismatch in analysis result. Expected '{repo_path}', got '{analysis.repo_path}'. Overwriting.")
                 analysis.repo_path = repo_path

            logger.info(f"Analysis complete for {repo_path}. Files processed: {analysis.total_files_changed}")
            # *** Return JSON string ***
            return analysis.model_dump_json(indent=2)

        except ValidationError as ve:
             # Error likely occurred within git_ops.analyze_repository creating the model
             error_msg = f"Pydantic validation error during analysis for '{repo_path}': {str(ve)}"
             logger.error(error_msg, exc_info=True)
             # Return a minimal valid result as JSON string in case of error
             error_analysis = RepositoryAnalysis(
                 repo_path=repo_path, # Use path from self.git_ops
                 error=error_msg
             )
             return error_analysis.model_dump_json(indent=2)
        except Exception as e:
            error_msg = f"Error analyzing repository '{repo_path}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return a minimal valid result as JSON string in case of error
            error_analysis = RepositoryAnalysis(
                repo_path=repo_path, # Use path from self.git_ops
                error=f"Analysis failed: {str(e)}"
            )
            return error_analysis.model_dump_json(indent=2)

# --- END OF FILE repo_analyzer.py ---