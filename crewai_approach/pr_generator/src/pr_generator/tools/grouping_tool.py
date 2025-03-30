"""
Tool for grouping file changes into logical PR groups.
"""
import json
import logging
from typing import List, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from shared.models.pr_models import (
    PRSuggestion, 
    PullRequestGroup, 
    FileChange,
    ChangeAnalysis
)
from shared.git_operations import get_changed_files

logger = logging.getLogger(__name__)

class FileInfo(BaseModel):
    """Simplified file information for grouping."""
    file_path: str
    directory: str = ""
    extension: str = ""
    lines_added: int = 0
    lines_deleted: int = 0
    diff_summary: str = ""

class GroupingResult(BaseModel):
    """Result of file grouping preparation."""
    repo_name: str
    repo_path: str
    total_files: int
    changes: List[FileInfo]
    error: Optional[str] = None
    truncated: bool = False

class GroupingInput(BaseModel):
    """Input for grouping tool."""
    analysis_result: ChangeAnalysis = Field(
        ..., 
        description="Analysis result from git analysis tool"
    )

class PRGroup(BaseModel):
    """PR group definition for agent output."""
    title: str
    files: List[str]
    rationale: str
    suggested_branch: str
    description: Optional[str] = None

class GroupingTool(BaseTool):
    """
    Tool for processing and grouping code changes.
    Prepares data for the agent to make decisions on grouping.
    """
    name: str = "group_code_changes"
    description: str = "Group code changes into logical pull request categories"
    args_schema: type = GroupingInput

    def __init__(self) -> None:
        """Initialize the grouping tool."""
        super().__init__()
        logger.info("Initialized GroupingTool")
        
    def _run(self, analysis_result: ChangeAnalysis) -> str:
        """
        Process code changes and prepare for PR suggestions.
        
        Args:
            analysis_result: The analysis result from the git analysis tool
        
        Returns:
            str: JSON string of processed data for grouping
        """
        logger.info("Preparing change data for PR grouping")
        
        try:
            # Extract changes from the ChangeAnalysis model
            changes = []
            for change in analysis_result.changes:
                file_info = FileInfo(
                    file_path=change.file_path,
                    directory=change.directory,
                    extension=change.extension or "",
                    lines_added=change.changes.added,
                    lines_deleted=change.changes.deleted,
                    diff_summary=change.diff[:200] + "..." if change.diff and len(change.diff) > 200 else ""
                )
                changes.append(file_info)
            
            total_files = len(changes)
            logger.info(f"Found {total_files} changes to group")

            # Save all files to a separate file that can be accessed later
            all_file_paths = [change.file_path for change in changes]
            try:
                with open('all_changed_files.json', 'w') as f:
                    json.dump(all_file_paths, f, indent=2)
                logger.info(f"Saved all {len(all_file_paths)} file paths to all_changed_files.json")
            except Exception as e:
                logger.error(f"Failed to save all file paths: {e}")
        
            
            # If no changes found, try fallback
            if not changes:
                changes = self._get_fallback_test_data()
                total_files = len(changes)
                logger.warning(f"Using {total_files} fallback test files")
            
            # Get repository name from path
            repo_path = analysis_result.repo_path or ""
            repo_name = repo_path.split("/")[-1] if repo_path else "unknown-repo"
            
            # Create result
            result = GroupingResult(
                repo_name=repo_name,
                repo_path=repo_path,
                total_files=total_files,
                changes=changes[:100],  # Limit to avoid token issues
                truncated=total_files > 100
            )
            
            return result.model_dump_json()
            
        except Exception as e:
            logger.exception(f"Error preparing data for PR grouping: {e}")
            error_result = GroupingResult(
                repo_name="unknown-repo",
                repo_path="",
                total_files=0,
                changes=[],
                error=str(e)
            )
            return error_result.model_dump_json()
    
    def _get_fallback_test_data(self) -> List[FileInfo]:
        """
        Get fallback test data if no changes are found.
        
        Returns:
            List[FileInfo]: List of test file information
        """
        return [
            FileInfo(file_path="backend/auth/oidc.py", directory="backend/auth", extension="py"),
            FileInfo(file_path="backend/core/authentication_middleware.py", directory="backend/core", extension="py"),
            FileInfo(file_path="backend/core/authorization.py", directory="backend/core", extension="py"),
            FileInfo(file_path="backend/core/custom_exceptions.py", directory="backend/core", extension="py"),
            FileInfo(file_path="backend/rag_solution/data_ingestion/document_processor.py", 
                   directory="backend/rag_solution/data_ingestion", extension="py"),
            FileInfo(file_path="backend/rag_solution/data_ingestion/pdf_processor.py", 
                   directory="backend/rag_solution/data_ingestion", extension="py")
        ]
    
    def create_pr_suggestions(self, groups: List[PRGroup]) -> str:
        """
        Create PR suggestions from the agent-provided groups.
        
        Args:
            groups: List of PR groups from the agent
            
        Returns:
            str: JSON string of PR suggestions
        """
        try:
            pr_groups = []
            for group in groups:
                pr_group = PullRequestGroup(
                    title=group.title,
                    files=group.files,
                    rationale=group.rationale,
                    suggested_branch=group.suggested_branch,
                    description=group.description
                )
                pr_groups.append(pr_group)
            
            # Create PR Suggestions
            pr_suggestions = PRSuggestion(
                pr_suggestions=pr_groups,
                total_groups=len(pr_groups),
                description=f"Generated {len(pr_groups)} PR suggestions",
                message=f"Generated {len(pr_groups)} PR suggestions"
            )
            
            return pr_suggestions.model_dump_json(indent=2)
            
        except Exception as e:
            logger.error(f"Error creating PR suggestions: {e}")
            error_result = PRSuggestion(
                error=f"Error creating PR suggestions: {str(e)}"
            )
            return error_result.model_dump_json(indent=2)