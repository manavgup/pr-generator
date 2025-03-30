"""
Pydantic models for PR generation and management.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
import logging
import os

logger = logging.getLogger(__name__)

class LineChanges(BaseModel):
    """Represents line changes in a file."""
    added: int = Field(default=0, description="Number of lines added")
    deleted: int = Field(default=0, description="Number of lines deleted")
    
    @property
    def total(self) -> int:
        """Total number of lines changed."""
        return self.added + self.deleted

class FileChange(BaseModel):
    """Model representing a single file change in a git repository."""
    model_config = ConfigDict(
        extra='forbid',  # Be strict about extra fields
    )
    
    file_path: str = Field(..., description="Path to the changed file")
    status: Optional[str] = Field(None, description="Git status of the file (modified, added, etc.)")
    changes: LineChanges = Field(default_factory=LineChanges, description="Line change details")
    diff: Optional[str] = Field(None, description="Git diff content for the file")
    
    @property
    def extension(self) -> Optional[str]:
        """File extension."""
        _, ext = os.path.splitext(self.file_path)
        return ext.lstrip('.') if ext else None
    
    @property
    def directory(self) -> str:
        """Directory containing the file."""
        return os.path.dirname(self.file_path) or "(root)"
    
    @property
    def is_deleted(self) -> bool:
        """Whether the file was deleted."""
        return self.changes.deleted > 0 and self.changes.added == 0

    @property
    def total_changes(self) -> int:
        return self.changes.added + self.changes.deleted

class ChangeGroup(BaseModel):
    """Represents a logical grouping of related file changes."""
    model_config = ConfigDict(
        extra='forbid',
    )
    
    name: str = Field(..., description="LLM-generated name for this group of changes")
    files: List[str] = Field(..., description="List of file paths in this group")

class ChangeClassificationRequest(BaseModel):
    """Request model for LLM-based change classification."""
    model_config = ConfigDict(
        extra='forbid',
    )
    
    files: List[str] = Field(..., description="List of file changes with metadata")

class PRSuggestionRequest(BaseModel):
    """Request model for generating PR suggestions from grouped changes."""
    model_config = ConfigDict(
        extra='forbid',
    )
    
    group: ChangeGroup
    detailed_diffs: List[str] = Field(..., description="Full diffs of files in the group")

class ChangeAnalysis(BaseModel):
    """Model representing the analysis of all changes in a git repository."""
    model_config = ConfigDict(
        extra='forbid',
    )
    
    changes: List[FileChange] = Field(default_factory=list, description="List of file changes")
    total_files_changed: int = Field(0, description="Total number of files changed")
    repo_path: Optional[str] = Field(None, description="Path to the git repository")
    
class PullRequestGroup(BaseModel):
    """Model representing a group of files for a potential PR."""
    model_config = ConfigDict(
        extra='forbid',
    )
    
    title: str = Field(..., description="PR title")
    files: List[str] = Field(default_factory=list, description="List of files in this PR group")
    rationale: str = Field(..., description="Explanation for grouping these changes")
    suggested_branch: str = Field(..., description="Git branch name for this PR")
    description: Optional[str] = Field(None, description="Detailed PR description")
    
    @property
    def primary_directory(self) -> Optional[str]:
        """Calculate the primary directory based on file paths."""
        if not self.files:
            return None
            
        # Count occurrences of each directory
        dir_counts = {}
        for file_path in self.files:
            directory = os.path.dirname(file_path) or "(root)"
            dir_counts[directory] = dir_counts.get(directory, 0) + 1
            
        # Return the most common directory
        return max(dir_counts.items(), key=lambda x: x[1])[0]

class PRGroupCollection(BaseModel):
    """Model representing the grouping of changes into potential PRs."""
    model_config = ConfigDict(
        extra='forbid',
    )
    
    description: Optional[str] = Field(None, description="Overall description of the grouping")
    pr_groups: List[PullRequestGroup] = Field(default_factory=list, description="List of PR groups")
    total_groups: int = Field(0, description="Total number of generated groups")
    grouping_strategy: Optional[str] = Field(None, description="Strategy used for grouping")
    error: Optional[str] = Field(None, description="Error message if grouping failed")

    @field_validator('pr_groups')
    @classmethod
    def validate_groups(cls, groups: List[PullRequestGroup]) -> List[PullRequestGroup]:
        """Filter out empty and invalid groups."""
        valid_groups = []
        for group in groups:
            if group.files and len(group.files) > 0:
                valid_groups.append(group)
            else:
                logger.warning(f"Filtering out empty group: {group.title}")
        
        # Set the total_groups after filtering
        if hasattr(cls, 'total_groups'):
            cls.total_groups = len(valid_groups)
        
        return valid_groups

class PRSuggestion(BaseModel):
    """Model representing the final PR suggestions."""
    model_config = ConfigDict(
        extra='forbid',
    )
    
    pr_suggestions: List[PullRequestGroup] = Field(
        default_factory=list, 
        description="List of PR suggestions"
    )
    total_groups: int = Field(0, description="Total number of generated groups")
    message: Optional[str] = Field(None, description="Additional message about the PR suggestions")
    error: Optional[str] = Field(None, description="Error message if PR suggestion generation failed")
    description: Optional[str] = Field(None, description="Overall description of the suggestions")
    validation_result: Optional[Dict[str, Any]] = Field(None, description="Validation results")

# Tool schema models for CrewAI tool schemas
class GitAnalysisToolInput(BaseModel):
    """Input for git analysis tool."""
    model_config = ConfigDict(extra='forbid')
    
    repo_path: Optional[str] = Field(None, description="Path to the git repository")
    query: Optional[str] = Field(None, description="Optional query to filter changes")

class CodeGroupingToolInput(BaseModel):
    """Input for grouping tool."""
    model_config = ConfigDict(extra='forbid')
    
    analysis_result: Dict[str, Any] = Field(..., description="Analysis result from git analysis tool")

class DirectorySummary(BaseModel):
    """Summary of files in a directory."""
    name: str = Field(..., description="Directory path")
    file_count: int = Field(..., description="Number of files in this directory")
    files: List[str] = Field(..., description="List of file paths in this directory")

class GitAnalysisOutput(BaseModel):
    """Output from GitAnalysisTool."""
    changes: List[FileChange] = Field(..., description="List of file changes")
    total_files_changed: int = Field(..., description="Total number of files changed")
    repo_path: str = Field(..., description="Path to the git repository")
    directory_summaries: List[DirectorySummary] = Field(..., description="Summaries of changes by directory")