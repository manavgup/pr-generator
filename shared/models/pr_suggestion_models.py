from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import Field, computed_field, field_validator

from .base_models import BaseModel, ConfigDict, logger
from .git_models import FileChange


class ChangeGroup(BaseModel):
    """Represents a logical grouping of related file changes."""
    model_config = ConfigDict(frozen=False)
    
    name: str = Field(..., description="Name for this group of changes")
    files: List[str] = Field(..., description="List of file paths in this group")
    
    @field_validator('files')
    @classmethod
    def ensure_unique_files(cls, v):
        """Ensure file paths are unique."""
        return list(dict.fromkeys(v))  # Remove duplicates while preserving order

class ChangeClassificationRequest(BaseModel):
    """Request model for LLM-based change classification."""
    model_config = ConfigDict(extra='forbid')
    files: List[Dict[str, Any]] = Field(..., description="List of file changes with metadata (consider using FileChange.model_dump() here)")

class PRSuggestionRequest(BaseModel):
    """Request model for generating PR suggestions from grouped changes."""
    model_config = ConfigDict(extra='forbid')
    group: ChangeGroup
    # Pass file paths; diffs can be fetched by the tool/agent using GitOperations
    # file_paths: List[Path] = Field(..., description="Paths of files in the group")
    # Or pass FileChange objects if more context needed:
    file_changes: List[FileChange] = Field(..., description="FileChange objects in the group")

class PullRequestGroup(BaseModel):
    """Model representing a group of files for a potential PR."""
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    title: str = Field(..., description="PR title")
    files: List[Path] = Field(default_factory=list, description="List of file paths (relative to repo root) in this PR group") # Use Path
    rationale: str = Field(..., description="Explanation for grouping these changes")
    suggested_branch: str = Field(..., description="Git branch name for this PR")
    description: Optional[str] = Field(None, description="Detailed PR description")

    @computed_field # type: ignore[misc]
    @property
    def primary_directory(self) -> Optional[Path]: # Return Path
        """Calculate the primary directory based on file paths."""
        if not self.files:
            return None

        dir_counts: Dict[Path, int] = {}
        for file_path in self.files:
            # Use the directory property logic from FileChange
            directory = file_path.parent if file_path.parent != Path('.') else Path('(root)')
            dir_counts[directory] = dir_counts.get(directory, 0) + 1

        # Return the most common directory Path object
        if not dir_counts:
            return None
        return max(dir_counts, key=dir_counts.get) # type: ignore


class PRGroupCollection(BaseModel):
    """Model representing the grouping of changes into potential PRs."""
    model_config = ConfigDict(extra='forbid')
    description: Optional[str] = Field(None, description="Overall description of the grouping")
    pr_groups: List[PullRequestGroup] = Field(default_factory=list, description="List of PR groups")
    grouping_strategy: Optional[str] = Field(None, description="Strategy used for grouping")
    error: Optional[str] = Field(None, description="Error message if grouping failed")

    # No need for total_groups field if it's just len(pr_groups)
    # Use a computed field if desired, or calculate on access
    @computed_field # type: ignore[misc]
    @property
    def total_groups(self) -> int:
        return len(self.pr_groups)

    # Field validator can stay if you want filtering logic during validation
    @field_validator('pr_groups')
    @classmethod
    def validate_groups(cls, groups: List[PullRequestGroup]) -> List[PullRequestGroup]:
        """Filter out empty groups during validation."""
        valid_groups = [group for group in groups if group.files]
        if len(valid_groups) < len(groups):
            logger.warning(f"Filtered out {len(groups) - len(valid_groups)} empty PR groups.")
        return valid_groups

class PRSuggestion(BaseModel):
    """Model representing the final PR suggestions."""
    model_config = ConfigDict(extra='forbid')
    pr_suggestions: List[PullRequestGroup] = Field(default_factory=list, description="List of PR suggestions")
    message: Optional[str] = Field(None, description="Additional message about the PR suggestions")
    error: Optional[str] = Field(None, description="Error message if PR suggestion generation failed")
    description: Optional[str] = Field(None, description="Overall description of the suggestions")
    validation_result: Optional[Dict[str, Any]] = Field(None, description="Validation results")

    @computed_field # type: ignore[misc]
    @property
    def total_groups(self) -> int:
        return len(self.pr_suggestions)