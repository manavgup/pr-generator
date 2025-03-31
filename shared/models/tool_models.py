from typing import List, Dict, Optional, Any
from pathlib import Path

from pydantic import Field, computed_field

from .base_models import BaseModel, ConfigDict
from .git_models import FileChange
from .directory_models import DirectorySummary

class GitAnalysisToolInput(BaseModel):
    """Input for git analysis tool."""
    model_config = ConfigDict(extra='forbid')
    repo_path: str # Keep as str here if tool input expects simple types
    query: Optional[str] = Field(None, description="Optional query (future use?)")
    include_stats: bool = Field(default=False, description="Whether to include line change stats")

class GitAnalysisOutput(BaseModel):
    """Output from GitAnalysisTool, using the updated FileChange model."""
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    changes: List[FileChange] = Field(..., description="List of detailed file changes")
    repo_path: Path = Field(..., description="Absolute path to the git repository") # Use Path
    directory_summaries: List[DirectorySummary] = Field(..., description="Summaries of changes by directory")

    @computed_field # type: ignore[misc]
    @property
    def total_files_changed(self) -> int:
        return len(self.changes)

class CodeGroupingToolInput(BaseModel):
    """Input for grouping tool."""
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    # Pass the full analysis output for context
    analysis_output: GitAnalysisOutput = Field(..., description="Analysis output from git analysis tool")