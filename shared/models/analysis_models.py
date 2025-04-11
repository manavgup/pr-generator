# analysis_models
import time
from typing import List, Dict, Set, Optional
from pydantic import Field, computed_field

from .base_models import BaseModel, ConfigDict
from .git_models import FileChange
from .directory_models import DirectorySummary

class RepositoryAnalysis(BaseModel):
    """Model for repository analysis results"""
    model_config = ConfigDict(
        extra='ignore'
    )

    # Use string path instead of Path
    repo_path: str = Field(..., description="Path to the git repository")
    file_changes: List[FileChange] = Field(default_factory=list, description="List of file changes")
    directory_summaries: List[DirectorySummary] = Field(default_factory=list, description="Summary of changes by directory")
    total_files_changed: int = Field(default=0, description="Total number of files with changes")
    total_lines_changed: int = Field(default=0, description="Total number of lines changed (added + deleted)")
    timestamp: float = Field(default_factory=lambda: time.time(), description="Timestamp of analysis")
    error: Optional[str] = Field(None, description="Optional field to report errors during analysis.")

    # Computed properties
    @computed_field(repr=False)  # type: ignore[misc]
    @property
    def extensions_summary(self) -> Dict[str, int]:
        """Summary of file extensions and their counts"""
        extensions: Dict[str, int] = {}
        for file_change in self.file_changes:
            ext = file_change.extension or "none"
            if ext not in extensions:
                extensions[ext] = 0
            extensions[ext] += 1
        return extensions
    
    @computed_field(repr=False)  # type: ignore[misc]
    def directories(self) -> List[str]:
        """List of unique directories with changes (as strings)."""
        # Return a sorted list instead of a set
        unique_dirs = {str(change.directory) for change in self.file_changes}
        return sorted(list(unique_dirs))
    
    def get_files_by_directory(self, directory: str) -> List[FileChange]:
        """Get all files in a specific directory."""
        return [change for change in self.file_changes if change.directory == directory]
    
    def get_files_by_extension(self, extension: str) -> List[FileChange]:
        """Get all files with a specific extension."""
        return [change for change in self.file_changes if change.extension == extension]
