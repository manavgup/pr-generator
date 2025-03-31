# analysis_models
import time
from typing import List, Dict, Set, Optional

from pydantic import Field, computed_field

from .base_models import BaseModel, ConfigDict
from .git_models import FileChange
from .directory_models import DirectorySummary

class RepositoryAnalysis(BaseModel):
    """Complete analysis of a git repository."""
    model_config = ConfigDict(frozen=False)
    
    repo_path: str = Field(..., description="Path to the git repository")
    file_changes: List[FileChange] = Field(default_factory=list, description="List of file changes")
    directory_summaries: List[DirectorySummary] = Field(default_factory=list, description="Summaries by directory")
    total_files_changed: int = Field(default=0, description="Total number of files changed")
    total_lines_changed: int = Field(default=0, description="Total number of lines changed")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of the analysis")
    
    @computed_field
    def extensions_summary(self) -> Dict[str, int]:
        """Summary of file extensions."""
        extensions: Dict[str, int] = {}
        for change in self.file_changes:
            ext = change.extension or "none"
            extensions[ext] = extensions.get(ext, 0) + 1
        return extensions
    
    @computed_field
    def directories(self) -> Set[str]:
        """Set of all directories with changes."""
        return {change.directory for change in self.file_changes}
    
    def get_files_by_directory(self, directory: str) -> List[FileChange]:
        """Get all files in a specific directory."""
        return [change for change in self.file_changes if change.directory == directory]
    
    def get_files_by_extension(self, extension: str) -> List[FileChange]:
        """Get all files with a specific extension."""
        return [change for change in self.file_changes if change.extension == extension]
