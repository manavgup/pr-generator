# directory_models.py
import os
from typing import List, Dict, Optional

from pydantic import Field, computed_field

from .base_models import BaseModel, ConfigDict


class DirectorySummary(BaseModel):
    """Summary of files in a directory."""
    model_config = ConfigDict(frozen=False)
    
    path: str = Field(..., description="Directory path")
    file_count: int = Field(default=0, description="Number of files in this directory")
    files: List[str] = Field(default_factory=list, description="List of file paths in this directory")
    total_changes: int = Field(default=0, description="Total number of line changes in this directory")
    extensions: Dict[str, int] = Field(default_factory=dict, description="Count of file extensions in this directory")
    
    @computed_field
    def is_root(self) -> bool:
        """Whether this is the root directory."""
        return self.path == "(root)"
    
    @computed_field
    def depth(self) -> int:
        """Depth of the directory in the file hierarchy."""
        if self.is_root:
            return 0
        return self.path.count(os.path.sep) + 1
    
    @computed_field
    def parent_directory(self) -> Optional[str]:
        """Parent directory path."""
        if self.is_root:
            return None
        parent = os.path.dirname(self.path)
        return parent if parent else "(root)"
