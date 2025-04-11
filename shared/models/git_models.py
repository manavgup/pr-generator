# Contains models directly related to Git operations
import uuid
from pathlib import Path
from typing import Optional, Dict, List

from pydantic import Field, computed_field

from .base_models import BaseModel, ConfigDict, FileType, FileStatusType

class LineChanges(BaseModel):
    """Represents aggregated line changes in a file (staged + unstaged)."""
    model_config = ConfigDict(extra='forbid')

    added: int = Field(default=0, description="Total lines added (staged + unstaged)")
    deleted: int = Field(default=0, description="Total lines deleted (staged + unstaged)")

    @property
    def total(self) -> int:
        """Total number of lines changed."""
        return self.added + self.deleted

class FileChange(BaseModel):
    """
    Model representing a single file change detected by git status,
    including metadata and optional summary statistics.
    """
    model_config = ConfigDict(
        extra='ignore',
        arbitrary_types_allowed=True # Allow pathlib.Path
    )
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this file change record.")

    # Change these to string paths for JSON serialization
    path: str = Field(..., description="The current path of the file relative to the repo root.")
    staged_status: FileStatusType = Field(..., description="Status of the file in the staging area (index).")
    unstaged_status: FileStatusType = Field(..., description="Status of the file in the working directory.")
    original_path: Optional[str] = Field(
        default=None,
        description="The original path if the file was renamed or copied (staged status R/C)."
    )
    file_type: FileType = Field(default=FileType.UNKNOWN, description="Type of file (text or binary)")
    changes: Optional[LineChanges] = Field(
        default=None, # Make optional, may not always be calculated
        description="Aggregated line change statistics (staged + unstaged)."
    )
    content_hash: Optional[str] = Field(default=None, description="Hash of the file content")
    token_estimate: Optional[int] = Field(default=None, description="Estimated tokens if diff is fetched")

    # Computed Fields (Using pathlib for computation but returning strings)
    @computed_field # type: ignore[misc]
    @property
    def directory(self) -> str:
        """The directory containing the file, relative to the repo root."""
        # Convert to Path for computation, then back to string
        path_obj = Path(self.path)
        parent = path_obj.parent
        return str(parent) if parent != Path('.') else "(root)"

    @computed_field # type: ignore[misc]
    @property
    def extension(self) -> Optional[str]:
        """The file extension, including the leading dot (e.g., '.py')."""
        path_obj = Path(self.path)
        return path_obj.suffix.lower() if path_obj.suffix else None

    @computed_field # type: ignore[misc]
    @property
    def filename(self) -> str:
        """The name of the file including the extension."""
        return Path(self.path).name

    # Convenience Properties based on Status
    @property
    def is_new(self) -> bool:
        """True if the file is newly added (staged) or untracked (unstaged)."""
        return self.staged_status == FileStatusType.ADDED or \
               self.unstaged_status == FileStatusType.UNTRACKED

    @property
    def is_deleted(self) -> bool:
        """True if the file is marked as deleted in either staged or unstaged state."""
        # Note: A file can be deleted in one state and modified/added in another!
        # This property checks if *any* deletion status exists.
        return self.staged_status == FileStatusType.DELETED or \
               self.unstaged_status == FileStatusType.DELETED

    @property
    def has_staged_changes(self) -> bool:
        """True if there are any changes staged for commit."""
        return self.staged_status != FileStatusType.NONE and \
               self.staged_status != FileStatusType.UNTRACKED # Untracked is inherently unstaged

    @property
    def has_unstaged_changes(self) -> bool:
        """True if there are any changes in the working directory not yet staged."""
        return self.unstaged_status != FileStatusType.NONE

    @property
    def total_changes(self) -> int:
        """Returns the total lines changed from the 'changes' field, or 0 if not calculated."""
        return self.changes.total if self.changes else 0

class DiffSummary(BaseModel):
    """Summary of a diff for context window management."""
    model_config = ConfigDict(frozen=False)
    
    original_size: int = Field(..., description="Original size of the diff in characters")
    summarized_size: int = Field(..., description="Size of the summarized diff in characters")
    key_sections: List[str] = Field(default_factory=list, description="Key sections extracted from diff")
    modification_patterns: Dict[str, int] = Field(default_factory=dict, description="Patterns of modifications")
    truncated: bool = Field(default=False, description="Whether the diff was truncated")
    
    @computed_field
    def compression_ratio(self) -> float:
        """Compression ratio of the summarized diff."""
        if self.original_size == 0:
            return 1.0
        return self.summarized_size / self.original_size