# Standard imports
from shared.utils.logging_utils import get_logger
from enum import Enum

# Pydantic imports
from pydantic import BaseModel, ConfigDict

logger = get_logger(__name__)

class FileType(str, Enum):
    """Enum for file types."""
    TEXT = "text"
    BINARY = "binary"
    UNKNOWN = "unknown"


class FileStatusType(str, Enum):
    """Represents the status of a file in Git."""
    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNTRACKED = "?"
    UNMERGED = "U"
    NONE = " " # Represents no change in this specific index/worktree location