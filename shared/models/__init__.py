"""
Shared models for PR generator and Git operations.

This module re-exports commonly used models for easier importing.
"""

# Base types
from .base_models import FileType, FileStatusType, BaseModel

# Git-related models
from .git_models import LineChanges, FileChange, DiffSummary

# Directory organization
from .directory_models import DirectorySummary  

# Analysis models
from .analysis_models import RepositoryAnalysis

from typing import List
# PR suggestion models
from .pr_suggestion_models import (
    ChangeGroup, 
    PullRequestGroup, 
    PRGroupCollection, 
    PRSuggestion, 
    ChangeClassificationRequest,
    PRSuggestionRequest
)

# Tool input/output models
from .tool_models import (
    GitAnalysisToolInput,
    GitAnalysisOutput,
    CodeGroupingToolInput
)

# Utility classes
from .utility_models import ProgressReporter

FileList = List[FileChange]
DirectoryList = List[DirectorySummary]