# START OF FILE batching_models.py
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from shared.models.git_models import FileChange # Assuming git_models is in the same parent dir
from .agent_models import PatternAnalysisResult, PRGroupingStrategy # Assuming agent_models is in the same parent dir
from shared.models.analysis_models import RepositoryAnalysis # Assuming analysis_models is in the same parent dir

# --- Input/Output for Batch Splitter ---

class BatchSplitterInput(BaseModel):
    """Input for the BatchSplitterTool."""
    repository_analysis: RepositoryAnalysis = Field(..., description="The overall analysis of the repository.")
    # Pass analysis results for potentially smarter splitting
    pattern_analysis: Optional[PatternAnalysisResult] = Field(None, description="Global pattern analysis results.")
    target_batch_size: int = Field(default=50, description="Desired number of files per batch.")
    # Add other potential strategy hints here if needed later

class BatchSplitterOutput(BaseModel):
    """Output of the BatchSplitterTool."""
    batches: List[List[str]] = Field(..., description="List of batches, where each inner list contains file paths (or file_ids).")
    strategy_used: str = Field(..., description="Description of the splitting strategy applied.")
    notes: Optional[str] = Field(None, description="Any relevant notes about the splitting.")

# --- Input/Output for Group Merger ---

class GroupMergingInput(BaseModel):
    """Input for the GroupMergingTool."""
    batch_grouping_results: List[PRGroupingStrategy] = Field(..., description="List of PRGroupingStrategy results, one from each processed batch.")
    original_repository_analysis: RepositoryAnalysis = Field(..., description="The original, full repository analysis for context and ensuring all files are covered.")
    # Pass analysis results for potentially smarter merging
    pattern_analysis: Optional[PatternAnalysisResult] = Field(None, description="Global pattern analysis results.")
    # Add other potential strategy hints here if needed later

class GroupMergingOutput(BaseModel):
    """Output of the GroupMergingTool."""
    merged_grouping_strategy: PRGroupingStrategy = Field(..., description="The final PRGroupingStrategy containing merged and potentially refined groups.")
    unmerged_files: List[str] = Field(default_factory=list, description="List of file paths that couldn't be merged or assigned to a group.") # Should ideally be empty
    notes: Optional[str] = Field(None, description="Any relevant notes about the merging process.")

# --- Context model for Worker Crew ---
# This defines what the Supervisor passes to the Worker Crew for each batch

class WorkerBatchContext(BaseModel):
    """Context provided to the worker crew for processing a single batch."""
    batch_file_paths: List[str] = Field(..., description="List of file paths (or file_ids) to process in this batch.")
    # Include full analysis/patterns if they fit and are useful for batch-level decisions
    # Alternatively, pass only summaries or references if too large
    repository_analysis: RepositoryAnalysis = Field(..., description="The full repository analysis (or a relevant subset/summary).")
    pattern_analysis: Optional[PatternAnalysisResult] = Field(None, description="Global pattern analysis results.")
    # Pass the globally chosen strategy
    grouping_strategy_decision: Any # Replace 'Any' with your actual GroupingStrategyDecision model if defined
    # Potentially other global context needed by worker tasks
    repo_path: str = Field(..., description="Path to the repository.")


# END OF FILE batching_models.py