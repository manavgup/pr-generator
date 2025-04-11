# START OF FILE batch_splitter_tool.py
from collections import defaultdict
import math
from pathlib import Path
import re
from typing import Type, List, Optional
from pydantic import BaseModel, Field

from .base_tool import BaseRepoTool 
from models.batching_models import BatchSplitterInput, BatchSplitterOutput
from shared.models.git_models import FileChange
from models.agent_models import PatternAnalysisResult
from shared.models.analysis_models import RepositoryAnalysis

from shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BatchSplitterToolSchema(BaseModel):
    """Input schema for BatchSplitterTool."""
    repository_analysis_json: str = Field(..., description="JSON string of the RepositoryAnalysis object.")
    pattern_analysis_json: Optional[str] = Field(None, description="JSON string of the PatternAnalysisResult object (optional).")
    target_batch_size: int = Field(default=50, description="Desired number of files per batch.")

class BatchSplitterTool(BaseRepoTool):
    name: str = "Batch Splitter Tool"
    description: str = "Splits a list of changed files into manageable batches based on size or other criteria."
    args_schema: Type[BaseModel] = BatchSplitterToolSchema

    def _run(
        self,
        repository_analysis_json: str,
        pattern_analysis_json: Optional[str] = None,
        target_batch_size: int = 50
    ) -> str:
        """Splits files into batches using adaptive sizing."""
        try:
            # Deserialize inputs
            repo_analysis = RepositoryAnalysis.model_validate_json(repository_analysis_json)
            
            # Calculate target complexity based on target_batch_size
            # This is a simple approximation - you may want to tune this
            target_complexity = target_batch_size * 1.5  # Average complexity per file * target size
            
            # Use adaptive batching
            batches = self._create_adaptive_batches(repo_analysis.file_changes, target_complexity)
            
            strategy_used = f"Adaptive complexity-based splitting into {len(batches)} batches"
            logger.info(f"Split into {len(batches)} adaptive batches based on file complexity")
            
            output = BatchSplitterOutput(
                batches=batches,
                strategy_used=strategy_used,
                notes=f"Target complexity was {target_complexity}"
            )
            return output.model_dump_json(indent=2)
            
        except Exception as e:
            logger.error(f"Error in BatchSplitterTool: {e}", exc_info=True)
            error_output = BatchSplitterOutput(batches=[], strategy_used="Error", 
                                            notes=f"Failed to split batches: {e}")
            return error_output.model_dump_json(indent=2)

    def _split_directory_files(self, files: List[str], target_size: int, 
                            pattern_analysis: Optional[PatternAnalysisResult]) -> List[List[str]]:
        """Split files from a large directory into related batches."""
        sub_batches = []
        
        # Use pattern information if available
        if pattern_analysis and pattern_analysis.naming_patterns:
            pattern_groups = defaultdict(list)
            matched_files = set()
            
            # Group by naming patterns
            for pattern in pattern_analysis.naming_patterns:
                for file in files:
                    filename = Path(file).name
                    if any(re.match(p, filename) for p in pattern.pattern):
                        pattern_groups[pattern.type].append(file)
                        matched_files.add(file)
            
            # Create batches from pattern groups
            for pattern_type, pattern_files in pattern_groups.items():
                # If pattern group too large, split by count
                if len(pattern_files) > target_size * 1.5:
                    for i in range(0, len(pattern_files), target_size):
                        sub_batches.append(pattern_files[i:i + target_size])
                else:
                    sub_batches.append(pattern_files)
            
            # Handle remaining files
            unmatched = [f for f in files if f not in matched_files]
            if unmatched:
                for i in range(0, len(unmatched), target_size):
                    sub_batches.append(unmatched[i:i + target_size])
        else:
            # Simple count-based splitting within directory
            for i in range(0, len(files), target_size):
                sub_batches.append(files[i:i + target_size])
                
        return sub_batches
    
    def _calculate_file_complexity(self, file_change: FileChange) -> float:
        """Calculate complexity score for a file based on size and type."""
        complexity = 1.0  # Base complexity
        
        # Factor in file size
        if file_change.changes:
            lines_changed = file_change.changes.added + file_change.changes.deleted
            if lines_changed > 500:
                complexity *= 3.0  # Very complex
            elif lines_changed > 200:
                complexity *= 2.0  # Moderately complex
            elif lines_changed > 50:
                complexity *= 1.5  # Somewhat complex
        
        # Factor in file type (can be customized based on your project)
        extension = Path(file_change.path).suffix
        if extension in ['.py', '.java', '.cpp']:
            complexity *= 1.5  # Code files are more complex
        elif extension in ['.yml', '.json', '.toml']:
            complexity *= 1.2  # Config files
        
        return complexity

    def _create_adaptive_batches(self, file_changes: List[FileChange], target_complexity: float) -> List[List[str]]:
        """Create batches based on file complexity instead of count."""
        batches = []
        current_batch = []
        current_complexity = 0.0
        
        # Sort files by complexity (descending) to distribute complex files better
        sorted_files = sorted(file_changes, 
                            key=lambda fc: self._calculate_file_complexity(fc),
                            reverse=True)
        
        for file_change in sorted_files:
            file_complexity = self._calculate_file_complexity(file_change)
            
            # If adding this file would exceed target complexity and batch isn't empty,
            # start a new batch
            if current_complexity + file_complexity > target_complexity and current_batch:
                batches.append([fc.path for fc in current_batch])
                current_batch = []
                current_complexity = 0.0
            
            # Add file to current batch
            current_batch.append(file_change)
            current_complexity += file_complexity
        
        # Add the last batch if it's not empty
        if current_batch:
            batches.append([fc.path for fc in current_batch])
        
        return batches

# END OF FILE batch_splitter_tool.py