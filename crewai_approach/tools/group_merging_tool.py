# START OF FILE group_merging_tool.py
import json
from typing import Type, List, Dict, Set, Optional, Any

# Assuming pydantic models are defined
from pydantic import BaseModel, Field, ValidationError

from .base_tool import BaseRepoTool
# Assuming models.batching_models contains GroupMergingOutput
from models.batching_models import GroupMergingOutput
# Import specific models including the Enum from agent_models
# --- Added Import ---
from models.agent_models import PRGroupingStrategy, PRGroup, PatternAnalysisResult, GroupingStrategyType
# --- End Added Import ---
from shared.models.analysis_models import RepositoryAnalysis
from shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

class GroupMergingToolSchema(BaseModel):
    """Input schema for GroupMergingTool."""
    batch_grouping_results: List[Dict[str, Any]] = Field(..., description="A list of PRGroupingStrategy objects (as dicts) from batches.")
    original_repository_analysis_json: str = Field(..., description="JSON string of the original, full RepositoryAnalysis object.")
    pattern_analysis_json: Optional[str] = Field(None, description="JSON string of the global PatternAnalysisResult object (optional).")


class GroupMergingTool(BaseRepoTool):
    name: str = "Group Merging Tool"
    description: str = "Merges PR grouping results from multiple batches into a single, coherent set of PR groups."
    args_schema: Type[BaseModel] = GroupMergingToolSchema

    def _run(
        self,
        batch_grouping_results: List[Dict[str, Any]],
        original_repository_analysis_json: str
    ) -> str:
        """Merges batch grouping results."""
        try:
            # --- Deserialize and Validate Inputs ---
            try:
                batch_results_data: List[PRGroupingStrategy] = [
                    PRGroupingStrategy.model_validate(item) for item in batch_grouping_results
                ]
                if not isinstance(batch_results_data, list):
                     raise ValueError("Input batch_grouping_results_json must be a JSON list.")
                # Validate each item - if this passes, we can trust the structure
                batch_results: List[PRGroupingStrategy] = [
                    PRGroupingStrategy.model_validate(item) for item in batch_results_data
                ]
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                logger.error(f"Failed to parse/validate batch_grouping_results_json: {e}", exc_info=True)
                raise ValueError("Input batch_grouping_results_json is invalid or does not match schema.") from e

            try:
                original_repo_analysis = RepositoryAnalysis.model_validate_json(original_repository_analysis_json)
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                logger.error(f"Failed to parse/validate original_repository_analysis_json: {e}", exc_info=True)
                raise ValueError("Input original_repository_analysis_json is invalid or does not match schema.") from e
            # --- End Input Validation ---

            logger.info(f"Received {len(batch_results)} valid PRGroupingStrategy objects to merge.")
            if not batch_results:
                # Handle case of empty but valid JSON list
                logger.warning("Received empty list of batch results. Returning empty strategy.")
                empty_strategy = PRGroupingStrategy(strategy_type=GroupingStrategyType.MIXED, groups=[], explanation="No batch results provided.")
                output = GroupMergingOutput(merged_grouping_strategy=empty_strategy, unmerged_files=[], notes="Empty input batch list.")
                return output.model_dump_json(indent=2)


            original_file_paths = set()
            if original_repo_analysis.file_changes: # Check if attribute exists
                 original_file_paths = {fc.path for fc in original_repo_analysis.file_changes if fc.path}


            # --- Merging Logic (Relying on Validated Structure) ---
            merged_groups: List[PRGroup] = []
            all_grouped_files: Set[str] = set()
            overall_explanation = "Merged groups from batches.\n"
            # Use Enum type hint, initialize properly
            merged_strategy_type: Optional[GroupingStrategyType] = None
            first_valid_strategy = None

            for batch_result in batch_results:
                 # Direct access is safe now due to validation
                 if not first_valid_strategy:
                      first_valid_strategy = batch_result.strategy_type # Should be Enum member
                 merged_groups.extend(batch_result.groups)
                 # Use .value for string representation in explanation
                 overall_explanation += f"\nBatch ({batch_result.strategy_type.value}): {batch_result.explanation or 'No explanation'}"
                 # Direct access, check if group.files is not None/empty
                 for group in batch_result.groups:
                      if group.files: # Checks for None or empty list
                           all_grouped_files.update(f for f in group.files if isinstance(f, str))

            # Decide final strategy type (simple: take first, default to MIXED)
            # More complex logic could check for consistency here
            merged_strategy_type = first_valid_strategy or GroupingStrategyType.MIXED

            logger.info(f"Initially merged {len(merged_groups)} groups covering {len(all_grouped_files)} files.")

            # --- Basic Deduplication & Unmerged Check ---
            final_groups: List[PRGroup] = []
            seen_files: Set[str] = set()
            for group in merged_groups:
                 # Direct access, check if not None/empty and is list
                 if group.files and isinstance(group.files, list):
                      unique_files_in_group = [f for f in group.files if isinstance(f, str) and f not in seen_files]
                      if unique_files_in_group:
                           # Modify group in place (simplest)
                           group.files = unique_files_in_group
                           final_groups.append(group)
                           seen_files.update(unique_files_in_group)


            unmerged_files = list(original_file_paths - seen_files)
            notes = f"Merged {len(batch_results)} batches (using strategy: {merged_strategy_type.value}). Found {len(unmerged_files)} unmerged files after simple concatenation & deduplication."
            if unmerged_files:
                logger.warning(f"Unmerged files after simple merge: {len(unmerged_files)}")


            # --- Construct Final Output ---
            merged_strategy = PRGroupingStrategy(
                strategy_type=merged_strategy_type, # Assign the determined Enum member
                groups=final_groups,
                explanation=overall_explanation.strip(),
                estimated_review_complexity=5.0, # Placeholder
                ungrouped_files=unmerged_files
            )

            output = GroupMergingOutput(
                merged_grouping_strategy=merged_strategy,
                unmerged_files=unmerged_files,
                notes=notes
            )
            return output.model_dump_json(indent=2)

        # --- MINIMUM FIX FOR ERROR HANDLING (Using Enum) ---
        except Exception as e:
            # Catches validation errors from above OR unexpected errors here
            error_msg = f"Error in GroupMergingTool: {e}"
            logger.error(error_msg, exc_info=True)
            # Return an error structure using a VALID strategy type Enum member
            error_strategy = PRGroupingStrategy(
                strategy_type=GroupingStrategyType.MIXED, # Use Enum member
                groups=[],
                explanation=f"Merging failed: {e}" # Error details in explanation
            )
            error_output = GroupMergingOutput(
                merged_grouping_strategy=error_strategy,
                unmerged_files=[], # Cannot determine unmerged files reliably on error
                notes=f"Failed to merge groups: {e}" # Error details in notes
            )
            return error_output.model_dump_json(indent=2)
        # --- END MINIMUM FIX ---

# END OF FILE group_merging_tool.py