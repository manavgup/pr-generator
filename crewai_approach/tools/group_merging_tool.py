"""
Group merging tool for merging PR grouping results from multiple batches.
"""
import json
import re
from typing import Type, List, Dict, Set, Optional, Any

# Assuming pydantic models are defined
from pydantic import BaseModel, Field, ValidationError

from .base_tool import BaseRepoTool
# Import specific models including the Enum from agent_models
from models.agent_models import PRGroupingStrategy, PRGroup, GroupingStrategyType
from models.batching_models import GroupMergingOutput
from shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

class GroupMergingToolSchema(BaseModel):
    """Input schema for GroupMergingTool using primitive types."""
    batch_grouping_results_json: List[str] = Field(..., description="A list of JSON strings representing PRGroupingStrategy objects from batches.")
    original_repository_analysis_json: str = Field(..., description="JSON string of the original, full RepositoryAnalysis object.")
    pattern_analysis_json: Optional[str] = Field(None, description="JSON string of the global PatternAnalysisResult object (optional).")


class GroupMergingTool(BaseRepoTool):
    name: str = "Group Merging Tool"
    description: str = "Merges PR grouping results from multiple batches into a single, coherent set of PR groups."
    args_schema: Type[BaseModel] = GroupMergingToolSchema

    def _run(
        self,
        batch_grouping_results_json: List[str],
        original_repository_analysis_json: str,
        pattern_analysis_json: Optional[str] = None
    ) -> str:
        """Merges batch grouping results."""
        # Echo received inputs for debugging
        logger.info(f"GroupMergingTool received {len(batch_grouping_results_json)} batch_grouping_results_json items")
        logger.info(f"GroupMergingTool received original_repository_analysis_json: {original_repository_analysis_json[:100]}...")
        if pattern_analysis_json:
            logger.info(f"GroupMergingTool received pattern_analysis_json: {pattern_analysis_json[:100]}...")
        
        try:
            # --- Validate Inputs ---
            if not isinstance(batch_grouping_results_json, list):
                raise ValueError("Input batch_grouping_results_json must be a list.")
                
            if not batch_grouping_results_json:
                # Handle case of empty but valid list
                logger.warning("Received empty list of batch results. Returning empty strategy.")
                empty_strategy = PRGroupingStrategy(
                    strategy_type=GroupingStrategyType.MIXED, 
                    groups=[], 
                    explanation="No batch results provided."
                )
                output = GroupMergingOutput(
                    merged_grouping_strategy=empty_strategy, 
                    unmerged_files=[], 
                    notes="Empty input batch list."
                )
                return output.model_dump_json(indent=2)
            
            # Sanitize original repository analysis JSON
            original_repository_analysis_json = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', original_repository_analysis_json)
                
            if not self._validate_json_string(original_repository_analysis_json):
                raise ValueError("Invalid original_repository_analysis_json provided")

            # --- Deserialize and Validate Batch Results with improved cleaning ---
            batch_results: List[Dict[str, Any]] = []
            for idx, result_json in enumerate(batch_grouping_results_json):
                # Clean up control characters
                sanitized_result_json = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', result_json)
                
                if not self._validate_json_string(sanitized_result_json):
                    logger.warning(f"Skipping invalid batch result JSON at index {idx}")
                    continue
                    
                try:
                    # Explicitly parse through JSON first for additional validation
                    parsed_json = json.loads(sanitized_result_json)
                    batch_results.append(parsed_json)
                except Exception as e:
                    logger.warning(f"Failed to parse batch result JSON: {e}")
                    continue
            
            if not batch_results:
                logger.warning("No valid batch results found. Returning empty strategy.")
                empty_strategy = PRGroupingStrategy(
                    strategy_type=GroupingStrategyType.MIXED, 
                    groups=[], 
                    explanation="No valid batch results provided."
                )
                output = GroupMergingOutput(
                    merged_grouping_strategy=empty_strategy, 
                    unmerged_files=[], 
                    notes="No valid batch results."
                )
                return output.model_dump_json(indent=2)
            
            # Extract file paths from original repository analysis
            original_file_paths = set(self._extract_file_paths(original_repository_analysis_json))
            
            # --- Merging Logic ---
            merged_groups: List[Dict[str, Any]] = []
            all_grouped_files: Set[str] = set()
            overall_explanation = "Merged groups from batches.\n"
            strategy_type_value = "mixed"  # Default
            
            for batch_result in batch_results:
                # Get strategy type from first valid batch if not set
                if strategy_type_value == "mixed":
                    strategy_type_value = batch_result.get("strategy_type", "mixed")
                
                # Extract and merge groups
                groups = batch_result.get("groups", [])
                merged_groups.extend(groups)
                
                # Add explanation
                batch_explanation = batch_result.get("explanation", "No explanation")
                overall_explanation += f"\nBatch ({strategy_type_value}): {batch_explanation}"
                
                # Track all files in groups
                for group in groups:
                    files = group.get("files", [])
                    if files:
                        all_grouped_files.update(f for f in files if isinstance(f, str))
            
            logger.info(f"Initially merged {len(merged_groups)} groups covering {len(all_grouped_files)} files.")
            
            # --- Deduplication ---
            final_groups: List[Dict[str, Any]] = []
            seen_files: Set[str] = set()
            
            for group in merged_groups:
                files = group.get("files", [])
                if files and isinstance(files, list):
                    unique_files_in_group = [f for f in files if isinstance(f, str) and f not in seen_files]
                    if unique_files_in_group:
                        # Create new group with unique files
                        new_group = dict(group)
                        new_group["files"] = unique_files_in_group
                        final_groups.append(new_group)
                        seen_files.update(unique_files_in_group)
            
            unmerged_files = list(original_file_paths - seen_files)
            notes = f"Merged {len(batch_results)} batches (using strategy: {strategy_type_value}). " \
                    f"Found {len(unmerged_files)} unmerged files after deduplication."
            
            if unmerged_files:
                logger.warning(f"Unmerged files after merge: {len(unmerged_files)}")
            
            # --- Convert to Pydantic Models ---
            try:
                strategy_type_enum = GroupingStrategyType(strategy_type_value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid strategy type '{strategy_type_value}'. Defaulting to MIXED.")
                strategy_type_enum = GroupingStrategyType.MIXED
            
            # Create PRGroup objects with validation
            final_pr_groups = []
            for group_dict in final_groups:
                try:
                    # Basic validation before conversion
                    if not isinstance(group_dict.get("files", []), list):
                        continue
                        
                    # Ensure all files are strings
                    files = [f for f in group_dict.get("files", []) if isinstance(f, str)]
                    
                    pr_group = PRGroup(
                        title=group_dict.get("title", "Untitled Group"),
                        files=files,
                        rationale=group_dict.get("rationale", "No rationale provided"),
                        estimated_size=group_dict.get("estimated_size", len(files)),
                        directory_focus=group_dict.get("directory_focus"),
                        feature_focus=group_dict.get("feature_focus"),
                        suggested_branch_name=group_dict.get("suggested_branch_name"),
                        suggested_pr_description=group_dict.get("suggested_pr_description")
                    )
                    final_pr_groups.append(pr_group)
                except Exception as e:
                    logger.warning(f"Error converting group to PRGroup: {e}")
                    continue
            
            # --- Create final strategy ---
            try:
                merged_strategy = PRGroupingStrategy(
                    strategy_type=strategy_type_enum,
                    groups=final_pr_groups,
                    explanation=overall_explanation.strip(),
                    estimated_review_complexity=5.0,
                    ungrouped_files=unmerged_files
                )
                
                # Validate the result by serializing and parsing
                result_json = merged_strategy.model_dump_json(indent=2)
                json.loads(result_json)  # Verify it's valid JSON
                logger.info(f"Successfully created merged strategy with {len(final_pr_groups)} groups")
                
                return result_json
                
            except Exception as e:
                # If validation fails, return a simplified error result
                logger.error(f"Error validating merged strategy: {e}")
                error_strategy = PRGroupingStrategy(
                    strategy_type=GroupingStrategyType.MIXED,
                    groups=[],
                    explanation=f"Merging failed during validation: {e}",
                    ungrouped_files=[]
                )
                return error_strategy.model_dump_json(indent=2)
            
        except Exception as e:
            error_msg = f"Error in GroupMergingTool: {e}"
            logger.error(error_msg, exc_info=True)
            
            error_strategy = PRGroupingStrategy(
                strategy_type=GroupingStrategyType.MIXED,
                groups=[],
                explanation=f"Merging failed: {e}",
                ungrouped_files=[]
            )
            return error_strategy.model_dump_json(indent=2)