# tools/batch_processor_tool.py
import json
import re
from typing import Type, List, Dict, Any, Set, Optional

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

# Base tool dependency
from .base_tool import BaseRepoTool

# Models needed for internal logic and return type
from crewai_approach.models.agent_models import (
    PRGroupingStrategy, PRGroup, GroupingStrategyType, PatternAnalysisResult,
    GroupingStrategyDecision # For parsing strategy decision
)
# Models needed only for DESERIALIZATION inside _run
from shared.models.analysis_models import RepositoryAnalysis, DirectorySummary, FileChange
from crewai_approach.models.batching_models import BatchSplitterOutput # For parsing batch output
from shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

# --- Input Schema for the Consolidated Tool ---
class BatchProcessorToolSchema(BaseModel):
    """Input schema for BatchProcessorTool using JSON strings for context."""
    batch_splitter_output_json: str = Field(..., description="REQUIRED: The full JSON string serialization of the BatchSplitterOutput object (contains the 'batches' list).")
    grouping_strategy_decision_json: str = Field(..., description="REQUIRED: The full JSON string serialization of the GroupingStrategyDecision object (contains 'strategy_type').")
    repository_analysis_json: str = Field(..., description="REQUIRED: The full JSON string serialization of the RepositoryAnalysis object (contains all 'file_changes').")
    pattern_analysis_json: Optional[str] = Field(None, description="Optional JSON string serialization of the PatternAnalysisResult object.")

class BatchProcessorTool(BaseRepoTool):
    name: str = "Batch Processor Tool"
    description: str = (
        "Processes all file batches sequentially based on provided strategy and context. "
        "Internally loops through batches from BatchSplitterOutput, applies grouping logic using "
        "RepositoryAnalysis and optional PatternAnalysis, and returns a JSON array string "
        "containing a PRGroupingStrategy result for each batch."
    )
    args_schema: Type[BaseModel] = BatchProcessorToolSchema

    # --- Helper to clean potential markdown/whitespace ---
    def _clean_json_string(self, json_string: Optional[str]) -> Optional[str]:
        if not json_string or not isinstance(json_string, str):
            return None
        try:
            # Remove leading/trailing whitespace and potential markdown fences
            cleaned = re.sub(r'^```json\s*', '', json_string.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()
            # Basic check if it looks like JSON
            if cleaned.startswith('{') or cleaned.startswith('['):
                 # Attempt a parse to validate
                 json.loads(cleaned)
                 return cleaned
            else:
                 logger.warning(f"Cleaned string doesn't look like JSON: {cleaned[:100]}...")
                 return None # Return None if it doesn't seem valid after cleaning
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not clean/validate JSON string: {e}. String: {json_string[:100]}...")
            return None # Return None on error

    def _run(
        self,
        batch_splitter_output_json: str,
        grouping_strategy_decision_json: str,
        repository_analysis_json: str,
        pattern_analysis_json: Optional[str] = None
    ) -> str:
        """
        Processes all batches sequentially. Parses context JSON strings, loops through batches,
        applies grouping logic internally, and returns a JSON array string of results.
        """
        logger.info(f"{self.name} starting execution.")
        all_batch_results: List[Dict] = [] # Store results as dicts for final JSON dump

        try:
            # 1. Clean and Parse Inputs
            cleaned_batch_output = self._clean_json_string(batch_splitter_output_json)
            cleaned_strategy_decision = self._clean_json_string(grouping_strategy_decision_json)
            cleaned_repo_analysis = self._clean_json_string(repository_analysis_json)
            cleaned_pattern_analysis = self._clean_json_string(pattern_analysis_json)

            if not cleaned_batch_output: raise ValueError("BatchSplitterOutput JSON is invalid or empty after cleaning.")
            if not cleaned_strategy_decision: raise ValueError("GroupingStrategyDecision JSON is invalid or empty after cleaning.")
            if not cleaned_repo_analysis: raise ValueError("RepositoryAnalysis JSON is invalid or empty after cleaning.")

            try:
                batch_output = BatchSplitterOutput.model_validate_json(cleaned_batch_output)
                strategy_decision = GroupingStrategyDecision.model_validate_json(cleaned_strategy_decision)
                repository_analysis = RepositoryAnalysis.model_validate_json(cleaned_repo_analysis)
            except (ValidationError, json.JSONDecodeError) as e:
                 logger.error(f"Pydantic validation/JSON parsing failed for core inputs: {e}", exc_info=True)
                 raise ValueError(f"Input JSON context parsing failed: {e}") from e

            pattern_analysis = PatternAnalysisResult() # Default empty
            if cleaned_pattern_analysis:
                try:
                    pattern_analysis = PatternAnalysisResult.model_validate_json(cleaned_pattern_analysis)
                except (ValidationError, json.JSONDecodeError) as e:
                    logger.warning(f"Could not parse optional pattern_analysis_json: {e}. Proceeding without.")

            # 2. Extract Key Info
            list_of_all_batches: List[List[str]] = getattr(batch_output, 'batches', [])
            strategy_type_str: str = getattr(strategy_decision, 'strategy_type', 'mixed') # Default if missing
            all_file_changes: List[FileChange] = getattr(repository_analysis, 'file_changes', [])
            directory_summaries: List[DirectorySummary] = getattr(repository_analysis, 'directory_summaries', [])

            if not list_of_all_batches:
                 logger.warning("No batches found in BatchSplitterOutput.")
                 return "[]" # Return empty JSON array

            try:
                strategy_type = GroupingStrategyType(strategy_type_str)
            except ValueError:
                logger.warning(f"Invalid strategy_type '{strategy_type_str}'. Defaulting to MIXED.")
                strategy_type = GroupingStrategyType.MIXED

            logger.info(f"Processing {len(list_of_all_batches)} batches using strategy: {strategy_type.value}")

            # 3. Internal Loop Through Batches
            for i, current_batch_paths in enumerate(list_of_all_batches):
                logger.info(f"--- Processing Batch {i} ({len(current_batch_paths)} files) ---")
                batch_groups: List[PRGroup] = []
                batch_explanation = f"Error processing batch {i}." # Default explanation
                batch_complexity = 1.0
                ungrouped_in_batch: List[str] = current_batch_paths # Default all ungrouped

                try:
                    batch_file_paths_set = set(current_batch_paths)
                    if not batch_file_paths_set:
                        logger.warning(f"Batch {i} is empty. Skipping.")
                        batch_explanation = f"Batch {i} was empty."
                        ungrouped_in_batch = []
                    else:
                        # Filter full analysis for this batch's files
                        batch_files_objects = [
                            fc for fc in all_file_changes
                            if hasattr(fc, 'path') and fc.path in batch_file_paths_set
                        ]
                        logger.debug(f"Found {len(batch_files_objects)} FileChange objects for batch {i}.")

                        if not batch_files_objects:
                             logger.warning(f"No FileChange objects found for paths in batch {i}.")
                             batch_explanation = f"No analysis data found for files in batch {i}."
                        else:
                            # Call internal grouping logic based on strategy
                            # These methods now return List[PRGroup]
                            if strategy_type == GroupingStrategyType.DIRECTORY_BASED:
                                batch_groups = self._group_by_directory(batch_files_objects, directory_summaries)
                            elif strategy_type == GroupingStrategyType.FEATURE_BASED:
                                batch_groups = self._group_by_feature(batch_files_objects, pattern_analysis)
                            elif strategy_type == GroupingStrategyType.MODULE_BASED:
                                batch_groups = self._group_by_module(batch_files_objects)
                            elif strategy_type == GroupingStrategyType.SIZE_BALANCED:
                                batch_groups = self._group_by_size(batch_files_objects)
                            elif strategy_type == GroupingStrategyType.MIXED:
                                batch_groups = self._group_mixed(batch_files_objects, pattern_analysis, directory_summaries)
                            else: # Fallback
                                batch_groups = self._group_by_directory(batch_files_objects, directory_summaries)

                            # Post-process groups for this batch
                            populated_groups: List[PRGroup] = []
                            for group in batch_groups:
                                group.files = [str(f) for f in group.files] # Ensure strings
                                if not group.files: continue # Skip empty groups
                                if not group.title: group.title = f"Chore: Grouped changes (Batch {i})"
                                if not group.rationale: group.rationale = f"Group created by {strategy_type.value} strategy for batch {i}."
                                group.suggested_branch_name = self._generate_branch_name(group.title)
                                group.suggested_pr_description = self._generate_pr_description(group, batch_files_objects) # Pass relevant context
                                populated_groups.append(group)

                            # Calculate ungrouped for *this* batch
                            all_grouped_in_batch = set().union(*(set(g.files) for g in populated_groups if g.files))
                            ungrouped_in_batch = list(batch_file_paths_set - all_grouped_in_batch)
                            batch_groups = populated_groups # Use the post-processed groups

                            batch_explanation = self._generate_strategy_explanation(strategy_type.value, batch_groups, len(current_batch_paths), i)
                            batch_complexity = self._estimate_review_complexity(batch_groups)

                            logger.info(f"Batch {i} resulted in {len(batch_groups)} groups, {len(ungrouped_in_batch)} ungrouped.")

                except Exception as batch_err:
                    logger.error(f"Error processing inside batch {i}: {batch_err}", exc_info=True)
                    # Keep defaults: empty batch_groups, error explanation, default complexity, all files ungrouped

                # Create the PRGroupingStrategy *object* for this batch
                batch_strategy_result = PRGroupingStrategy(
                    strategy_type=strategy_type,
                    groups=batch_groups,
                    explanation=batch_explanation,
                    estimated_review_complexity=batch_complexity,
                    ungrouped_files=ungrouped_in_batch
                )
                # Add its dictionary representation to the results list
                all_batch_results.append(batch_strategy_result.model_dump(mode='json')) # Use model_dump for Pydantic v2

            # 4. Serialize the final list of results
            final_json_output = json.dumps(all_batch_results, indent=2)
            logger.info(f"{self.name} finished successfully. Returning JSON array of {len(all_batch_results)} batch results.")
            return final_json_output

        except Exception as e:
            logger.error(f"FATAL Error in {self.name}: {e}", exc_info=True)
            # Return an empty JSON array string on fatal error to avoid breaking downstream tasks
            return "[]"

    # --- Internal Grouping Methods (Copied/Adapted from FileGrouperTool) ---
    # IMPORTANT: Ensure these methods now correctly use the arguments passed
    # (e.g., batch_files_objects, pattern_analysis, directory_summaries)
    # and return List[PRGroup].

    def _group_by_directory(self, batch_files_objects: List[FileChange], directory_summaries: List[DirectorySummary]) -> List[PRGroup]:
        # (Keep implementation as before)
        groups: List[PRGroup] = []
        dir_to_files = defaultdict(list)
        for file_ctx in batch_files_objects:
            directory = file_ctx.directory or "(root)"
            if directory and file_ctx.path:
                dir_to_files[directory].append(file_ctx.path)
        for directory, files in dir_to_files.items():
            if files:
                groups.append(PRGroup(title=f"Refactor: Changes in directory '{directory}'", files=files, rationale=f"Batch changes focused within the '{directory}' directory.", directory_focus=directory, estimated_size=len(files)))
        return groups

    def _group_by_feature(self, batch_files_objects: List[FileChange], pattern_analysis: PatternAnalysisResult) -> List[PRGroup]:
        # (Keep implementation as before)
        groups: List[PRGroup] = []
        assigned_files: Set[str] = set()
        current_batch_paths: Set[str] = {fc.path for fc in batch_files_objects if fc.path}
        if not current_batch_paths: return []
        def add_group_if_valid(title, files_from_pattern, rationale, feature_focus):
            if not isinstance(files_from_pattern, list): return
            valid_files = [f for f in files_from_pattern if isinstance(f, str) and f in current_batch_paths and f not in assigned_files]
            if valid_files:
                groups.append(PRGroup(title=title, files=valid_files, rationale=rationale, feature_focus=feature_focus, estimated_size=len(valid_files)))
                assigned_files.update(valid_files)
        if pattern_analysis.naming_patterns:
            for pattern in pattern_analysis.naming_patterns:
                 pattern_type = getattr(pattern, 'type', None); pattern_matches = getattr(pattern, 'matches', None)
                 if pattern_type and pattern_matches: add_group_if_valid(f"Feature: Relates to {pattern_type}", pattern_matches, f"Batch changes related to {pattern_type} based on file naming patterns.", pattern_type)
        if pattern_analysis.similar_names:
            for similar_group in pattern_analysis.similar_names:
                base_pattern = getattr(similar_group, 'base_pattern', None); similar_files = getattr(similar_group, 'files', None)
                if base_pattern and similar_files: add_group_if_valid(f"Feature: Files related to '{base_pattern}'", similar_files, f"Batch files sharing the common base pattern '{base_pattern}'.", base_pattern)
        if hasattr(pattern_analysis, 'common_patterns') and pattern_analysis.common_patterns:
             common_patterns_obj = pattern_analysis.common_patterns
             if hasattr(common_patterns_obj, 'common_prefixes'):
                 for prefix_group in common_patterns_obj.common_prefixes:
                      prefix = getattr(prefix_group, 'pattern_value', None); prefix_files = getattr(prefix_group, 'files', None)
                      if prefix and prefix_files: add_group_if_valid(f"Feature: Files with prefix '{prefix}'", prefix_files, f"Batch files sharing the common prefix '{prefix}'.", f"prefix-{prefix}")
             if hasattr(common_patterns_obj, 'common_suffixes'):
                 for suffix_group in common_patterns_obj.common_suffixes:
                      suffix = getattr(suffix_group, 'pattern_value', None); suffix_files = getattr(suffix_group, 'files', None)
                      if suffix and suffix_files: add_group_if_valid(f"Feature: Files with suffix '{suffix}'", suffix_files, f"Batch files sharing the common suffix '{suffix}'.", f"suffix-{suffix}")
        remaining_files_paths = [fc.path for fc in batch_files_objects if fc.path and fc.path not in assigned_files]
        if remaining_files_paths:
             groups.append(PRGroup(title="Feature: Other Related Changes", files=remaining_files_paths, rationale="Remaining files in the batch, potentially related by feature context.", feature_focus="misc-feature", estimated_size=len(remaining_files_paths)))
        return groups

    def _group_by_module(self, batch_files_objects: List[FileChange]) -> List[PRGroup]:
        # (Keep implementation as before)
        logger.info(f"Starting module-based grouping for {len(batch_files_objects)} files")
        module_groups = defaultdict(list)
        for file_ctx in batch_files_objects:
            extension = file_ctx.extension or "(noext)"
            if file_ctx.path:
                module_groups[extension].append(file_ctx.path)
        logger.info(f"Created {len(module_groups)} potential module groups")
        groups = []
        for module, files in module_groups.items():
            if files:
                module_display = module.replace(".", "") if module != "(noext)" else "NoExtension"
                module_display = module_display or "NoExtension"
                groups.append(PRGroup(title=f"Chore: {module_display} module changes", files=files, rationale=f"Batch changes grouped by file type '{module}'.", feature_focus=f"module-{module_display}", estimated_size=len(files)))
        logger.debug(f"Created {len(groups)} module groups with {sum(len(g.files) for g in groups)} total files")
        return groups

    def _group_by_size(self, batch_files_objects: List[FileChange]) -> List[PRGroup]:
        # (Keep implementation as before)
        groups: List[PRGroup] = []
        file_paths_in_batch = [fc.path for fc in batch_files_objects if fc.path]
        num_files = len(file_paths_in_batch)
        if num_files == 0: return []
        # Simple split into 1 or 2 groups for simplicity within a batch
        num_groups = 1 if num_files <= 10 else 2 # Adjust threshold as needed
        batch_size = (num_files + num_groups - 1) // num_groups
        for i in range(num_groups):
            start_index = i * batch_size; end_index = min((i + 1) * batch_size, num_files)
            group_files = file_paths_in_batch[start_index:end_index]
            if group_files:
                 part_num = i + 1
                 groups.append(PRGroup(title=f"Chore: Batch Changes (Part {part_num}/{num_groups})", files=group_files, rationale=f"Part {part_num} of the batch, grouped for balanced size.", feature_focus=f"size-balanced-{part_num}", estimated_size=len(group_files)))
        return groups

    def _group_mixed(self, batch_files_objects: List[FileChange], pattern_analysis: PatternAnalysisResult, directory_summaries: List[DirectorySummary]) -> List[PRGroup]:
        # (Keep implementation as before)
        feature_groups = self._group_by_feature(batch_files_objects, pattern_analysis)
        assigned_files: Set[str] = set()
        filtered_feature_groups: List[PRGroup] = []
        for group in feature_groups:
            if group.files and isinstance(group.files, list):
                unique_files = [f for f in group.files if isinstance(f, str) and f not in assigned_files]
                if unique_files:
                    group.files = unique_files; filtered_feature_groups.append(group); assigned_files.update(unique_files)
        remaining_files_objs = [fc for fc in batch_files_objects if fc.path and fc.path not in assigned_files]
        if remaining_files_objs:
            directory_groups_for_remaining = self._group_by_directory(remaining_files_objs, directory_summaries)
            combined_groups = filtered_feature_groups + [g for g in directory_groups_for_remaining if g.files]
        else:
            combined_groups = filtered_feature_groups
        # Ensure unique files across combined groups if necessary (optional refinement)
        final_groups: List[PRGroup] = []
        final_assigned: Set[str] = set()
        for group in combined_groups:
             unique_files = [f for f in group.files if f not in final_assigned]
             if unique_files:
                  group.files = unique_files
                  final_groups.append(group)
                  final_assigned.update(unique_files)
        return final_groups


    # --- Helper Methods (Copied/Adapted from FileGrouperTool) ---
    def _generate_branch_name(self, title: str) -> str:
        # (Keep implementation as before)
        if not title: title = "untitled-group"
        branch_name = title.lower()
        branch_name = re.sub(r'[_\s/:]+', '-', branch_name)
        branch_name = re.sub(r'[^\w\-]+', '', branch_name)
        branch_name = branch_name.strip('-')
        branch_name = branch_name[:70]
        if not branch_name: branch_name = 'fix-unnamed-changes'
        return f"feature/{branch_name}"

    def _generate_pr_description(self, group: PRGroup, batch_files_context: List[FileChange]) -> str:
        # (Keep implementation as before)
        group_file_paths = set(group.files)
        group_file_changes = [fc for fc in batch_files_context if hasattr(fc, 'path') and fc.path in group_file_paths]
        description = f"## PR Suggestion: {group.title}\n\n";
        if group.rationale: description += f"**Rationale:** {group.rationale}\n\n"
        description += "**Files Changed in this Group:**\n";
        for file_path in sorted(group.files)[:15]:
             fc = next((f for f in group_file_changes if f.path == file_path), None); status_char = "?"
             if fc:
                  st_val = getattr(fc, 'staged_status', None); ut_val = getattr(fc, 'unstaged_status', None)
                  if st_val and st_val != " ": status_char = st_val
                  elif ut_val and ut_val != " ": status_char = ut_val
                  else: status_char = ' '
             description += f"- `{file_path}` ({status_char})\n"
        if len(group.files) > 15: description += f"- ... and {len(group.files) - 15} more file(s)\n"
        total_added = sum(getattr(fc.changes, 'added', 0) for fc in group_file_changes if hasattr(fc, 'changes') and fc.changes)
        total_deleted = sum(getattr(fc.changes, 'deleted', 0) for fc in group_file_changes if hasattr(fc, 'changes') and fc.changes)
        if total_added > 0 or total_deleted > 0: description += f"\n**Approximate Changes:** +{total_added} lines, -{total_deleted} lines\n"
        return description

    def _generate_strategy_explanation(self, strategy_value: str, groups: List[PRGroup], batch_file_count: int, batch_index: int) -> str:
        # (Keep implementation as before, maybe add batch index)
        group_count = len(groups); grouped_files_count = sum(len(g.files) for g in groups if g.files); strategy_name = strategy_value.replace("_", " ").title()
        explanation = (f"Batch {batch_index}: Applied **{strategy_name}** grouping to {batch_file_count} files. Resulted in {group_count} group(s) covering {grouped_files_count} files. "); return explanation

    def _estimate_review_complexity(self, groups: List[PRGroup]) -> float:
        # (Keep implementation as before)
        if not groups: return 1.0
        group_count = len(groups)
        total_files = sum(len(g.files) for g in groups if g.files)
        max_files = max((len(g.files) for g in groups if g.files), default=0)
        complexity = 1.0 + (group_count * 0.5) + (total_files * 0.1) + (max_files * 0.05)
        return min(10.0, max(1.0, round(complexity, 1)))