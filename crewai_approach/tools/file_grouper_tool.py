# START OF FILE file_grouper_tool.py (Re-Revised with Multi-Strategy Support)
import json
import re
from typing import Type, List, Dict, Any, Set
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from .base_tool import BaseRepoTool 
from models.batching_models import WorkerBatchContext 
from models.agent_models import PRGroupingStrategy, PRGroup, GroupingStrategyType, PatternAnalysisResult
from shared.models.analysis_models import RepositoryAnalysis, DirectorySummary
from shared.models.git_models import FileChange, FileStatusType
from shared.utils.logging_utils import get_logger

logger = get_logger(__name__)

class FileGrouperToolSchema(BaseModel):
    """Input schema for FileGrouperTool (takes batch context)."""
    worker_batch_context_json: str = Field(..., description="JSON string of the WorkerBatchContext object containing batch details and global context.")

class FileGrouperTool(BaseRepoTool):
    name: str = "File Grouper Tool"
    description: str = "Groups a specific batch of related files into logical PR groups based on a chosen strategy (directory, feature, module, size, mixed)."
    args_schema: Type[BaseModel] = FileGrouperToolSchema
    # repo_path might still be needed by BaseRepoTool or internal logic

    def _run(self, worker_batch_context_json: str) -> str:
        """Generates PR groups for a given batch of files using the specified strategy."""
        groups: List[PRGroup] = []
        strategy_type: GroupingStrategyType = GroupingStrategyType.MIXED # Default
        batch_files: List[FileChange] = []
        batch_file_paths: Set[str] = set()

        try:
            # 1. Deserialize the context
            batch_context = WorkerBatchContext.model_validate_json(worker_batch_context_json)
            strategy_type = batch_context.grouping_strategy_decision.strategy_type # Get strategy from context
            batch_file_paths = set(batch_context.batch_file_paths) # Set of paths for this batch

            logger.info(f"FileGrouperTool processing batch of {len(batch_file_paths)} files.")
            logger.info(f"Using global strategy: {strategy_type}")

            # 2. Filter the full analysis to get FileChange objects for this batch
            # Ensure we only work with FileChange objects corresponding to paths in our batch
            batch_files = [
                fc for fc in batch_context.repository_analysis.file_changes
                if fc.path in batch_file_paths
            ]

            if not batch_files:
                logger.warning("No files found matching the provided batch paths for grouping.")
                strategy_result = PRGroupingStrategy(
                    strategy_type=strategy_type,
                    groups=[],
                    explanation="No files in this batch to group.",
                    ungrouped_files=list(batch_file_paths) # The original paths are ungrouped
                )
                return strategy_result.model_dump_json(indent=2)

            # Access necessary context parts (handle potential None)
            pattern_analysis = batch_context.pattern_analysis or PatternAnalysisResult() # Use empty if None
            directory_summaries = batch_context.repository_analysis.directory_summaries or []

            # 3. Choose grouping function based on strategy type (Dispatch)
            if strategy_type == GroupingStrategyType.DIRECTORY_BASED:
                groups = self._group_by_directory(batch_files, directory_summaries)
            elif strategy_type == GroupingStrategyType.FEATURE_BASED:
                groups = self._group_by_feature(batch_files, pattern_analysis)
            elif strategy_type == GroupingStrategyType.MODULE_BASED:
                groups = self._group_by_module(batch_files)
            elif strategy_type == GroupingStrategyType.SIZE_BALANCED:
                groups = self._group_by_size(batch_files)
            elif strategy_type == GroupingStrategyType.MIXED:
                groups = self._group_mixed(batch_files, pattern_analysis, directory_summaries)
            else:
                logger.warning(f"Unknown strategy '{strategy_type}', defaulting to directory-based for this batch.")
                groups = self._group_by_directory(batch_files, directory_summaries)

            # 4. Post-processing and Output Formatting
            # Ensure essential fields are populated (title, files, rationale minimum)
            # Add generated branch names and descriptions
            populated_groups: List[PRGroup] = []
            for group in groups:
                 # Ensure files are string paths
                 group.files = [str(f) for f in group.files] # Assume files might be Path objects internally
                 if not group.files: # Skip empty groups potentially created by logic
                     continue
                 if not group.title: # Add default title if missing
                      group.title = "Chore: Grouped changes"
                 if not group.rationale: # Add default rationale
                      group.rationale = f"Group created by {strategy_type} strategy for this batch."

                 # Generate branch name and description
                 group.suggested_branch_name = self._generate_branch_name(group.title)
                 group.suggested_pr_description = self._generate_pr_description(group, batch_files) # Pass batch_files for context
                 populated_groups.append(group)


            # 5. Check for ungrouped files within the batch
            all_grouped_files_in_batch: Set[str] = set()
            for group in populated_groups:
                all_grouped_files_in_batch.update(group.files)

            ungrouped_in_batch = list(batch_file_paths - all_grouped_files_in_batch)
            if ungrouped_in_batch:
                 logger.warning(f"{len(ungrouped_in_batch)} files from the batch remain ungrouped by strategy {strategy_type}.")

            # 6. Create the final PRGroupingStrategy output object
            strategy_explanation = self._generate_strategy_explanation(strategy_type.value, populated_groups, len(batch_files))
            estimated_complexity = self._estimate_review_complexity(populated_groups)

            strategy_result = PRGroupingStrategy(
                strategy_type=strategy_type,
                groups=populated_groups,
                explanation=strategy_explanation,
                estimated_review_complexity=estimated_complexity,
                ungrouped_files=ungrouped_in_batch
            )
            return strategy_result.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Error grouping files with strategy {strategy_type}: {e}", exc_info=True)
            error_result = PRGroupingStrategy(
                strategy_type=strategy_type,
                groups=[],
                explanation=f"Error during file grouping for batch: {e}",
                estimated_review_complexity=1.0,
                ungrouped_files=list(batch_file_paths) # All files in batch are considered ungrouped on error
                # Consider adding an error field to PRGroupingStrategy if needed
            )
            return error_result.model_dump_json(indent=2)

    # --- Grouping Strategy Implementations ---
    # These now accept List[FileChange] for the batch and return List[PRGroup]

    def _group_by_directory(self, batch_files: List[FileChange],
                           directory_summaries: List[DirectorySummary]) -> List[PRGroup]:
        """Group files by directory."""
        groups: List[PRGroup] = []
        dir_to_files = defaultdict(list)

        for file_change in batch_files:
            # Use the directory property from FileChange model
            directory = file_change.directory # This is already a string relative path or '(root)'
            dir_to_files[directory].append(file_change.path)

        for directory, files in dir_to_files.items():
            if files:
                groups.append(PRGroup(
                    title=f"Refactor: Changes in directory '{directory}'",
                    files=files,
                    rationale=f"Batch changes focused within the '{directory}' directory.",
                    directory_focus=directory,
                    estimated_size=len(files) # Simple estimate
                ))
        return groups

    def _group_by_feature(self, batch_files: List[FileChange],
                         pattern_analysis: PatternAnalysisResult) -> List[PRGroup]:
        """Group files by inferred feature using pattern analysis."""
        groups: List[PRGroup] = []
        assigned_files: Set[str] = set() # Track files assigned to avoid duplicates from patterns

       
        current_batch_paths: Set[str] = {fc.path for fc in batch_files}
        if not current_batch_paths: # Early exit if batch is empty
             return []

        # Helper to add group if files exist and haven't been fully assigned
        def add_group_if_valid(title, files_from_pattern, rationale, feature_focus):
            valid_files = [
                f for f in files_from_pattern
                if f in current_batch_paths and f not in assigned_files
            ]
            if valid_files:
                groups.append(PRGroup(
                    title=title, files=valid_files, rationale=rationale,
                    feature_focus=feature_focus, estimated_size=len(valid_files)
                ))
                assigned_files.update(valid_files)

        # Process naming patterns (if applicable to batch files)
        for pattern in pattern_analysis.naming_patterns:
            pattern_type = pattern.type

            if pattern_type and pattern.matches:
                 add_group_if_valid(
                     title=f"Feature: Relates to {pattern_type}",
                     files=pattern.matches,
                     rationale=f"Batch changes related to {pattern_type} based on file naming patterns.",
                     feature_focus=pattern_type
                 )

        # Process similar names
        for similar_group in pattern_analysis.similar_names:
            base_pattern = similar_group.base_pattern

            if base_pattern and similar_group.files:
                 add_group_if_valid(
                      title=f"Feature: Files related to '{base_pattern}'",
                      files=similar_group.files,
                      rationale=f"Batch files sharing the common base pattern '{base_pattern}'.",
                      feature_focus=base_pattern
                 )

        # Process common patterns (Prefixes/Suffixes)
        if pattern_analysis.common_patterns:
            for prefix_group in pattern_analysis.common_patterns.common_prefixes:
                prefix = prefix_group.pattern_value

                if prefix and prefix_group.files:
                     add_group_if_valid(
                         title=f"Feature: Files with prefix '{prefix}'",
                         files=prefix_group.files,
                         rationale=f"Batch files sharing the common prefix '{prefix}'.",
                         feature_focus=f"prefix-{prefix}"
                     )

            for suffix_group in pattern_analysis.common_patterns.common_suffixes:
                suffix = suffix_group.files

                if suffix and suffix_group.files:
                     add_group_if_valid(
                         title=f"Feature: Files with suffix '{suffix}'",
                         files=suffix_group.files,
                         rationale=f"Batch files sharing the common suffix '{suffix}'.",
                         feature_focus=f"suffix-{suffix}"
                     )
        return groups

    def _group_by_module(self, batch_files: List[FileChange]) -> List[PRGroup]:
        """Group files by module based on file extension."""
        module_groups = defaultdict(list)
        for file_change in batch_files:
            extension = file_change.extension or "(noext)" # Use extension property, handle None
            module_groups[extension].append(file_change.path)

        groups: List[PRGroup] = []
        for module, files in module_groups.items():
            if files:
                module_display = module.replace(".", "") if module != "(noext)" else "NoExtension"
                module_display = module_display or "NoExtension" # Ensure not empty
                groups.append(PRGroup(
                    title=f"Chore: {module_display} module changes",
                    files=files,
                    rationale=f"Batch changes grouped by file type '{module}'.",
                    feature_focus=f"module-{module_display}",
                    estimated_size=len(files)
                ))
        return groups

    def _group_by_size(self, batch_files: List[FileChange]) -> List[PRGroup]:
        """Group files to create roughly balanced PRs within the batch."""
        groups: List[PRGroup] = []
        file_paths_in_batch = [fc.path for fc in batch_files]
        num_files = len(file_paths_in_batch)

        # Simple split: Aim for 2 groups if enough files, otherwise 1
        if num_files == 0:
            return []
        elif num_files <= 5: # Small batch, single group
             num_groups = 1
        else:
             num_groups = 2 # Aim for two groups

        batch_size = (num_files + num_groups - 1) // num_groups # Ceiling division

        for i in range(num_groups):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, num_files)
            group_files = file_paths_in_batch[start_index:end_index]
            if group_files:
                 part_num = i + 1
                 groups.append(PRGroup(
                     title=f"Chore: Batch Changes (Part {part_num}/{num_groups})",
                     files=group_files,
                     rationale=f"Part {part_num} of the batch, grouped for balanced size.",
                     feature_focus=f"size-balanced-{part_num}",
                     estimated_size=len(group_files)
                 ))
        return groups

    def _group_mixed(self, batch_files: List[FileChange],
                    pattern_analysis: PatternAnalysisResult,
                    directory_summaries: List[DirectorySummary]) -> List[PRGroup]:
        """Mixed approach for the batch: prioritize features, fallback to directory."""
        # Prioritize feature grouping first
        feature_groups = self._group_by_feature(batch_files, pattern_analysis)
        assigned_files: Set[str] = {file for group in feature_groups for file in group.files}

        # Find remaining files
        remaining_files = [fc for fc in batch_files if fc.path not in assigned_files]

        # Group remaining files by directory
        if remaining_files:
            directory_groups_for_remaining = self._group_by_directory(remaining_files, directory_summaries)
            # Combine, ensuring no empty groups are added
            combined_groups = feature_groups + [g for g in directory_groups_for_remaining if g.files]
        else:
            combined_groups = feature_groups

        # Simple combination for now. Could add logic to merge small directory groups or relate them.
        return combined_groups


    # --- Helper Methods (Adapted from old tool) ---

    def _generate_branch_name(self, title: str) -> str:
        """Generate a Git branch name from a PR title."""
        if not title: title = "untitled-group"
        branch_name = title.lower()
        # Replace common separators with hyphen
        branch_name = re.sub(r'[_\s/:]+', '-', branch_name)
        # Remove invalid characters
        branch_name = re.sub(r'[^\w\-]+', '', branch_name)
        # Remove leading/trailing hyphens
        branch_name = branch_name.strip('-')
        # Limit length (optional, Git has limits but they are long)
        branch_name = branch_name[:70]
        # Ensure it's not empty
        if not branch_name: branch_name = 'fix-unnamed-changes'
        return f"feature/{branch_name}" # Or chore/fix depending on title?

    def _generate_pr_description(self, group: PRGroup, batch_files_context: List[FileChange]) -> str:
        """Generate a PR description for a group."""
        # Find FileChange objects for files in this group to get more context if needed
        group_file_paths = set(group.files)
        group_file_changes = [fc for fc in batch_files_context if fc.path in group_file_paths]

        description = f"## PR Suggestion: {group.title}\n\n"
        if group.rationale:
            description += f"**Rationale:** {group.rationale}\n\n"

        description += "**Files Changed in this Group:**\n"
        # List files, maybe add status?
        for file_path in sorted(group.files)[:15]: # Limit list length
             # Find matching FileChange object for status (optional)
             fc = next((f for f in group_file_changes if f.path == file_path), None)
             status_char = "?"
             if fc:
                  # Simple status: Staged takes precedence
                  st = fc.staged_status
                  ut = fc.unstaged_status
                  if st != FileStatusType.NONE: status_char = st.value
                  elif ut != FileStatusType.NONE: status_char = ut.value
                  else: status_char = ' ' # Should not happen if it's a change
             description += f"- `{file_path}` ({status_char})\n"

        if len(group.files) > 15:
            description += f"- ... and {len(group.files) - 15} more file(s)\n"

        # Add more context? e.g., total lines changed in group?
        total_added = sum(fc.changes.added for fc in group_file_changes if fc.changes)
        total_deleted = sum(fc.changes.deleted for fc in group_file_changes if fc.changes)
        if total_added > 0 or total_deleted > 0:
             description += f"\n**Approximate Changes:** +{total_added} lines, -{total_deleted} lines\n"

        return description

    def _generate_strategy_explanation(self, strategy_value: str, groups: List[PRGroup], batch_file_count: int) -> str:
        """Generate an explanation for the grouping strategy applied to the batch."""
        group_count = len(groups)
        grouped_files_count = sum(len(group.files) for group in groups)
        strategy_name = strategy_value.replace("_", " ").title()

        explanation = (
            f"Applied **{strategy_name}** grouping strategy to organize the {batch_file_count} files in this batch. "
            f"Resulted in {group_count} logical group(s) containing {grouped_files_count} files. "
        )
        # Add specific strategy notes if desired
        return explanation

    def _estimate_review_complexity(self, groups: List[PRGroup]) -> float:
        """Estimate the review complexity (1-10 scale) for the groups in this batch."""
        if not groups: return 1.0
        # Simple estimate based on number of groups and files
        group_count = len(groups)
        total_files = sum(len(group.files) for group in groups)
        max_files = max((len(group.files) for group in groups), default=0)

        complexity = 1.0 + (group_count * 0.5) + (total_files * 0.1) + (max_files * 0.05)
        return min(10.0, max(1.0, round(complexity, 1)))


# END OF FILE file_grouper_tool.py