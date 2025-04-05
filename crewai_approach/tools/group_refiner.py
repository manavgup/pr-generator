"""
Group refiner tool for refining and balancing PR groups.
"""
import re
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from pathlib import Path

# Pydantic imports
from pydantic import BaseModel, Field

# Logging import
from shared.utils.logging_utils import get_logger

# Model imports
from models.agent_models import GroupingStrategyType, PRGroup, PRGroupingStrategy, PRValidationResult
# Base Tool import
from .base_tools import BaseRepoTool # Inherit from the common base

logger = get_logger(__name__)

# Define the input schema for the tool
class GroupRefinerInput(BaseModel):
    """Input schema for the GroupRefiner tool."""
    repo_path: str = Field(..., description="Path to the git repository (required by BaseRepoTool)")
    grouping_strategy: PRGroupingStrategy = Field(..., description="The PR grouping strategy object to refine")
    validation_result: PRValidationResult = Field(..., description="Validation result data from the GroupValidator tool")

# Inherit from BaseRepoTool
class GroupRefiner(BaseRepoTool):
    """
    Tool for refining and balancing PR groups based on validation results.
    Takes an existing grouping strategy and validation feedback, then applies
    heuristics to improve the groups (e.g., balancing size, fixing duplicates).
    Outputs the refined PRGroupingStrategy.
    """

    name: str = "Group Refiner"
    description: str = "Refines and balances PR groups based on validation results for optimal review experience"
    # Define the input schema
    args_schema: type[BaseModel] = GroupRefinerInput

    # _run now accepts **kwargs as defined in BaseRepoTool
    def _run(self, **kwargs) -> PRGroupingStrategy:
        """
        Refine and balance PR groups based on validation results.

        Args:
            **kwargs: Expects 'repo_path', 'grouping_strategy', and 'validation_result'
                      based on args_schema.

        Returns:
            Refined PRGroupingStrategy object.
        """
        repo_path = kwargs.get("repo_path")
        grouping_strategy = kwargs.get("grouping_strategy")
        validation_result = kwargs.get("validation_result")

        # --- Input Validation and Parsing ---
        if not grouping_strategy or not validation_result:
            missing = []
            if not grouping_strategy: missing.append("'grouping_strategy'")
            if not validation_result: missing.append("'validation_result'")
            error_msg = f"Missing required arguments for Group Refiner: {', '.join(missing)}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure inputs are Pydantic models (parse if dict)
        try:
            if isinstance(grouping_strategy, dict):
                grouping_strategy = PRGroupingStrategy(**grouping_strategy)
            if isinstance(validation_result, dict):
                validation_result = PRValidationResult(**validation_result)
        except Exception as e:
            logger.error(f"Failed to parse input arguments into Pydantic models: {e}")
            raise ValueError("Invalid input structure for grouping_strategy or validation_result.") from e

        logger.info(f"Running Group Refiner Tool on {repo_path}")

        # --- Core Logic ---

        # If validation already passed, no refinement needed
        if validation_result.is_valid:
            logger.info("Validation passed, no refinement needed.")
            return grouping_strategy

        logger.info("Refining PR groups based on validation issues...")
        strategy_type = grouping_strategy.strategy_type
        # Deep copy is crucial here to avoid modifying the original context object
        try:
             original_groups = [g.model_copy(deep=True) for g in grouping_strategy.groups]
             original_ungrouped = grouping_strategy.ungrouped_files.copy()
        except Exception as e:
             logger.error(f"Error deep copying grouping strategy: {e}")
             # Fallback or raise error - for now, we'll raise
             raise RuntimeError("Failed to prepare groups for refinement.") from e

        refined_groups = original_groups # Work on the copy

        # Process validation issues
        issues = validation_result.issues

        # Keep track of all files to ensure none are lost in refinement
        all_files_initial = set()
        for group in original_groups:
            all_files_initial.update(group.files)
        all_files_initial.update(original_ungrouped)

        # --- Apply Refinement Steps ---
        # Note: These helpers modify refined_groups in place
        self._handle_empty_groups(refined_groups, issues)
        self._handle_duplicate_files(refined_groups, issues) # Handles duplicates across the *current* refined_groups
        self._handle_missing_files(refined_groups, original_ungrouped, issues) # Adds missing/ungrouped
        self._handle_related_files(refined_groups, issues) # Consolidates related pairs
        self._handle_imbalanced_groups(refined_groups, issues) # Rebalances sizes

        # Final pass updates
        self._update_missing_metadata(refined_groups) # Fills in missing titles/branches/descriptions

        # --- Verify Completeness Post-Refinement ---
        refined_all_files = set()
        for group in refined_groups:
            refined_all_files.update(group.files)

        missing_after_refinement = all_files_initial - refined_all_files
        if missing_after_refinement:
            logger.warning(f"Files missing after refinement: {missing_after_refinement}. Adding to miscellaneous group.")
            # Find or create a miscellaneous group
            misc_group = next((g for g in refined_groups if g.feature_focus == "miscellaneous"), None)
            if misc_group:
                 current_files = set(misc_group.files)
                 new_files_to_add = list(missing_after_refinement - current_files)
                 if new_files_to_add:
                      misc_group.files.extend(new_files_to_add)
                      misc_group.estimated_size = len(misc_group.files)
                 logger.info(f"Added {len(new_files_to_add)} files to existing miscellaneous group.")
            else:
                logger.info("Creating new miscellaneous group.")
                refined_groups.append(PRGroup(
                    title="Miscellaneous Unassigned Changes",
                    files=list(missing_after_refinement),
                    rationale="These files were unassigned or became unassigned during the refinement process and are grouped here to ensure completeness.",
                    estimated_size=len(missing_after_refinement),
                    directory_focus=None,
                    feature_focus="miscellaneous",
                    suggested_branch_name="feature/misc-unassigned-changes",
                    suggested_pr_description=f"## Miscellaneous Unassigned Changes\n\nThis PR includes {len(missing_after_refinement)} files that could not be assigned to other logical groups during refinement."
                ))
        elif len(all_files_initial) != len(refined_all_files):
            logger.error(f"File count mismatch after refinement! Initial: {len(all_files_initial)}, Refined: {len(refined_all_files)}. Some files might be duplicated.")
            # This indicates a potential bug in duplicate handling or other logic


        # --- Update Explanation and Create Final Strategy Object ---
        refined_explanation = (
            f"{grouping_strategy.explanation}\n\n"
            "Refinement Applied: Groups were adjusted based on validation feedback to improve "
            "completeness, balance, coherence, and metadata."
        )

        # Recalculate complexity based on refined groups
        final_complexity = self._estimate_review_complexity(refined_groups)

        refined_strategy = PRGroupingStrategy(
            strategy_type=strategy_type,
            groups=refined_groups,
            explanation=refined_explanation,
            estimated_review_complexity=final_complexity,
            ungrouped_files=[] # Should be empty after refinement adds misc group
        )

        logger.info(f"Refinement complete. Final group count: {len(refined_strategy.groups)}")
        return refined_strategy

    # --- Helper methods (_handle_*, _choose_*, _find_*, _consolidate*, _rebalance*, etc.) ---
    # Keep these methods as they contain the core logic. Ensure they operate correctly
    # on the List[PRGroup] passed to them. Add logging within helpers for debugging.

    def _handle_empty_groups(self, groups: List[PRGroup], issues: List[Any]) -> None:
        empty_group_issues = [iss for iss in issues if iss.issue_type == "empty_group"]
        if not empty_group_issues: return

        empty_group_titles = {title for iss in empty_group_issues for title in iss.affected_groups}
        logger.debug(f"Removing empty groups: {empty_group_titles}")

        initial_count = len(groups)
        groups[:] = [g for g in groups if g.title not in empty_group_titles or g.files] # Modify list in-place
        removed_count = initial_count - len(groups)
        if removed_count > 0:
             logger.info(f"Removed {removed_count} empty group(s).")

    def _handle_duplicate_files(self, groups: List[PRGroup], issues: List[Any]) -> None:
        duplicate_file_issues = [iss for iss in issues if iss.issue_type == "duplicate_files"]
        if not duplicate_file_issues: return

        logger.debug("Handling duplicate files across groups...")
        file_to_groups_indices = defaultdict(list)
        for i, group in enumerate(groups):
            for file_path in group.files:
                file_to_groups_indices[file_path].append(i)

        duplicates = {fp: idxs for fp, idxs in file_to_groups_indices.items() if len(idxs) > 1}
        resolved_count = 0

        for file_path, group_indices in duplicates.items():
            if len(group_indices) <= 1: continue # Should not happen based on filter, but safe check

            candidate_groups = [groups[i] for i in group_indices]
            best_group_local_idx = self._choose_best_group_for_file(file_path, candidate_groups)
            best_group_global_idx = group_indices[best_group_local_idx]
            logger.debug(f"File '{file_path}' duplicated in groups {group_indices}. Assigning to group index {best_group_global_idx}.")

            # Remove file from all groups except the chosen one
            for i, group_idx in enumerate(group_indices):
                if group_idx != best_group_global_idx:
                    try:
                        groups[group_idx].files.remove(file_path)
                        groups[group_idx].estimated_size = len(groups[group_idx].files) # Update size
                        resolved_count += 1
                    except ValueError:
                         # This might happen if the file was already removed by another refinement step, log warning
                         logger.warning(f"Could not remove duplicate file '{file_path}' from group index {group_idx} (title: '{groups[group_idx].title}'). It might have been removed already.")
        if resolved_count > 0:
             logger.info(f"Resolved {resolved_count} instances of duplicate files.")

    def _handle_missing_files(self,
                           groups: List[PRGroup],
                           ungrouped_files: List[str],
                           issues: List[Any]) -> None:
        missing_file_issues = [iss for iss in issues if iss.issue_type in ["missing_files", "ungrouped_files"]]
        all_unassigned = set(ungrouped_files) # Start with explicitly ungrouped

        for issue in missing_file_issues:
            if issue.issue_type == "missing_files":
                # Crude parsing from description - adjust if format changes
                match = re.search(r"group: (.*)", issue.description)
                if match:
                    files_str = match.group(1).split("...")[0] # Get part before ellipsis
                    found_files = {p.strip().strip("'\"`") for p in files_str.split(',')}
                    all_unassigned.update(found_files)
                    logger.debug(f"Identified potentially missing files from issue description: {found_files}")

        if not all_unassigned:
            logger.debug("No missing or ungrouped files identified to handle.")
            return

        logger.debug(f"Attempting to assign {len(all_unassigned)} missing/ungrouped files...")
        assigned_count = 0
        still_unassigned = set()

        for file_path in all_unassigned:
            best_group = self._find_best_group_for_file(file_path, groups)
            if best_group:
                if file_path not in best_group.files: # Avoid adding duplicates within a group
                     best_group.files.append(file_path)
                     best_group.estimated_size = len(best_group.files) # Update size
                     assigned_count += 1
                     logger.debug(f"Assigned missing file '{file_path}' to group '{best_group.title}'.")
                else:
                     logger.warning(f"Missing file '{file_path}' was already present in target group '{best_group.title}'.")

            else:
                still_unassigned.add(file_path)

        logger.info(f"Assigned {assigned_count} missing/ungrouped files to existing groups.")

        # If files remain unassigned after trying to find best fit, they will be caught
        # by the final completeness check and put into the miscellaneous group.
        if still_unassigned:
             logger.info(f"{len(still_unassigned)} files remain unassigned after initial pass, will be added to misc group if needed.")


    def _handle_related_files(self, groups: List[PRGroup], issues: List[Any]) -> None:
        related_file_issues = [iss for iss in issues if iss.issue_type in ["separated_test_impl", "separated_model_schema"]]
        if not related_file_issues: return

        logger.debug("Handling separated related files (test/impl, model/schema)...")
        consolidated_count = 0

        for issue in related_file_issues:
            # Improved parsing
            file1_match = re.search(r"file '([^']+)' is in group '([^']+)'", issue.description, re.IGNORECASE)
            file2_match = re.search(r"(?:implementation|schema) '([^']+)' is in group '([^']+)'", issue.description, re.IGNORECASE)

            if file1_match and file2_match:
                file1, group1_title = file1_match.groups()
                file2, group2_title = file2_match.groups()

                if group1_title != group2_title:
                     logger.debug(f"Consolidating related files: '{file1}' (in '{group1_title}') and '{file2}' (in '{group2_title}')")
                     self._consolidate_files(groups, file1, file2)
                     consolidated_count += 1
            else:
                 logger.warning(f"Could not parse related file paths/groups from issue: {issue.description}")

        if consolidated_count > 0:
             logger.info(f"Consolidated {consolidated_count} pairs of related files into the same group.")


    def _handle_imbalanced_groups(self, groups: List[PRGroup], issues: List[Any]) -> None:
        # Consider issues indicating imbalance or specific oversized groups
        imbalance_triggered = any(iss.issue_type in ["imbalanced_groups", "oversized_group"] for iss in issues)
        if not imbalance_triggered: return
        if len(groups) < 2: return # Need at least two groups to rebalance

        logger.debug("Handling imbalanced/oversized groups...")
        # --- Simple Rebalancing Attempt ---
        # Find largest and smallest groups
        groups.sort(key=lambda g: len(g.files), reverse=True) # Sort in-place, largest first
        largest_group = groups[0]
        smallest_group = groups[-1]

        # Define imbalance/oversize criteria (can be tuned)
        is_oversized = len(largest_group.files) > 20
        is_imbalanced = len(largest_group.files) > 5 * len(smallest_group.files) if len(smallest_group.files) > 0 else False

        if is_oversized or is_imbalanced:
             logger.info(f"Attempting rebalance. Largest group '{largest_group.title}' ({len(largest_group.files)} files), Smallest group '{smallest_group.title}' ({len(smallest_group.files)} files).")
             # Move one file from largest to smallest as a simple heuristic
             # More sophisticated logic could move multiple based on targets
             if len(largest_group.files) > 1: # Can only move if largest has > 1 file
                  # Select a file to move (e.g., the first one lexicographically)
                  file_to_move = min(largest_group.files)
                  logger.debug(f"Moving file '{file_to_move}' from '{largest_group.title}' to '{smallest_group.title}'.")
                  self._move_files_between_groups(largest_group, smallest_group, [file_to_move])
             else:
                  logger.warning(f"Cannot rebalance: Largest group '{largest_group.title}' only has one file.")
        else:
             logger.debug("Groups considered sufficiently balanced/sized based on current heuristics.")

    def _update_missing_metadata(self, groups: List[PRGroup]) -> None:
        logger.debug("Checking and updating missing metadata (titles, branches, descriptions)...")
        updated_count = 0
        for i, group in enumerate(groups):
             group_updated = False
             # Ensure title exists
             if not group.title:
                  group.title = f"Unnamed Group {i+1}"
                  logger.warning(f"Group at index {i} had no title, assigned default: '{group.title}'")
                  group_updated = True

             # Add missing branch name
             if not group.suggested_branch_name:
                  title = group.title
                  branch_name = title.lower().replace(" ", "-")
                  branch_name = re.sub(r'[^\w\-]+', '', branch_name).strip('-') # Allow alphanumeric and hyphen
                  branch_name = re.sub(r'[-]+', '-', branch_name) # Collapse multiple hyphens
                  group.suggested_branch_name = f"feature/{branch_name[:50]}" # Limit length
                  logger.debug(f"Generated branch name for '{title}': '{group.suggested_branch_name}'")
                  group_updated = True

             # Add missing PR description
             if not group.suggested_pr_description:
                  title = group.title
                  rationale = group.rationale or "Grouped related changes."
                  files = group.files

                  description = f"## {title}\n\n"
                  description += f"{rationale}\n\n"
                  description += f"### Files Changed ({len(files)})\n\n"
                  # List first 10 files, indicate if more
                  for file_path in sorted(files)[:10]:
                       description += f"- `{file_path}`\n"
                  if len(files) > 10:
                       description += f"- ... and {len(files) - 10} more file(s)\n"
                  group.suggested_pr_description = description
                  logger.debug(f"Generated PR description for '{title}'.")
                  group_updated = True

             if group_updated:
                  updated_count += 1
        if updated_count > 0:
            logger.info(f"Updated missing metadata for {updated_count} group(s).")


    # --- Core Heuristic/Utility Helpers ---
    # _choose_best_group_for_file, _find_best_group_for_file, _consolidate_files,
    # _rebalance_groups (could be simplified as above or kept complex),
    # _select_files_to_move (and its sub-helpers), _move_files_between_groups
    # --- Add estimate_review_complexity ---
    def _estimate_review_complexity(self, groups: List[PRGroup]) -> float:
        """Estimate the review complexity (1-10 scale) of the PR groups."""
        if not groups: return 1.0

        group_count = len(groups)
        total_files = sum(len(g.files) for g in groups)
        avg_files_per_group = total_files / group_count if group_count > 0 else 0
        max_files_in_group = max((len(g.files) for g in groups), default=0)
        # Complexity score calculation (example heuristic)
        complexity = 1.0
        complexity += min(3.0, group_count / 2.0)      # Penalty for too many PRs
        complexity += min(3.0, avg_files_per_group / 5.0) # Penalty for large average size
        complexity += min(3.0, max_files_in_group / 10.0) # Penalty for very large individual PRs
        # Add penalty for single-file groups if there are many groups?
        single_file_groups = sum(1 for g in groups if len(g.files) == 1)
        if group_count > 3 and single_file_groups > group_count / 2:
             complexity += 1.0 # Penalty if more than half are single files in a multi-PR scenario

        return round(min(10.0, max(1.0, complexity)), 1) # Clamp between 1 and 10

    # Include other required helpers (_choose_best_group_for_file, _find_best_group_for_file, etc.) here
    # Make sure their logic is sound and they handle PRGroup objects correctly.
    # Add logging within them for better traceability.

    def _choose_best_group_for_file(self, file_path: str, candidate_groups: List[PRGroup]) -> int:
        """Heuristic to choose the best group for a file among candidates (e.g., for duplicates)."""
        if not candidate_groups: return 0 # Should not happen if called correctly
        if len(candidate_groups) == 1: return 0 # Only one choice

        path = Path(file_path)
        directory = str(path.parent) if path.parent != Path('.') else "(root)"
        extension = path.suffix.lower()
        best_score = -1
        best_idx = 0

        logger.debug(f"Choosing best group for '{file_path}' among {len(candidate_groups)} candidates...")

        for i, group in enumerate(candidate_groups):
            score = 0
            # Factor 1: Directory Focus Match (Strong signal)
            if group.directory_focus and directory.startswith(group.directory_focus):
                score += 10
                # Bonus for exact match
                if directory == group.directory_focus:
                    score += 5

            # Factor 2: Feature Focus Match (Medium signal)
            if group.feature_focus and group.feature_focus != "miscellaneous":
                 # Check if feature focus is part of the path segments
                 if group.feature_focus in path.parts or group.feature_focus in path.stem.lower():
                      score += 8

            # Factor 3: File Extension Affinity (Lower signal)
            group_extensions = {Path(f).suffix.lower() for f in group.files if Path(f).suffix}
            if extension in group_extensions:
                score += 3
                # Bonus if the group primarily contains this extension type
                if len(group.files) > 0 and group_extensions.count(extension) / len(group.files) > 0.5:
                    score += 2

            # Factor 4: Group Cohesion (Check if siblings are present)
            siblings = {str(p) for p in path.parent.glob('*') if p.is_file()} & set(group.files)
            if len(siblings) > 0:
                 score += 4 # Bonus if other files from same dir are already here

            # Factor 5: Group Size Penalty (Discourage adding to already large groups)
            if len(group.files) > 15: score -= 5
            elif len(group.files) > 10: score -= 2

            logger.debug(f"  - Group '{group.title}' (idx {i}): Score {score}")
            if score > best_score:
                best_score = score
                best_idx = i

        logger.debug(f"Best group chosen: Index {best_idx} ('{candidate_groups[best_idx].title}') with score {best_score}")
        return best_idx

    def _find_best_group_for_file(self, file_path: str, groups: List[PRGroup]) -> Optional[PRGroup]:
        """Heuristic to find the best *existing* group for an ungrouped file."""
        if not groups: return None

        path = Path(file_path)
        directory = str(path.parent) if path.parent != Path('.') else "(root)"
        filename = path.name
        extension = path.suffix.lower()

        group_scores = []
        logger.debug(f"Finding best group for unassigned file '{file_path}'...")

        for i, group in enumerate(groups):
            score = 0
            group_files = group.files
            if not group_files: continue # Skip empty groups

            # Factor 1: Test/Implementation Pairing (Very strong signal)
            is_test_file = filename.startswith("test_") or filename.endswith("Test.java") # Example patterns
            if is_test_file:
                 # Try to find corresponding implementation file in the group
                 base_name = filename.split("test_")[-1] if filename.startswith("test_") else filename.replace("Test.java",".java") # Simplified
                 impl_candidates = [f for f in group_files if Path(f).name == base_name]
                 if impl_candidates: score += 30
            else:
                 # Check if a corresponding test file exists in the group
                 test_name1 = f"test_{filename}"
                 test_name2 = f"{path.stem}Test.java" # Example
                 test_candidates = [f for f in group_files if Path(f).name in [test_name1, test_name2]]
                 if test_candidates: score += 30

            # Factor 2: Directory Focus Match
            if group.directory_focus and directory.startswith(group.directory_focus):
                score += 15
                if directory == group.directory_focus: score += 5

            # Factor 3: Sibling Files Present
            siblings = {str(p) for p in path.parent.glob('*') if p.is_file()} & set(group.files)
            if len(siblings) > 0: score += 10

            # Factor 4: Feature Focus Match
            if group.feature_focus and group.feature_focus != "miscellaneous":
                 if group.feature_focus in path.parts or group.feature_focus in path.stem.lower():
                      score += 8

            # Factor 5: Extension Affinity
            group_extensions = {Path(f).suffix.lower() for f in group_files if Path(f).suffix}
            if extension in group_extensions:
                score += 5

            # Factor 6: Group Size Preference (Prefer smaller, non-empty groups)
            group_size = len(group_files)
            if 1 <= group_size < 5: score += 4
            elif 5 <= group_size < 10: score += 2
            elif group_size >= 15: score -= 5 # Penalty for large groups

            if score > 0: # Only consider groups with a positive score
                 group_scores.append((group, score))
            logger.debug(f"  - Group '{group.title}' (idx {i}): Score {score}")


        if not group_scores:
            logger.debug(f"No suitable group found for '{file_path}'.")
            return None

        group_scores.sort(key=lambda x: x[1], reverse=True)
        best_group, best_score = group_scores[0]
        # Add a minimum score threshold if desired
        min_score_threshold = 5
        if best_score >= min_score_threshold:
             logger.debug(f"Best group found: '{best_group.title}' with score {best_score}")
             return best_group
        else:
             logger.debug(f"Best group '{best_group.title}' score {best_score} below threshold {min_score_threshold}.")
             return None

    def _consolidate_files(self, groups: List[PRGroup], file1: str, file2: str) -> None:
        """Ensures two related files end up in the same group."""
        file1_indices = [i for i, g in enumerate(groups) if file1 in g.files]
        file2_indices = [i for i, g in enumerate(groups) if file2 in g.files]

        file1_group_idx = file1_indices[0] if file1_indices else None
        file2_group_idx = file2_indices[0] if file2_indices else None

        if file1_group_idx is not None and file2_group_idx is not None:
            # Both files are in groups
            if file1_group_idx == file2_group_idx: return # Already together

            # Move file from smaller group to larger group (or arbitrarily if equal)
            group1 = groups[file1_group_idx]
            group2 = groups[file2_group_idx]
            if len(group1.files) <= len(group2.files): # Move file1 to group2
                logger.debug(f"Consolidating: Moving '{file1}' from '{group1.title}' to '{group2.title}' (joining '{file2}')")
                self._move_files_between_groups(group1, group2, [file1])
            else: # Move file2 to group1
                logger.debug(f"Consolidating: Moving '{file2}' from '{group2.title}' to '{group1.title}' (joining '{file1}')")
                self._move_files_between_groups(group2, group1, [file2])

        elif file1_group_idx is not None: # Only file1 is grouped, add file2
            group1 = groups[file1_group_idx]
            if file2 not in group1.files:
                 logger.debug(f"Consolidating: Adding '{file2}' to '{group1.title}' (joining '{file1}')")
                 group1.files.append(file2)
                 group1.estimated_size = len(group1.files)

        elif file2_group_idx is not None: # Only file2 is grouped, add file1
             group2 = groups[file2_group_idx]
             if file1 not in group2.files:
                  logger.debug(f"Consolidating: Adding '{file1}' to '{group2.title}' (joining '{file2}')")
                  group2.files.append(file1)
                  group2.estimated_size = len(group2.files)
        else:
             # Neither file is currently grouped - they might be added later by _handle_missing_files
             logger.warning(f"Cannot consolidate '{file1}' and '{file2}' as neither is currently in a group.")

    def _move_files_between_groups(self, source_group: PRGroup, target_group: PRGroup, files_to_move: List[str]) -> None:
        """Moves files and updates metadata."""
        moved_count = 0
        for file_path in files_to_move:
            if file_path in source_group.files:
                source_group.files.remove(file_path)
                if file_path not in target_group.files: # Avoid adding duplicates to target
                     target_group.files.append(file_path)
                moved_count += 1
            else:
                 logger.warning(f"Attempted to move file '{file_path}' from '{source_group.title}', but it was not found.")

        if moved_count > 0:
             # Update sizes
             source_group.estimated_size = len(source_group.files)
             target_group.estimated_size = len(target_group.files)
             # Optionally update rationale/description (simple update)
             source_group.rationale += f" (Note: {moved_count} file(s) moved to '{target_group.title}' during refinement)."
             target_group.rationale += f" (Note: Includes {moved_count} file(s) moved from '{source_group.title}' during refinement)."
             logger.debug(f"Moved {moved_count} file(s) from '{source_group.title}' to '{target_group.title}'.")


    # Remove _rebalance_groups and its sub-helpers (_select_files_*) if the simple
    # rebalancing in _handle_imbalanced_groups is sufficient. If more complex
    # balancing is needed, ensure they are correctly implemented. For now, relying
    # on the simplified logic in _handle_imbalanced_groups.