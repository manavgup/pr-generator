"""
Group refiner tool for refining and balancing PR groups.
"""
import re
import json
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from .base_tools import BaseRepoTool

logger = get_logger(__name__)

class GroupRefinerInput(BaseModel):
    """Input schema for the GroupRefiner tool."""
    repo_path: str = Field(..., description="Path to the git repository (required by BaseRepoTool)")
    grouping_strategy: Dict[str, Any] = Field(..., description="The PR grouping strategy to refine")
    validation_result: Dict[str, Any] = Field(..., description="Validation result data from the GroupValidator tool")

class GroupRefiner(BaseRepoTool):
    """
    Tool for refining and balancing PR groups based on validation results.
    Takes an existing grouping strategy and validation feedback, then applies
    heuristics to improve the groups (e.g., balancing size, fixing duplicates).
    Outputs the refined PRGroupingStrategy.
    """

    name: str = "Group Refiner"
    description: str = "Refines and balances PR groups based on validation results for optimal review experience"
    args_schema: type[BaseModel] = GroupRefinerInput

    def _run(self, **kwargs) -> str:
        """
        Refine and balance PR groups based on validation results.

        Args:
            **kwargs: Expects 'repo_path', 'grouping_strategy', and 'validation_result'
                      based on args_schema.

        Returns:
            JSON string containing the refined PR grouping strategy
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
            return json.dumps({"error": error_msg})

        try:
            logger.info(f"Running Group Refiner Tool on {repo_path}")

            # --- Core Logic ---
            # If validation already passed, no refinement needed
            if validation_result.get("is_valid", False):
                logger.info("Validation passed, no refinement needed.")
                return json.dumps(grouping_strategy, indent=2)

            logger.info("Refining PR groups based on validation issues...")
            strategy_type = grouping_strategy.get("strategy_type", "mixed")
            
            # Deep copy groups to avoid modifying the original
            original_groups = []
            for group in grouping_strategy.get("groups", []):
                # Manual deep copy for dictionaries
                copied_group = {}
                for key, value in group.items():
                    if key == "files":
                        # Deep copy for list of files
                        copied_group[key] = list(value) if value else []
                    else:
                        copied_group[key] = value
                original_groups.append(copied_group)
                
            # Copy ungrouped files list
            original_ungrouped = list(grouping_strategy.get("ungrouped_files", []))

            refined_groups = original_groups  # Work on the copy

            # Process validation issues
            issues = validation_result.get("issues", [])

            # Keep track of all files to ensure none are lost in refinement
            all_files_initial = set()
            for group in original_groups:
                all_files_initial.update(group.get("files", []))
            all_files_initial.update(original_ungrouped)

            # --- Apply Refinement Steps ---
            # Note: These helpers modify refined_groups in place
            self._handle_empty_groups(refined_groups, issues)
            self._handle_duplicate_files(refined_groups, issues)  # Handles duplicates across the *current* refined_groups
            self._handle_missing_files(refined_groups, original_ungrouped, issues)  # Adds missing/ungrouped
            self._handle_related_files(refined_groups, issues)  # Consolidates related pairs
            self._handle_imbalanced_groups(refined_groups, issues)  # Rebalances sizes

            # Final pass updates
            self._update_missing_metadata(refined_groups)  # Fills in missing titles/branches/descriptions

            # --- Verify Completeness Post-Refinement ---
            refined_all_files = set()
            for group in refined_groups:
                refined_all_files.update(group.get("files", []))

            missing_after_refinement = all_files_initial - refined_all_files
            if missing_after_refinement:
                logger.warning(f"Files missing after refinement: {missing_after_refinement}. Adding to miscellaneous group.")
                # Find or create a miscellaneous group
                misc_group = next((g for g in refined_groups if g.get("feature_focus") == "miscellaneous"), None)
                if misc_group:
                    current_files = set(misc_group.get("files", []))
                    new_files_to_add = list(missing_after_refinement - current_files)
                    if new_files_to_add:
                        misc_group["files"].extend(new_files_to_add)
                        misc_group["estimated_size"] = len(misc_group["files"])
                    logger.info(f"Added {len(new_files_to_add)} files to existing miscellaneous group.")
                else:
                    logger.info("Creating new miscellaneous group.")
                    missing_files_list = list(missing_after_refinement)
                    refined_groups.append({
                        "title": "Miscellaneous Unassigned Changes",
                        "files": missing_files_list,
                        "rationale": "These files were unassigned or became unassigned during the refinement process and are grouped here to ensure completeness.",
                        "estimated_size": len(missing_files_list),
                        "directory_focus": None,
                        "feature_focus": "miscellaneous",
                        "suggested_branch_name": "feature/misc-unassigned-changes",
                        "suggested_pr_description": f"## Miscellaneous Unassigned Changes\n\nThis PR includes {len(missing_files_list)} files that could not be assigned to other logical groups during refinement."
                    })
            elif len(all_files_initial) != len(refined_all_files):
                logger.error(f"File count mismatch after refinement! Initial: {len(all_files_initial)}, Refined: {len(refined_all_files)}. Some files might be duplicated.")

            # --- Update Explanation and Create Final Strategy Object ---
            explanation = grouping_strategy.get("explanation", "")
            refined_explanation = (
                f"{explanation}\n\n"
                "Refinement Applied: Groups were adjusted based on validation feedback to improve "
                "completeness, balance, coherence, and metadata."
            )

            # Recalculate complexity based on refined groups
            final_complexity = self._estimate_review_complexity(refined_groups)

            # Build the refined strategy result
            refined_strategy = {
                "strategy_type": strategy_type,
                "groups": refined_groups,
                "explanation": refined_explanation,
                "estimated_review_complexity": final_complexity,
                "ungrouped_files": []  # Should be empty after refinement adds misc group
            }

            logger.info(f"Refinement complete. Final group count: {len(refined_strategy['groups'])}")
            return json.dumps(refined_strategy, indent=2)

        except Exception as e:
            error_msg = f"Error refining PR groups: {str(e)}"
            logger.error(error_msg)
            
            # Return a serialized error response
            error_result = {
                "error": error_msg,
                "strategy_type": grouping_strategy.get("strategy_type", "unknown") if grouping_strategy else "unknown",
                "groups": grouping_strategy.get("groups", []) if grouping_strategy else [],
                "explanation": "Failed to refine groups due to an error.",
                "estimated_review_complexity": grouping_strategy.get("estimated_review_complexity", 5.0) if grouping_strategy else 5.0,
                "ungrouped_files": grouping_strategy.get("ungrouped_files", []) if grouping_strategy else []
            }
            
            return json.dumps(error_result, indent=2)

    # --- Helper methods modified to work with dictionaries ---
    
    def _handle_empty_groups(self, groups: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> None:
        """Remove empty groups identified in validation issues."""
        empty_group_issues = [iss for iss in issues if iss.get("issue_type") == "empty_group"]
        if not empty_group_issues:
            return

        empty_group_titles = set()
        for issue in empty_group_issues:
            affected_groups = issue.get("affected_groups", [])
            empty_group_titles.update(affected_groups)
            
        logger.debug(f"Removing empty groups: {empty_group_titles}")

        initial_count = len(groups)
        
        # Filter groups with either a title not in empty_group_titles or with files
        groups_to_keep = []
        for group in groups:
            title = group.get("title", "")
            files = group.get("files", [])
            if title not in empty_group_titles or files:
                groups_to_keep.append(group)
                
        # Modify list in-place to replace contents
        groups[:] = groups_to_keep
        
        removed_count = initial_count - len(groups)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} empty group(s).")

    def _handle_duplicate_files(self, groups: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> None:
        """Handle duplicate files across groups."""
        duplicate_file_issues = [iss for iss in issues if iss.get("issue_type") == "duplicate_files"]
        if not duplicate_file_issues:
            return

        logger.debug("Handling duplicate files across groups...")
        
        # Build a mapping from files to the group indices that contain them
        file_to_groups_indices = defaultdict(list)
        for i, group in enumerate(groups):
            for file_path in group.get("files", []):
                file_to_groups_indices[file_path].append(i)

        duplicates = {fp: idxs for fp, idxs in file_to_groups_indices.items() if len(idxs) > 1}
        resolved_count = 0

        for file_path, group_indices in duplicates.items():
            if len(group_indices) <= 1:
                continue  # Should not happen based on filter, but safe check

            # Get candidate groups for this file
            candidate_groups = [groups[i] for i in group_indices]
            best_group_local_idx = self._choose_best_group_for_file(file_path, candidate_groups)
            best_group_global_idx = group_indices[best_group_local_idx]
            
            logger.debug(f"File '{file_path}' duplicated in groups {group_indices}. Assigning to group index {best_group_global_idx}.")

            # Remove file from all groups except the chosen one
            for i, group_idx in enumerate(group_indices):
                if group_idx != best_group_global_idx:
                    try:
                        group_files = groups[group_idx].get("files", [])
                        if file_path in group_files:
                            group_files.remove(file_path)
                            groups[group_idx]["files"] = group_files
                            groups[group_idx]["estimated_size"] = len(group_files)  # Update size
                            resolved_count += 1
                    except Exception as e:
                        logger.warning(f"Could not remove duplicate file '{file_path}' from group index {group_idx}: {str(e)}")
        
        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} instances of duplicate files.")

    def _handle_missing_files(self, groups: List[Dict[str, Any]], ungrouped_files: List[str], issues: List[Dict[str, Any]]) -> None:
        """Handle missing or ungrouped files by assigning them to appropriate groups."""
        missing_file_issues = [iss for iss in issues if iss.get("issue_type") in ["missing_files", "ungrouped_files"]]
        all_unassigned = set(ungrouped_files)  # Start with explicitly ungrouped

        for issue in missing_file_issues:
            if issue.get("issue_type") == "missing_files":
                # Extract missing files from the description
                desc = issue.get("description", "")
                match = re.search(r"Some files are not included in any group: (.*)", desc)
                if match:
                    files_str = match.group(1).split("...")[0]  # Get part before ellipsis
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
                files = best_group.get("files", [])
                if file_path not in files:  # Avoid adding duplicates within a group
                    files.append(file_path)
                    best_group["files"] = files
                    best_group["estimated_size"] = len(files)  # Update size
                    assigned_count += 1
                    logger.debug(f"Assigned missing file '{file_path}' to group '{best_group.get('title', 'Unknown')}'.")
                else:
                    logger.warning(f"Missing file '{file_path}' was already present in target group '{best_group.get('title', 'Unknown')}'.")
            else:
                still_unassigned.add(file_path)

        logger.info(f"Assigned {assigned_count} missing/ungrouped files to existing groups.")

        # If files remain unassigned after trying to find best fit, they will be caught
        # by the final completeness check and put into the miscellaneous group.
        if still_unassigned:
            logger.info(f"{len(still_unassigned)} files remain unassigned after initial pass, will be added to misc group if needed.")

    # The rest of your helper methods would be similarly refactored to handle dictionaries
    # For brevity, I'll include just a few more key methods

    def _handle_related_files(self, groups: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> None:
        """Consolidate related files (like tests and implementations) into the same groups."""
        related_file_issues = [iss for iss in issues if iss.get("issue_type") in ["separated_test_impl", "separated_model_schema"]]
        if not related_file_issues: 
            return

        logger.debug("Handling separated related files (test/impl, model/schema)...")
        consolidated_count = 0

        for issue in related_file_issues:
            desc = issue.get("description", "")
            
            # Extract file paths and group names using regular expressions
            file1_match = re.search(r"file '([^']+)' is in group '([^']+)'", desc, re.IGNORECASE)
            file2_match = re.search(r"(?:implementation|schema) '([^']+)' is in group '([^']+)'", desc, re.IGNORECASE)

            if file1_match and file2_match:
                file1, group1_title = file1_match.groups()
                file2, group2_title = file2_match.groups()

                if group1_title != group2_title:
                    logger.debug(f"Consolidating related files: '{file1}' (in '{group1_title}') and '{file2}' (in '{group2_title}')")
                    self._consolidate_files(groups, file1, file2)
                    consolidated_count += 1
            else:
                logger.warning(f"Could not parse related file paths/groups from issue: {desc}")

        if consolidated_count > 0:
            logger.info(f"Consolidated {consolidated_count} pairs of related files into the same group.")

    def _handle_imbalanced_groups(self, groups: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> None:
        """Rebalance groups to address size imbalances."""
        # Check if there are issues indicating imbalance or specific oversized groups
        imbalance_triggered = any(iss.get("issue_type") in ["imbalanced_groups", "oversized_group"] for iss in issues)
        if not imbalance_triggered or len(groups) < 2:
            return

        logger.debug("Handling imbalanced/oversized groups...")
        
        # Sort groups by size (files count), largest first
        groups.sort(key=lambda g: len(g.get("files", [])), reverse=True)
        
        largest_group = groups[0]
        smallest_group = groups[-1]

        # Define imbalance/oversize criteria
        largest_files = len(largest_group.get("files", []))
        smallest_files = len(smallest_group.get("files", []))
        
        is_oversized = largest_files > 20
        is_imbalanced = smallest_files > 0 and largest_files > 5 * smallest_files

        if is_oversized or is_imbalanced:
            logger.info(
                f"Attempting rebalance. Largest group '{largest_group.get('title', 'Unknown')}' "
                f"({largest_files} files), Smallest group '{smallest_group.get('title', 'Unknown')}' "
                f"({smallest_files} files)."
            )
            
            # Simple rebalancing - move one file from largest to smallest group
            if largest_files > 1:  # Can only move if largest has > 1 file
                largest_files_list = largest_group.get("files", [])
                smallest_files_list = smallest_group.get("files", [])
                
                # Choose a file to move (simplest approach - first one alphabetically)
                if largest_files_list:
                    file_to_move = sorted(largest_files_list)[0]
                    logger.debug(f"Moving file '{file_to_move}' from '{largest_group.get('title', 'Unknown')}' to '{smallest_group.get('title', 'Unknown')}'.")
                    
                    # Remove from source
                    largest_files_list.remove(file_to_move)
                    largest_group["files"] = largest_files_list
                    largest_group["estimated_size"] = len(largest_files_list)
                    
                    # Add to target
                    if file_to_move not in smallest_files_list:
                        smallest_files_list.append(file_to_move)
                        smallest_group["files"] = smallest_files_list
                        smallest_group["estimated_size"] = len(smallest_files_list)
                    
                    # Update rationale
                    largest_group["rationale"] = largest_group.get("rationale", "") + f" (Note: Some files moved to '{smallest_group.get('title', 'Unknown')}' during refinement.)"
                    smallest_group["rationale"] = smallest_group.get("rationale", "") + f" (Note: Includes files moved from '{largest_group.get('title', 'Unknown')}' during refinement.)"
            else:
                logger.warning(f"Cannot rebalance: Largest group '{largest_group.get('title', 'Unknown')}' only has one file.")
        else:
            logger.debug("Groups considered sufficiently balanced/sized based on current heuristics.")

    def _estimate_review_complexity(self, groups: List[Dict[str, Any]]) -> float:
        """Estimate the review complexity (1-10 scale) of the PR groups."""
        if not groups:
            return 1.0

        group_count = len(groups)
        total_files = sum(len(group.get("files", [])) for group in groups)
        avg_files_per_group = total_files / group_count if group_count > 0 else 0
        max_files_in_group = max((len(group.get("files", [])) for group in groups), default=0)
        
        # Complexity score calculation
        complexity = 1.0
        complexity += min(3.0, group_count / 2.0)       # Penalty for too many PRs
        complexity += min(3.0, avg_files_per_group / 5.0)  # Penalty for large average size
        complexity += min(3.0, max_files_in_group / 10.0)  # Penalty for very large individual PRs
        
        # Add penalty for single-file groups if there are many groups
        single_file_groups = sum(1 for g in groups if len(g.get("files", [])) == 1)
        if group_count > 3 and single_file_groups > group_count / 2:
            complexity += 1.0  # Penalty if more than half are single files in a multi-PR scenario

        return round(min(10.0, max(1.0, complexity)), 1)  # Clamp between 1 and 10

    # Add two key helper methods for finding best groups

    def _choose_best_group_for_file(self, file_path: str, candidate_groups: List[Dict[str, Any]]) -> int:
        """Determine the best group for a file among multiple candidates."""
        if not candidate_groups:
            return 0  # Should not happen if called correctly
        if len(candidate_groups) == 1:
            return 0  # Only one choice

        path = Path(file_path)
        directory = str(path.parent) if path.parent != Path('.') else "(root)"
        extension = path.suffix.lower()
        best_score = -1
        best_idx = 0

        logger.debug(f"Choosing best group for '{file_path}' among {len(candidate_groups)} candidates...")

        for i, group in enumerate(candidate_groups):
            score = 0
            # Factor 1: Directory Focus Match (Strong signal)
            dir_focus = group.get("directory_focus")
            if dir_focus and directory.startswith(dir_focus):
                score += 10
                # Bonus for exact match
                if directory == dir_focus:
                    score += 5

            # Factor 2: Feature Focus Match (Medium signal)
            feature_focus = group.get("feature_focus")
            if feature_focus and feature_focus != "miscellaneous":
                # Check if feature focus is part of the path segments
                if feature_focus in str(path.parts) or feature_focus in path.stem.lower():
                    score += 8

            # Factor 3: File Extension Affinity (Lower signal)
            group_files = group.get("files", [])
            group_extensions = {Path(f).suffix.lower() for f in group_files if Path(f).suffix}
            if extension in group_extensions:
                score += 3
                # Bonus if the group primarily contains this extension type
                ext_count = sum(1 for f in group_files if Path(f).suffix.lower() == extension)
                if len(group_files) > 0 and ext_count / len(group_files) > 0.5:
                    score += 2

            # Factor 4: Group Size Penalty (Discourage adding to already large groups)
            if len(group_files) > 15:
                score -= 5
            elif len(group_files) > 10:
                score -= 2

            logger.debug(f"  - Group '{group.get('title', 'Unknown')}' (idx {i}): Score {score}")
            if score > best_score:
                best_score = score
                best_idx = i

        logger.debug(f"Best group chosen: Index {best_idx} ('{candidate_groups[best_idx].get('title', 'Unknown')}') with score {best_score}")
        return best_idx

    def _find_best_group_for_file(self, file_path: str, groups: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best existing group for an ungrouped file."""
        if not groups:
            return None

        path = Path(file_path)
        directory = str(path.parent) if path.parent != Path('.') else "(root)"
        filename = path.name
        extension = path.suffix.lower()

        group_scores = []
        logger.debug(f"Finding best group for unassigned file '{file_path}'...")

        for i, group in enumerate(groups):
            score = 0
            group_files = group.get("files", [])
            if not group_files:
                continue  # Skip empty groups

            # Factor 1: Directory Focus Match
            dir_focus = group.get("directory_focus")
            if dir_focus and directory.startswith(dir_focus):
                score += 15
                if directory == dir_focus:
                    score += 5

            # Factor 2: Feature Focus Match
            feature_focus = group.get("feature_focus")
            if feature_focus and feature_focus != "miscellaneous":
                if feature_focus in str(path.parts) or feature_focus in path.stem.lower():
                    score += 8

            # Factor 3: Extension Affinity
            group_extensions = {Path(f).suffix.lower() for f in group_files if Path(f).suffix}
            if extension in group_extensions:
                score += 5

            # Factor 4: Group Size Preference (Prefer smaller, non-empty groups)
            group_size = len(group_files)
            if 1 <= group_size < 5:
                score += 4
            elif 5 <= group_size < 10:
                score += 2
            elif group_size >= 15:
                score -= 5  # Penalty for large groups

            if score > 0:  # Only consider groups with a positive score
                group_scores.append((group, score))
            logger.debug(f"  - Group '{group.get('title', 'Unknown')}' (idx {i}): Score {score}")

        if not group_scores:
            logger.debug(f"No suitable group found for '{file_path}'.")
            return None

        group_scores.sort(key=lambda x: x[1], reverse=True)
        best_group, best_score = group_scores[0]
        
        # Minimum score threshold to consider a group acceptable
        min_score_threshold = 5
        if best_score >= min_score_threshold:
            logger.debug(f"Best group found: '{best_group.get('title', 'Unknown')}' with score {best_score}")
            return best_group
        else:
            logger.debug(f"Best group '{best_group.get('title', 'Unknown')}' score {best_score} below threshold {min_score_threshold}.")
            return None
            
    def _update_missing_metadata(self, groups: List[Dict[str, Any]]) -> None:
        """Fill in missing metadata like titles, branch names, and descriptions."""
        logger.debug("Checking and updating missing metadata (titles, branches, descriptions)...")
        updated_count = 0
        
        for i, group in enumerate(groups):
            group_updated = False
            
            # Ensure title exists
            if not group.get("title"):
                group["title"] = f"Unnamed Group {i+1}"
                logger.warning(f"Group at index {i} had no title, assigned default: '{group['title']}'")
                group_updated = True

            # Add missing branch name
            if not group.get("suggested_branch_name"):
                title = group.get("title", f"group{i+1}")
                branch_name = title.lower().replace(" ", "-")
                branch_name = re.sub(r'[^\w\-]+', '', branch_name).strip('-')  # Allow alphanumeric and hyphen
                branch_name = re.sub(r'[-]+', '-', branch_name)  # Collapse multiple hyphens
                group["suggested_branch_name"] = f"feature/{branch_name[:50]}"  # Limit length
                logger.debug(f"Generated branch name for '{title}': '{group['suggested_branch_name']}'")
                group_updated = True

            # Add missing PR description
            if not group.get("suggested_pr_description"):
                title = group.get("title", f"Group {i+1}")
                rationale = group.get("rationale", "Grouped related changes.")
                files = group.get("files", [])

                description = f"## {title}\n\n"
                description += f"{rationale}\n\n"
                description += f"### Files Changed ({len(files)})\n\n"
                
                # List first 10 files, indicate if more
                for file_path in sorted(files)[:10]:
                    description += f"- `{file_path}`\n"
                if len(files) > 10:
                    description += f"- ... and {len(files) - 10} more file(s)\n"
                    
                group["suggested_pr_description"] = description
                logger.debug(f"Generated PR description for '{title}'.")
                group_updated = True

            if group_updated:
                updated_count += 1
                
        if updated_count > 0:
            logger.info(f"Updated missing metadata for {updated_count} group(s).")
            
    def _consolidate_files(self, groups: List[Dict[str, Any]], file1: str, file2: str) -> None:
        """Ensure two related files end up in the same group."""
        # Find which groups contain each file
        file1_indices = []
        file2_indices = []
        
        for i, group in enumerate(groups):
            files = group.get("files", [])
            if file1 in files:
                file1_indices.append(i)
            if file2 in files:
                file2_indices.append(i)

        file1_group_idx = file1_indices[0] if file1_indices else None
        file2_group_idx = file2_indices[0] if file2_indices else None

        if file1_group_idx is not None and file2_group_idx is not None:
            # Both files are in groups
            if file1_group_idx == file2_group_idx:
                return  # Already together

            # Get the groups
            group1 = groups[file1_group_idx]
            group2 = groups[file2_group_idx]
            
            # Move file from smaller group to larger group (or arbitrarily if equal)
            group1_files = len(group1.get("files", []))
            group2_files = len(group2.get("files", []))
            
            if group1_files <= group2_files:
                # Move file1 to group2
                logger.debug(f"Consolidating: Moving '{file1}' from '{group1.get('title', 'Unknown')}' to '{group2.get('title', 'Unknown')}' (joining '{file2}')")
                self._move_file(group1, group2, file1)
            else:
                # Move file2 to group1
                logger.debug(f"Consolidating: Moving '{file2}' from '{group2.get('title', 'Unknown')}' to '{group1.get('title', 'Unknown')}' (joining '{file1}')")
                self._move_file(group2, group1, file2)

        elif file1_group_idx is not None:
            # Only file1 is grouped, add file2
            group1 = groups[file1_group_idx]
            files = group1.get("files", [])
            if file2 not in files:
                logger.debug(f"Consolidating: Adding '{file2}' to '{group1.get('title', 'Unknown')}' (joining '{file1}')")
                files.append(file2)
                group1["files"] = files
                group1["estimated_size"] = len(files)

        elif file2_group_idx is not None:
            # Only file2 is grouped, add file1
            group2 = groups[file2_group_idx]
            files = group2.get("files", [])
            if file1 not in files:
                logger.debug(f"Consolidating: Adding '{file1}' to '{group2.get('title', 'Unknown')}' (joining '{file2}')")
                files.append(file1)
                group2["files"] = files
                group2["estimated_size"] = len(files)
        else:
            # Neither file is currently grouped - they might be added later by _handle_missing_files
            logger.warning(f"Cannot consolidate '{file1}' and '{file2}' as neither is currently in a group.")

    def _move_file(self, source_group: Dict[str, Any], target_group: Dict[str, Any], file_path: str) -> None:
        """Move a file from one group to another."""
        source_files = source_group.get("files", [])
        target_files = target_group.get("files", [])
        
        # Remove from source if present
        if file_path in source_files:
            source_files.remove(file_path)
            source_group["files"] = source_files
            source_group["estimated_size"] = len(source_files)
            
            # Add to target if not already there
            if file_path not in target_files:
                target_files.append(file_path)
                target_group["files"] = target_files
                target_group["estimated_size"] = len(target_files)
                
                # Update rationale
                source_title = source_group.get("title", "Unknown group")
                target_title = target_group.get("title", "Unknown group")
                
                source_group["rationale"] = source_group.get("rationale", "") + f" (Note: File '{file_path}' moved to '{target_title}' during refinement.)"
                target_group["rationale"] = target_group.get("rationale", "") + f" (Note: Includes file '{file_path}' moved from '{source_title}' during refinement.)"
                
                logger.debug(f"Moved file '{file_path}' from '{source_title}' to '{target_title}'.")
            else:
                logger.warning(f"File '{file_path}' already exists in target group '{target_group.get('title', 'Unknown')}', not duplicating.")
        else:
            logger.warning(f"File '{file_path}' not found in source group '{source_group.get('title', 'Unknown')}', cannot move.")