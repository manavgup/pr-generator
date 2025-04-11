"""
Group validator tool for validating PR groups against best practices.
"""
from typing import List, Dict, Any, Set, Type
from pathlib import Path
import re
import json

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from .base_tool import BaseRepoTool
from models.agent_models import PRGroupingStrategy, PRValidationResult, GroupValidationIssue, GroupingStrategyType

logger = get_logger(__name__)

class GroupValidatorToolSchema(BaseModel):
    pr_grouping_strategy_json: str = Field(..., description="JSON string of the PRGroupingStrategy object to validate.")
    # Optional flag to indicate final validation, if specific rules apply
    is_final_validation: bool = Field(default=False, description="Set to true if this is the final validation after merging batches.")

class GroupValidatorTool(BaseRepoTool):
    name: str = "Group Validator Tool"
    description: str = "Validates a set of proposed PR groups against predefined rules and best practices."
    args_schema: Type[BaseModel] = GroupValidatorToolSchema

    def _run(self, pr_grouping_strategy_json: str, is_final_validation: bool = False) -> str:
        """Validates the PR groups."""
        try:
            grouping_strategy = PRGroupingStrategy.model_validate_json(pr_grouping_strategy_json)
            logger.info(f"Validating {len(grouping_strategy.groups)} groups. Final validation: {is_final_validation}")

            issues: List[GroupValidationIssue] = []

            # --- Apply Validation Rules ---
            # These rules operate on the 'grouping_strategy.groups' list.
            # The rules themselves likely don't need to know if it's a batch or final,
            # unless a rule specifically relates to overall repository completeness.

            # Example Rule 1: Check for empty groups
            empty_groups = [g.title for g in grouping_strategy.groups if not g.files]
            if empty_groups:
                issues.append(GroupValidationIssue(
                    severity="high", issue_type="Empty Group",
                    description="Groups should not be empty.",
                    affected_groups=empty_groups,
                    recommendation="Remove or merge these groups."
                ))

            # Example Rule 2: Check for excessively large groups (placeholder)
            # size_limit = 500 # Example line count or complexity score
            # large_groups = [g.title for g in grouping_strategy.groups if g.estimated_size > size_limit]
            # if large_groups:
            #     issues.append(GroupValidationIssue(
            #         severity="medium", issue_type="Group Size",
            #         description=f"Groups exceed estimated size limit ({size_limit}).",
            #         affected_groups=large_groups,
            #         recommendation="Consider splitting these groups further."
            #     ))

            # Example Rule 3: Check for duplicate files across groups
            files_in_groups: Dict[str, List[str]] = {}
            all_files_set: List[str] = []
            for group in grouping_strategy.groups:
                for file_path in group.files:
                    if file_path not in files_in_groups:
                        files_in_groups[file_path] = []
                    files_in_groups[file_path].append(group.title)
                all_files_set.extend(group.files)

            duplicates = {fp: titles for fp, titles in files_in_groups.items() if len(titles) > 1}
            if duplicates:
                # Create one issue summarizing duplicates
                affected_titles = list(set(t for titles in duplicates.values() for t in titles))
                issues.append(GroupValidationIssue(
                    severity="high", issue_type="Duplicate Files",
                    description=f"Files found in multiple groups: {list(duplicates.keys())[:5]}...", # Show some examples
                    affected_groups=affected_titles,
                    recommendation="Ensure each file belongs to only one PR group. Refine merging logic or group definitions."
                ))

            # Example Rule 4: Check for ungrouped files (more relevant in final validation)
            if is_final_validation and grouping_strategy.ungrouped_files:
                 issues.append(GroupValidationIssue(
                    severity="medium", issue_type="Ungrouped Files",
                    description=f"{len(grouping_strategy.ungrouped_files)} files remain ungrouped.",
                    affected_groups=[], # Not specific to a group
                    recommendation="Assign ungrouped files to existing groups or create a 'miscellaneous' group."
                ))

            # --- End Validation Rules ---

            is_valid = not issues
            validation_notes = f"Validation complete. Found {len(issues)} issues."
            if not is_valid:
                validation_notes += " Issues detected, refinement may be required."
            logger.info(validation_notes)

            result = PRValidationResult(
                is_valid=is_valid,
                issues=issues,
                validation_notes=validation_notes,
                strategy_type=grouping_strategy.strategy_type
            )
            return result.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Error in GroupValidatorTool: {e}", exc_info=True)
            error_result = PRValidationResult(
                is_valid=False,
                issues=[GroupValidationIssue(severity="critical", issue_type="Tool Error", description=f"Validation failed: {e}", affected_groups=[], recommendation="Debug tool.")],
                validation_notes=f"Validation process failed: {e}",
                strategy_type=GroupingStrategyType.MIXED # Or some default
            )
            return error_result.model_dump_json(indent=2)

# END OF FILE group_validator_tool.py