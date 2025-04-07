"""
Group validator tool for validating PR groups against best practices.
"""
from typing import List, Dict, Any, Set
from pathlib import Path
import re
import json

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from .base_tools import BaseRepoTool

logger = get_logger(__name__)

class GroupValidatorInput(BaseModel):
    """Input schema for the GroupValidator tool."""
    repo_path: str = Field(..., description="Path to the git repository")
    pr_grouping_strategy: Dict[str, Any] = Field(..., description="PR grouping strategy to validate")
    repository_analysis: Dict[str, Any] = Field(..., description="Repository analysis data")

class GroupValidatorTool(BaseRepoTool):
    """Tool for validating PR groups against best practices."""

    name: str = "Group Validator"
    description: str = "Validates PR groups against best practices"
    args_schema: type[BaseModel] = GroupValidatorInput

    def _run(self, repo_path: str, pr_grouping_strategy: Dict[str, Any], repository_analysis: Dict[str, Any], **kwargs) -> str:
        """
        Validate PR groups against best practices.

        Args:
            repo_path: Path to the git repository
            pr_grouping_strategy: PR grouping strategy as dictionary
            repository_analysis: Repository analysis data as dictionary
            **kwargs: Additional arguments (ignored)

        Returns:
            JSON string containing validation results
        """
        logger.info("Validating PR groups")

        try:
            strategy_type = pr_grouping_strategy.get("strategy_type", "mixed")
            groups = pr_grouping_strategy.get("groups", [])
            ungrouped_files = pr_grouping_strategy.get("ungrouped_files", [])

            # Get all file paths from repo analysis
            all_files = set()
            for fc in repository_analysis.get("file_changes", []):
                file_path = fc.get("path", "")
                if file_path:
                    all_files.add(file_path)

            # Validate that all files are included in groups
            issues = []
            validation_passed = True

            # Check for completeness
            all_grouped_files = set()
            for group in groups:
                all_grouped_files.update(group.get("files", []))

            missing_files = all_files - all_grouped_files
            if missing_files and not ungrouped_files:
                validation_passed = False
                missing_examples = sorted(list(missing_files))[:5]
                has_more = len(missing_files) > 5
                missing_desc = ', '.join(missing_examples)
                if has_more:
                    missing_desc += "..."
                    
                issues.append({
                    "severity": "high",
                    "issue_type": "missing_files",
                    "description": f"Some files are not included in any group: {missing_desc}",
                    "affected_groups": [],
                    "recommendation": "Ensure all files are included in at least one group."
                })
            elif ungrouped_files:
                validation_passed = False
                ungrouped_examples = ungrouped_files[:5]
                has_more = len(ungrouped_files) > 5
                ungrouped_desc = ', '.join(ungrouped_examples)
                if has_more:
                    ungrouped_desc += "..."
                    
                issues.append({
                    "severity": "medium",
                    "issue_type": "ungrouped_files",
                    "description": f"Some files are explicitly marked as ungrouped: {ungrouped_desc}",
                    "affected_groups": [],
                    "recommendation": "Consider creating additional groups or adding these files to existing groups."
                })

            # Check group sizes
            for i, group in enumerate(groups):
                group_title = group.get("title", f"Group {i+1}")
                files = group.get("files", [])

                # Check for empty groups
                if not files:
                    validation_passed = False
                    issues.append({
                        "severity": "high",
                        "issue_type": "empty_group",
                        "description": f"Group '{group_title}' has no files.",
                        "affected_groups": [group_title],
                        "recommendation": "Remove this empty group or add files to it."
                    })
                    continue

                # Check for oversized groups
                if len(files) > 20:
                    validation_passed = False
                    issues.append({
                        "severity": "medium",
                        "issue_type": "oversized_group",
                        "description": f"Group '{group_title}' has {len(files)} files, which may be too large for effective review.",
                        "affected_groups": [group_title],
                        "recommendation": "Consider splitting this group into smaller, more focused PRs."
                    })

                # Check for undersized groups
                if len(files) == 1 and len(groups) > 1:
                    issues.append({
                        "severity": "low",
                        "issue_type": "undersized_group",
                        "description": f"Group '{group_title}' has only one file: {files[0]}",
                        "affected_groups": [group_title],
                        "recommendation": "Consider combining with another group or adding related files."
                    })

            # Check for imbalanced group sizes
            if len(groups) >= 2:
                group_sizes = [len(group.get("files", [])) for group in groups]
                max_size = max(group_sizes)
                min_size = min(group_sizes)

                if max_size > 5 * min_size:
                    # Only flag as an issue if the largest group is at least 5x the smallest and has multiple files
                    validation_passed = False
                    large_group = groups[group_sizes.index(max_size)].get("title", f"Group {group_sizes.index(max_size)+1}")
                    small_group = groups[group_sizes.index(min_size)].get("title", f"Group {group_sizes.index(min_size)+1}")

                    issues.append({
                        "severity": "medium",
                        "issue_type": "imbalanced_groups",
                        "description": f"Large size disparity between groups: '{large_group}' has {max_size} files while '{small_group}' has only {min_size}.",
                        "affected_groups": [large_group, small_group],
                        "recommendation": "Consider rebalancing groups for more even review workload."
                    })

            # Check for duplicate files across groups
            file_to_groups = {}
            for i, group in enumerate(groups):
                group_title = group.get("title", f"Group {i+1}")
                for file_path in group.get("files", []):
                    if file_path not in file_to_groups:
                        file_to_groups[file_path] = []
                    file_to_groups[file_path].append(group_title)

            duplicate_files = {file_path: group_list for file_path, group_list in file_to_groups.items() if len(group_list) > 1}

            if duplicate_files:
                validation_passed = False
                duplicate_examples = list(duplicate_files.items())[:3]
                description = "Files appear in multiple groups: " + ", ".join(
                    f"'{file_path}' in {', '.join(groups)}" for file_path, groups in duplicate_examples
                )

                if len(duplicate_files) > 3:
                    description += f" and {len(duplicate_files) - 3} more"

                affected_groups = set()
                for groups_list in duplicate_files.values():
                    affected_groups.update(groups_list)

                issues.append({
                    "severity": "high",
                    "issue_type": "duplicate_files",
                    "description": description,
                    "affected_groups": list(affected_groups),
                    "recommendation": "Each file should appear in exactly one group to avoid confusion and duplicate work."
                })

            # Check for missing titles or descriptions
            for i, group in enumerate(groups):
                group_title = group.get("title", "")
                branch_name = group.get("suggested_branch_name", "")
                description = group.get("suggested_pr_description", "")

                if not group_title:
                    validation_passed = False
                    issues.append({
                        "severity": "medium",
                        "issue_type": "missing_title",
                        "description": f"Group at index {i} is missing a title.",
                        "affected_groups": [f"Group {i+1}"],
                        "recommendation": "Add a clear, descriptive title to each PR group."
                    })

                if not branch_name:
                    validation_passed = False
                    group_identifier = group_title or f"at index {i}"
                    issues.append({
                        "severity": "low",
                        "issue_type": "missing_branch_name",
                        "description": f"Group '{group_identifier}' is missing a suggested branch name.",
                        "affected_groups": [group_title or f"Group {i+1}"],
                        "recommendation": "Add a suggested branch name to each PR group."
                    })

                if not description:
                    validation_passed = False
                    group_identifier = group_title or f"at index {i}"
                    issues.append({
                        "severity": "medium",
                        "issue_type": "missing_description",
                        "description": f"Group '{group_identifier}' is missing a PR description.",
                        "affected_groups": [group_title or f"Group {i+1}"],
                        "recommendation": "Add a comprehensive description to each PR group."
                    })

            # Check for test files that aren't grouped with their implementation
            test_impl_issues = self._validate_test_implementation_pairing(groups)
            if test_impl_issues:
                validation_passed = False
                issues.extend(test_impl_issues)

            # Check for model files that aren't grouped with related schema files
            model_schema_issues = self._validate_model_schema_pairing(groups)
            if model_schema_issues:
                validation_passed = False
                issues.extend(model_schema_issues)

            # Sort issues by severity (high first, then medium, then low)
            severity_order = {"high": 0, "medium": 1, "low": 2}
            issues.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 99))

            # Generate validation notes
            validation_notes = self._generate_validation_notes(validation_passed, issues)

            # Create result dictionary
            result = {
                "is_valid": validation_passed,
                "issues": issues,
                "validation_notes": validation_notes,
                "strategy_type": strategy_type
            }

            # Return serialized JSON
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Error validating PR groups: {str(e)}"
            logger.error(error_msg)
            
            # Return a serialized error response
            error_result = {
                "is_valid": False,
                "issues": [{
                    "severity": "high",
                    "issue_type": "validation_error",
                    "description": f"Error during validation: {str(e)}",
                    "affected_groups": [],
                    "recommendation": "Check input data and try again."
                }],
                "validation_notes": f"Validation failed with error: {str(e)}",
                "strategy_type": pr_grouping_strategy.get("strategy_type", "unknown"),
                "error": error_msg
            }
            
            return json.dumps(error_result, indent=2)
            
    def _validate_test_implementation_pairing(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate that test files are grouped with their implementation.

        Args:
            groups: List of group dictionaries

        Returns:
            List of validation issues as dictionaries
        """
        issues = []

        # Create a mapping of file stems to their groups
        stem_to_group = {}
        for i, group in enumerate(groups):
            group_title = group.get("title", f"Group {i+1}")
            for file_path in group.get("files", []):
                path = Path(file_path)
                filename = path.name

                # Extract stem for both test and implementation files
                stem = filename
                is_test = False

                # Handle pytest style test files
                if filename.startswith("test_"):
                    stem = filename[5:]  # Remove "test_"
                    is_test = True

                # Handle Java/JUnit style test files
                elif filename.endswith("Test.java"):
                    stem = filename[:-9]  # Remove "Test.java"
                    is_test = True

                # Store mapping
                if stem not in stem_to_group:
                    stem_to_group[stem] = {"test": None, "impl": None, "test_path": None, "impl_path": None}

                if is_test:
                    stem_to_group[stem]["test"] = group_title
                    stem_to_group[stem]["test_path"] = file_path
                else:
                    stem_to_group[stem]["impl"] = group_title
                    stem_to_group[stem]["impl_path"] = file_path

        # Check for test and implementation files in different groups
        for stem, info in stem_to_group.items():
            test_group = info["test"]
            impl_group = info["impl"]

            if test_group and impl_group and test_group != impl_group:
                issues.append({
                    "severity": "medium",
                    "issue_type": "separated_test_impl",
                    "description": f"Test file '{info['test_path']}' is in group '{test_group}' but its implementation '{info['impl_path']}' is in group '{impl_group}'.",
                    "affected_groups": [test_group, impl_group],
                    "recommendation": "Tests and their implementations should typically be in the same PR for consistent review."
                })

        return issues

    def _validate_model_schema_pairing(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate that model files are grouped with their schema files.

        Args:
            groups: List of group dictionaries

        Returns:
            List of validation issues as dictionaries
        """
        issues = []

        # Create a mapping of model base names to their groups
        model_to_group = {}
        schema_to_group = {}

        # First pass: identify model and schema files
        for i, group in enumerate(groups):
            group_title = group.get("title", f"Group {i+1}")
            for file_path in group.get("files", []):
                path = Path(file_path)
                filename = path.name.lower()

                # Check for model files
                if "model" in filename:
                    # Extract the base name (part before "model")
                    match = re.search(r'([a-z0-9_]+)model', filename)
                    if match:
                        base_name = match.group(1)
                        model_to_group[base_name] = {
                            "group": group_title,
                            "path": file_path
                        }

                # Check for schema files
                if "schema" in filename:
                    # Extract the base name (part before "schema")
                    match = re.search(r'([a-z0-9_]+)schema', filename)
                    if match:
                        base_name = match.group(1)
                        schema_to_group[base_name] = {
                            "group": group_title,
                            "path": file_path
                        }

        # Second pass: check if model and schema files are in different groups
        for base_name, model_info in model_to_group.items():
            if base_name in schema_to_group:
                schema_info = schema_to_group[base_name]

                if model_info["group"] != schema_info["group"]:
                    issues.append({
                        "severity": "medium",
                        "issue_type": "separated_model_schema",
                        "description": f"Model file '{model_info['path']}' is in group '{model_info['group']}' but its schema '{schema_info['path']}' is in group '{schema_info['group']}'.",
                        "affected_groups": [model_info["group"], schema_info["group"]],
                        "recommendation": "Models and their schemas should typically be in the same PR for consistent review."
                    })

        return issues

    def _generate_validation_notes(self, is_valid: bool, issues: List[Dict[str, Any]]) -> str:
        """
        Generate validation notes based on results.

        Args:
            is_valid: Whether validation passed
            issues: List of validation issues

        Returns:
            Validation notes
        """
        if is_valid:
            return "All PR groups passed validation."
        else:
            num_issues = len(issues)
            return f"PR groups failed validation with {num_issues} issue(s)."