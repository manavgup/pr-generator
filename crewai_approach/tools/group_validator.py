"""
Group validator tool for validating PR groups against best practices.
"""
from typing import List, Dict, Any, Set
from pathlib import Path
import re

from pydantic import BaseModel, Field

from shared.utils.logging_utils import get_logger
from models.agent_models import GroupingStrategyType, PRGroup, PRGroupingStrategy, PRValidationResult, GroupValidationIssue
from shared.models.analysis_models import RepositoryAnalysis
from .base_tools import BaseRepoTool

logger = get_logger(__name__)

class GroupValidatorInput(BaseModel):
    """Input schema for the GroupValidator tool."""
    repo_path: str = Field(..., description="Path to the git repository")
    pr_grouping_strategy: PRGroupingStrategy = Field(..., description="PR grouping strategy to validate")
    repository_analysis: RepositoryAnalysis = Field(..., description="Repository analysis data")

class GroupValidatorTool(BaseRepoTool):
    """Tool for validating PR groups against best practices."""

    name: str = "Group Validator"
    description: str = "Validates PR groups against best practices"
    args_schema: type[BaseModel] = GroupValidatorInput

    def _run(self, repo_path: str, pr_grouping_strategy: PRGroupingStrategy, repository_analysis: RepositoryAnalysis, **kwargs) -> PRValidationResult:
        """
        Validate PR groups against best practices.

        Args:
            repo_path: Path to the git repository
            pr_grouping_strategy: PR grouping strategy to validate
            repository_analysis: Repository analysis data

        Returns:
            PRValidationResult with validation results
        """
        logger.info("Validating PR groups")

        strategy_type = pr_grouping_strategy.strategy_type
        groups = pr_grouping_strategy.groups
        ungrouped_files = pr_grouping_strategy.ungrouped_files

        # Get all file paths from repo analysis
        all_files = {str(fc.path) for fc in repository_analysis.file_changes}

        # Validate that all files are included in groups
        issues: List[GroupValidationIssue] = []
        validation_passed = True

        # Check for completeness
        all_grouped_files = set()
        for group in groups:
            all_grouped_files.update(group.files)

        missing_files = all_files - all_grouped_files
        if missing_files and not ungrouped_files:
            validation_passed = False
            issues.append(GroupValidationIssue(
                severity="high",
                issue_type="missing_files",
                description=f"Some files are not included in any group: {', '.join(sorted(list(missing_files)[:5]))}{'...' if len(missing_files) > 5 else ''}",
                affected_groups=[],
                recommendation="Ensure all files are included in at least one group."
            ))
        elif ungrouped_files:
            validation_passed = False
            issues.append(GroupValidationIssue(
                severity="medium",
                issue_type="ungrouped_files",
                description=f"Some files are explicitly marked as ungrouped: {', '.join(sorted(ungrouped_files[:5]))}{'...' if len(ungrouped_files) > 5 else ''}",
                affected_groups=[],
                recommendation="Consider creating additional groups or adding these files to existing groups."
            ))

        # Check group sizes
        for i, group in enumerate(groups):
            group_title = group.title
            files = group.files

            # Check for empty groups
            if not files:
                validation_passed = False
                issues.append(GroupValidationIssue(
                    severity="high",
                    issue_type="empty_group",
                    description=f"Group '{group_title}' has no files.",
                    affected_groups=[group_title],
                    recommendation="Remove this empty group or add files to it."
                ))
                continue

            # Check for oversized groups
            if len(files) > 20:
                validation_passed = False
                issues.append(GroupValidationIssue(
                    severity="medium",
                    issue_type="oversized_group",
                    description=f"Group '{group_title}' has {len(files)} files, which may be too large for effective review.",
                    affected_groups=[group_title],
                    recommendation="Consider splitting this group into smaller, more focused PRs."
                ))

            # Check for undersized groups
            if len(files) == 1 and len(groups) > 1:
                issues.append(GroupValidationIssue(
                    severity="low",
                    issue_type="undersized_group",
                    description=f"Group '{group_title}' has only one file: {files[0]}",
                    affected_groups=[group_title],
                    recommendation="Consider combining with another group or adding related files."
                ))

        # Check for imbalanced group sizes
        if len(groups) >= 2:
            group_sizes = [len(group.files) for group in groups]
            max_size = max(group_sizes)
            min_size = min(group_sizes)

            if max_size > 5 * min_size:
                # Only flag as an issue if the largest group is at least 5x the smallest and has multiple files
                validation_passed = False
                large_group = groups[group_sizes.index(max_size)].title
                small_group = groups[group_sizes.index(min_size)].title

                issues.append(GroupValidationIssue(
                    severity="medium",
                    issue_type="imbalanced_groups",
                    description=f"Large size disparity between groups: '{large_group}' has {max_size} files while '{small_group}' has only {min_size}.",
                    affected_groups=[large_group, small_group],
                    recommendation="Consider rebalancing groups for more even review workload."
                ))

        # Check for duplicate files across groups
        file_to_groups: Dict[str, List[str]] = {}
        for i, group in enumerate(groups):
            group_title = group.title
            for file_path in group.files:
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
            for groups in duplicate_files.values():
                affected_groups.update(groups)

            issues.append(GroupValidationIssue(
                severity="high",
                issue_type="duplicate_files",
                description=description,
                affected_groups=list(affected_groups),
                recommendation="Each file should appear in exactly one group to avoid confusion and duplicate work."
            ))

        # Check for missing titles or descriptions
        for i, group in enumerate(groups):
            group_title = group.title
            branch_name = group.suggested_branch_name
            description = group.suggested_pr_description

            if not group_title:
                validation_passed = False
                issues.append(GroupValidationIssue(
                    severity="medium",
                    issue_type="missing_title",
                    description=f"Group at index {i} is missing a title.",
                    affected_groups=[f"Group {i+1}"],
                    recommendation="Add a clear, descriptive title to each PR group."
                ))

            if not branch_name:
                validation_passed = False
                issues.append(GroupValidationIssue(
                    severity="low",
                    issue_type="missing_branch_name",
                    description=f"Group '{group_title or f'at index {i}'}' is missing a suggested branch name.",
                    affected_groups=[group_title or f"Group {i+1}"],
                    recommendation="Add a suggested branch name to each PR group."
                ))

            if not description:
                validation_passed = False
                issues.append(GroupValidationIssue(
                    severity="medium",
                    issue_type="missing_description",
                    description=f"Group '{group_title or f'at index {i}'}' is missing a PR description.",
                    affected_groups=[group_title or f"Group {i+1}"],
                    recommendation="Add a comprehensive description to each PR group."
                ))

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
        issues.sort(key=lambda x: severity_order.get(x.severity, 99))

        # Generate validation notes
        validation_notes = self._generate_validation_notes(validation_passed, issues)

        return PRValidationResult(
            is_valid=validation_passed,
            issues=issues,
            validation_notes=validation_notes,
            strategy_type=strategy_type
        )

    def _validate_test_implementation_pairing(self, groups: List[PRGroup]) -> List[GroupValidationIssue]:
        """
        Validate that test files are grouped with their implementation.

        Args:
            groups: List of group dictionaries

        Returns:
            List of validation issues
        """
        issues: List[GroupValidationIssue] = []

        # Create a mapping of file stems to their groups
        stem_to_group = {}
        for i, group in enumerate(groups):
            group_title = group.title
            for file_path in group.files:
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
                issues.append(GroupValidationIssue(
                    severity="medium",
                    issue_type="separated_test_impl",
                    description=f"Test file '{info['test_path']}' is in group '{test_group}' but its implementation '{info['impl_path']}' is in group '{impl_group}'.",
                    affected_groups=[test_group, impl_group],
                    recommendation="Tests and their implementations should typically be in the same PR for consistent review."
                ))

        return issues

    def _validate_model_schema_pairing(self, groups: List[PRGroup]) -> List[GroupValidationIssue]:
        """
        Validate that model files are grouped with their schema files.

        Args:
            groups: List of group dictionaries

        Returns:
            List of validation issues
        """
        issues: List[GroupValidationIssue] = []

        # Create a mapping of model base names to their groups
        model_to_group = {}
        schema_to_group = {}

        # First pass: identify model and schema files
        for i, group in enumerate(groups):
            group_title = group.title
            for file_path in group.files:
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
                    issues.append(GroupValidationIssue(
                        severity="medium",
                        issue_type="separated_model_schema",
                        description=f"Model file '{model_info['path']}' is in group '{model_info['group']}' but its schema '{schema_info['path']}' is in group '{schema_info['group']}'.",
                        affected_groups=[model_info["group"], schema_info["group"]],
                        recommendation="Models and their schemas should typically be in the same PR for consistent review."
                    ))

        return issues

    def _generate_validation_notes(self, is_valid: bool, issues: List[GroupValidationIssue]) -> str:
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

    def _generate_strategy_explanation(self, strategy_type: GroupingStrategyType, groups: List[PRGroup]) -> str:
        """Generate an explanation for the grouping strategy."""
        group_count = len(groups)
        total_files = sum(len(group.files) for group in groups)

        strategy_name = strategy_type.value.replace("_", " ").title()

        explanation = (
            f"Applied {strategy_name} grouping strategy to organize {total_files} files "
            f"into {group_count} logical pull requests. "
        )

        if strategy_type == GroupingStrategyType.DIRECTORY_BASED:
            explanation += "Files were grouped based on their directory structure to maintain cohesion."
        elif strategy_type == GroupingStrategyType.FEATURE_BASED:
            explanation += "Files were grouped based on inferred features and related functionality."
        elif strategy_type == GroupingStrategyType.MODULE_BASED:
            explanation += "Files were grouped based on their types and likely module boundaries."
        elif strategy_type == GroupingStrategyType.SIZE_BALANCED:
            explanation += "Files were grouped to create balanced, manageable pull requests."
        elif strategy_type == GroupingStrategyType.MIXED:
            explanation += "A mixed strategy was applied, combining directory structure and feature grouping."

        return explanation

    def _estimate_review_complexity(self, groups: List[PRGroup]) -> float:
        """Estimate the review complexity (1-10 scale) of the PR groups."""
        if not groups:
            return 1.0

        # Factors that increase complexity:
        # 1. Number of groups
        # 2. Average files per group
        # 3. Maximum files in any group
        # 4. Presence of ungrouped files

        group_count = len(groups)
        total_files = sum(len(group.files) for group in groups)
        avg_files_per_group = total_files / group_count if group_count > 0 else 0
        max_files_in_group = max((len(group.files) for group in groups), default=0)

        # Calculate complexity score (1-10 scale)
        complexity = 1.0

        # Factor 1: Number of groups
        complexity += min(3.0, group_count / 2)

        # Factor 2: Average files per group
        complexity += min(3.0, avg_files_per_group / 5)

        # Factor 3: Maximum files in any group
        complexity += min(3.0, max_files_in_group / 10)

        # Normalize to 1-10 scale
        complexity = min(10.0, max(1.0, complexity))

        return round(complexity, 1)

    def _generate_branch_name(self, title: str) -> str:
        """Generate a Git branch name from a PR title."""
        # Simple conversion: lowercase, replace spaces with hyphens, remove special chars
        branch_name = title.lower().replace(" ", "-")
        branch_name = re.sub(r'[^\w\-]', '', branch_name)
        return f"feature/{branch_name}"

    def _generate_pr_description(self, group: Dict[str, Any], file_changes: List[Any]) -> str:
        """Generate a PR description for a group."""
        title = group.get("title", "Untitled PR")
        rationale = group.get("rationale", "")
        files = group.get("files", [])

        description = f"## {title}\n\n"
        if rationale:
            description += f"{rationale}\n\n"

        description += "## Files Changed\n\n"
        for file_path in sorted(files)[:10]:
            description += f"- `{file_path}`\n"

        if len(files) > 10:
            description += f"- ... and {len(files) - 10} more file(s)\n"

        return description