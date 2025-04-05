"""
File grouper tool for organizing files into logical PR groups.
"""
from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path
import re

from pydantic import BaseModel, Field

from shared.utils.logging_utils import get_logger
from models.agent_models import GroupingStrategyType, PRGroup, PRGroupingStrategy
from shared.models.analysis_models import RepositoryAnalysis
from models.agent_models import PatternAnalysisResult, RepositoryMetrics
from .base_tools import BaseRepoTool

logger = get_logger(__name__)


class FileGrouperInput(BaseModel):
    """Input schema for the FileGrouper tool."""
    repo_path: str = Field(..., description="Path to the git repository")
    strategy: GroupingStrategyType = Field(..., description="The grouping strategy to use")
    repository_analysis: RepositoryAnalysis = Field(..., description="Repository analysis data")
    pattern_analysis: PatternAnalysisResult = Field(..., description="Pattern analysis results")
    repository_metrics: RepositoryMetrics = Field(..., description="Repository-wide metrics")


class FileGrouperTool(BaseRepoTool):
    """Tool for grouping files into logical PR suggestions."""

    name: str = "File Grouper"
    description: str = "Groups files into logical PR suggestions based on selected strategies"
    args_schema: type[BaseModel] = FileGrouperInput

    def _run(self, repo_path: str, strategy: GroupingStrategyType, repository_analysis: RepositoryAnalysis, pattern_analysis: PatternAnalysisResult, repository_metrics: RepositoryMetrics, **kwargs) -> PRGroupingStrategy:
        """
        Group files into logical PR suggestions based on the selected strategy.

        Args:
            repo_path: Path to the git repository
            strategy: The grouping strategy to use
            repository_analysis: Repository analysis data
            pattern_analysis: Pattern analysis data
            repository_metrics: Metrics for what the repo analysis is.

        Returns:
            PRGroupingStrategy with logical PR groups
        """
        logger.info(f"Grouping files using strategy: {strategy}")

        file_changes = repository_analysis.file_changes

        # Extract file paths
        file_paths = [str(fc.path) for fc in file_changes]

        if not file_paths:
            return PRGroupingStrategy(
                strategy_type=strategy,
                groups=[],
                explanation="No file changes to group",
                ungrouped_files=[]
            )

        # Choose grouping function based on strategy type
        if strategy == GroupingStrategyType.DIRECTORY_BASED:
            groups = self._group_by_directory(file_paths, file_changes, repository_analysis.directory_summaries)
        elif strategy == GroupingStrategyType.FEATURE_BASED:
            groups = self._group_by_feature(file_paths, file_changes, pattern_analysis)
        elif strategy == GroupingStrategyType.MODULE_BASED:
            groups = self._group_by_module(file_paths, file_changes)
        elif strategy == GroupingStrategyType.SIZE_BALANCED:
            groups = self._group_by_size(file_paths, file_changes)
        elif strategy == GroupingStrategyType.MIXED:
            groups = self._group_mixed(file_paths, file_changes, pattern_analysis, repository_analysis.directory_summaries)
        else:
            # Default to directory-based
            groups = self._group_by_directory(file_paths, file_changes, repository_analysis.directory_summaries)

        # Create PR groups
        pr_groups = []
        for group in groups:
            if not group["files"]:
                continue

            pr_groups.append(PRGroup(
                title=group["title"],
                files=group["files"],
                rationale=group["rationale"],
                estimated_size=len(group["files"]),
                directory_focus=group.get("directory_focus"),
                feature_focus=group.get("feature_focus"),
                suggested_branch_name=self._generate_branch_name(group["title"]),
                suggested_pr_description=self._generate_pr_description(group, file_changes)
            ))

        # Check for any files that weren't grouped
        all_grouped_files = set()
        for group in pr_groups:
            all_grouped_files.update(group.files)

        ungrouped = [f for f in file_paths if f not in all_grouped_files]

        # Create a strategy result
        strategy_result = PRGroupingStrategy(
            strategy_type=strategy,
            groups=pr_groups,
            explanation=self._generate_strategy_explanation(strategy, pr_groups),
            estimated_review_complexity=self._estimate_review_complexity(pr_groups),
            ungrouped_files=ungrouped
        )

        return strategy_result

    def _group_by_directory(self,
                            file_paths: List[str],
                            file_changes: List[Any],
                            directory_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group files by directory."""
        groups = []
        dir_to_files = defaultdict(list)

        for file_path in file_paths:
            # Determine directory and add to corresponding list
            directory = Path(file_path).parent if Path(file_path).parent != Path('.') else Path('(root)')
            dir_to_files[str(directory)].append(file_path)

        for directory, files in dir_to_files.items():
            if files:
                directory_focus = directory
                groups.append({
                    "title": f"Changes in {directory}",
                    "files": files,
                    "rationale": f"These changes are focused within the '{directory}' directory.",
                    "directory_focus": directory_focus
                })

        return groups

    def _group_by_feature(self,
                          file_paths: List[str],
                          file_changes: List[Any],
                          pattern_analysis: PatternAnalysisResult) -> List[Dict[str, Any]]:
        """Group files by inferred feature."""
        groups = []
        # Combine patterns and relationship info
        all_patterns = pattern_analysis.naming_patterns + pattern_analysis.similar_names + \
                       pattern_analysis.common_patterns.common_prefixes + pattern_analysis.common_patterns.common_suffixes

        for pattern in all_patterns:
            if hasattr(pattern, 'matches'):
                files = pattern.matches
                if files:
                    groups.append({
                        "title": f"Related to {pattern.type}",
                        "files": files,
                        "rationale": f"These changes are related to {pattern.type} as identified by file naming patterns."
                    })

        return groups
    
    def _group_by_module(self,
                         file_paths: List[str],
                         file_changes: List[Any]) -> List[Dict[str, Any]]:
        """Group files by module based on file type."""
        module_groups = defaultdict(list)

        for file_path in file_paths:
            extension = Path(file_path).suffix.lower()
            module_groups[extension].append(file_path)
        
        groups = []
        for module, files in module_groups.items():
            if files:
                groups.append({
                    "title": f"{module.title()} Module Changes",
                    "files": files,
                    "rationale": f"These changes are focused within the {module} module (based on file types)."
                })
        
        return groups

    def _group_by_size(self,
                       file_paths: List[str],
                       file_changes: List[Any]) -> List[Dict[str, Any]]:
        """Group files to create balanced PRs."""
        # For simplicity, just create two groups. In real scenario, use more sophisticated balancing logic.
        groups = []
        group1 = file_paths[:len(file_paths) // 2]
        group2 = file_paths[len(file_paths) // 2:]

        if group1:
            groups.append({
                "title": "Part 1 - Size Balanced Changes",
                "files": group1,
                "rationale": "This group includes the first half of changes to balance PR size."
            })
        if group2:
            groups.append({
                "title": "Part 2 - Size Balanced Changes",
                "files": group2,
                "rationale": "This group includes the second half of changes to balance PR size."
            })

        return groups

    def _group_mixed(self,
                        file_paths: List[str],
                        file_changes: List[Any],
                        pattern_analysis: PatternAnalysisResult,
                        directory_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Mixed approach combining directory structure and feature grouping."""
            # First, group by directory
            directory_groups = self._group_by_directory(file_paths, file_changes, directory_summaries)
            # Then, group by feature to refine the groupings
            feature_groups = self._group_by_feature(file_paths, file_changes, pattern_analysis)
            # Combine these two approaches (e.g., merge feature-based groupings into directory-based groupings)

            combined_groups = directory_groups + feature_groups

            return combined_groups

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