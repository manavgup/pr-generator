"""
File grouper tool for organizing files into logical PR groups.
"""
from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path
import re
import json

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from .base_tools import BaseRepoTool

logger = get_logger(__name__)

class FileGrouperInput(BaseModel):
    """Input schema for the FileGrouper tool."""
    repo_path: str = Field(..., description="Path to the git repository")
    strategy: str = Field(..., description="The grouping strategy to use")
    repository_analysis: Dict[str, Any] = Field(..., description="Repository analysis data")
    pattern_analysis: Dict[str, Any] = Field(..., description="Pattern analysis results")
    repository_metrics: Dict[str, Any] = Field(..., description="Repository-wide metrics")

class FileGrouperTool(BaseRepoTool):
    """Tool for grouping files into logical PR suggestions."""

    name: str = "File Grouper"
    description: str = "Groups files into logical PR suggestions based on selected strategies"
    args_schema: type[BaseModel] = FileGrouperInput

    def _run(self, repo_path: str, strategy: str, repository_analysis: Dict[str, Any], 
             pattern_analysis: Dict[str, Any], repository_metrics: Dict[str, Any], **kwargs) -> str:
        """
        Group files into logical PR suggestions based on the selected strategy.

        Args:
            repo_path: Path to the git repository
            strategy: The grouping strategy to use (string value)
            repository_analysis: Repository analysis data as dictionary
            pattern_analysis: Pattern analysis data as dictionary
            repository_metrics: Repository metrics as dictionary
            **kwargs: Additional arguments (ignored)

        Returns:
            JSON string containing PR grouping strategy
        """
        logger.info(f"Grouping files using strategy: {strategy}")

        try:
            # Extract file changes
            file_changes = repository_analysis.get("file_changes", [])
            
            # Extract file paths
            file_paths = []
            for fc in file_changes:
                file_path = fc.get("path", "")
                if file_path:
                    file_paths.append(file_path)

            if not file_paths:
                return json.dumps({
                    "strategy_type": strategy,
                    "groups": [],
                    "explanation": "No file changes to group",
                    "estimated_review_complexity": 1.0,
                    "ungrouped_files": []
                }, indent=2)

            # Choose grouping function based on strategy type
            if strategy == "directory_based":
                groups = self._group_by_directory(file_paths, file_changes, repository_analysis.get("directory_summaries", []))
            elif strategy == "feature_based":
                groups = self._group_by_feature(file_paths, file_changes, pattern_analysis)
            elif strategy == "module_based":
                groups = self._group_by_module(file_paths, file_changes)
            elif strategy == "size_balanced":
                groups = self._group_by_size(file_paths, file_changes)
            elif strategy == "mixed":
                groups = self._group_mixed(file_paths, file_changes, pattern_analysis, repository_analysis.get("directory_summaries", []))
            else:
                # Default to directory-based
                logger.warning(f"Unknown strategy '{strategy}', defaulting to directory-based")
                groups = self._group_by_directory(file_paths, file_changes, repository_analysis.get("directory_summaries", []))

            # Create PR groups
            pr_groups = []
            for group in groups:
                files = group.get("files", [])
                if not files:
                    continue

                pr_groups.append({
                    "title": group.get("title", "Untitled Group"),
                    "files": files,
                    "rationale": group.get("rationale", ""),
                    "estimated_size": len(files),
                    "directory_focus": group.get("directory_focus"),
                    "feature_focus": group.get("feature_focus"),
                    "suggested_branch_name": self._generate_branch_name(group.get("title", "untitled")),
                    "suggested_pr_description": self._generate_pr_description(group, file_changes)
                })

            # Check for any files that weren't grouped
            all_grouped_files = set()
            for group in pr_groups:
                all_grouped_files.update(group.get("files", []))

            ungrouped = [f for f in file_paths if f not in all_grouped_files]

            # Create a strategy result
            strategy_result = {
                "strategy_type": strategy,
                "groups": pr_groups,
                "explanation": self._generate_strategy_explanation(strategy, pr_groups),
                "estimated_review_complexity": self._estimate_review_complexity(pr_groups),
                "ungrouped_files": ungrouped
            }

            return json.dumps(strategy_result, indent=2)
            
        except Exception as e:
            error_msg = f"Error grouping files: {str(e)}"
            logger.error(error_msg)
            
            # Return a serialized error response
            error_result = {
                "strategy_type": strategy,
                "groups": [],
                "explanation": f"Error during file grouping: {str(e)}",
                "estimated_review_complexity": 1.0,
                "ungrouped_files": file_paths if 'file_paths' in locals() else [],
                "error": error_msg
            }
            
            return json.dumps(error_result, indent=2)

    def _group_by_directory(self, file_paths: List[str], file_changes: List[Dict[str, Any]], 
                           directory_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group files by directory."""
        groups = []
        dir_to_files = defaultdict(list)

        for file_path in file_paths:
            # Determine directory and add to corresponding list
            path = Path(file_path)
            directory = str(path.parent) if path.parent != Path('.') else "(root)"
            dir_to_files[directory].append(file_path)

        for directory, files in dir_to_files.items():
            if files:
                groups.append({
                    "title": f"Changes in {directory}",
                    "files": files,
                    "rationale": f"These changes are focused within the '{directory}' directory.",
                    "directory_focus": directory,
                    "feature_focus": None
                })

        return groups

    def _group_by_feature(self, file_paths: List[str], file_changes: List[Dict[str, Any]], 
                         pattern_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group files by inferred feature."""
        groups = []
        
        # Process naming patterns
        naming_patterns = pattern_analysis.get("naming_patterns", [])
        for pattern in naming_patterns:
            pattern_type = pattern.get("type")
            matches = pattern.get("matches", [])
            if pattern_type and matches:
                groups.append({
                    "title": f"Related to {pattern_type}",
                    "files": matches,
                    "rationale": f"These changes are related to {pattern_type} as identified by file naming patterns.",
                    "directory_focus": None,
                    "feature_focus": pattern_type
                })
                
        # Process similar names
        similar_names = pattern_analysis.get("similar_names", [])
        for similar_group in similar_names:
            base_pattern = similar_group.get("base_pattern")
            files = similar_group.get("files", [])
            if base_pattern and files:
                groups.append({
                    "title": f"Files related to {base_pattern}",
                    "files": files,
                    "rationale": f"These files share the common base pattern '{base_pattern}'.",
                    "directory_focus": None,
                    "feature_focus": base_pattern
                })
                
        # Process common patterns
        common_patterns = pattern_analysis.get("common_patterns", {})
        common_prefixes = common_patterns.get("common_prefixes", [])
        common_suffixes = common_patterns.get("common_suffixes", [])
        
        for prefix_group in common_prefixes:
            prefix = prefix_group.get("pattern_value")
            files = prefix_group.get("files", [])
            if prefix and files:
                groups.append({
                    "title": f"Files with prefix '{prefix}'",
                    "files": files,
                    "rationale": f"These files share the common prefix '{prefix}'.",
                    "directory_focus": None,
                    "feature_focus": f"prefix-{prefix}"
                })
                
        for suffix_group in common_suffixes:
            suffix = suffix_group.get("pattern_value")
            files = suffix_group.get("files", [])
            if suffix and files:
                groups.append({
                    "title": f"Files with suffix '{suffix}'",
                    "files": files,
                    "rationale": f"These files share the common suffix '{suffix}'.",
                    "directory_focus": None,
                    "feature_focus": f"suffix-{suffix}"
                })

        return groups
    
    def _group_by_module(self, file_paths: List[str], file_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group files by module based on file type."""
        module_groups = defaultdict(list)

        for file_path in file_paths:
            extension = Path(file_path).suffix.lower()
            # Use empty string extension if none exists
            if not extension:
                extension = "(no extension)"
            module_groups[extension].append(file_path)
        
        groups = []
        for module, files in module_groups.items():
            if files:
                module_display = module if module != "(no extension)" else "Files without extension"
                groups.append({
                    "title": f"{module_display} Module Changes",
                    "files": files,
                    "rationale": f"These changes are focused within the {module} module (based on file types).",
                    "directory_focus": None,
                    "feature_focus": f"module-{module}"
                })
        
        return groups

    def _group_by_size(self, file_paths: List[str], file_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group files to create balanced PRs."""
        # For simplicity, just create two groups. In real scenario, use more sophisticated balancing logic.
        groups = []
        group1 = file_paths[:len(file_paths) // 2]
        group2 = file_paths[len(file_paths) // 2:]

        if group1:
            groups.append({
                "title": "Part 1 - Size Balanced Changes",
                "files": group1,
                "rationale": "This group includes the first half of changes to balance PR size.",
                "directory_focus": None,
                "feature_focus": "size-balanced-1"
            })
        if group2:
            groups.append({
                "title": "Part 2 - Size Balanced Changes",
                "files": group2,
                "rationale": "This group includes the second half of changes to balance PR size.",
                "directory_focus": None,
                "feature_focus": "size-balanced-2"
            })

        return groups

    def _group_mixed(self, file_paths: List[str], file_changes: List[Dict[str, Any]], 
                    pattern_analysis: Dict[str, Any], directory_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mixed approach combining directory structure and feature grouping."""
        # First, group by directory
        directory_groups = self._group_by_directory(file_paths, file_changes, directory_summaries)
        # Then, group by feature to refine the groupings
        feature_groups = self._group_by_feature(file_paths, file_changes, pattern_analysis)
        
        # Combine these two approaches
        # For simplicity in this refactoring, we'll just return the combination of both
        combined_groups = directory_groups + feature_groups

        return combined_groups

    def _generate_branch_name(self, title: str) -> str:
        """Generate a Git branch name from a PR title."""
        # Simple conversion: lowercase, replace spaces with hyphens, remove special chars
        branch_name = title.lower().replace(" ", "-")
        branch_name = re.sub(r'[^\w\-]', '', branch_name)
        return f"feature/{branch_name}"

    def _generate_pr_description(self, group: Dict[str, Any], file_changes: List[Dict[str, Any]]) -> str:
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

    def _generate_strategy_explanation(self, strategy: str, groups: List[Dict[str, Any]]) -> str:
        """Generate an explanation for the grouping strategy."""
        group_count = len(groups)
        total_files = sum(len(group.get("files", [])) for group in groups)

        strategy_name = strategy.replace("_", " ").title()

        explanation = (
            f"Applied {strategy_name} grouping strategy to organize {total_files} files "
            f"into {group_count} logical pull requests. "
        )

        if strategy == "directory_based":
            explanation += "Files were grouped based on their directory structure to maintain cohesion."
        elif strategy == "feature_based":
            explanation += "Files were grouped based on inferred features and related functionality."
        elif strategy == "module_based":
            explanation += "Files were grouped based on their types and likely module boundaries."
        elif strategy == "size_balanced":
            explanation += "Files were grouped to create balanced, manageable pull requests."
        elif strategy == "mixed":
            explanation += "A mixed strategy was applied, combining directory structure and feature grouping."

        return explanation

    def _estimate_review_complexity(self, groups: List[Dict[str, Any]]) -> float:
        """Estimate the review complexity (1-10 scale) of the PR groups."""
        if not groups:
            return 1.0

        # Factors that increase complexity:
        # 1. Number of groups
        # 2. Average files per group
        # 3. Maximum files in any group
        # 4. Presence of ungrouped files

        group_count = len(groups)
        total_files = sum(len(group.get("files", [])) for group in groups)
        avg_files_per_group = total_files / group_count if group_count > 0 else 0
        max_files_in_group = max((len(group.get("files", [])) for group in groups), default=0)

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