"""
Repository metrics calculator for PR recommendation.
"""
from typing import Dict, List, Any
import json
import math

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from shared.models.analysis_models import RepositoryAnalysis
from models.agent_models import RepositoryMetrics
from .base_tools import BaseRepoTool

logger = get_logger(__name__)


class RepositoryMetricsInput(BaseModel):
    """Input schema for Repository Metrics Calculator Tool."""
    repository_analysis: Dict[str, Any] = Field(
        ..., 
        description="Repository analysis data"
    )


class RepositoryMetricsCalculator(BaseRepoTool):
    """
    Tool for calculating objective metrics about a repository's changes.
    Focuses on extracting and calculating metrics without making recommendations.
    """

    name: str = "Repository Metrics Calculator"
    description: str = """
    Calculates objective metrics about a repository's changes.
    Provides insights into directory structure, file distribution, and change patterns.
    Returns factual information without making strategy recommendations.
    """
    args_schema: type[BaseModel] = RepositoryMetricsInput

    def _run(self, repository_analysis: Dict[str, Any], **kwargs) -> str:
        """
        Calculate objective metrics from repository analysis data.

        Args:
            repository_analysis: Repository analysis data as dictionary
            **kwargs: Additional arguments (ignored)

        Returns:
            JSON string containing calculated metrics
        """
        repo_path = repository_analysis.get("repo_path", "unknown")
        logger.info(f"Calculating repository metrics for {repo_path}")

        try:
            # Extract data from repository_analysis
            total_files_changed = repository_analysis.get("total_files_changed", 0)
            total_lines_changed = repository_analysis.get("total_lines_changed", 0)
            directory_summaries = repository_analysis.get("directory_summaries", [])
            file_changes = repository_analysis.get("file_changes", [])
            
            # Calculate extension summary if not provided directly
            extensions_summary = repository_analysis.get("extensions_summary", {})
            if not extensions_summary:
                extensions_summary = self._extract_extensions_summary(file_changes)
            
            # Calculate directory metrics
            directory_metrics = self._calculate_directory_metrics(
                directory_summaries=directory_summaries,
                total_files=total_files_changed
            )

            # Calculate file type metrics
            file_type_metrics = self._calculate_file_type_metrics(
                extensions_summary=extensions_summary
            )

            # Calculate change distribution metrics
            change_metrics = self._calculate_change_metrics(
                file_changes=file_changes
            )

            # Calculate complexity indicators (objective factors only)
            complexity_indicators = self._calculate_complexity_indicators(
                directory_summaries=directory_summaries,
                total_files=total_files_changed,
                total_lines=total_lines_changed,
                file_changes=file_changes
            )

            # Construct the result
            result = {
                "repo_path": str(repo_path),
                "total_files_changed": total_files_changed,
                "total_lines_changed": total_lines_changed,
                "directory_metrics": directory_metrics,
                "file_type_metrics": file_type_metrics,
                "change_metrics": change_metrics,
                "complexity_indicators": complexity_indicators
            }

            # Return serialized JSON
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"Error calculating repository metrics: {str(e)}"
            logger.error(error_msg)
            
            # Return a serialized error response with minimal valid structure
            error_result = {
                "repo_path": str(repo_path),
                "total_files_changed": 0,
                "total_lines_changed": 0,
                "directory_metrics": {},
                "file_type_metrics": {},
                "change_metrics": {},
                "complexity_indicators": [],
                "error": error_msg
            }
            
            return json.dumps(error_result, indent=2)
            
    def _extract_extensions_summary(self, file_changes: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Extract extensions summary from file changes.
        
        Args:
            file_changes: List of file change dictionaries
            
        Returns:
            Dictionary mapping extensions to counts
        """
        extensions = {}
        for change in file_changes:
            ext = change.get("extension", "none")
            extensions[ext] = extensions.get(ext, 0) + 1
        return extensions

    def _calculate_directory_metrics(self, directory_summaries: List[Dict[str, Any]], total_files: int) -> Dict[str, Any]:
        """
        Calculate metrics related to directory structure.

        Args:
            directory_summaries: List of directory summary dictionaries
            total_files: Total number of files changed

        Returns:
            Directory metrics dictionary
        """
        if not directory_summaries:
            return {
                "directory_count": 0,
                "max_files_per_directory": 0,
                "avg_files_per_directory": 0,
                "directory_concentration": 0,
                "directory_depth": {"max": 0, "avg": 0},
                "directories_with_multiple_file_types": 0
            }

        # Calculate basic metrics
        directory_count = len(directory_summaries)
        file_counts = [d.get("file_count", 0) for d in directory_summaries]
        max_files = max(file_counts) if file_counts else 0
        avg_files = sum(file_counts) / len(file_counts) if file_counts else 0

        # Calculate directory concentration
        concentration = max_files / total_files if total_files > 0 else 0

        # Calculate directory depths
        depths = []
        for dir_summary in directory_summaries:
            path = dir_summary.get("path", "")
            if path == "(root)":
                depth = 0
            else:
                depth = path.count('/') + 1  # Count separators plus 1
            depths.append(depth)

        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0

        # Count directories with multiple file types
        multi_type_dirs = 0
        for dir_summary in directory_summaries:
            extensions = dir_summary.get("extensions", {})
            if len(extensions) > 1:
                multi_type_dirs += 1

        return {
            "directory_count": directory_count,
            "max_files_per_directory": max_files,
            "avg_files_per_directory": round(avg_files, 2),
            "directory_concentration": round(concentration, 2),
            "directory_depth": {"max": max_depth, "avg": round(avg_depth, 2)},
            "directories_with_multiple_file_types": multi_type_dirs
        }

    def _calculate_file_type_metrics(self, extensions_summary: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate metrics related to file types.

        Args:
            extensions_summary: Dictionary mapping extensions to counts

        Returns:
            File type metrics dictionary
        """
        if not extensions_summary:
            return {
                "file_type_count": 0,
                "primary_file_type": None,
                "primary_file_type_percentage": 0,
                "file_type_distribution": {}
            }

        # Calculate file type metrics
        file_type_count = len(extensions_summary)
        total_files = sum(extensions_summary.values())

        # Find primary file type
        primary_file_type = max(extensions_summary.items(), key=lambda x: x[1])[0] if extensions_summary else None
        primary_count = max(extensions_summary.values()) if extensions_summary else 0
        primary_percentage = (primary_count / total_files * 100) if total_files > 0 else 0

        # Calculate percentage distribution
        distribution = {}
        for ext, count in extensions_summary.items():
            distribution[ext] = round((count / total_files * 100), 1) if total_files > 0 else 0

        return {
            "file_type_count": file_type_count,
            "primary_file_type": primary_file_type,
            "primary_file_type_percentage": round(primary_percentage, 1),
            "file_type_distribution": distribution
        }

    def _calculate_change_metrics(self, file_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics related to change patterns.

        Args:
            file_changes: List of file change dictionaries

        Returns:
            Change metrics dictionary
        """
        if not file_changes:
            return {
                "avg_lines_per_file": 0,
                "max_lines_changed": 0,
                "files_with_large_changes": 0,
                "files_with_small_changes": 0,
                "change_size_distribution": {"large": 0, "medium": 0, "small": 0}
            }

        # Calculate lines changed per file
        lines_per_file = []
        large_changes = 0
        small_changes = 0
        large_threshold = 100  # Arbitrary threshold for "large" changes
        small_threshold = 10   # Arbitrary threshold for "small" changes

        for change in file_changes:
            # Get total changes, handling possible missing fields
            if "changes" in change and change["changes"]:
                changes_obj = change["changes"]
                if isinstance(changes_obj, dict):
                    total_lines = changes_obj.get("added", 0) + changes_obj.get("deleted", 0)
                else:
                    # Handle case where changes is not a dictionary
                    total_lines = 0
            else:
                # Get total_changes directly if present
                total_lines = change.get("total_changes", 0)

            lines_per_file.append(total_lines)

            if total_lines >= large_threshold:
                large_changes += 1
            elif total_lines <= small_threshold:
                small_changes += 1

        # Calculate basic metrics
        avg_lines = sum(lines_per_file) / len(lines_per_file) if lines_per_file else 0
        max_lines = max(lines_per_file) if lines_per_file else 0

        # Calculate change size distribution
        medium_changes = len(file_changes) - large_changes - small_changes

        return {
            "avg_lines_per_file": round(avg_lines, 1),
            "max_lines_changed": max_lines,
            "files_with_large_changes": large_changes,
            "files_with_small_changes": small_changes,
            "change_size_distribution": {
                "large": large_changes,
                "medium": medium_changes,
                "small": small_changes
            }
        }

    def _calculate_complexity_indicators(self, 
                                        directory_summaries: List[Dict[str, Any]],
                                        total_files: int,
                                        total_lines: int,
                                        file_changes: List[Dict[str, Any]]) -> List[str]:
        """
        Calculate objective complexity indicators.

        Args:
            directory_summaries: List of directory summary dictionaries
            total_files: Total number of files changed
            total_lines: Total number of lines changed
            file_changes: List of file change dictionaries

        Returns:
            List of complexity indicator strings
        """
        indicators = []

        # Indicator: Large number of files
        if total_files > 50:
            indicators.append("large_file_count")

        # Indicator: Large number of lines
        if total_lines > 1000:
            indicators.append("large_line_count")

        # Indicator: Many directories
        if len(directory_summaries) > 10:
            indicators.append("many_directories")

        # Calculate file counts by directory to check concentration
        dir_file_counts = {}
        for dir_summary in directory_summaries:
            dir_file_counts[dir_summary.get("path", "")] = dir_summary.get("file_count", 0)

        max_dir_files = max(dir_file_counts.values()) if dir_file_counts else 0

        # Indicator: High concentration in one directory
        if max_dir_files > 0 and total_files > 0:
            concentration = max_dir_files / total_files
            if concentration > 0.7:
                indicators.append("high_directory_concentration")
            elif concentration < 0.3 and len(directory_summaries) > 3:
                indicators.append("highly_distributed_changes")

        # Indicator: Many file types
        # Collect unique extensions from file changes
        extensions = set()
        for change in file_changes:
            ext = change.get("extension")
            if ext:
                extensions.add(ext)
                
        extension_count = len(extensions)
        if extension_count > 5:
            indicators.append("many_file_types")

        # Indicator: Cross-cutting changes (changes in many directories with different file types)
        if len(directory_summaries) > 3 and extension_count > 3:
            indicators.append("potential_cross_cutting_changes")

        return indicators