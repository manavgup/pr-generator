"""
Repository metrics calculator for PR recommendation.
"""
from typing import Dict, List, Any, Type, Optional
import json
import math
from collections import defaultdict

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from crewai_approach.models.agent_models import RepositoryMetrics  # For result model validation
from .base_tool import BaseRepoTool

logger = get_logger(__name__)


class RepositoryMetricsSchema(BaseModel):
    """Input schema for Repository Metrics Calculator Tool using primitive types."""
    repository_analysis_json: str = Field(
        ...,
        description="JSON string serialization of the RepositoryAnalysis object."
    )


class RepositoryMetricsCalculator(BaseRepoTool):
    """
    Tool for calculating objective metrics about a repository's changes.
    Focuses on extracting and calculating metrics without making recommendations.
    """

    name: str = "Repository Metrics Calculator"
    description: str = """
    Calculates objective metrics about a repository's changes based on RepositoryAnalysis JSON.
    Provides insights into directory structure, file distribution, and change patterns.
    Returns factual information without making strategy recommendations.
    """
    args_schema: Type[BaseModel] = RepositoryMetricsSchema

    def _run(self, repository_analysis_json: str) -> str:
        """
        Calculate objective metrics from repository analysis JSON data.

        Args:
            repository_analysis_json: JSON string of RepositoryAnalysis data.

        Returns:
            JSON string containing calculated metrics (RepositoryMetrics object).
        """
        # Echo received inputs for debugging
        logger.info(f"RepositoryMetricsCalculator received repository_analysis_json: {repository_analysis_json[:100]}...")

        try:
            # Validate the input JSON
            if not repository_analysis_json or not isinstance(repository_analysis_json, str):
                raise ValueError("Invalid repository_analysis_json provided")

            # Extract basic repository info
            repo_info = self._extract_repository_info(repository_analysis_json)
            repo_path = repo_info.get("repo_path", self._repo_path)
            total_files_changed = repo_info.get("total_files_changed", 0)
            total_lines_changed = repo_info.get("total_lines_changed", 0)
            
            logger.info(f"Calculating repository metrics for {repo_path}")
            
            # Extract file metadata
            file_metadata = self._extract_file_metadata(repository_analysis_json)
            
            # Extract directory summaries
            directory_summaries = self._extract_directory_summaries(repository_analysis_json)
            
            # Extract file extensions
            extensions_summary = repo_info.get("extensions_summary", {})
            if not extensions_summary and file_metadata:
                # Calculate from file metadata if not provided
                extensions_summary = defaultdict(int)
                for file_info in file_metadata:
                    ext = file_info.get("extension") or "none"
                    extensions_summary[ext] += 1
                extensions_summary = dict(extensions_summary)

            # Calculate directory metrics
            directory_metrics = self._calculate_directory_metrics(
                directory_summaries=directory_summaries,
                total_files=total_files_changed
            )

            # Calculate file type metrics
            file_type_metrics = self._calculate_file_type_metrics(
                extensions_summary=extensions_summary,
                total_files=total_files_changed
            )

            # Calculate change distribution metrics
            change_metrics = self._calculate_change_metrics(
                file_metadata=file_metadata
            )

            # Calculate complexity indicators (objective factors only)
            complexity_indicators = self._calculate_complexity_indicators(
                directory_summaries=directory_summaries,
                total_files=total_files_changed,
                total_lines=total_lines_changed,
                file_metadata=file_metadata
            )

            # Construct the result using the RepositoryMetrics Pydantic model
            metrics_result = RepositoryMetrics(
                repo_path=repo_path,
                total_files_changed=total_files_changed,
                total_lines_changed=total_lines_changed,
                directory_metrics=directory_metrics,
                file_type_metrics=file_type_metrics,
                change_metrics=change_metrics,
                complexity_indicators=complexity_indicators
            )

            # Return serialized JSON using model_dump_json
            logger.info(f"Metrics calculation complete for {repo_path}")
            return metrics_result.model_dump_json(indent=2)

        except ValidationError as ve:
            error_msg = f"Pydantic validation error calculating metrics: {str(ve)}"
            logger.error(error_msg, exc_info=True)
            # Return error JSON
            error_data = {"error": error_msg, "repo_path": "unknown"}
            return json.dumps(error_data, indent=2)
        except json.JSONDecodeError as je:
            error_msg = f"Failed to decode input repository_analysis_json: {str(je)}"
            logger.error(error_msg, exc_info=True)
            error_data = {"error": error_msg, "repo_path": "unknown"}
            return json.dumps(error_data, indent=2)
        except Exception as e:
            error_msg = f"Unexpected error calculating repository metrics: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_data = {"error": error_msg, "repo_path": "unknown"}
            # Attempt to create a minimal error structure matching RepositoryMetrics if possible
            try:
                error_metrics = RepositoryMetrics(
                     repo_path="unknown",
                     total_files_changed=0, total_lines_changed=0,
                     directory_metrics={}, file_type_metrics={}, change_metrics={},
                     complexity_indicators=[], error=error_msg
                )
                return error_metrics.model_dump_json(indent=2)
            except:  # Fallback if error model fails
                 return json.dumps(error_data, indent=2)

    # --- Helper Methods (Refactored to use primitives) ---

    def _calculate_directory_metrics(self, directory_summaries: List[Dict[str, Any]], total_files: int) -> Dict[str, Any]:
        """Calculate metrics related to directory structure."""
        if not directory_summaries:
            # Return structure matching RepositoryMetrics.directory_metrics
            return {
                "directory_count": 0, "max_files_per_directory": 0,
                "avg_files_per_directory": 0.0, "directory_concentration": 0.0,
                "directory_depth": {"max": 0, "avg": 0.0},
                "directories_with_multiple_file_types": 0
            }

        directory_count = len(directory_summaries)
        
        # Extract file counts from summaries
        file_counts = [d.get("file_count", 0) for d in directory_summaries]
        max_files = max(file_counts) if file_counts else 0
        
        # Use actual total_files passed in for average calculation base
        avg_files = total_files / directory_count if directory_count > 0 else 0.0

        concentration = max_files / total_files if total_files > 0 else 0.0

        depths = []
        for dir_summary in directory_summaries:
            # Get depth from the summary if available, otherwise calculate it
            if "depth" in dir_summary:
                depths.append(dir_summary["depth"])
            else:
                path = dir_summary.get("path", "")
                if path == "(root)":
                    depths.append(0)
                else:
                    depths.append(path.count('/') + 1)

        max_depth = max(depths) if depths else 0
        avg_depth = sum(depths) / len(depths) if depths else 0.0

        multi_type_dirs = sum(1 for d in directory_summaries 
                            if len(d.get("extensions", {})) > 1)

        return {
            "directory_count": directory_count,
            "max_files_per_directory": max_files,
            "avg_files_per_directory": round(avg_files, 2),
            "directory_concentration": round(concentration, 2),
            "directory_depth": {"max": max_depth, "avg": round(avg_depth, 2)},
            "directories_with_multiple_file_types": multi_type_dirs
        }

    def _calculate_file_type_metrics(self, extensions_summary: Dict[str, int], total_files: int) -> Dict[str, Any]:
        """Calculate metrics related to file types."""
        if not extensions_summary or total_files == 0:
             # Return structure matching RepositoryMetrics.file_type_metrics
             return {
                 "file_type_count": 0, "primary_file_type": None,
                 "primary_file_type_percentage": 0.0, "file_type_distribution": {}
             }

        file_type_count = len(extensions_summary)

        primary_file_type = max(extensions_summary, key=extensions_summary.get, default=None)
        primary_count = extensions_summary.get(primary_file_type, 0)
        primary_percentage = (primary_count / total_files * 100) if total_files > 0 else 0.0

        distribution = {
            ext: round((count / total_files * 100), 1)
            for ext, count in extensions_summary.items()
        }

        return {
            "file_type_count": file_type_count,
            "primary_file_type": primary_file_type,
            "primary_file_type_percentage": round(primary_percentage, 1),
            "file_type_distribution": distribution
        }

    def _calculate_change_metrics(self, file_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics related to change patterns."""
        if not file_metadata:
            # Return structure matching RepositoryMetrics.change_metrics
             return {
                 "avg_lines_per_file": 0.0, "max_lines_changed": 0,
                 "files_with_large_changes": 0, "files_with_small_changes": 0,
                 "change_size_distribution": {"large": 0, "medium": 0, "small": 0}
             }

        lines_per_file = []
        large_changes = 0
        small_changes = 0
        large_threshold = 100
        small_threshold = 10

        for file_info in file_metadata:
            # Get total_changes from the metadata
            total_lines = file_info.get("total_changes", 0)
            if not total_lines and ("added_lines" in file_info or "deleted_lines" in file_info):
                total_lines = file_info.get("added_lines", 0) + file_info.get("deleted_lines", 0)
                
            lines_per_file.append(total_lines)

            if total_lines >= large_threshold:
                large_changes += 1
            elif total_lines <= small_threshold:
                small_changes += 1

        avg_lines = sum(lines_per_file) / len(lines_per_file) if lines_per_file else 0.0
        max_lines = max(lines_per_file) if lines_per_file else 0
        medium_changes = len(file_metadata) - large_changes - small_changes

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
                                        file_metadata: List[Dict[str, Any]]) -> List[str]:
        """Calculate objective complexity indicators."""
        indicators = []
        if total_files > 50: indicators.append("large_file_count")
        if total_lines > 1000: indicators.append("large_line_count")
        if len(directory_summaries) > 10: indicators.append("many_directories")

        # Use metrics calculated earlier if possible, or recalculate concisely
        dir_metrics = self._calculate_directory_metrics(directory_summaries, total_files)
        if dir_metrics["directory_concentration"] > 0.7:
            indicators.append("high_directory_concentration")
        elif dir_metrics["directory_concentration"] < 0.3 and dir_metrics["directory_count"] > 3:
            indicators.append("highly_distributed_changes")

        # Count unique extensions in file metadata
        extensions = set()
        for file_info in file_metadata:
            ext = file_info.get("extension")
            if ext:
                extensions.add(ext)
                
        extension_count = len(extensions)
        if extension_count > 5: indicators.append("many_file_types")

        if dir_metrics["directory_count"] > 3 and extension_count > 3:
            indicators.append("potential_cross_cutting_changes")

        return sorted(list(set(indicators)))  # Ensure unique and sorted
