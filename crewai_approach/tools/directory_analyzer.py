"""
Directory analyzer tool for analyzing directory structure patterns.
"""
from typing import List, Dict, Any
from pathlib import Path
import math
import os
import json

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from .base_tools import BaseRepoTool

logger = get_logger(__name__)

class DirectoryAnalyzerInput(BaseModel):
    """Input schema for the DirectoryAnalyzer tool."""
    repo_path: str = Field(..., description="Path to the git repository (required by BaseRepoTool)")
    repository_analysis: Dict[str, Any] = Field(..., description="Repository analysis data")

class DirectoryAnalyzer(BaseRepoTool):
    """
    Tool for analyzing directory structure from repository analysis data.
    Identifies organizational patterns, hierarchy, complexity, and potential feature groupings.
    """

    name: str = "Directory Analyzer"
    description: str = (
        "Analyzes directory structure from RepositoryAnalysis data to identify organizational patterns, "
        "hierarchy, complexity, and potential feature groupings."
    )
    args_schema: type[BaseModel] = DirectoryAnalyzerInput

    def _run(self, **kwargs) -> str:
        """
        Analyze directory structure to identify organizational patterns.

        Args:
            **kwargs: Expects 'repo_path' and 'repository_analysis' based on args_schema.

        Returns:
            JSON string containing directory analysis information
        """
        repo_path = kwargs.get("repo_path")
        repository_analysis = kwargs.get("repository_analysis")

        try:
            logger.info(f"Running Directory Analyzer Tool on {repo_path}")

            if not repository_analysis:
                logger.error("Missing 'repository_analysis' in arguments for Directory Analyzer.")
                return json.dumps({
                    "directory_count": 0,
                    "max_depth": 0,
                    "avg_files_per_directory": 0.0,
                    "directory_complexity": [],
                    "parent_child_relationships": [],
                    "potential_feature_directories": [],
                    "error": "Missing repository_analysis data"
                }, indent=2)

            # Access data from the repository_analysis dictionary
            directory_summaries = repository_analysis.get("directory_summaries", [])

            if not directory_summaries:
                # Return an empty result if no summaries
                logger.warning("No directory summaries found in repository_analysis. Returning empty result.")
                return json.dumps({
                    "directory_count": 0,
                    "max_depth": 0,
                    "avg_files_per_directory": 0.0,
                    "directory_complexity": [],
                    "parent_child_relationships": [],
                    "potential_feature_directories": []
                }, indent=2)

            # Extract paths (filter out None or empty paths)
            directories = []
            for d in directory_summaries:
                dir_path = d.get("path", "")
                if dir_path and dir_path != '.':
                    directories.append(Path(dir_path))

            # Calculate directory hierarchy
            hierarchy = self._calculate_hierarchy(directories)

            # Calculate directory complexity
            directory_complexity_list = []
            for dir_summary in directory_summaries:
                dir_path = dir_summary.get("path", "")
                if not dir_path:
                    continue  # Skip if path is missing
                
                # Extract data from directory summary
                file_count = dir_summary.get("file_count", 0)
                extensions = dir_summary.get("extensions", {})
                total_changes = dir_summary.get("total_changes", 0)
                
                # Simple complexity heuristic
                complexity_score = min(10, (
                    file_count * 0.3 +
                    len(extensions) * 1.0 +
                    math.log10(max(1, total_changes)) * 2.0  # Avoid log(0)
                ))
                
                directory_complexity_list.append({
                    "path": dir_path,
                    "file_count": file_count,
                    "changed_file_count": file_count,
                    "extension_counts": extensions,
                    "estimated_complexity": round(complexity_score, 2)
                })

            # Calculate directory relatedness
            relatedness_matrix = self._calculate_relatedness(directory_summaries)

            # Identify potential feature directories
            potential_features = self._identify_potential_features(directory_summaries, relatedness_matrix)

            # Calculate avg files per changed directory
            total_files = repository_analysis.get("total_files_changed", 0)
            num_changed_dirs = len([ds for ds in directory_summaries if ds.get("path", "")])
            avg_files = 0.0
            if total_files > 0 and num_changed_dirs > 0:
                avg_files = round(total_files / num_changed_dirs, 2)

            # Construct the result
            result = {
                "directory_count": num_changed_dirs,
                "max_depth": hierarchy.get("max_depth", 0),
                "avg_files_per_directory": avg_files,
                "directory_complexity": directory_complexity_list,
                "parent_child_relationships": hierarchy.get("relationships", []),
                "potential_feature_directories": potential_features
            }

            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Error analyzing directory structure: {str(e)}"
            logger.error(error_msg)
            
            # Return a serialized error response
            error_result = {
                "directory_count": 0,
                "max_depth": 0,
                "avg_files_per_directory": 0.0,
                "directory_complexity": [],
                "parent_child_relationships": [],
                "potential_feature_directories": [],
                "error": error_msg
            }
            
            return json.dumps(error_result, indent=2)

    def _calculate_hierarchy(self, directories: List[Path]) -> Dict[str, Any]:
        """
        Calculate directory hierarchy information.

        Args:
            directories: List of Path objects representing directories with changes

        Returns:
            Dictionary with hierarchy information
        """
        if not directories:
            return {"max_depth": 0, "avg_files_per_dir": 0.0, "relationships": []}

        max_depth = 0
        dir_strings = set()
        for d in directories:
            # Handle Path objects correctly
            path_str = str(d)
            if path_str == "(root)":  # Check for special root marker
                dir_strings.add("(root)")
            elif path_str == '.':  # Treat '.' as root
                dir_strings.add("(root)")
            else:
                dir_strings.add(path_str)
                # Calculate depth based on parts
                max_depth = max(max_depth, len(d.parts))

        relationships = []
        for dir_str in dir_strings:
            if dir_str == "(root)":
                continue
            current_path = Path(dir_str)
            # Handle cases where parent is '.' (root) or higher up
            parent_path = current_path.parent
            parent_str = str(parent_path) if str(parent_path) != '.' else "(root)"

            if parent_str in dir_strings:
                relationships.append({
                    "parent": parent_str,
                    "child": dir_str
                })

        # Placeholder - calculation moved to _run
        avg_files_per_dir = 0.0

        return {
            "max_depth": max_depth,
            "avg_files_per_dir": avg_files_per_dir,
            "relationships": relationships
        }

    def _calculate_relatedness(self, directory_summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate relatedness between directories based on common file extensions.

        Args:
            directory_summaries: List of directory summary dictionaries

        Returns:
            Dictionary mapping directory pairs to relatedness scores
        """
        relatedness = {}

        for i, dir1 in enumerate(directory_summaries):
            dir1_path = dir1.get("path", "")
            if not dir1_path:
                continue
                
            dir1_extensions = set(dir1.get("extensions", {}).keys())
            relatedness[dir1_path] = {}

            for j, dir2 in enumerate(directory_summaries):
                if i == j:
                    continue
                    
                dir2_path = dir2.get("path", "")
                if not dir2_path:
                    continue
                    
                dir2_extensions = set(dir2.get("extensions", {}).keys())

                if not dir1_extensions or not dir2_extensions:
                    relatedness[dir1_path][dir2_path] = 0.0
                    continue

                intersection = len(dir1_extensions.intersection(dir2_extensions))
                union = len(dir1_extensions.union(dir2_extensions))
                similarity = intersection / union if union > 0 else 0.0
                relatedness[dir1_path][dir2_path] = similarity

        return relatedness

    def _identify_potential_features(self, directory_summaries: List[Dict[str, Any]],
                              relatedness_matrix: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Identify potential feature directories based on file types and relatedness.

        Args:
            directory_summaries: List of directory summary dictionaries
            relatedness_matrix: Directory relatedness matrix

        Returns:
            List of potential feature directories as dictionaries
        """
        potential_features = []

        for dir_summary in directory_summaries:
            dir_path = dir_summary.get("path", "")
            if not dir_path or dir_path == "(root)":
                continue

            extensions = dir_summary.get("extensions", {})
            is_diverse = len(extensions) >= 3
            related_dirs = []

            if dir_path in relatedness_matrix:
                for other_dir, similarity in relatedness_matrix[dir_path].items():
                    if similarity > 0.5:
                        related_dirs.append(other_dir)

            is_cross_cutting = len(related_dirs) >= 2

            if is_diverse or is_cross_cutting:
                potential_features.append({
                    "directory": dir_path,
                    "is_diverse": is_diverse,
                    "is_cross_cutting": is_cross_cutting,
                    "file_types": list(extensions.keys()),
                    "related_directories": related_dirs,
                    "confidence": 0.7 if is_diverse and is_cross_cutting else 0.5
                })

        return potential_features