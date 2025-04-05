"""
Directory analyzer tool for analyzing directory structure patterns.
"""
from typing import List, Dict, Any
from pathlib import Path
import math
import os # Import os for path operations if needed

# Corrected crewAI import (BaseTool is not needed directly)
# Inherit from our specific BaseRepoTool
from .base_tools import BaseRepoTool
from .repo_analyzer import RepositoryAnalysis
# Import Pydantic for input schema
from pydantic import BaseModel, Field

from shared.utils.logging_utils import get_logger
# Import the specific models needed
from models.agent_models import DirectoryComplexity, DirectoryAnalysisResult, ParentChildRelation, PotentialFeatureDirectory
from shared.models.analysis_models import RepositoryAnalysis # Import RepositoryAnalysis

logger = get_logger(__name__)

# Define the input schema for the tool
# It MUST include repo_path for BaseRepoTool's _before_run
class DirectoryAnalyzerInput(BaseModel):
    """Input schema for the DirectoryAnalyzer tool."""
    repo_path: str = Field(..., description="Path to the git repository (required by BaseRepoTool)")
    repository_analysis: RepositoryAnalysis = Field(..., description="RepositoryAnalysis object containing repository analysis data")

# Inherit from BaseRepoTool
class DirectoryAnalyzer(BaseRepoTool):
    """
    Tool for analyzing directory structure from RepositoryAnalysis data.
    Identifies organizational patterns, hierarchy, complexity, and potential feature groupings.
    Outputs a structured DirectoryAnalysisResult.
    """

    name: str = "Directory Analyzer"
    description: str = (
        "Analyzes directory structure from RepositoryAnalysis data to identify organizational patterns, "
        "hierarchy, complexity, and potential feature groupings. Outputs a structured DirectoryAnalysisResult."
    )
    # Define the input schema
    args_schema: type[BaseModel] = DirectoryAnalyzerInput

    # _run now accepts **kwargs as defined in BaseRepoTool
    def _run(self, **kwargs) -> DirectoryAnalysisResult:
        """
        Analyze directory structure to identify organizational patterns.

        Args:
            **kwargs: Expects 'repo_path' and 'repository_analysis' based on args_schema.

        Returns:
            DirectoryAnalysisResult object with directory analysis information.
        """
        repo_path = kwargs.get("repo_path")
        repository_analysis = kwargs.get("repository_analysis")

        if not repository_analysis:
             logger.error("Missing 'repository_analysis' in arguments for Directory Analyzer.")
             # Return an empty or error state appropriate for your design
             # For now, returning an empty result:
             return DirectoryAnalysisResult(
                 directory_count=0, max_depth=0, avg_files_per_directory=0.0,
                 directory_complexity=[], parent_child_relationships=[],
                 potential_feature_directories=[]
             )

        # Ensure repository_analysis is the correct type if needed (crewAI might pass a dict)
        if isinstance(repository_analysis, dict):
             try:
                 repository_analysis = RepositoryAnalysis(**repository_analysis)
             except Exception as e:
                 logger.error(f"Failed to parse 'repository_analysis' dictionary into Pydantic model: {e}")
                 # Handle error appropriately
                 raise ValueError("Invalid repository_analysis structure provided.") from e

        logger.info(f"Running Directory Analyzer Tool on {repo_path}") # repo_path available from kwargs

        # Access data directly from the typed input object
        directory_summaries = repository_analysis.directory_summaries

        if not directory_summaries:
             # Return an empty result if no summaries
             logger.warning("No directory summaries found in repository_analysis. Returning empty result.")
             return DirectoryAnalysisResult(
                 directory_count=0,
                 max_depth=0,
                 avg_files_per_directory=0.0,
                 directory_complexity=[],
                 parent_child_relationships=[],
                 potential_feature_directories=[]
             )

        # Extract paths using list comprehension
        # Ensure paths are treated correctly relative to the repo root if needed
        # Using the path strings directly from DirectorySummary is usually sufficient
        # Filter out potential None or empty paths just in case
        directories = [Path(d.path) for d in directory_summaries if d.path and d.path != '.']

        # Calculate directory hierarchy
        hierarchy = self._calculate_hierarchy(directories)

        # Calculate directory complexity
        directory_complexity_list: List[DirectoryComplexity] = []
        for dir_summary in directory_summaries:
            if not dir_summary.path: continue # Skip if path is missing
            # Simple complexity heuristic
            complexity_score = min(10, (
                dir_summary.file_count * 0.3 +
                len(dir_summary.extensions) * 1.0 +
                math.log10(max(1, dir_summary.total_changes)) * 2.0 # Avoid log(0)
            ))
            directory_complexity_list.append(DirectoryComplexity(
                path=dir_summary.path,
                file_count=dir_summary.file_count,
                changed_file_count=dir_summary.file_count,
                extension_counts=dir_summary.extensions,
                estimated_complexity=round(complexity_score, 2)
            ))

        # Calculate directory relatedness
        # Pass the original list of DirectorySummary objects
        relatedness_matrix = self._calculate_relatedness(directory_summaries)

        # Identify potential feature directories
        potential_features = self._identify_potential_features(directory_summaries, relatedness_matrix)

        # Calculate avg files per changed directory
        avg_files = 0.0
        num_changed_dirs = len([ds for ds in directory_summaries if ds.path]) # Count dirs with paths
        if repository_analysis.total_files_changed > 0 and num_changed_dirs > 0:
             avg_files = round(repository_analysis.total_files_changed / num_changed_dirs, 2)

        # Construct and return the structured result object
        return DirectoryAnalysisResult(
            directory_count=num_changed_dirs,
            max_depth=hierarchy.get("max_depth", 0),
            avg_files_per_directory=avg_files,
            directory_complexity=directory_complexity_list,
            parent_child_relationships=[ParentChildRelation(**rel) for rel in hierarchy.get("relationships", [])],
            potential_feature_directories=[PotentialFeatureDirectory(**feat) for feat in potential_features]
        )

    # --- Helper methods remain the same ---
    # _calculate_hierarchy, _calculate_relatedness, _identify_potential_features
    # Note: Ensure these helpers correctly handle DirectorySummary objects if passed directly

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
            if path_str == "(root)": # Check for special root marker
                dir_strings.add("(root)")
            elif path_str == '.': # Treat '.' as root
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
                relationships.append({"parent": parent_str, "child": dir_str})

        # Placeholder - calculation moved to _run
        avg_files_per_dir = 0.0

        return {
            "max_depth": max_depth,
            "avg_files_per_dir": avg_files_per_dir,
            "relationships": relationships
        }

    # Type hint List[Any] for now, ensure it receives DirectorySummary objects
    def _calculate_relatedness(self, directory_summaries: List[Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate relatedness between directories based on common file extensions.

        Args:
            directory_summaries: List of DirectorySummary objects

        Returns:
            Dictionary mapping directory pairs to relatedness scores
        """
        relatedness = {}
        summaries = directory_summaries

        for i, dir1 in enumerate(summaries):
            if not hasattr(dir1, 'path') or not dir1.path: continue
            dir1_path = dir1.path
            dir1_extensions = set(getattr(dir1, 'extensions', {}).keys())

            relatedness[dir1_path] = {}

            for j, dir2 in enumerate(summaries):
                if i == j: continue
                if not hasattr(dir2, 'path') or not dir2.path: continue

                dir2_path = dir2.path
                dir2_extensions = set(getattr(dir2, 'extensions', {}).keys())

                if not dir1_extensions or not dir2_extensions:
                    relatedness[dir1_path][dir2_path] = 0.0
                    continue

                intersection = len(dir1_extensions.intersection(dir2_extensions))
                union = len(dir1_extensions.union(dir2_extensions))
                similarity = intersection / union if union > 0 else 0.0
                relatedness[dir1_path][dir2_path] = similarity

        return relatedness

    # Type hint List[Any] for now, ensure it receives DirectorySummary objects
    def _identify_potential_features(self,
                               directory_summaries: List[Any],
                               relatedness_matrix: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Identify potential feature directories based on file types and relatedness.

        Args:
            directory_summaries: List of DirectorySummary objects
            relatedness_matrix: Directory relatedness matrix

        Returns:
            List of potential feature directories as dictionaries
        """
        potential_features = []
        summaries = directory_summaries

        for dir_summary in summaries:
            if not hasattr(dir_summary, 'path') or not dir_summary.path or dir_summary.path == "(root)":
                continue

            dir_path = dir_summary.path
            extensions = getattr(dir_summary, 'extensions', {})
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