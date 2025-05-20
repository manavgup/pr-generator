"""
Directory analyzer tool for analyzing directory structure patterns.
"""
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
import math
import json

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from crewai_approach.models.agent_models import DirectoryAnalysisResult, DirectoryComplexity, ParentChildRelation, PotentialFeatureDirectory
from .base_tool import BaseRepoTool

logger = get_logger(__name__)

class DirectoryAnalyzerSchema(BaseModel):
    """Input schema for the DirectoryAnalyzer tool using primitive types."""
    repository_analysis_json: str = Field(..., description="JSON string serialization of the RepositoryAnalysis object.")

class DirectoryAnalyzer(BaseRepoTool):
    """
    Tool for analyzing directory structure from repository analysis data.
    Identifies organizational patterns, hierarchy, complexity, and potential feature groupings.
    """

    name: str = "Directory Analyzer"
    description: str = (
        "Analyzes directory structure from RepositoryAnalysis JSON data to identify organizational patterns, "
        "hierarchy, complexity, and potential feature groupings. Returns a DirectoryAnalysisResult JSON."
    )
    args_schema: Type[BaseModel] = DirectoryAnalyzerSchema

    def _run(self, repository_analysis_json: str) -> str:
        """
        Analyze directory structure to identify organizational patterns.

        Args:
            repository_analysis_json: JSON string of RepositoryAnalysis data.

        Returns:
            JSON string containing directory analysis information (DirectoryAnalysisResult).
        """
        # Echo received inputs for debugging
        logger.info(f"DirectoryAnalyzer received repository_analysis_json: {repository_analysis_json[:100]}...")
        
        try:
            logger.info("Running Directory Analyzer Tool...")
            
            # Validate the input JSON
            if not self._validate_json_string(repository_analysis_json):
                raise ValueError("Invalid repository_analysis_json provided")
            
            # Extract directory summaries
            directory_summaries = self._extract_directory_summaries(repository_analysis_json)
            
            # Extract repository info
            repo_info = self._extract_repository_info(repository_analysis_json)
            total_files_changed = repo_info.get("total_files_changed", 0)
            
            if not directory_summaries:
                logger.warning("No directory summaries found in repository_analysis. Returning empty result.")
                # Return a valid empty result using the DirectoryAnalysisResult model
                empty_result = DirectoryAnalysisResult(
                    directory_count=0, max_depth=0, avg_files_per_directory=0.0,
                    directory_complexity=[], parent_child_relationships=[], 
                    potential_feature_directories=[]
                )
                return empty_result.model_dump_json(indent=2)

            # Extract directory paths
            directory_paths = [ds.get("path", "") for ds in directory_summaries if ds.get("path")]
            
            # Extract directory strings
            directory_strings = []
            for dir_summary in directory_summaries:
                dir_path_str = dir_summary.get("path", "")
                if dir_path_str and dir_path_str != '.':  # Filter out empty or '.' which means root
                    directory_strings.append(dir_path_str)
                elif dir_path_str == '.' or dir_path_str == '(root)':  # Handle explicit root markers
                    directory_strings.append("(root)")

            # Convert to Path objects for hierarchy analysis
            directories_paths = [Path(d) for d in directory_strings if d != "(root)"]
            
            # Calculate directory hierarchy
            hierarchy_result = self._calculate_hierarchy(directories_paths, set(directory_strings))

            # Calculate directory complexity
            directory_complexity_results = []
            for dir_summary in directory_summaries:
                dir_path = dir_summary.get("path", "")
                if not dir_path:
                    continue

                file_count = dir_summary.get("file_count", 0)
                extensions = dir_summary.get("extensions", {})
                total_changes = dir_summary.get("total_changes", 0)

                # Simple complexity heuristic
                complexity_score = min(10.0, (
                    (file_count * 0.3) +
                    (len(extensions) * 1.0) +
                    (math.log10(max(1, total_changes)) * 2.0)  # Avoid log(0)
                ))

                directory_complexity_results.append(DirectoryComplexity(
                    path=dir_path,
                    file_count=file_count,
                    changed_file_count=file_count,  # Mapping directly from summary file_count
                    extension_counts=extensions,
                    estimated_complexity=round(complexity_score, 2)
                ))

            # Calculate directory relatedness
            relatedness_matrix = self._calculate_relatedness(directory_summaries)

            # Identify potential feature directories
            potential_features = self._identify_potential_features(directory_summaries, relatedness_matrix)

            # Calculate avg files per changed directory
            num_changed_dirs = len({ds.get("path", "") for ds in directory_summaries if ds.get("path", "")})  # Count unique dirs
            avg_files = 0.0
            if total_files_changed > 0 and num_changed_dirs > 0:
                avg_files = round(total_files_changed / num_changed_dirs, 2)

            # Construct the final result using the DirectoryAnalysisResult model
            result = DirectoryAnalysisResult(
                directory_count=num_changed_dirs,
                max_depth=hierarchy_result["max_depth"],
                avg_files_per_directory=avg_files,
                directory_complexity=directory_complexity_results,
                parent_child_relationships=hierarchy_result["relationships"],
                potential_feature_directories=potential_features
            )

            logger.info("Directory analysis complete.")
            return result.model_dump_json(indent=2)

        except ValidationError as ve:
            error_msg = f"Pydantic validation error during directory analysis: {str(ve)}"
            logger.error(error_msg, exc_info=True)
            error_result = DirectoryAnalysisResult(
                directory_count=0, max_depth=0, avg_files_per_directory=0.0,
                directory_complexity=[], parent_child_relationships=[],
                potential_feature_directories=[], error=error_msg
            )
            return error_result.model_dump_json(indent=2)
        except json.JSONDecodeError as je:
            error_msg = f"Failed to decode input repository_analysis_json: {str(je)}"
            logger.error(error_msg, exc_info=True)
            error_result = DirectoryAnalysisResult(
                 directory_count=0, max_depth=0, avg_files_per_directory=0.0,
                 directory_complexity=[], parent_child_relationships=[],
                 potential_feature_directories=[], error=error_msg
             )
            return error_result.model_dump_json(indent=2)
        except Exception as e:
            error_msg = f"Unexpected error analyzing directory structure: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_result = DirectoryAnalysisResult(
                 directory_count=0, max_depth=0, avg_files_per_directory=0.0,
                 directory_complexity=[], parent_child_relationships=[],
                 potential_feature_directories=[], error=error_msg
             )
            return error_result.model_dump_json(indent=2)

    # --- Helper Methods (Updated for dictionary data instead of model objects) ---

    def _calculate_hierarchy(self, directories: List[Path], dir_strings_set: set) -> Dict[str, Any]:
        """
        Calculate directory hierarchy information.

        Args:
            directories: List of Path objects representing changed directories (excluding root).
            dir_strings_set: Set of all directory path strings including '(root)'.

        Returns:
            Dictionary with max_depth and relationships list.
        """
        if not directories and "(root)" not in dir_strings_set:
            return {"max_depth": 0, "relationships": []}

        max_depth = 0
        for d_path in directories:
            # Calculate depth based on parts for non-root paths
             max_depth = max(max_depth, len(d_path.parts))

        relationships_list: List[ParentChildRelation] = []
        processed_children = set()  # Avoid duplicate relationships if structure is deep

        for dir_str in dir_strings_set:
            if dir_str == "(root)" or dir_str in processed_children:
                continue

            current_path = Path(dir_str)
            parent_path = current_path.parent

            # Determine parent string representation
            parent_str = str(parent_path)
            if parent_str == '.':
                parent_str = "(root)"

            # Check if the calculated parent exists in the set of changed directories
            if parent_str in dir_strings_set:
                 relationships_list.append(ParentChildRelation(
                    parent=parent_str,
                    child=dir_str
                ))
                 processed_children.add(dir_str)

        return {
            "max_depth": max_depth,
            "relationships": relationships_list
        }

    def _calculate_relatedness(self, directory_summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate relatedness between directories based on common file extensions.

        Args:
            directory_summaries: List of DirectorySummary dictionaries.

        Returns:
            Dictionary mapping directory pairs to relatedness scores (Jaccard Index).
        """
        relatedness: Dict[str, Dict[str, float]] = {}
        dir_paths = [ds.get("path", "") for ds in directory_summaries if ds.get("path", "")]  # Get valid paths

        for i, dir1 in enumerate(directory_summaries):
            dir1_path = dir1.get("path", "")
            if not dir1_path:
                continue

            # Extensions should be a dict {ext: count}, get keys for set operations
            dir1_extensions = set(dir1.get("extensions", {}).keys())
            relatedness[dir1_path] = {}  # Initialize inner dict

            for j, dir2 in enumerate(directory_summaries):
                if i == j:
                    continue

                dir2_path = dir2.get("path", "")
                if not dir2_path:
                    continue

                dir2_extensions = set(dir2.get("extensions", {}).keys())

                if not dir1_extensions or not dir2_extensions:
                    # If one directory has no typed files, similarity is 0
                    similarity = 0.0
                else:
                    intersection = len(dir1_extensions.intersection(dir2_extensions))
                    union = len(dir1_extensions.union(dir2_extensions))
                    # Jaccard Index
                    similarity = intersection / union if union > 0 else 0.0

                relatedness[dir1_path][dir2_path] = round(similarity, 3)  # Store rounded score

        return relatedness

    def _identify_potential_features(self, directory_summaries: List[Dict[str, Any]],
                                     relatedness_matrix: Dict[str, Dict[str, float]]) -> List[PotentialFeatureDirectory]:
        """
        Identify potential feature directories based on file types and relatedness.

        Args:
            directory_summaries: List of DirectorySummary dictionaries.
            relatedness_matrix: Directory relatedness matrix.

        Returns:
            List of PotentialFeatureDirectory objects.
        """
        potential_features_list: List[PotentialFeatureDirectory] = []
        diversity_threshold = 3  # Min number of extensions to be considered diverse
        relatedness_threshold = 0.5  # Min similarity score to be considered related
        cross_cutting_threshold = 2  # Min number of related directories

        for dir_summary in directory_summaries:
            dir_path = dir_summary.get("path", "")
            if not dir_path or dir_path == "(root)":
                continue  # Skip root or invalid paths

            extensions = dir_summary.get("extensions", {})
            file_types = list(extensions.keys())

            is_diverse = len(file_types) >= diversity_threshold

            related_dirs: List[str] = []
            if dir_path in relatedness_matrix:
                for other_dir, similarity in relatedness_matrix[dir_path].items():
                    if similarity >= relatedness_threshold:
                        related_dirs.append(other_dir)

            is_cross_cutting = len(related_dirs) >= cross_cutting_threshold

            # Identify as potential feature if diverse or significantly cross-cutting
            if is_diverse or is_cross_cutting:
                # Simple confidence heuristic
                confidence = 0.5
                if is_diverse:
                    confidence += 0.2
                if is_cross_cutting:
                    confidence += 0.2
                if is_diverse and is_cross_cutting:
                    confidence += 0.1  # Bonus

                potential_features_list.append(PotentialFeatureDirectory(
                    directory=dir_path,
                    is_diverse=is_diverse,
                    is_cross_cutting=is_cross_cutting,
                    file_types=file_types,
                    related_directories=related_dirs,
                    confidence=min(1.0, round(confidence, 2))  # Cap at 1.0
                ))

        return potential_features_list
