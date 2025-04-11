# --- START OF FILE directory_analyzer_tool.py ---

"""
Directory analyzer tool for analyzing directory structure patterns.
"""
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
import math
import json

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
# Assuming RepositoryAnalysis and DirectorySummary are correctly defined here
from shared.models.analysis_models import RepositoryAnalysis, DirectorySummary

from .base_tool import BaseRepoTool

logger = get_logger(__name__)

# --- Pydantic Models for Output Structure ---

class DirectoryComplexityInfo(BaseModel):
    """Detailed complexity info for a single directory."""
    path: str = Field(description="Path to the directory.")
    file_count: int = Field(description="Total number of files changed within this directory (recursive).")
    # changed_file_count seems redundant if file_count already means changed files in summary
    # Kept original field name for now, maps directly from DirectorySummary.file_count
    changed_file_count: int = Field(description="Number of files directly changed in this directory (from summary).")
    extension_counts: Dict[str, int] = Field(description="Count of changed files per extension in this directory.")
    estimated_complexity: float = Field(description="Calculated complexity score (heuristic).")

class DirectoryRelationship(BaseModel):
    """Represents a parent-child relationship between changed directories."""
    parent: str = Field(description="Path of the parent directory.")
    child: str = Field(description="Path of the child directory.")

class PotentialFeatureDirectory(BaseModel):
    """Information about a directory potentially representing a feature."""
    directory: str = Field(description="Path to the potential feature directory.")
    is_diverse: bool = Field(description="Indicates if the directory contains diverse file types (>= 3).")
    is_cross_cutting: bool = Field(description="Indicates if the directory is related to multiple other directories.")
    file_types: List[str] = Field(description="List of file extensions found in this directory.")
    related_directories: List[str] = Field(description="List of other directories with high relatedness score (> 0.5).")
    confidence: float = Field(description="Confidence score that this directory represents a feature.")

class DirectoryAnalysisResult(BaseModel):
    """Output model for the Directory Analyzer tool."""
    directory_count: int = Field(description="Total number of distinct directories with changes.")
    max_depth: int = Field(description="Maximum depth of any changed directory.")
    avg_files_per_directory: float = Field(description="Average number of changed files per changed directory.")
    directory_complexity: List[DirectoryComplexityInfo] = Field(description="List of complexity details for each changed directory.")
    parent_child_relationships: List[DirectoryRelationship] = Field(description="List of identified parent-child relationships among changed directories.")
    potential_feature_directories: List[PotentialFeatureDirectory] = Field(description="List of directories identified as potentially representing features.")
    error: Optional[str] = Field(None, description="Error message if analysis failed.")


class DirectoryAnalyzerSchema(BaseModel):
    """Input schema for the DirectoryAnalyzer tool."""
    # Input is the JSON string from RepoAnalyzerTool
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
        try:
            logger.info("Running Directory Analyzer Tool...")
            # Deserialize the input JSON string to a Pydantic object
            repository_analysis = RepositoryAnalysis.model_validate_json(repository_analysis_json)

            # --- Use attribute access on the Pydantic object ---
            directory_summaries_list = repository_analysis.directory_summaries if repository_analysis.directory_summaries else []
            total_files_changed = repository_analysis.total_files_changed

            if not directory_summaries_list:
                logger.warning("No directory summaries found in repository_analysis. Returning empty result.")
                # Return a valid empty result using the Pydantic model
                empty_result = DirectoryAnalysisResult(
                    directory_count=0, max_depth=0, avg_files_per_directory=0.0,
                    directory_complexity=[], parent_child_relationships=[], potential_feature_directories=[]
                )
                return empty_result.model_dump_json(indent=2)

            # Extract paths correctly
            directories_paths: List[Path] = []
            directory_strings: List[str] = [] # Keep track of original string paths
            for dir_summary in directory_summaries_list:
                 # Use attribute access
                 dir_path_str = dir_summary.path
                 if dir_path_str and dir_path_str != '.': # Filter out empty or '.' which means root
                     directories_paths.append(Path(dir_path_str))
                     directory_strings.append(dir_path_str)
                 elif dir_path_str == '.' or dir_path_str == '(root)': # Handle explicit root markers
                      directory_strings.append("(root)")


            # Calculate directory hierarchy
            hierarchy_result = self._calculate_hierarchy(directories_paths, set(directory_strings))

            # Calculate directory complexity
            directory_complexity_results: List[DirectoryComplexityInfo] = []
            for dir_summary in directory_summaries_list:
                # Use attribute access
                dir_path = dir_summary.path
                if not dir_path: continue # Skip summaries without a path

                file_count = dir_summary.file_count
                # extensions should be Dict[str, int] based on RepositoryAnalysis model
                extensions = dir_summary.extensions if dir_summary.extensions else {}
                total_changes = dir_summary.total_changes

                # Simple complexity heuristic
                complexity_score = min(10.0, (
                    (file_count * 0.3) +
                    (len(extensions) * 1.0) +
                    (math.log10(max(1, total_changes)) * 2.0) # Avoid log(0)
                ))

                directory_complexity_results.append(DirectoryComplexityInfo(
                    path=dir_path,
                    file_count=file_count,
                    changed_file_count=file_count, # Mapping directly from summary file_count
                    extension_counts=extensions,
                    estimated_complexity=round(complexity_score, 2)
                ))

            # Calculate directory relatedness
            relatedness_matrix = self._calculate_relatedness(directory_summaries_list)

            # Identify potential feature directories
            potential_features = self._identify_potential_features(directory_summaries_list, relatedness_matrix)

            # Calculate avg files per changed directory
            num_changed_dirs = len({ds.path for ds in directory_summaries_list if ds.path}) # Count unique dirs
            avg_files = 0.0
            if total_files_changed > 0 and num_changed_dirs > 0:
                avg_files = round(total_files_changed / num_changed_dirs, 2)

            # Construct the final result using the Pydantic model
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

    # --- Helper Methods (Updated with Attribute Access and Type Hints) ---

    def _calculate_hierarchy(self, directories: List[Path], dir_strings_set: set[str]) -> Dict[str, Any]:
        """
        Calculate directory hierarchy information.

        Args:
            directories: List of Path objects representing changed directories (excluding root).
            dir_strings_set: Set of all directory path strings including '(root)'.

        Returns:
            Dictionary with max_depth and relationships list.
        """
        if not directories and "(root)" not in dir_strings_set :
            return {"max_depth": 0, "relationships": []}

        max_depth = 0
        for d_path in directories:
            # Calculate depth based on parts for non-root paths
             max_depth = max(max_depth, len(d_path.parts))

        relationships_list: List[DirectoryRelationship] = []
        processed_children = set() # Avoid duplicate relationships if structure is deep

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
                 relationships_list.append(DirectoryRelationship(
                    parent=parent_str,
                    child=dir_str
                ))
                 processed_children.add(dir_str)

        return {
            "max_depth": max_depth,
            "relationships": relationships_list
        }

    def _calculate_relatedness(self, directory_summaries: List[DirectorySummary]) -> Dict[str, Dict[str, float]]:
        """
        Calculate relatedness between directories based on common file extensions.

        Args:
            directory_summaries: List of DirectorySummary objects.

        Returns:
            Dictionary mapping directory pairs to relatedness scores (Jaccard Index).
        """
        relatedness: Dict[str, Dict[str, float]] = {}
        dir_paths = [ds.path for ds in directory_summaries if ds.path] # Get valid paths

        for i, dir1 in enumerate(directory_summaries):
            # Use attribute access
            dir1_path = dir1.path
            if not dir1_path: continue

            # Extensions should be a dict {ext: count}, get keys for set operations
            dir1_extensions = set(dir1.extensions.keys()) if dir1.extensions else set()
            relatedness[dir1_path] = {} # Initialize inner dict

            for j, dir2 in enumerate(directory_summaries):
                if i == j: continue

                dir2_path = dir2.path
                if not dir2_path: continue

                dir2_extensions = set(dir2.extensions.keys()) if dir2.extensions else set()

                if not dir1_extensions or not dir2_extensions:
                    # If one directory has no typed files, similarity is 0
                    similarity = 0.0
                else:
                    intersection = len(dir1_extensions.intersection(dir2_extensions))
                    union = len(dir1_extensions.union(dir2_extensions))
                    # Jaccard Index
                    similarity = intersection / union if union > 0 else 0.0

                relatedness[dir1_path][dir2_path] = round(similarity, 3) # Store rounded score

        return relatedness

    def _identify_potential_features(self, directory_summaries: List[DirectorySummary],
                                     relatedness_matrix: Dict[str, Dict[str, float]]) -> List[PotentialFeatureDirectory]:
        """
        Identify potential feature directories based on file types and relatedness.

        Args:
            directory_summaries: List of DirectorySummary objects.
            relatedness_matrix: Directory relatedness matrix.

        Returns:
            List of PotentialFeatureDirectory objects.
        """
        potential_features_list: List[PotentialFeatureDirectory] = []
        diversity_threshold = 3 # Min number of extensions to be considered diverse
        relatedness_threshold = 0.5 # Min similarity score to be considered related
        cross_cutting_threshold = 2 # Min number of related directories

        for dir_summary in directory_summaries:
            # Use attribute access
            dir_path = dir_summary.path
            if not dir_path or dir_path == "(root)": continue # Skip root or invalid paths

            extensions = dir_summary.extensions if dir_summary.extensions else {}
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
                if is_diverse: confidence += 0.2
                if is_cross_cutting: confidence += 0.2
                if is_diverse and is_cross_cutting: confidence += 0.1 # Bonus

                potential_features_list.append(PotentialFeatureDirectory(
                    directory=dir_path,
                    is_diverse=is_diverse,
                    is_cross_cutting=is_cross_cutting,
                    file_types=file_types,
                    related_directories=related_dirs,
                    confidence=min(1.0, round(confidence, 2)) # Cap at 1.0
                ))

        return potential_features_list

# --- END OF FILE directory_analyzer_tool.py ---