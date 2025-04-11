# --- START OF FILE pattern_analyzer_tool.py ---

"""
Pattern analyzer tool for identifying patterns in file changes.
"""
from typing import List, Dict, Any, Optional, Set, Type # Added Type
import re
import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
# Assuming RepositoryAnalysis and FileChange are correctly defined here
from shared.models.analysis_models import RepositoryAnalysis, FileChange
from shared.utils.logging_utils import get_logger
from .base_tool import BaseRepoTool

logger = get_logger(__name__)

class PatternAnalyzerToolSchema(BaseModel):
    """Input schema for the PatternAnalyzer tool."""
    repository_analysis_json: str = Field(..., description="JSON string serialization of the RepositoryAnalysis object.")

# Placeholder definition for the structure returned by the tool.
# In a real scenario, this would likely be imported from models.agent_models
class PatternAnalysisResultStructure(BaseModel):
    naming_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    similar_names: List[Dict[str, Any]] = Field(default_factory=list)
    common_patterns: Dict[str, List[Dict[str, Any]]] = Field(default_factory=lambda: {"common_prefixes": [], "common_suffixes": []})
    related_files: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_summary: str = ""
    confidence: float = 0.0
    error: Optional[str] = None


class PatternAnalyzerTool(BaseRepoTool):
    """Tool for identifying patterns in file changes to detect related modifications."""

    name: str = "Pattern Analyzer"
    description: str = "Identifies patterns in file changes based on repository analysis JSON. Returns PatternAnalysisResultStructure JSON."
    args_schema: Type[BaseModel] = PatternAnalyzerToolSchema

    def _run(self, repository_analysis_json: str) -> str:
        """
        Analyze patterns in file changes to detect related modifications.

        Args:
            repository_analysis_json: JSON string of RepositoryAnalysis data.

        Returns:
            JSON string containing pattern analysis information (PatternAnalysisResultStructure).
        """
        repo_path = self.git_ops.repo_path if hasattr(self, 'git_ops') and self.git_ops else "unknown"
        logger.info(f"Analyzing file change patterns for {repo_path}")

        try:
            # Deserialize input JSON to Pydantic object
            repository_analysis = RepositoryAnalysis.model_validate_json(repository_analysis_json)

            # --- Corrected Data Extraction ---
            # Access attribute directly from the Pydantic object
            file_changes_list: List[FileChange] = repository_analysis.file_changes if repository_analysis.file_changes else []

            if not file_changes_list:
                 logger.warning(f"No file changes found in analysis for {repo_path}. Returning empty pattern result.")
                 result = PatternAnalysisResultStructure(analysis_summary="No file changes found.")
                 return result.model_dump_json(indent=2)

            # Extract file names and paths using attribute access
            file_names: List[str] = []
            file_paths: List[str] = []
            for fc in file_changes_list:
                # Use the path attribute from the FileChange object
                file_path_str = fc.path
                if file_path_str: # Ensure path exists
                    file_paths.append(file_path_str)
                    # Derive filename from the path attribute
                    filename = Path(file_path_str).name
                    file_names.append(filename)
                else:
                    logger.warning("Found a FileChange object with no path.")
            # --- End Corrected Data Extraction ---


            # Analyze file naming patterns (using file_names list)
            naming_patterns = self._analyze_naming_patterns(file_names)

            # Find files with similar names (using file_names and file_paths lists)
            similar_names = self._find_similar_names(file_names, file_paths)

            # Analyze common prefixes/suffixes (using file_names and file_paths lists)
            common_patterns = self._analyze_common_patterns(file_names, file_paths)

            # Detect file pairs that often change together (using the list of FileChange objects)
            related_files = self._detect_related_files(file_changes_list)

            # Construct the result using the placeholder structure
            # Note: If an actual PatternAnalysisResult model exists, use that instead.
            result = PatternAnalysisResultStructure(
                naming_patterns=naming_patterns,
                similar_names=similar_names,
                common_patterns={
                    "common_prefixes": common_patterns["common_prefixes"],
                    "common_suffixes": common_patterns["common_suffixes"]
                },
                related_files=related_files,
                analysis_summary="Pattern analysis completed successfully.",
                confidence=0.8 # Placeholder confidence
            )

            logger.info(f"Pattern analysis complete for {repo_path}.")
            return result.model_dump_json(indent=2)

        except ValidationError as ve:
            error_msg = f"Pydantic validation error during pattern analysis for {repo_path}: {str(ve)}"
            logger.error(error_msg, exc_info=True)
            error_result = PatternAnalysisResultStructure(analysis_summary=error_msg, confidence=0.0, error=error_msg)
            return error_result.model_dump_json(indent=2)
        except json.JSONDecodeError as je:
            error_msg = f"Failed to decode input repository_analysis_json for {repo_path}: {str(je)}"
            logger.error(error_msg, exc_info=True)
            error_result = PatternAnalysisResultStructure(analysis_summary=error_msg, confidence=0.0, error=error_msg)
            return error_result.model_dump_json(indent=2)
        except Exception as e:
            # Catch potential AttributeErrors if FileChange model is missing expected fields
            error_msg = f"Unexpected error analyzing file patterns for {repo_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_result = PatternAnalysisResultStructure(analysis_summary=error_msg, confidence=0.0, error=error_msg)
            return error_result.model_dump_json(indent=2)


    # --- Helper Methods (Input types updated, internal logic checked) ---

    def _analyze_naming_patterns(self, file_names: List[str]) -> List[Dict[str, Any]]:
        """Analyze naming patterns in files."""
        # This method already works with a list of strings, no changes needed internally.
        patterns = []
        test_pattern = re.compile(r'^test_(.+)\.py$')
        test_files = [name for name in file_names if test_pattern.match(name)]
        if test_files:
            patterns.append({
                "pattern": "test_*.py",
                "matches": test_files,
                "type": "test_files",
                "description": "Python test files"
            })

        interface_pattern = re.compile(r'^I([A-Z].+)\.(?:py|java|ts|cs)$')
        interface_files = [name for name in file_names if interface_pattern.match(name)]
        if interface_files:
            patterns.append({
                "pattern": "I*.{py,java,ts,cs}",
                "matches": interface_files,
                "type": "interface_files",
                "description": "Interface files with I prefix"
            })

        component_pattern = re.compile(r'^(.+)Component\.(?:tsx|jsx|js|ts)$')
        component_files = [name for name in file_names if component_pattern.match(name)]
        if component_files:
            patterns.append({
                "pattern": "*Component.{tsx,jsx,js,ts}",
                "matches": component_files,
                "type": "component_files",
                "description": "UI component files"
            })

        model_pattern = re.compile(r'^(.+)Model\.(?:py|java|ts|cs)$')
        model_files = [name for name in file_names if model_pattern.match(name)]
        if model_files:
            patterns.append({
                "pattern": "*Model.{py,java,ts,cs}",
                "matches": model_files,
                "type": "model_files",
                "description": "Model files"
            })
        return patterns

    def _find_similar_names(self, file_names: List[str], file_paths: List[str]) -> List[Dict[str, Any]]:
        """Find files with similar names."""
        # This method works with lists of strings, internal logic is okay.
        # Ensure name_to_path mapping is correct.
        if len(file_names) != len(file_paths):
             logger.error(f"Mismatch between file_names ({len(file_names)}) and file_paths ({len(file_paths)}) lengths.")
             return [] # Cannot reliably map

        similar_groups = []
        processed = set()
        name_to_path = dict(zip(file_names, file_paths))

        for i, name1 in enumerate(file_names):
            if name1 in processed: continue
            base_name1 = Path(name1).stem # Use stem to get name without final suffix

            similar_found = []
            for j, name2 in enumerate(file_names):
                if i == j or name2 in processed: continue
                base_name2 = Path(name2).stem

                # Check if base names are similar enough (adjust logic if needed)
                # Original logic: one contained in the other. Stem matching might be better.
                # if base_name1 in base_name2 or base_name2 in base_name1:
                # Let's use a slightly different check: exact stem match or test_ prefix match
                is_similar = (base_name1 == base_name2 or
                              base_name1 == f"test_{base_name2}" or
                              base_name2 == f"test_{base_name1}")

                if is_similar:
                    similar_found.append(name2)

            if similar_found:
                # Group includes the current file (name1) and similar files found
                current_group_names = [name1] + similar_found
                # Get paths for all names in the group
                current_group_paths = [name_to_path[name] for name in current_group_names if name in name_to_path]

                similar_groups.append({
                    "base_pattern": base_name1, # Use base name of the first file as pattern key
                    "files": current_group_paths
                })
                processed.update(current_group_names) # Mark all in the group as processed

        return similar_groups

    def _analyze_common_patterns(self, file_names: List[str], file_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze common prefixes and suffixes in file names."""
        # This method works with lists of strings, internal logic is okay.
        # Ensure name_to_path mapping is correct.
        if len(file_names) != len(file_paths):
             logger.error(f"Mismatch between file_names ({len(file_names)}) and file_paths ({len(file_paths)}) lengths.")
             return {"common_prefixes": [], "common_suffixes": []}

        prefixes = defaultdict(list)
        suffixes = defaultdict(list)
        name_to_path = dict(zip(file_names, file_paths))

        for name in file_names:
            # Extract prefix (alphanumeric before first underscore or dot)
            prefix_match = re.match(r'^([a-zA-Z0-9]+)[_.]', name)
            if prefix_match:
                prefix = prefix_match.group(1)
                if name in name_to_path: prefixes[prefix].append(name_to_path[name])

            # Extract suffix (alphanumeric after last underscore, before extension)
            # Requires at least one underscore before the potential suffix
            suffix_match = re.match(r'^.+_([a-zA-Z0-9]+)\.[^.]+$', name)
            if suffix_match:
                suffix = suffix_match.group(1)
                if name in name_to_path: suffixes[suffix].append(name_to_path[name])

        # Filter to keep only prefixes/suffixes associated with more than one file
        common_prefixes_dict = {prefix: files for prefix, files in prefixes.items() if len(files) > 1}
        common_suffixes_dict = {suffix: files for suffix, files in suffixes.items() if len(files) > 1}

        # Convert to the list-of-dictionaries format expected by the output structure
        common_prefix_groups = [
            {"pattern_type": "prefix", "pattern_value": p, "files": f}
            for p, f in common_prefixes_dict.items()
        ]
        common_suffix_groups = [
            {"pattern_type": "suffix", "pattern_value": s, "files": f}
            for s, f in common_suffixes_dict.items()
        ]

        return {
            "common_prefixes": common_prefix_groups,
            "common_suffixes": common_suffix_groups
        }

    def _detect_related_files(self, file_changes_list: List[FileChange]) -> List[Dict[str, Any]]:
        """Detect files that are likely related based on heuristics."""
        # This method now receives List[FileChange]. Access attributes correctly.
        related_groups = []

        # Group by directory (already correct using attribute access internally if FileChange model has 'directory')
        # dir_to_files = defaultdict(list)
        # for fc in file_changes_list:
        #     directory = fc.directory # Assumes FileChange has directory attr
        #     if fc.path: dir_to_files[directory].append(fc.path)

        # --- Look for implementation/test pairs ---
        impl_test_pairs = []
        processed_impl = set() # Avoid duplicating pairs if both impl and test trigger match

        for fc in file_changes_list:
            if not fc.path or fc.path in processed_impl: continue

            # Use Path object for easier manipulation
            current_path_obj = Path(fc.path)
            filename = current_path_obj.name
            file_stem = current_path_obj.stem # Name without final suffix
            file_suffix = current_path_obj.suffix # e.g., ".py"

            # Potential implementation file pattern (simple check)
            if not filename.startswith("test_") and file_suffix in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cs']:
                # Look for corresponding test file
                expected_test_stem = f"test_{file_stem}"
                for other_fc in file_changes_list:
                    if not other_fc.path: continue
                    other_path_obj = Path(other_fc.path)
                    # Check stem and suffix match, and ensure they are in same/related directory (optional check)
                    if other_path_obj.stem == expected_test_stem and other_path_obj.suffix == file_suffix:
                        # Basic check passed, add pair
                         impl_test_pairs.append({
                            "file1": fc.path,
                            "file2": other_fc.path,
                            "relation_type": "implementation_test",
                            "base_name": file_stem
                        })
                         processed_impl.add(fc.path) # Mark impl as processed
                         processed_impl.add(other_fc.path) # Mark test as processed
                         break # Found test, move to next file

        if impl_test_pairs:
            related_groups.append({
                "type": "implementation_test_pairs",
                "pairs": impl_test_pairs
            })

        # --- Look for model/schema pairs (example) ---
        # Simplified version, adjust regex/logic as needed for specific project conventions
        model_schema_pairs = []
        processed_model = set()

        for fc in file_changes_list:
            if not fc.path or fc.path in processed_model: continue
            current_path_obj = Path(fc.path)
            filename = current_path_obj.name

            # Potential model file pattern (e.g., *Model.py)
            model_match = re.match(r'^(.+)(?:Model|Dto)\.(py|js|ts|cs|java)$', filename)
            if model_match:
                base_name = model_match.group(1)
                # Look for corresponding schema/validator file (e.g., *Schema.py or *Validator.py)
                schema_pattern = re.compile(f'^{base_name}(?:Schema|Validator)\.(py|js|ts|cs|java)$')
                for other_fc in file_changes_list:
                    if not other_fc.path: continue
                    other_filename = Path(other_fc.path).name
                    if schema_pattern.match(other_filename):
                        model_schema_pairs.append({
                            "file1": fc.path,
                            "file2": other_fc.path,
                            "relation_type": "model_schema_validator",
                            "base_name": base_name
                        })
                        processed_model.add(fc.path)
                        processed_model.add(other_fc.path)
                        break

        if model_schema_pairs:
             related_groups.append({
                "type": "model_schema_pairs",
                "pairs": model_schema_pairs
            })

        # Add more relationship detection logic here (e.g., Component/Stylesheet)

        return related_groups

# --- END OF FILE pattern_analyzer_tool.py ---