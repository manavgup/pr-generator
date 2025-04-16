"""
Pattern analyzer tool for identifying patterns in file changes.
"""
from typing import List, Dict, Any, Optional, Set, Type
import re
import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from models.agent_models import PatternAnalysisResult
from shared.utils.logging_utils import get_logger
from .base_tool import BaseRepoTool

logger = get_logger(__name__)

class PatternAnalyzerToolSchema(BaseModel):
    """Input schema for the PatternAnalyzer tool using primitive types only."""
    file_paths: List[str] = Field(..., description="List of file paths to analyze for patterns")
    # Optional parameter to provide directory info if available
    directory_to_files: Optional[Dict[str, List[str]]] = Field(None, description="Optional mapping of directories to files")

class SimplifiedFileInfo(BaseModel):
    """Minimal file information needed for pattern analysis."""
    path: str
    name: str = ""
    directory: str = ""
    extension: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if self.path and not self.name:
            path_obj = Path(self.path)
            self.name = path_obj.name
            self.directory = str(path_obj.parent)
            self.extension = path_obj.suffix


class PatternAnalyzerTool(BaseRepoTool):
    """Tool for identifying patterns in file changes to detect related modifications."""

    name: str = "Pattern Analyzer"
    description: str = "Identifies patterns in file changes based on file paths. Returns PatternAnalysisResult JSON."
    args_schema: Type[BaseModel] = PatternAnalyzerToolSchema

    def _run(
        self,
        file_paths: List[str],
        directory_to_files: Optional[Dict[str, List[str]]] = None
    ) -> str:
        """
        Analyze patterns in file changes to detect related modifications.

        Args:
            file_paths: List of file paths to analyze
            directory_to_files: Optional dictionary mapping directories to file lists

        Returns:
            JSON string containing pattern analysis information (PatternAnalysisResult).
        """
        # Echo received inputs for debugging
        logger.info(f"PatternAnalyzerTool received {len(file_paths)} file paths")
        if directory_to_files:
            logger.info(f"PatternAnalyzerTool received directory mapping with {len(directory_to_files)} directories")
        
        repo_path = self.git_ops.repo_path if hasattr(self, 'git_ops') and self.git_ops else "unknown"
        logger.info(f"Analyzing file change patterns for {repo_path}")

        try:
            if not file_paths:
                 logger.warning(f"No file paths provided for analysis. Returning empty pattern result.")
                 result = PatternAnalysisResult(analysis_summary="No file paths provided.")
                 return result.model_dump_json(indent=2)

            # --- Prepare simplified file information ---
            file_info_list: List[SimplifiedFileInfo] = []
            for file_path in file_paths:
                if file_path:  # Ensure not empty
                    file_info = SimplifiedFileInfo(path=file_path)
                    file_info_list.append(file_info)

            # Extract file names and paths
            file_names: List[str] = []
            file_paths_processed: List[str] = []
            for fi in file_info_list:
                if fi.path:  # Ensure path exists
                    file_paths_processed.append(fi.path)
                    file_names.append(fi.name)
            # --- End data preparation ---

            # Analyze file naming patterns
            naming_patterns = self._analyze_naming_patterns(file_names)

            # Find files with similar names
            similar_names = self._find_similar_names(file_names, file_paths_processed)

            # Analyze common prefixes/suffixes
            common_patterns = self._analyze_common_patterns(file_names, file_paths_processed)

            # Detect file pairs that often change together
            related_files = self._detect_related_files(file_info_list)

            # Construct the result using the proper model
            result = PatternAnalysisResult(
                naming_patterns=naming_patterns,
                similar_names=similar_names,
                common_patterns={
                    "common_prefixes": common_patterns["common_prefixes"],
                    "common_suffixes": common_patterns["common_suffixes"]
                },
                related_files=related_files,
                analysis_summary="Pattern analysis completed successfully.",
                confidence=0.8  # Placeholder confidence
            )

            logger.info(f"Pattern analysis complete for {repo_path}.")
            return result.model_dump_json(indent=2)

        except Exception as e:
            # Catch all errors and return a valid PatternAnalysisResult with error info
            error_msg = f"Error analyzing file patterns: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_result = PatternAnalysisResult(
                analysis_summary=error_msg, 
                confidence=0.0, 
                error=error_msg
            )
            return error_result.model_dump_json(indent=2)


    # --- Helper Methods (Updated for simplified file info) ---

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

                # Check if base names are similar enough
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

    def _detect_related_files(self, file_info_list: List[SimplifiedFileInfo]) -> List[Dict[str, Any]]:
        """Detect files that are likely related based on heuristics."""
        # Updated to work with SimplifiedFileInfo instead of FileChange
        related_groups = []

        # --- Look for implementation/test pairs ---
        impl_test_pairs = []
        processed_impl = set() # Avoid duplicating pairs if both impl and test trigger match

        for file_info in file_info_list:
            if not file_info.path or file_info.path in processed_impl: continue

            filename = file_info.name
            file_stem = Path(filename).stem # Name without final suffix
            file_suffix = Path(filename).suffix # e.g., ".py"

            # Potential implementation file pattern (simple check)
            if not filename.startswith("test_") and file_suffix in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cs']:
                # Look for corresponding test file
                expected_test_stem = f"test_{file_stem}"
                for other_file in file_info_list:
                    if not other_file.path: continue
                    other_filename = other_file.name
                    other_path_obj = Path(other_filename)
                    # Check stem and suffix match
                    if other_path_obj.stem == expected_test_stem and other_path_obj.suffix == file_suffix:
                        # Basic check passed, add pair
                         impl_test_pairs.append({
                            "file1": file_info.path,
                            "file2": other_file.path,
                            "relation_type": "implementation_test",
                            "base_name": file_stem
                        })
                         processed_impl.add(file_info.path) # Mark impl as processed
                         processed_impl.add(other_file.path) # Mark test as processed
                         break # Found test, move to next file

        if impl_test_pairs:
            related_groups.append({
                "type": "implementation_test_pairs",
                "pairs": impl_test_pairs
            })

        # --- Look for model/schema pairs (example) ---
        model_schema_pairs = []
        processed_model = set()

        for file_info in file_info_list:
            if not file_info.path or file_info.path in processed_model: continue
            filename = file_info.name

            # Potential model file pattern (e.g., *Model.py)
            model_match = re.match(r'^(.+)(?:Model|Dto)\.(py|js|ts|cs|java)$', filename)
            if model_match:
                base_name = model_match.group(1)
                # Look for corresponding schema/validator file (e.g., *Schema.py or *Validator.py)
                schema_pattern = re.compile(rf'^{base_name}(?:Schema|Validator)\.(py|js|ts|cs|java)$')
                for other_file in file_info_list:
                    if not other_file.path: continue
                    other_filename = other_file.name
                    if schema_pattern.match(other_filename):
                        model_schema_pairs.append({
                            "file1": file_info.path,
                            "file2": other_file.path,
                            "relation_type": "model_schema_validator",
                            "base_name": base_name
                        })
                        processed_model.add(file_info.path)
                        processed_model.add(other_file.path)
                        break

        if model_schema_pairs:
             related_groups.append({
                "type": "model_schema_pairs",
                "pairs": model_schema_pairs
            })

        return related_groups