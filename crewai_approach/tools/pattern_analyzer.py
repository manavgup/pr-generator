"""
Pattern analyzer tool for identifying patterns in file changes.
"""
from typing import List, Dict, Any, Optional, Set
import re
import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from shared.utils.logging_utils import get_logger
from .base_tools import BaseRepoTool

logger = get_logger(__name__)

class PatternAnalyzerInput(BaseModel):
    """Input schema for the PatternAnalyzer tool."""
    repo_path: str = Field(..., description="Path to the git repository")
    repository_analysis: Dict[str, Any] = Field(..., description="Repository analysis data")

class PatternAnalyzerTool(BaseRepoTool):
    """Tool for identifying patterns in file changes to detect related modifications."""

    name: str = "Pattern Analyzer"
    description: str = "Identifies patterns in file changes to detect related modifications"
    args_schema: type[BaseModel] = PatternAnalyzerInput

    def _run(self, repo_path: str, repository_analysis: Dict[str, Any], **kwargs) -> str:
        """
        Analyze patterns in file changes to detect related modifications.

        Args:
            repo_path: Path to the git repository
            repository_analysis: Repository analysis data as dictionary
            **kwargs: Additional arguments (ignored)

        Returns:
            JSON string containing pattern analysis information
        """
        logger.info("Analyzing file change patterns")

        try:
            # Extract file changes from repository_analysis
            file_changes = repository_analysis.get("file_changes", [])
            
            # Extract file names and extensions from the file changes
            file_names = []
            file_paths = []
            for fc in file_changes:
                # Extract filename and path, handling possible field names and formats
                filename = fc.get("filename", Path(fc.get("path", "")).name)
                file_names.append(filename)
                file_paths.append(fc.get("path", ""))

            # Analyze file naming patterns
            naming_patterns = self._analyze_naming_patterns(file_names)

            # Find files with similar names
            similar_names = self._find_similar_names(file_names, file_paths)

            # Analyze common prefixes/suffixes
            common_patterns = self._analyze_common_patterns(file_names, file_paths)

            # Detect file pairs that often change together
            related_files = self._detect_related_files(file_changes)

            # Construct the result
            result = {
                "naming_patterns": naming_patterns,
                "similar_names": similar_names,
                "common_patterns": {
                    "common_prefixes": common_patterns["common_prefixes"],
                    "common_suffixes": common_patterns["common_suffixes"]
                },
                "related_files": related_files,
                "analysis_summary": "Pattern analysis completed",
                "confidence": 0.8
            }

            # Return serialized JSON
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Error analyzing file patterns: {str(e)}"
            logger.error(error_msg)
            
            # Return a serialized error response
            error_result = {
                "naming_patterns": [],
                "similar_names": [],
                "common_patterns": {"common_prefixes": [], "common_suffixes": []},
                "related_files": [],
                "analysis_summary": f"Error during pattern analysis: {str(e)}",
                "confidence": 0.0,
                "error": error_msg
            }
            
            return json.dumps(error_result, indent=2)

    def _analyze_naming_patterns(self, file_names: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze naming patterns in files.

        Args:
            file_names: List of file names

        Returns:
            List of identified naming patterns as dictionaries
        """
        patterns = []

        # Look for common patterns
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
        """
        Find files with similar names.

        Args:
            file_names: List of file names
            file_paths: List of file paths

        Returns:
            List of groups of similar file names as dictionaries
        """
        similar_groups = []
        processed = set()

        # Create a mapping from file names to paths
        name_to_path = dict(zip(file_names, file_paths))

        for i, name1 in enumerate(file_names):
            if name1 in processed:
                continue

            # Get base name without extension
            base_name1 = re.sub(r'\.[^.]+$', name1, 1)

            # Find similar names
            similar = []
            similar_paths = []
            for j, name2 in enumerate(file_names):
                if i == j or name2 in processed:
                    continue

                base_name2 = re.sub(r'\.[^.]+$', name2, 1)

                # Check if base names are similar (one contained in the other)
                if base_name1 in base_name2 or base_name2 in base_name1:
                    similar.append(name2)
                    similar_paths.append(name_to_path[name2])

            if similar:
                similar.append(name1)  # Include the original name
                similar_paths.append(name_to_path[name1])  # Include the original path
                similar_groups.append({
                    "base_pattern": base_name1,
                    "files": similar_paths
                })
                processed.update(similar)

        return similar_groups

    def _analyze_common_patterns(self, file_names: List[str], file_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze common prefixes and suffixes in file names.

        Args:
            file_names: List of file names
            file_paths: List of file paths

        Returns:
            Dictionary with common prefix and suffix information
        """
        prefixes = defaultdict(list)
        suffixes = defaultdict(list)

        # Create a mapping from file names to paths
        name_to_path = dict(zip(file_names, file_paths))

        # Extract prefixes (before first underscore or dot)
        for name in file_names:
            prefix_match = re.match(r'^([a-zA-Z0-9]+)[_.]', name)
            if prefix_match:
                prefix = prefix_match.group(1)
                prefixes[prefix].append(name_to_path[name])

            # Extract suffixes (after last underscore but before extension)
            suffix_match = re.match(r'^.*_([a-zA-Z0-9]+)\.[^.]+$', name)
            if suffix_match:
                suffix = suffix_match.group(1)
                suffixes[suffix].append(name_to_path[name])

        # Filter to keep only prefixes/suffixes with multiple files
        common_prefixes = {prefix: files for prefix, files in prefixes.items() if len(files) > 1}
        common_suffixes = {suffix: files for suffix, files in suffixes.items() if len(files) > 1}

        # Convert to required format
        common_prefix_groups = [
            {"pattern_type": "prefix", "pattern_value": p, "files": f} 
            for p, f in common_prefixes.items()
        ]
        
        common_suffix_groups = [
            {"pattern_type": "suffix", "pattern_value": s, "files": f} 
            for s, f in common_suffixes.items()
        ]

        return {
            "common_prefixes": common_prefix_groups,
            "common_suffixes": common_suffix_groups
        }

    def _detect_related_files(self, file_changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect files that are likely related based on heuristics.

        Args:
            file_changes: List of file change dictionaries

        Returns:
            List of related file groups as dictionaries
        """
        related_groups = []

        # Group by directory
        dir_to_files = defaultdict(list)
        for fc in file_changes:
            directory = fc.get("directory", str(Path(fc.get("path", "")).parent))
            file_path = fc.get("path", "")
            if file_path:
                dir_to_files[directory].append(file_path)

        # Look for implementation/test pairs
        impl_test_pairs = []
        for fc in file_changes:
            filename = fc.get("filename", Path(fc.get("path", "")).name)
            file_path = fc.get("path", "")

            # Check if this is an implementation file with a corresponding test
            impl_match = re.match(r'^([a-zA-Z0-9_]+)\.(py|js|ts|jsx|tsx)$', filename)
            if impl_match:
                base_name = impl_match.group(1)
                ext = impl_match.group(2)

                # Look for corresponding test file
                test_pattern = f"test_{base_name}.{ext}"
                for other_fc in file_changes:
                    other_filename = other_fc.get("filename", Path(other_fc.get("path", "")).name)
                    if other_filename == test_pattern:
                        impl_test_pairs.append({
                            "file1": file_path,
                            "file2": other_fc.get("path", ""),
                            "relation_type": "implementation_test",
                            "base_name": base_name
                        })

        # Look for model/schema pairs
        model_schema_pairs = []
        for fc in file_changes:
            filename = fc.get("filename", Path(fc.get("path", "")).name)

            # Check if this is a model file
            model_match = re.match(r'^(.+)Model\.(py|js|ts|cs|java)$', filename)
            if model_match:
                base_name = model_match.group(1)

                # Look for corresponding schema file
                schema_pattern = f"{base_name}Schema"
                for other_fc in file_changes:
                    other_filename = other_fc.get("filename", Path(other_fc.get("path", "")).name)
                    if schema_pattern in other_filename:
                        model_schema_pairs.append({
                            "file1": fc.get("path", ""),
                            "file2": other_fc.get("path", ""),
                            "relation_type": "model_schema",
                            "base_name": base_name
                        })

        # Combine all related groups
        if impl_test_pairs:
            related_groups.append({
                "type": "implementation_test_pairs",
                "pairs": impl_test_pairs
            })

        if model_schema_pairs:
            related_groups.append({
                "type": "model_schema_pairs",
                "pairs": model_schema_pairs
            })

        return related_groups