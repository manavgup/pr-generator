"""
Group manager for PR generation.
"""
import logging
import os
from typing import Dict, List, Set, Any

from shared.utils.logging_utils import log_operation

logger = logging.getLogger(__name__)


class GroupManager:
    """
    Manages file groups for PR generation.
    
    Responsibilities:
    - Splitting large groups into manageable PRs
    - Adding groups for files not covered by initial analysis
    - Managing group structure and file distribution
    """
    
    def __init__(self, max_files_per_pr: int = 20):
        """
        Initialize the GroupManager.
        
        Args:
            max_files_per_pr: Maximum number of files per PR
        """
        self.max_files_per_pr = max_files_per_pr
    
    @log_operation("Splitting large group")
    def split_group(self, group: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a large group into smaller ones.
        
        Args:
            group: Group dictionary to split
            
        Returns:
            List of smaller group dictionaries
        """
        files = group.get("files", [])
        title = group.get("title", "Untitled PR")
        reasoning = group.get("reasoning", "")
        branch_name = group.get("branch_name", "")
        description = group.get("description", "")
        
        # If group is small enough, return as is
        if len(files) <= self.max_files_per_pr:
            return [group]
        
        # Calculate number of parts needed
        num_parts = (len(files) + self.max_files_per_pr - 1) // self.max_files_per_pr
        logger.info(f"Splitting group '{title}' into {num_parts} parts")
        
        # Split files into chunks
        chunks = []
        for i in range(num_parts):
            start_idx = i * self.max_files_per_pr
            end_idx = min(start_idx + self.max_files_per_pr, len(files))
            chunk_files = files[start_idx:end_idx]
            
            # Create a chunk group
            chunk = {
                "title": f"{title} (Part {i+1}/{num_parts})",
                "files": chunk_files,
                "reasoning": reasoning,
                "description": description,
                "branch_name": f"{branch_name}-part-{i+1}" if branch_name else None
            }
            chunks.append(chunk)
        
        return chunks
    
    @log_operation("Adding groups for unassigned files")
    def add_missing_files_group(self, groups: List[Dict[str, Any]], all_files: Set[str]) -> List[Dict[str, Any]]:
        """
        Add a group for files that aren't included in any existing group.
        
        Args:
            groups: List of group dictionaries
            all_files: Set of all changed files
            
        Returns:
            Updated list of group dictionaries
        """
        # Collect all files that are already grouped
        grouped_files = set()
        for group in groups:
            grouped_files.update(group.get("files", []))
        
        # Find files that aren't in any group
        missing_files = list(all_files - grouped_files)
        
        # If there are missing files, add a group for them
        if missing_files:
            logger.info(f"Adding groups for {len(missing_files)} ungrouped files")
            
            # Organize missing files by directory for better grouping
            files_by_dir = self._group_files_by_directory(missing_files)
            
            # For each directory with files, create a separate group
            for dir_name, dir_files in files_by_dir.items():
                if dir_files:  # Only create groups with files
                    groups.append(self._create_directory_group(dir_name, dir_files))
        
        return groups
    
    def _group_files_by_directory(self, files: List[str]) -> Dict[str, List[str]]:
        """
        Group files by their directory.
        
        Args:
            files: List of file paths
            
        Returns:
            Dictionary mapping directory names to lists of files
        """
        files_by_dir = {}
        for file_path in files:
            dir_name = os.path.dirname(file_path) or "(root)"
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(file_path)
        return files_by_dir
    
    def _create_directory_group(self, dir_name: str, files: List[str]) -> Dict[str, Any]:
        """
        Create a group for files in a specific directory.
        
        Args:
            dir_name: Directory name
            files: List of files in the directory
            
        Returns:
            Group dictionary
        """
        # Create a readable directory name for the title
        readable_dir = dir_name.replace('/', ' ').replace('_', ' ').title()
        
        return {
            "title": f"Update {readable_dir} Files",
            "files": files,
            "reasoning": f"Changes to files in the {dir_name} directory",
            "suggested_branch": f"update-{dir_name.replace('/', '-').lower()}"
        }