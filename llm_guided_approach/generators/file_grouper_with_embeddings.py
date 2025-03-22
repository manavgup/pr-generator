"""
File grouper with embedding support using LLMService for PR generation.
"""
import logging
import os
from typing import Dict, List, Tuple, Any, Set
import random

from llm_guided_approach.llm_service import LLMService
from shared.utils.logging_utils import log_operation, log_llm_prompt
from shared.models.pr_models import FileChange

logger = logging.getLogger(__name__)


class FileGrouperWithEmbeddings:
    """
    Groups files into logical pull requests using embeddings for semantic similarity.
    
    Responsibilities:
    - Creating embeddings for file changes using the LLMService
    - Using similarity to group related files
    - Creating prompts for LLM-based file grouping on semantically related chunks
    - Processing LLM responses into structured groups
    """
    
    def __init__(self, llm_service: LLMService, verbose: bool = False):
        """
        Initialize the FileGrouperWithEmbeddings.
        
        Args:
            llm_service: LLM service for interacting with language models and getting embeddings
            verbose: Enable verbose logging
        """
        self.llm_service = llm_service
        self.verbose = verbose
    
    @log_operation("Grouping files with embeddings")
    def group_files(self, changes: List[FileChange], max_files_per_group: int = 10) -> List[Dict[str, Any]]:
        """
        Group files into logical pull requests using embeddings for semantic similarity.
        
        Args:
            changes: List of file changes
            max_files_per_group: Maximum number of files per group
            
        Returns:
            List of group dictionaries
        """
        if len(changes) <= max_files_per_group:
            # If we have a small number of changes, just use the standard approach
            return self._standard_grouping(changes)
        
        # Get file contents for embedding
        file_texts = {}
        for change in changes:
            if change.diff:
                file_texts[change.file_path] = change.diff
            else:
                # If diff is not available, use file path as a fallback
                file_texts[change.file_path] = change.file_path
        
        # Create embeddings for all changes
        logger.info(f"Creating embeddings for {len(changes)} files")
        
        # Generate embeddings using the llm_service
        embeddings = {}
        try:
            # Get all texts in a list, preserving the order
            text_list = list(file_texts.values())
            file_paths = list(file_texts.keys())
            
            # Get embeddings for all texts
            embedding_vectors = self.llm_service.get_embeddings(text_list)
            
            # Map file paths to embeddings
            for i, file_path in enumerate(file_paths):
                if i < len(embedding_vectors):
                    embeddings[file_path] = embedding_vectors[i]
                else:
                    logger.warning(f"Missing embedding for {file_path}")
        except Exception as e:
            logger.exception(f"Error getting embeddings: {e}")
            return self._fallback_grouping_by_directory(changes, max_files_per_group)
        
        # Create semantic groups based on embeddings
        logger.info("Creating semantic groups based on embeddings")
        semantic_groups = self._group_by_similarity(changes, embeddings, max_files_per_group)
        
        # Process each semantic group with the LLM
        logger.info(f"Processing {len(semantic_groups)} semantic groups with LLM")
        all_groups = []
        
        for i, group_changes in enumerate(semantic_groups):
            logger.info(f"Processing semantic group {i+1}/{len(semantic_groups)} with {len(group_changes)} files")
            
            group_result = self._process_semantic_group(group_changes)
            all_groups.extend(group_result)
        
        logger.info(f"Created {len(all_groups)} total groups across all semantic chunks")
        return all_groups
    
    def _standard_grouping(self, changes: List[FileChange]) -> List[Dict[str, Any]]:
        """
        Standard grouping approach for a small number of files.
        
        Args:
            changes: List of file changes
            
        Returns:
            List of group dictionaries
        """
        grouping_prompt = self.llm_service.create_grouping_prompt(changes)
        log_llm_prompt("File grouping prompt", grouping_prompt, self.verbose)
        
        logger.info("Getting groupings from LLM service")
        result = self.llm_service.analyze_changes(grouping_prompt)
        
        groups = self._validate_groups(result)
        logger.info(f"Created {len(groups)} file groups")
        return groups
    
    def _process_semantic_group(self, changes: List[FileChange]) -> List[Dict[str, Any]]:
        """
        Process a semantic group with the LLM.
        
        Args:
            changes: List of file changes in the semantic group
            
        Returns:
            List of group dictionaries
        """
        # Create prompt for just this semantic group
        grouping_prompt = self.llm_service.create_grouping_prompt(changes)
        log_llm_prompt(f"Semantic group prompt ({len(changes)} files)", grouping_prompt, self.verbose)
        
        # Get suggestions from the LLM
        result = self.llm_service.analyze_changes(grouping_prompt)
        
        # Validate and normalize the groups
        groups = self._validate_groups(result)
        logger.info(f"Created {len(groups)} groups for semantic chunk of {len(changes)} files")
        
        return groups
    
    def _group_by_similarity(
        self, 
        changes: List[FileChange], 
        embeddings: Dict[str, List[float]], 
        max_files_per_group: int
    ) -> List[List[FileChange]]:
        """
        Group files by similarity using embeddings.
        
        This implements a simple greedy algorithm:
        1. Start with a random file as a seed
        2. Find the most similar files to add to the group
        3. When the group reaches max size, start a new group
        4. Repeat until all files are assigned to groups
        
        Args:
            changes: List of file changes
            embeddings: Dictionary mapping file paths to embedding vectors
            max_files_per_group: Maximum number of files per group
            
        Returns:
            List of groups, where each group is a list of FileChange objects
        """
        # Quick validation - make sure we have embeddings for all files
        for change in changes:
            if change.file_path not in embeddings:
                logger.warning(f"No embedding found for {change.file_path}, using fallback grouping")
                return self._fallback_grouping_by_directory(changes, max_files_per_group)
        
        # Initialize groups
        groups = []
        current_group = []
        remaining_changes = changes.copy()
        
        # Function to calculate similarity between two embeddings (cosine similarity)
        def similarity(vec1, vec2):
            if not vec1 or not vec2:
                return 0.0
                
            # Dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Magnitudes
            mag1 = sum(a * a for a in vec1) ** 0.5
            mag2 = sum(b * b for b in vec2) ** 0.5
            
            # Avoid division by zero
            if mag1 * mag2 == 0:
                return 0.0
                
            return dot_product / (mag1 * mag2)
        
        # While we still have files to assign
        while remaining_changes:
            # If we're starting a new group, pick a random seed file
            if not current_group:
                seed_file = random.choice(remaining_changes)
                current_group.append(seed_file)
                remaining_changes.remove(seed_file)
                continue
            
            # If the current group is full, add it to groups and start a new one
            if len(current_group) >= max_files_per_group:
                groups.append(current_group)
                current_group = []
                continue
            
            # Find the most similar file to the current group
            best_similarity = -1.0
            best_match = None
            
            # Get the average embedding of the current group
            group_embedding = [0.0] * len(embeddings[current_group[0].file_path])
            for change in current_group:
                file_embedding = embeddings[change.file_path]
                for i in range(len(group_embedding)):
                    group_embedding[i] += file_embedding[i] / len(current_group)
            
            # Find the most similar file
            for change in remaining_changes:
                file_embedding = embeddings[change.file_path]
                sim = similarity(group_embedding, file_embedding)
                
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = change
            
            # Add the best match to the current group
            if best_match:
                current_group.append(best_match)
                remaining_changes.remove(best_match)
            else:
                # This shouldn't happen, but just in case
                break
        
        # Add the last group if it's not empty
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _fallback_grouping_by_directory(self, changes: List[FileChange], max_files_per_group: int) -> List[List[FileChange]]:
        """
        Fallback grouping method that uses directory structure.
        
        Args:
            changes: List of file changes
            max_files_per_group: Maximum number of files per group
            
        Returns:
            List of groups, where each group is a list of FileChange objects
        """
        # Group by directory
        by_directory = {}
        for change in changes:
            dir_name = change.directory
            if dir_name not in by_directory:
                by_directory[dir_name] = []
            by_directory[dir_name].append(change)
        
        # Create groups, splitting large directories if needed
        groups = []
        for dir_name, dir_changes in by_directory.items():
            if len(dir_changes) <= max_files_per_group:
                groups.append(dir_changes)
            else:
                # Split large directories into smaller chunks
                for i in range(0, len(dir_changes), max_files_per_group):
                    chunk = dir_changes[i:i + max_files_per_group]
                    groups.append(chunk)
        
        return groups
    
    def _validate_groups(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate and normalize groups from the analyzer.
        
        Args:
            result: LLM result dictionary
            
        Returns:
            List of validated group dictionaries
        """
        if not result or "groups" not in result:
            logger.error("LLM returned invalid result")
            return []
        
        groups = result["groups"]
        
        # Validate each group has required fields
        validated_groups = []
        for i, group in enumerate(groups):
            if not self._is_valid_group(group):
                logger.warning(f"Skipping invalid group at index {i}")
                continue
            
            # Normalize field names
            normalized_group = self._normalize_group(group)
            validated_groups.append(normalized_group)
        
        return validated_groups
    
    def _is_valid_group(self, group: Dict[str, Any]) -> bool:
        """
        Check if a group has all required fields.
        
        Args:
            group: Group dictionary
            
        Returns:
            True if the group is valid, False otherwise
        """
        required_fields = ["files", "title"]
        for field in required_fields:
            if field not in group or not group[field]:
                return False
        return True
    
    def _normalize_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize field names to ensure consistency.
        
        Args:
            group: Group dictionary
            
        Returns:
            Normalized group dictionary
        """
        normalized = {
            "title": group.get("title", "Untitled PR"),
            "files": group.get("files", []),
            "reasoning": group.get("reasoning", group.get("rationale", "")),
            "branch_name": group.get("branch_name", self._generate_branch_name(group.get("title", ""))),
            "description": group.get("description", "")
        }
        return normalized
    
    def _generate_branch_name(self, title: str) -> str:
        """
        Generate a git branch name from a PR title.
        
        Args:
            title: PR title
            
        Returns:
            Git branch name
        """
        # Replace spaces with hyphens, remove special characters
        branch = title.lower().replace(" ", "-")
        branch = ''.join(c for c in branch if c.isalnum() or c == '-')
        
        # Truncate to reasonable length
        return branch[:50]