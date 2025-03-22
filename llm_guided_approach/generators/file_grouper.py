"""
File grouper using LLMService for PR generation.
"""
import logging
from typing import Dict, List, Any

from llm_guided_approach.llm_service import LLMService
from shared.utils.logging_utils import log_operation, log_llm_prompt
from shared.models.pr_models import FileChange

logger = logging.getLogger(__name__)


class FileGrouper:
    """
    Groups files into logical pull requests using an LLM.
    
    Responsibilities:
    - Creating prompts for LLM-based file grouping
    - Processing LLM responses into structured groups
    - Validating grouping results for consistency
    """
    
    def __init__(self, llm_service: LLMService, verbose: bool = False):
        """
        Initialize the FileGrouper.
        
        Args:
            llm_service: LLM service for interacting with language models
            verbose: Enable verbose logging
        """
        self.llm_service = llm_service
        self.verbose = verbose
    
    @log_operation("Grouping files")
    def group_files(self, changes: List[FileChange]) -> List[Dict[str, Any]]:
        """
        Group files into logical pull requests.
        
        Args:
            changes: List of file changes
            
        Returns:
            List of group dictionaries
        """
        # Create prompt for the LLM to group the changes
        grouping_prompt = self.llm_service.create_grouping_prompt(changes)
        
        # Log the prompt if verbose mode is enabled
        log_llm_prompt("File grouping prompt", grouping_prompt, self.verbose)
        
        # Get initial groupings from the LLM service
        logger.info("Getting initial groupings from LLM")
        result = self.llm_service.analyze_changes(grouping_prompt)
        
        # Process and validate the result
        groups = self._validate_groups(result)
        
        logger.info(f"Created {len(groups)} initial file groups")
        return groups
    
    def _validate_groups(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate and normalize groups from the LLM.
        
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
        """Strict group validation"""
        required_fields = {
            "files": (list, lambda x: len(x) >= 1),
            "title": (str, lambda x: len(x) >= 10),
            "reasoning": (str, lambda x: len(x) >= 20)
        }
        
        for field, (type_check, validator) in required_fields.items():
            if field not in group or \
            not isinstance(group[field], type_check) or \
            not validator(group[field]):
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