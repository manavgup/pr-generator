"""
PR description generator using LLMService.
"""
import logging
from typing import Dict, List, Any, Optional

from llm_guided_approach.llm_service import LLMService
from shared.utils.logging_utils import log_operation, log_llm_prompt

logger = logging.getLogger(__name__)


class DescriptionGenerator:
    """
    Generates detailed PR descriptions using an LLM.
    
    Responsibilities:
    - Creating prompts for LLM-based description generation
    - Processing LLM responses into structured descriptions
    - Formatting descriptions consistently
    """
    
    def __init__(self, llm_service: LLMService, verbose: bool = False):
        """
        Initialize the DescriptionGenerator.
        
        Args:
            llm_service: LLM service for generating descriptions
            verbose: Enable verbose logging
        """
        self.llm_service = llm_service
        self.verbose = verbose
    
    @log_operation("Generating PR description")
    def generate_description(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed PR description for a group.
        
        Args:
            group: Group dictionary
            
        Returns:
            Dictionary with detailed PR content
        """
        try:
            # Create the prompt for detailed description
            detail_prompt = self._create_description_prompt(group)
            
            # Log the prompt if verbose mode is enabled
            log_llm_prompt("Description prompt", detail_prompt, self.verbose)
            
            # Get detailed description using the LLM service
            detail_result = self.llm_service.analyze_changes(detail_prompt)
            
            # Extract description from result
            description = self._extract_description(detail_result, group)
            
            # Update the group with the new description
            detailed_group = group.copy()
            detailed_group["description"] = description
            
            return detailed_group
            
        except Exception as e:
            logger.warning(f"Error generating detailed description: {e}")
            # Return the original group if there's an error
            return group
    
    def _create_description_prompt(self, group: Dict[str, Any]) -> str:
        """
        Create a prompt for generating a detailed PR description.
        
        Args:
            group: Group dictionary
            
        Returns:
            Prompt string for the LLM
        """
        files_str = "\n".join([f"- {file}" for file in group.get("files", [])])
        
        return f"""
        Create a detailed pull request description for the following group of files:
        
        Title: {group.get('title', 'Untitled PR')}
        Files:
        {files_str}
        
        Rationale: {group.get('reasoning', '')}
        
        The description should include:
        1. A concise summary of the changes
        2. The purpose and impact of these changes
        3. Any technical considerations or implementation details
        4. Any breaking changes or migration steps required
        
        Respond with a JSON object containing a detailed description field.
        """
    
    def _extract_description(self, result: Dict[str, Any], group: Dict[str, Any]) -> str:
        """
        Extract description from LLM result.
        
        Args:
            result: LLM result dictionary
            group: Original group dictionary
            
        Returns:
            Description string
        """
        # Try to find description in the result
        description = self._find_description_in_result(result)
        
        # If we couldn't find a description, use the reasoning from the original group
        if not description:
            description = group.get("reasoning", "")
            logger.warning("Could not extract description from LLM result, using reasoning instead")
        
        return description
    
    def _find_description_in_result(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Find description field in a potentially nested result dictionary.
        
        Args:
            result: LLM result dictionary
            
        Returns:
            Description string if found, None otherwise
        """
        # Check for direct 'description' field
        if isinstance(result, dict):
            # Direct field in the result
            if "description" in result:
                return result["description"]
            
            # Search in nested dictionaries (up to one level)
            for key, value in result.items():
                if isinstance(value, dict) and "description" in value:
                    return value["description"]
        
        return None