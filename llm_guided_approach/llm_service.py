"""
LLM Service implementation that uses llm_clients package.
"""
import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union

from llm_clients import (
    get_client, GenerationParams, Message, MessageRole,
    AnthropicConfig, OpenAIConfig, OllamaConfig, WatsonxConfig
)
from llm_clients.interfaces import ProviderType
from pydantic import BaseModel, ValidationError
from shared.config.llm_config import LLMConfig, LLMProvider
from shared.models.pr_models import FileChange
from shared.utils.logging_utils import log_llm_prompt
from collections import defaultdict

logger = logging.getLogger(__name__)

class GroupValidationModel(BaseModel):
    files: List[str]
    title: str
    description: str
    reasoning: str
    suggested_branch: Optional[str] = None

class LLMService:
    """
    A service for interacting with Language Models using the llm_clients package.
    Enhanced with robust JSON handling and error recovery.
    """
    
    def __init__(self, config: Union[LLMConfig, Dict[str, Any]]):
        # Initialization remains the same
        if isinstance(config, dict):
            self.config = LLMConfig(**config)
        else:
            self.config = config
            
        provider_map = {
            LLMProvider.OPENAI: "openai",
            LLMProvider.OLLAMA: "ollama",
        }
        
        provider_name = provider_map.get(self.config.provider, self.config.provider.value)
        client_config = self._create_client_config()
        
        self.client = get_client(
            provider=provider_name, 
            model_id=self.config.model,
            config=client_config
        )
        logger.info(f"Initialized LLM Service with provider: {provider_name}, model: {self.config.model}")

    def analyze_changes(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze changes with enhanced JSON handling and validation.
        """
        try:
            logger.info(f"Sending prompt to {self.config.provider.value} model: {self.config.model}")
            log_llm_prompt("Grouping Prompt", prompt)
            
            messages = [
                Message(role=MessageRole.SYSTEM, content=self._system_prompt()),
                Message(role=MessageRole.USER, content=prompt)
            ]
            
            response = self.client.generate_with_messages(messages, self._generation_params())
            log_llm_prompt("Grouping Prompt Response", response)
            
            # JSON processing pipeline
            result = self._process_response(response)
            return self._validate_and_deduplicate(result)
            
        except Exception as e:
            logger.exception(f"Error getting LLM analysis: {e}")
            return {"groups": []}

    def _process_response(self, response: str) -> Dict[str, Any]:
        """Process LLM response through validation pipeline"""
        # First try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Then try extraction from text
        extracted = self._extract_json_from_text(response)
        if extracted:
            return extracted
            
        # Final fallback with validation
        return self._validate_loose_json(response)

    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        """Improved JSON extraction with markdown support"""
        json_pattern = r'(?s)(?:```json\s*)(.*?)(?:```)|(\{.*?\})'
        matches = re.finditer(json_pattern, text)
        
        for match in matches:
            json_str = next((g for g in match.groups() if g), None)
            if json_str:
                try:
                    return json.loads(json_str.strip())
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse attempt failed: {e}")
                    continue
        return None

    def _validate_loose_json(self, text: str) -> Dict[str, Any]:
        """Fallback validation for malformed JSON"""
        try:
            # Attempt to find the first valid JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except Exception as e:
            logger.warning(f"Loose JSON validation failed: {e}")
            return {"groups": []}

    def _validate_and_deduplicate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate group structure and remove duplicates"""
        if not isinstance(result, dict):
            return {"groups": []}
            
        validated_groups = []
        seen_files = set()
        
        for group in result.get("groups", []):
            try:
                # Validate against Pydantic model
                valid_group = GroupValidationModel(**group)
                files_tuple = tuple(sorted(valid_group.files))
                
                if files_tuple not in seen_files:
                    seen_files.add(files_tuple)
                    validated_groups.append(valid_group.model_dump())
            except ValidationError as e:
                logger.warning(f"Invalid group structure: {e}")
                
        return {"groups": validated_groups}

    def _system_prompt(self) -> str:
        """System prompt with JSON enforcement"""
        return """You are a technical lead analyzing git changes. 
        Respond ONLY with valid JSON using this structure:
        {
            "groups": [{
                "files": ["path1", "path2"],
                "title": "type(scope): desc",
                "description": "...",
                "reasoning": "..."
            }]
        }
        """

    def _generation_params(self) -> GenerationParams:
        """Consolidated generation parameters"""
        return GenerationParams(
            temperature=self.config.temperature or 0.2,
            max_tokens=self.config.max_tokens or 4000,
            top_p=0.9,
            response_format={"type": "json_object"} if self.config.provider == LLMProvider.OPENAI else None
        )
  
    def _create_client_config(self) -> Any:
        """
        Create provider-specific client configuration.
        
        Returns:
            Provider-specific configuration object
        """
        if self.config.provider == LLMProvider.OPENAI:
            return OpenAIConfig(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
                default_system_prompt="You are a technical lead analyzing git changes. Always respond with valid JSON."
            )
        elif self.config.provider == LLMProvider.OLLAMA:
            return OllamaConfig(
                base_url=self.config.base_url or "http://localhost:11434",
                request_timeout=120.0,  # Longer timeout for Ollama
                use_formatter=True      # Use proper prompt formatting
            )
        # Add other providers as needed
        
        # Return None if no specific config is needed
        return None
    
    def analyze_changes(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze changes with the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Dictionary with the analysis result
        """
        try:
            logger.info(f"Sending prompt to {self.config.provider.value} model: {self.config.model}")
            log_llm_prompt("Grouping Prompt", prompt)  # Log the prompt for debugging
            # Create generation parameters
            params = GenerationParams(
                temperature=self.config.temperature or 0.2,
                max_tokens=self.config.max_tokens or 4000,
                # OpenAI can use JSON mode if using their API directly
                top_p=0.9  # A reasonable default for most LLMs
            )
            
            # Create a message-based prompt for better structure
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are a technical lead analyzing git changes. Always respond with valid JSON."
                ),
                Message(
                    role=MessageRole.USER,
                    content=prompt
                )
            ]
            
            # Generate using message format
            response = self.client.generate_with_messages(messages, params)
            log_llm_prompt("Grouping Prompt Response", response)  # Log the prompt for debugging
            # Try to parse as JSON
            try:
                result = json.loads(response)
                return self._deduplicate_groups(result)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse response as JSON: {e}")
                
                # Try to extract JSON from the response
                extracted_json = self._extract_json_from_text(response)
                if extracted_json:
                    return self._deduplicate_groups(extracted_json)
                    
                # If we can't parse JSON, return an empty result
                logger.error("Could not extract valid JSON from response")
                return {"groups": []}
                
        except Exception as e:
            logger.exception(f"Error getting LLM analysis: {e}")
            return {"groups": []}
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # For Ollama, we need to use a specific embedding model
            if self.config.provider == LLMProvider.OLLAMA:
                # Use an appropriate embedding model
                return self.client.get_embeddings(texts, model_id="granite-embedding")
            
            # For other providers, use their default embedding model
            return self.client.get_embeddings(texts)
        except Exception as e:
            logger.exception(f"Error getting embeddings: {e}")
            return [[] for _ in texts]  # Return empty embeddings on error
    
    def format_changes_for_prompt(self, changes: List[FileChange]) -> str:
        """Format file changes with diff snippets"""
        dir_changes = defaultdict(list)
        
        for change in changes:
            file_info = {
                "file": change.file_path,
                "status": "deleted" if change.is_deleted else "modified",
                "changes": f"+{change.changes.added}/-{change.changes.deleted}",
                "diff_snippet": (change.diff[:500] + "...") if change.diff else "No diff available"
            }
            dir_changes[change.directory].append(file_info)
        
        return json.dumps({
            "changes_by_directory": dir_changes,
            "total_files": len(changes)
        }, indent=2)
    
    def create_grouping_prompt(self, changes: List[FileChange]) -> str:
        """Create prompt for the LLM to analyze and group changes with branch names."""
        changes_str = self.format_changes_for_prompt(changes)
        
        return f"""Analyze these Git changes and suggest logical pull request groupings.

        Changes to analyze:
        {changes_str}

        Consider these factors when grouping:
        1. Related functionality and features
        2. Dependencies between changes
        3. Testing impact and risk level
        4. Architecture and component boundaries
        5. Each PR should have a clear, single responsibility

        DO NOT create empty groups with no files.
        AVOID generic titles - be specific about what changed.
        ENSURE each group has a complete set of related files.

        For each group, provide:
        1. A clear, specific title using semantic commit format (e.g., "feat(auth): implement OAuth2 flow")
        2. A detailed description explaining what changes were made and why
        3. Technical reasoning explaining why these files belong together
        4. A suggested git branch name (kebab-case, lowercase, no spaces, max 50 chars)

        Respond with a JSON object in this format:
        {{
            "groups": [
                {{
                    "files": ["path/to/file1", "path/to/file2"],
                    "title": "feat(component): detailed change description",
                    "description": "Comprehensive explanation of changes and purpose",
                    "reasoning": "Technical explanation of why these files form a logical group",
                    "suggested_branch": "feat-component-meaningful-name"
                }}
            ]
        }}
        """

    def _deduplicate_groups(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate groups from results."""
        if not isinstance(result, dict) or 'groups' not in result:
            return {"groups": []}
            
        seen_files = set()
        unique_groups = []
        for group in result['groups']:
            if 'files' not in group:
                continue
                
            files_tuple = tuple(sorted(group['files']))
            if files_tuple not in seen_files:
                seen_files.add(files_tuple)
                unique_groups.append(group)
        
        result['groups'] = unique_groups
        return result
    
    def _extract_json_from_text(self, text: str) -> Optional[dict]:
        # Match both ```json and regular JSON
        json_pattern = r'(?:```json\n?)(.*?)(?:```)|({.*})'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            json_str = match.group(1) or match.group(2)
            if json_str:
                try:
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue
        return None
        
    def test_connection(self) -> bool:
        """Test the LLM connection."""
        try:
            # Try a simple prompt to test connection
            response = self.client.generate(
                "Return a short response",
                GenerationParams(max_tokens=10)
            )
            return bool(response.strip())
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False