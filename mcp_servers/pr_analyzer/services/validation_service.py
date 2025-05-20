# mcp_servers/pr_analyzer/services/validation_service.py
"""Validation service for PR group validation and refinement."""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from crewai_approach.tools.group_validator_tool import GroupValidatorTool
from crewai_approach.tools.group_refiner_tool import GroupRefinerTool

logger = logging.getLogger(__name__)

class ValidationService:
    """Service for coordinating PR validation operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_cache = {}
        
    def _get_tool(self, tool_class, repo_path: str):
        """Get or create a tool instance."""
        cache_key = f"{tool_class.__name__}:{repo_path}"
        if cache_key not in self.tool_cache:
            self.tool_cache[cache_key] = tool_class(repo_path)
        return self.tool_cache[cache_key]
    
    async def validate_groups(
        self,
        pr_grouping_strategy: Dict[str, Any],
        is_final_validation: bool = False
    ) -> Dict[str, Any]:
        """Validate suggested PR groupings."""
        try:
            # Extract repo path from strategy
            repo_path = pr_grouping_strategy.get("repo_path", "")
            if not repo_path and "groups" in pr_grouping_strategy:
                # Try to extract from first group
                groups = pr_grouping_strategy["groups"]
                if groups and isinstance(groups[0], dict):
                    first_file = groups[0].get("files", [""])[0]
                    repo_path = str(Path(first_file).parent.parent) if first_file else ""
            
            validator = self._get_tool(GroupValidatorTool, repo_path or ".")
            
            result_json = validator._run(
                pr_grouping_strategy_json=json.dumps(pr_grouping_strategy),
                is_final_validation=is_final_validation
            )
            
            return json.loads(result_json)
            
        except Exception as e:
            logger.error(f"Error validating groups: {e}", exc_info=True)
            raise
    
    async def refine_groups(
        self,
        pr_grouping_strategy: Dict[str, Any],
        validation_result: Dict[str, Any],
        original_repository_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Refine PR groups based on validation results."""
        try:
            # Extract repo path
            repo_path = pr_grouping_strategy.get("repo_path", "")
            if not repo_path and original_repository_analysis:
                repo_path = original_repository_analysis.get("repo_path", "")
            
            refiner = self._get_tool(GroupRefinerTool, repo_path or ".")
            
            result_json = refiner._run(
                pr_grouping_strategy_json=json.dumps(pr_grouping_strategy),
                pr_validation_result_json=json.dumps(validation_result),
                original_repository_analysis_json=json.dumps(original_repository_analysis) if original_repository_analysis else None
            )
            
            return json.loads(result_json)
            
        except Exception as e:
            logger.error(f"Error refining groups: {e}", exc_info=True)
            raise
    
    async def check_completeness(
        self,
        pr_grouping_strategy: Dict[str, Any],
        original_repository_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if all files from original analysis are included in PR groups."""
        try:
            # Extract all files from original analysis
            original_files = set()
            if "file_changes" in original_repository_analysis:
                for fc in original_repository_analysis["file_changes"]:
                    if "path" in fc:
                        original_files.add(fc["path"])
            
            # Extract all files from PR groups
            grouped_files = set()
            if "groups" in pr_grouping_strategy:
                for group in pr_grouping_strategy["groups"]:
                    if "files" in group:
                        grouped_files.update(group["files"])
            
            # Find missing files
            missing_files = original_files - grouped_files
            
            return {
                "is_complete": len(missing_files) == 0,
                "total_files": len(original_files),
                "grouped_files": len(grouped_files),
                "missing_files": list(missing_files),
                "completeness_percentage": (len(grouped_files) / len(original_files) * 100) if original_files else 100
            }
            
        except Exception as e:
            logger.error(f"Error checking completeness: {e}", exc_info=True)
            raise