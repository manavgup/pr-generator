# mcp_servers/pr_analyzer/tools/validation.py
"""PR validation and refinement tools."""
import json
import logging
from typing import Dict, Any, Optional, List, Annotated

from fastmcp import FastMCP
from mcp_servers.pr_analyzer.services.validation_service import ValidationService
from pydantic import Field

logger = logging.getLogger(__name__)

def register_validation_tools(server: FastMCP, tool_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
    """Register validation-related tools."""
    # Create service instance
    validation_service = ValidationService(global_config)
    registered = []
    
    # Tool 1: validate_pr_groups
    async def validate_pr_groups_func(
        pr_grouping_strategy: Annotated[Dict[str, Any], Field(description="PR grouping strategy to validate")],
        is_final_validation: Annotated[bool, Field(description="Whether this is final validation after merging")] = False
    ) -> List[Dict[str, Any]]:
        """Validate a proposed PR grouping strategy."""
        try:
            result = await validation_service.validate_groups(
                pr_grouping_strategy=pr_grouping_strategy,
                is_final_validation=is_final_validation
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error validating PR groups: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Validation failed: {str(e)}"})}]
    
    server.tool()(validate_pr_groups_func)
    registered.append("validate_pr_groups")
    
    # Tool 2: refine_pr_groups
    async def refine_pr_groups_func(
        pr_grouping_strategy: Annotated[Dict[str, Any], Field(description="PR grouping strategy to refine")],
        validation_result: Annotated[Dict[str, Any], Field(description="Validation results with issues")],
        original_repository_analysis: Annotated[Optional[Dict[str, Any]], Field(description="Optional repository analysis for completeness check")] = None
    ) -> List[Dict[str, Any]]:
        """Refine PR groups based on validation results."""
        try:
            result = await validation_service.refine_groups(
                pr_grouping_strategy=pr_grouping_strategy,
                validation_result=validation_result,
                original_repository_analysis=original_repository_analysis
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error refining PR groups: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Refinement failed: {str(e)}"})}]
    
    server.tool()(refine_pr_groups_func)
    registered.append("refine_pr_groups")
    
    # Tool 3: check_pr_completeness
    async def check_pr_completeness_func(
        pr_grouping_strategy: Annotated[Dict[str, Any], Field(description="PR grouping strategy to check")],
        original_repository_analysis: Annotated[Dict[str, Any], Field(description="Original repository analysis")]
    ) -> List[Dict[str, Any]]:
        """Check if all files from original analysis are included in PR groups."""
        try:
            result = await validation_service.check_completeness(
                pr_grouping_strategy=pr_grouping_strategy,
                original_repository_analysis=original_repository_analysis
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error checking completeness: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Completeness check failed: {str(e)}"})}]
    
    server.tool()(check_pr_completeness_func)
    registered.append("check_pr_completeness")
    
    return registered
