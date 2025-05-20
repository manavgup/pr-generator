# mcp_servers/pr_analyzer/tools/workflow.py
"""Complete workflow tools."""
import json
import logging
from typing import Dict, Any, Optional, List, Annotated

from fastmcp import FastMCP
from mcp_servers.pr_analyzer.services.workflow_service import WorkflowService
from pydantic import Field

logger = logging.getLogger(__name__)

def register_workflow_tools(server: FastMCP, tool_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
    """Register workflow tools."""
    # Create service instance
    workflow_service = WorkflowService(global_config)
    registered = []
    
    # Tool 1: complete_pr_workflow
    async def complete_pr_workflow_func(
        repo_path: Annotated[str, Field(description="Path to the git repository")],
        strategy: Annotated[Optional[str], Field(description="Grouping strategy to use", enum=["directory", "feature", "module", "balanced", "mixed", None])] = None,
        max_files_per_pr: Annotated[int, Field(description="Maximum files per PR")] = 30,
        target_batch_size: Annotated[int, Field(description="Target batch size for processing")] = 50,
        validate: Annotated[bool, Field(description="Whether to validate groups")] = True,
        generate_metadata: Annotated[bool, Field(description="Whether to generate PR metadata")] = True
    ) -> List[Dict[str, Any]]:
        """Run the complete PR generation workflow."""
        try:
            result = await workflow_service.run_complete_workflow(
                repo_path=repo_path,
                strategy=strategy,
                max_files_per_pr=max_files_per_pr,
                target_batch_size=target_batch_size,
                validate=validate,
                generate_metadata=generate_metadata
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Workflow failed: {str(e)}"})}]
    
    server.tool()(complete_pr_workflow_func)
    registered.append("complete_pr_workflow")
    
    # Tool 2: generate_pr_metadata
    async def generate_pr_metadata_func(
        pr_group: Annotated[Dict[str, Any], Field(description="PR group to generate metadata for")],
        template: Annotated[str, Field(description="Template to use for generation", enum=["standard", "minimal", "detailed"])] = "standard",
        repository_analysis: Annotated[Optional[Dict[str, Any]], Field(description="Optional repository analysis for context")] = None
    ) -> List[Dict[str, Any]]:
        """Generate PR metadata including title, description, and labels."""
        try:
            result = await workflow_service.generate_pr_metadata(
                pr_group=pr_group,
                template=template,
                repository_analysis=repository_analysis
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error generating PR metadata: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Failed to generate PR metadata: {str(e)}"})}]
    
    server.tool()(generate_pr_metadata_func)
    registered.append("generate_pr_metadata")
    
    # Tool 3: export_pr_groups
    async def export_pr_groups_func(
        pr_grouping_strategy: Annotated[Dict[str, Any], Field(description="PR grouping strategy to export")],
        format: Annotated[str, Field(description="Export format", enum=["json", "markdown", "csv"])] = "json",
        include_diffs: Annotated[bool, Field(description="Whether to include file diffs")] = False
    ) -> List[Dict[str, Any]]:
        """Export PR groups in various formats."""
        try:
            result = await workflow_service.export_pr_groups(
                pr_grouping_strategy=pr_grouping_strategy,
                format=format,
                include_diffs=include_diffs
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error exporting PR groups: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Export failed: {str(e)}"})}]
    
    server.tool()(export_pr_groups_func)
    registered.append("export_pr_groups")
    
    return registered
