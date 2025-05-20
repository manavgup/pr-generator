# mcp_servers/pr_analyzer/tools/grouping.py
"""PR grouping tools."""
import json
import logging
from typing import Dict, Any, Optional, List, Annotated

from fastmcp import FastMCP
from mcp_servers.pr_analyzer.services.grouping_service import GroupingService
from pydantic import Field

logger = logging.getLogger(__name__)

def register_grouping_tools(server: FastMCP, tool_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
    """Register grouping-related tools."""
    # Create service instance
    grouping_service = GroupingService(global_config)
    registered = []
    
    # Tool 1: suggest_pr_boundaries
    async def suggest_pr_boundaries_func(
        analysis: Annotated[Dict[str, Any], Field(description="Repository analysis data")],
        strategy: Annotated[Optional[str], Field(description="Grouping strategy (directory, feature, module, balanced, mixed)", enum=["directory", "feature", "module", "balanced", "mixed", None])] = None,
        max_files_per_pr: Annotated[int, Field(description="Maximum files per PR")] = 30,
        target_batch_size: Annotated[int, Field(description="Target batch size for processing")] = 50
    ) -> List[Dict[str, Any]]:
        """Suggest logical PR boundaries based on repository analysis."""
        try:
            result = await grouping_service.suggest_pr_boundaries(
                repository_analysis=analysis,
                strategy=strategy,
                max_files_per_pr=max_files_per_pr,
                target_batch_size=target_batch_size
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error suggesting PR boundaries: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Failed to suggest PR boundaries: {str(e)}"})}]
    
    server.tool()(suggest_pr_boundaries_func)
    registered.append("suggest_pr_boundaries")
    
    # Tool 2: select_grouping_strategy
    async def select_grouping_strategy_func(
        repository_analysis: Annotated[Dict[str, Any], Field(description="Repository analysis data")],
        repository_metrics: Annotated[Optional[Dict[str, Any]], Field(description="Optional pre-calculated metrics")] = None,
        pattern_analysis: Annotated[Optional[Dict[str, Any]], Field(description="Optional pattern analysis results")] = None
    ) -> List[Dict[str, Any]]:
        """Select optimal grouping strategy based on repository characteristics."""
        try:
            result = await grouping_service.select_grouping_strategy(
                repository_analysis=repository_analysis,
                repository_metrics=repository_metrics,
                pattern_analysis=pattern_analysis
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Strategy selection failed: {str(e)}"})}]
    
    server.tool()(select_grouping_strategy_func)
    registered.append("select_grouping_strategy")
    
    # Tool 3: split_into_batches
    async def split_into_batches_func(
        repository_analysis: Annotated[Dict[str, Any], Field(description="Repository analysis data")],
        pattern_analysis: Annotated[Optional[Dict[str, Any]], Field(description="Optional pattern analysis results")] = None,
        target_batch_size: Annotated[int, Field(description="Target number of files per batch")] = 50
    ) -> List[Dict[str, Any]]:
        """Split files into manageable batches for processing."""
        try:
            result = await grouping_service.split_into_batches(
                repository_analysis=repository_analysis,
                pattern_analysis=pattern_analysis,
                target_batch_size=target_batch_size
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error splitting batches: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Batch splitting failed: {str(e)}"})}]
    
    server.tool()(split_into_batches_func)
    registered.append("split_into_batches")
    
    # Tool 4: merge_batch_results
    async def merge_batch_results_func(
        batch_results: Annotated[List[Dict[str, Any]], Field(description="List of batch processing results")],
        original_analysis: Annotated[Dict[str, Any], Field(description="Original repository analysis")]
    ) -> List[Dict[str, Any]]:
        """Merge PR grouping results from multiple batches."""
        try:
            result = await grouping_service.merge_batch_results(
                batch_results=batch_results,
                original_analysis=original_analysis
            )
            
            return [{"type": "text", "text": json.dumps(result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error merging results: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Result merging failed: {str(e)}"})}]
    
    server.tool()(merge_batch_results_func)
    registered.append("merge_batch_results")
    
    return registered
