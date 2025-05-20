# mcp_servers/pr_analyzer/services/grouping_service.py
"""Grouping service for PR boundary suggestions."""
import json
import logging
from typing import Dict, Any, Optional, List

from crewai_approach.tools.grouping_strategy_selector_tool import GroupingStrategySelector
from crewai_approach.tools.batch_splitter_tool import BatchSplitterTool
from crewai_approach.tools.batch_processor_tool import BatchProcessorTool
from crewai_approach.tools.group_merging_tool import GroupMergingTool

logger = logging.getLogger(__name__)

class GroupingService:
    """Service for coordinating PR grouping operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_cache = {}
        
    def _get_tool(self, tool_class, repo_path: str):
        """Get or create a tool instance."""
        cache_key = f"{tool_class.__name__}:{repo_path}"
        if cache_key not in self.tool_cache:
            self.tool_cache[cache_key] = tool_class(repo_path)
        return self.tool_cache[cache_key]
    
    async def suggest_pr_boundaries(
        self,
        repository_analysis: Dict[str, Any],
        strategy: Optional[str] = None,
        max_files_per_pr: int = 30,
        target_batch_size: int = 50
    ) -> Dict[str, Any]:
        """Suggest logical PR boundaries based on repository analysis."""
        try:
            repo_path = repository_analysis.get("repo_path", "")
            
            # Step 1: Select strategy if not provided
            if not strategy:
                strategy_result = await self.select_grouping_strategy(repository_analysis)
                strategy = strategy_result.get("strategy_type", "mixed")
            
            # Step 2: Split into batches
            batch_result = await self.split_into_batches(
                repository_analysis=repository_analysis,
                target_batch_size=target_batch_size
            )
            
            # Step 3: Process batches
            processor = self._get_tool(BatchProcessorTool, repo_path)
            batch_results_json = processor._run(
                batch_splitter_output_json=json.dumps(batch_result),
                grouping_strategy_decision_json=json.dumps({"strategy_type": strategy}),
                repository_analysis_json=json.dumps(repository_analysis)
            )
            
            batch_results = json.loads(batch_results_json)
            
            # Step 4: Merge results
            merged_result = await self.merge_batch_results(
                batch_results=batch_results,
                original_analysis=repository_analysis
            )
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Error suggesting PR boundaries: {e}", exc_info=True)
            raise
    
    async def select_grouping_strategy(
        self,
        repository_analysis: Dict[str, Any],
        repository_metrics: Optional[Dict[str, Any]] = None,
        pattern_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Select optimal grouping strategy."""
        try:
            repo_path = repository_analysis.get("repo_path", "")
            selector = self._get_tool(GroupingStrategySelector, repo_path)
            
            result_json = selector._run(
                repository_analysis_json=json.dumps(repository_analysis),
                repository_metrics_json=json.dumps(repository_metrics) if repository_metrics else None,
                pattern_analysis_json=json.dumps(pattern_analysis) if pattern_analysis else None
            )
            
            return json.loads(result_json)
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}", exc_info=True)
            raise
    
    async def split_into_batches(
        self,
        repository_analysis: Dict[str, Any],
        pattern_analysis: Optional[Dict[str, Any]] = None,
        target_batch_size: int = 50
    ) -> Dict[str, Any]:
        """Split files into manageable batches."""
        try:
            repo_path = repository_analysis.get("repo_path", "")
            splitter = self._get_tool(BatchSplitterTool, repo_path)
            
            result_json = splitter._run(
                repository_analysis_json=json.dumps(repository_analysis),
                pattern_analysis_json=json.dumps(pattern_analysis) if pattern_analysis else None,
                target_batch_size=target_batch_size
            )
            
            return json.loads(result_json)
            
        except Exception as e:
            logger.error(f"Error splitting batches: {e}", exc_info=True)
            raise
    
    async def merge_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        original_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge PR grouping results from multiple batches."""
        try:
            repo_path = original_analysis.get("repo_path", "")
            merger = self._get_tool(GroupMergingTool, repo_path)
            
            result_json = merger._run(
                batch_grouping_results_json=json.dumps(batch_results),
                original_repository_analysis_json=json.dumps(original_analysis)
            )
            
            return json.loads(result_json)
            
        except Exception as e:
            logger.error(f"Error merging results: {e}", exc_info=True)
            raise
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        strategies = self.config.get("strategies", {})
        if strategy_name in strategies:
            return strategies[strategy_name]
        return {"error": f"Strategy '{strategy_name}' not found"}