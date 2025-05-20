# mcp_servers/pr_analyzer/tools/analysis.py
"""Repository analysis tools."""
import json
import logging
from typing import Dict, Any, Optional, List, Annotated

from fastmcp import FastMCP
from mcp_servers.pr_analyzer.services.analysis_service import AnalysisService
from pydantic import Field

logger = logging.getLogger(__name__)

def to_serializable(obj):
    """Convert an object to a JSON serializable format."""
    if hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in obj.__dict__.items() 
                if not k.startswith("_")}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        try:
            # Check if obj is JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def register_analysis_tools(server: FastMCP, tool_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
    """Register analysis-related tools."""
    # Create service instance
    analysis_service = AnalysisService(global_config)
    registered = []
    
    logger.info("Registering analyze_repository tool...")
    
    # Tool 1: analyze_repository
    async def analyze_repository_func(
        repo_path: Annotated[str, Field(description="Path to the local git repository")],
        max_files: Annotated[Optional[int], Field(description="Maximum number of changed files to analyze fully")] = None,
        max_batch_size: Annotated[Optional[int], Field(description="Target maximum files per processing batch.")] = None,
        verbose: Annotated[int, Field(description="Increase verbosity level (0, 1, or 2).")] = 0,
        manager_llm: Annotated[Optional[str], Field(description="LLM model to use for the manager agent/process.")] = None
    ) -> List[Dict[str, Any]]:
        """Analyze uncommitted changes in a repository using the CrewAI script."""
        logger.info(f"analyze_repository_func called with repo_path={repo_path}, max_files={max_files}, verbose={verbose}")
        try:
            # Process streaming results into a final result
            all_results = []
            final_result = None
            
            # Collect streaming results
            async for content in analysis_service.analyze_repository(
                repo_path=repo_path,
                max_files=max_files,
                max_batch_size=max_batch_size,
                verbose=verbose,
                manager_llm=manager_llm
            ):
                logger.debug(f"Received content: {content}")
                
                # Store intermediate results
                if isinstance(content, dict):
                    # Convert any non-serializable content to serializable form
                    serializable_content = to_serializable(content)
                    all_results.append(serializable_content)
                    
                    # If this looks like a final result, store it separately
                    if "final_analysis" in content or "repository_analysis" in content:
                        final_result = serializable_content
            
            # Return the results in a consistent format
            if final_result:
                # If we have a clear final result, use that
                return [{"type": "text", "text": json.dumps(final_result, indent=2)}]
            elif all_results:
                # Otherwise return all collected results
                return [{"type": "text", "text": json.dumps(all_results, indent=2)}]
            else:
                # Fallback if no results were collected
                return [{"type": "text", "text": json.dumps({"status": "completed", "message": "Analysis completed but no results were returned"})}]
            
        except Exception as e:
            logger.error(f"Error analyzing repository: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Analysis failed: {str(e)}"})}]
    
    # Register the tool with the server
    logger.info("Applying server.tool() decorator to analyze_repository_func...")
    try:
        # Register with both names for compatibility
        server.tool(name="analyze_repository")(analyze_repository_func)
        logger.info("Successfully registered analyze_repository tool")
        registered.append("analyze_repository")
        
        # Also register with namespaced name for consistency
        server.tool(name="analysis.analyze_repository")(analyze_repository_func)
        logger.info("Successfully registered analysis.analyze_repository tool")
        registered.append("analysis.analyze_repository")
    except Exception as e:
        logger.error(f"Failed to register analyze_repository tool: {e}", exc_info=True)
    
    # Tool 2: analyze_patterns
    async def analyze_patterns_func(
        file_paths: Annotated[List[str], Field(description="List of file paths to analyze")],
        repository_path: Annotated[str, Field(description="Path to the repository")],
        directory_to_files: Annotated[Optional[Dict[str, List[str]]], Field(description="Optional mapping of directories to files")] = None
    ) -> List[Dict[str, Any]]:
        """Analyze patterns in file changes."""
        try:
            result = await analysis_service.analyze_patterns(
                file_paths=file_paths,
                repository_path=repository_path,
                directory_to_files=directory_to_files
            )
            
            # Convert result to serializable format
            serializable_result = to_serializable(result)
            return [{"type": "text", "text": json.dumps(serializable_result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Pattern analysis failed: {str(e)}"})}]
    
    # Register with both names for compatibility
    server.tool(name="analyze_patterns")(analyze_patterns_func)
    registered.append("analyze_patterns")
    
    server.tool(name="analysis.analyze_patterns")(analyze_patterns_func)
    registered.append("analysis.analyze_patterns")
    
    # Tool 3: calculate_metrics
    async def calculate_metrics_func(
        repository_analysis: Annotated[Dict[str, Any], Field(description="Repository analysis data")]
    ) -> List[Dict[str, Any]]:
        """Calculate repository metrics."""
        try:
            result = await analysis_service.calculate_metrics(
                repository_analysis=repository_analysis
            )
            
            # Convert result to serializable format
            serializable_result = to_serializable(result)
            return [{"type": "text", "text": json.dumps(serializable_result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Metrics calculation failed: {str(e)}"})}]
    
    # Register with both names for compatibility
    server.tool(name="calculate_metrics")(calculate_metrics_func)
    registered.append("calculate_metrics")
    
    server.tool(name="analysis.calculate_metrics")(calculate_metrics_func)
    registered.append("analysis.calculate_metrics")
    
    # Tool 4: analyze_directory_structure
    async def analyze_directory_structure_func(
        repository_analysis: Annotated[Dict[str, Any], Field(description="Repository analysis data")]
    ) -> List[Dict[str, Any]]:
        """Analyze directory structure and relationships."""
        try:
            result = await analysis_service.analyze_directory_structure(
                repository_analysis=repository_analysis
            )
            
            # Convert result to serializable format
            serializable_result = to_serializable(result)
            return [{"type": "text", "text": json.dumps(serializable_result, indent=2)}]
            
        except Exception as e:
            logger.error(f"Error analyzing directory structure: {e}", exc_info=True)
            return [{"type": "text", "text": json.dumps({"error": f"Directory analysis failed: {str(e)}"})}]
    
    # Register with both names for compatibility
    server.tool(name="analyze_directory_structure")(analyze_directory_structure_func)
    registered.append("analyze_directory_structure")
    
    server.tool(name="analysis.analyze_directory_structure")(analyze_directory_structure_func)
    registered.append("analysis.analyze_directory_structure")
    
    return registered