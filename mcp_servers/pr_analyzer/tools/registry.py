# mcp_servers/pr_analyzer/tools/registry.py
"""Tool registry for the MCP server."""
import logging
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from mcp_servers.pr_analyzer.tools.analysis import register_analysis_tools
from mcp_servers.pr_analyzer.tools.grouping import register_grouping_tools
from mcp_servers.pr_analyzer.tools.validation import register_validation_tools
from mcp_servers.pr_analyzer.tools.workflow import register_workflow_tools

logger = logging.getLogger(__name__)

def register_tools(mcp: FastMCP, config: Dict[str, Any]):
    """Register all tools with the MCP server."""
    tool_config = config.get('tools', {})
    
    # Keep track of registered tools
    registered_tools = []
    
    # Register each tool category
    tool_registries = [
        ('analysis', register_analysis_tools),
        ('grouping', register_grouping_tools),
        ('validation', register_validation_tools),
        ('workflow', register_workflow_tools)
    ]
    
    # Log the available tools in the server before registration
    logger.info(f"Available tools before registration: {mcp.list_tools()}")
    
    for category, register_func in tool_registries:
        try:
            logger.info(f"Registering {category} tools...")
            count = register_func(mcp, tool_config, config)
            if count:
                tool_names = [f"{category}.{name}" for name in count]
                registered_tools.extend(tool_names)
                logger.info(f"Successfully registered {len(count)} {category} tools: {', '.join(count)}")
            else:
                logger.warning(f"No {category} tools were registered")
        except Exception as e:
            logger.error(f"Failed to register {category} tools: {e}", exc_info=True)
            if config.get('strict_mode', False):
                raise
    
    # Log the available tools in the server after registration
    logger.info(f"Available tools after registration: {mcp.list_tools()}")
    logger.info(f"Total tools registered: {len(registered_tools)}")
    return registered_tools
