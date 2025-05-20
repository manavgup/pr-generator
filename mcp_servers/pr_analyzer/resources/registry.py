# mcp_servers/pr_analyzer/resources/registry.py
"""Resource registry for the MCP server."""
import logging
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from mcp_servers.pr_analyzer.resources.templates import register_template_resources
from mcp_servers.pr_analyzer.resources.strategies import register_strategy_resources
from mcp_servers.pr_analyzer.resources.rules import register_rule_resources

logger = logging.getLogger(__name__)

def register_resources(mcp: FastMCP, config: Dict[str, Any]):
    """Register all resources with the MCP server."""
    resource_config = config.get('resources', {})
    
    # Keep track of registered resources
    registered_resources = []
    
    # Register each resource category
    resource_registries = [
        ('templates', register_template_resources),
        ('strategies', register_strategy_resources),
        ('rules', register_rule_resources)
    ]
    
    for category, register_func in resource_registries:
        try:
            logger.info(f"Registering {category} resources...")
            count = register_func(mcp, resource_config, config)
            registered_resources.extend([f"{category}.{name}" for name in count])
            logger.info(f"Successfully registered {len(count)} {category} resources")
        except Exception as e:
            logger.error(f"Failed to register {category} resources: {e}", exc_info=True)
            if config.get('strict_mode', False):
                raise
    
    logger.info(f"Total resources registered: {len(registered_resources)}")
    return registered_resources