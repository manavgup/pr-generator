#!/usr/bin/env python3
"""
PR Analyzer MCP Server - Main Entry Point
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Use fastmcp imports
from fastmcp import FastMCP

# Import our modules
from mcp_servers.pr_analyzer.tools.registry import register_tools
from mcp_servers.pr_analyzer.resources.registry import register_resources
from mcp_servers.pr_analyzer.config.loader import load_config, setup_logging

logger = logging.getLogger(__name__)

# Load configuration
logger.info("Loading configuration from server.yaml...")
config = load_config('server.yaml')
logger.info(f"Configuration loaded: {config.keys()}")

# Setup logging
logger.info("Setting up logging...")
setup_logging(config.get('logging', {}))
logger.info("Logging setup complete")

# Create the MCP server instance as a global variable
server_config = config.get('server', {})
logger.info(f"Creating FastMCP server with config: {server_config}")
server = FastMCP(
    name=server_config.get('name', 'PR Analyzer'),
    instructions=server_config.get('instructions', 'This server provides PR analysis tools and resources.')
)
logger.info(f"FastMCP server created with name: {server.name}")

# Register all components
logger.info("Registering MCP tools...")
registered_tools = register_tools(server, config)
logger.info(f"Registered tools: {registered_tools}")

logger.info("Registering MCP resources...")
registered_resources = register_resources(server, config)
logger.info(f"Registered resources: {registered_resources}")

# Get the tools
available_tools = server.list_tools()
logger.info(f"Available tools after registration: {available_tools}")

# FIXED: Extract just the tool names for comparison
tool_names = [tool.name for tool in available_tools if hasattr(tool, 'name')]
logger.info(f"Tool names: {tool_names}")

# FIXED: Check if the tools exist by name
if not tool_names:
    logger.warning("No tools are available in the server after registration!")
elif "analyze_repository" not in tool_names and "analysis.analyze_repository" not in tool_names:
    logger.warning("Neither 'analyze_repository' nor 'analysis.analyze_repository' tool is available in the server!")
else:
    logger.info("Repository analysis tools are available")

logger.info("Starting PR Analyzer MCP Server...")

if __name__ == "__main__":
    # Run the server with default STDIO transport
    server.run()
    # Alternatively, to use HTTP transport:
    # server.run(transport="streamable-http", host="127.0.0.1", port=9000)