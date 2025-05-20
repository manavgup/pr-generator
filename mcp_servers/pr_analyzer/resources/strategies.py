# mcp_servers/pr_analyzer/resources/strategies.py
"""Strategy information resources."""
import logging
from typing import Dict, Any, List, Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger(__name__)

# Strategy definitions
STRATEGIES = {
    "directory": {
        "name": "Directory-based",
        "description": "Groups files by their directory structure",
        "pros": [
            "Natural organization following project structure",
            "Easy to understand and review",
            "Maintains logical boundaries"
        ],
        "cons": [
            "May split related changes across directories",
            "Can create very small or very large PRs"
        ],
        "best_for": "Well-organized projects with clear directory structure"
    },
    "feature": {
        "name": "Feature-based",
        "description": "Groups files that implement the same feature",
        "pros": [
            "Keeps related functionality together",
            "Easier to understand the purpose of changes",
            "Better for cross-cutting concerns"
        ],
        "cons": [
            "Requires pattern analysis",
            "May be less accurate for unfamiliar codebases"
        ],
        "best_for": "Feature development and refactoring"
    },
    "module": {
        "name": "Module-based",
        "description": "Groups files by their module or file type",
        "pros": [
            "Groups similar file types together",
            "Good for technology-specific reviews",
            "Consistent grouping approach"
        ],
        "cons": [
            "May separate related functionality",
            "Less intuitive for feature changes"
        ],
        "best_for": "Technology upgrades or module-specific changes"
    },
    "balanced": {
        "name": "Size-balanced",
        "description": "Creates evenly-sized PRs for easier review",
        "pros": [
            "Consistent PR sizes",
            "Easier to review",
            "Predictable review time"
        ],
        "cons": [
            "May split logical units",
            "Less semantic meaning"
        ],
        "best_for": "Large changesets that need to be split"
    },
    "mixed": {
        "name": "Mixed strategy",
        "description": "Combines multiple strategies for optimal grouping",
        "pros": [
            "Flexible approach",
            "Can adapt to different change patterns",
            "Best of multiple strategies"
        ],
        "cons": [
            "More complex logic",
            "Less predictable results"
        ],
        "best_for": "Complex projects with varied change patterns"
    }
}

def register_strategy_resources(mcp: FastMCP, resource_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
    """Register strategy resources."""
    registered = []
    
    @mcp.resource("strategies://list")
    async def list_strategies() -> Dict[str, Any]:
        """List available grouping strategies."""
        return {
            "strategies": list(STRATEGIES.keys()),
            "default": global_config.get("strategies", {}).get("default", "directory")
        }
    
    registered.append("list")
    
    @mcp.resource("strategies://{strategy_name}")
    async def get_strategy_info(strategy_name: Annotated[str, Field(description="Name of the strategy to retrieve")]) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        if strategy_name not in STRATEGIES:
            available = ", ".join(STRATEGIES.keys())
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")
        
        return STRATEGIES[strategy_name]
    
    registered.append("get")
    
    @mcp.resource("strategies://comparison")
    async def compare_strategies() -> Dict[str, Any]:
        """Get a comparison of all strategies."""
        comparison = {}
        for name, info in STRATEGIES.items():
            comparison[name] = {
                "description": info["description"],
                "best_for": info["best_for"]
            }
        return comparison
    
    registered.append("comparison")
    
    return registered
