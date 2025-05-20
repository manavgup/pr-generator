# mcp_servers/pr_analyzer/resources/templates.py
"""PR template resources."""
import logging
from typing import Dict, Any, List, Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger(__name__)

# Template definitions
TEMPLATES = {
    "standard": """## Description
{description}

## Changes
{changes}

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
""",
    "minimal": """{description}

Changes:
{changes}
""",
    "detailed": """## Summary
{summary}

## Motivation
{motivation}

## Changes
{changes}

## Testing Strategy
{testing_strategy}

## Performance Impact
{performance_impact}

## Security Considerations
{security_considerations}

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Performance impact assessed
"""
}

def register_template_resources(mcp: FastMCP, resource_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
    """Register template resources."""
    registered = []
    
    @mcp.resource("templates://list")
    async def list_templates() -> Dict[str, Any]:
        """List available PR templates."""
        return {
            "templates": list(TEMPLATES.keys()),
            "default": resource_config.get("default", "standard")
        }
    
    registered.append("list")
    
    @mcp.resource("templates://{template_name}")
    async def get_template(template_name: Annotated[str, Field(description="Name of the template to retrieve")]) -> str:
        """Get a specific PR template."""
        if template_name not in TEMPLATES:
            available = ", ".join(TEMPLATES.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        return TEMPLATES[template_name]
    
    registered.append("get")
    
    @mcp.resource("templates://config")
    async def get_template_config() -> Dict[str, Any]:
        """Get template configuration."""
        return {
            "default": resource_config.get("default", "standard"),
            "available": list(TEMPLATES.keys()),
            "custom_enabled": resource_config.get("custom_enabled", False)
        }
    
    registered.append("config")
    
    return registered
