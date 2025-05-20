# mcp_servers/pr_analyzer/resources/rules.py
"""Validation rule resources."""
import logging
from typing import Dict, Any, List, Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

logger = logging.getLogger(__name__)

# Validation rule definitions
VALIDATION_RULES = {
    "size_check": {
        "name": "PR Size Check",
        "description": "Validates that PRs are not too large",
        "parameters": {
            "max_files": 50,
            "max_size_mb": 100
        },
        "severity": "warning"
    },
    "conflict_check": {
        "name": "Conflict Check",
        "description": "Checks for potential merge conflicts",
        "parameters": {
            "check_dependencies": True
        },
        "severity": "error"
    },
    "test_coverage": {
        "name": "Test Coverage",
        "description": "Ensures test files are included when needed",
        "parameters": {
            "require_tests": True,
            "test_patterns": ["test_", "_test.py", ".test."]
        },
        "severity": "warning"
    },
    "file_duplication": {
        "name": "File Duplication",
        "description": "Prevents files from appearing in multiple PRs",
        "parameters": {},
        "severity": "error"
    },
    "empty_groups": {
        "name": "Empty Groups",
        "description": "Removes empty PR groups",
        "parameters": {},
        "severity": "error"
    }
}

def register_rule_resources(mcp: FastMCP, resource_config: Dict[str, Any], global_config: Dict[str, Any]) -> List[str]:
    """Register validation rule resources."""
    registered = []
    
    @mcp.resource("rules://list")
    async def list_rules() -> Dict[str, Any]:
        """List available validation rules."""
        return {
            "rules": list(VALIDATION_RULES.keys()),
            "enabled": resource_config.get("enabled_rules", list(VALIDATION_RULES.keys()))
        }
    
    registered.append("list")
    
    @mcp.resource("rules://{rule_name}")
    async def get_rule_info(rule_name: Annotated[str, Field(description="Name of the rule to retrieve")]) -> Dict[str, Any]:
        """Get information about a specific validation rule."""
        if rule_name not in VALIDATION_RULES:
            available = ", ".join(VALIDATION_RULES.keys())
            raise ValueError(f"Rule '{rule_name}' not found. Available: {available}")
        
        return VALIDATION_RULES[rule_name]
    
    registered.append("get")
    
    @mcp.resource("rules://config")
    async def get_rules_config() -> Dict[str, Any]:
        """Get validation rules configuration."""
        enabled_rules = resource_config.get("enabled_rules", list(VALIDATION_RULES.keys()))
        
        config = {}
        for rule_name in enabled_rules:
            if rule_name in VALIDATION_RULES:
                rule = VALIDATION_RULES[rule_name]
                config[rule_name] = {
                    "enabled": True,
                    "severity": rule["severity"],
                    "parameters": rule["parameters"]
                }
        
        return config
    
    registered.append("config")
    
    return registered
