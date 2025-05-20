import jwt
from typing import Dict, Optional, Any, List
import asyncio
from functools import wraps

class AuthenticationError(Exception):
    """Custom exception for authentication failures"""
    pass

class AuthorizationError(Exception):
    """Custom exception for authorization failures"""
    pass

class SecurityManager:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.public_key = self._load_public_key()
        self.allowed_tools = self.config.get("allowed_tools", {})
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads security configuration from a YAML file."""
        # This is a placeholder. In a real implementation, you'd load and parse the YAML.
        # For now, returning a dummy structure based on the plan.
        return {
            "authentication": {
                "provider": "jwt",
                "public_key_path": "/secrets/jwt_public.pem",
                "issuer": "pr-generator-auth",
                "audience": "pr-generator-mcp"
            },
            "authorization": {
                "default_policy": "deny",
                "tool_permissions": {
                    "analyze_repository": {
                        "roles": ["developer", "admin"],
                        "scopes": ["repo:read"]
                    },
                    "suggest_pr_boundaries": {
                        "roles": ["developer", "admin"],
                        "scopes": ["repo:read", "ai:use"]
                    },
                    "create_pull_request": {
                        "roles": ["developer", "admin"],
                        "scopes": ["repo:write", "pr:create"]
                    }
                },
                "rate_limits": [
                    {"role": "developer", "requests_per_hour": 100},
                    {"role": "admin", "requests_per_hour": 1000}
                ]
            },
            "audit": {
                "enabled": True,
                "log_level": "info",
                "sensitive_fields": ["auth_token", "api_key"]
            }
        }

    def _load_public_key(self) -> str:
        """Loads the public key for JWT verification."""
        # This is a placeholder. In a real implementation, you'd load the key from a file.
        return "dummy_public_key" # Replace with actual key loading

    def _get_user_roles(self, user_id: str) -> List[str]:
        """Retrieves user roles (placeholder)."""
        # This is a placeholder. In a real implementation, you'd fetch user roles from a user management system.
        # For now, assume a default role for demonstration.
        return ["developer"]

    def authenticate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            # In a real implementation, you'd use the loaded public_key
            payload = jwt.decode(
                token,
                self.public_key, # Use the loaded public key
                algorithms=["RS256"],
                issuer=self.config["authentication"]["issuer"],
                audience=self.config["authentication"]["audience"]
            )
            return payload
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def authorize_tool_access(self, user_id: str, tool_name: str) -> bool:
        """Check if user can access specific tool"""
        user_roles = self._get_user_roles(user_id)
        tool_requirements = self.allowed_tools.get(tool_name, {})
        
        required_roles = tool_requirements.get("roles", [])
        
        # Check if user has any of the required roles
        if required_roles and not any(role in user_roles for role in required_roles):
            return False
            
        # Add more sophisticated scope/permission checks here if needed
        
        return True
    
    def secure_tool(self, tool_name: str):
        """Decorator to secure tool access"""
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                # Extract auth context from MCP request
                auth_context = kwargs.get("_mcp_auth_context", {})
                user_id = auth_context.get("user_id")
                
                if not user_id:
                    raise AuthenticationError("No authentication provided")
                
                if not self.authorize_tool_access(user_id, tool_name):
                    raise AuthorizationError(
                        f"User {user_id} not authorized for tool {tool_name}"
                    )
                
                return await func(self, *args, **kwargs)
            return wrapper
        return decorator
