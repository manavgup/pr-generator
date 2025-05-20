from typing import Callable, Any
import asyncio
from functools import wraps
import logging

class MCPError(Exception):
    """Base exception for MCP operations"""
    pass

class ToolExecutionError(MCPError):
    """Error during tool execution"""
    pass

class ConnectionError(MCPError):
    """MCP connection error"""
    pass

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retry_config = {
            "max_attempts": 3,
            "backoff_factor": 2,
            "max_delay": 30
        }
    
    def with_retry(self, exceptions=(Exception,)):
        """Decorator for automatic retry with exponential backoff"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                delay = 1
                
                for attempt in range(self.retry_config["max_attempts"]):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < self.retry_config["max_attempts"] - 1:
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s"
                            )
                            await asyncio.sleep(delay)
                            delay = min(
                                delay * self.retry_config["backoff_factor"],
                                self.retry_config["max_delay"]
                            )
                        else:
                            self.logger.error(f"All attempts failed: {e}")
                
                raise last_exception
            return wrapper
        return decorator
    
    async def handle_tool_error(self, tool_name: str, error: Exception):
        """Centralized tool error handling"""
        error_context = {
            "tool": tool_name,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        
        self.logger.error(f"Tool error: {error_context}")
        
        # Send to monitoring system
        await self.send_to_monitoring(error_context)
        
        # Determine if error is recoverable
        if isinstance(error, ConnectionError):
            raise MCPError("Connection lost. Please retry.")
        else:
            raise ToolExecutionError(f"Tool {tool_name} failed: {error}")
