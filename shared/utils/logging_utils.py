"""
Enhanced logging configuration for PR generation.
"""
import os
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any

# Add file permission constants for Windows compatibility
LOG_DIR = "logs"

def configure_logging(log_file: Optional[str] = None, verbose: bool = False) -> logging.Logger:
    """
    Configure logging with simplified file handling.
    
    Args:
        log_file: Path to log file (if None, a timestamped file will be used)
        verbose: Whether to enable verbose logging
        
    Returns:
        The root logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Set default log file if not provided
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"pr_generator_{timestamp}.log")
    
    # Set the root logger level
    root_logger = logging.getLogger()
    
    # Remove existing handlers to prevent duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(log_level)
    
    # Create a formatter that includes the logger name and line number
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Configure file handler - simpler approach
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Log some initial messages
    root_logger.info(f"Logging configured: verbose={verbose}, log_file={log_file}")
    
    # Reduce noise from third-party libraries
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("litellm").setLevel(logging.INFO)
    
    # Set specific loggers for our own modules to INFO
    logging.getLogger("pr_generator").setLevel(logging.INFO)
    logging.getLogger("shared").setLevel(logging.INFO)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    Ensures consistent logging configuration throughout the application.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class LoggingContext:
    """
    Context manager for temporarily changing logging settings.
    
    Example:
        with LoggingContext(level=logging.DEBUG):
            # Code executed with DEBUG logging
        # Outside the context, original logging level is restored
    """
    def __init__(self, logger=None, level=None, handler=None, close=True):
        self.logger = logger or logging.getLogger()
        self.level = level
        self.handler = handler
        self.close = close
        self.old_level = self.logger.level
        self.old_handlers = self.logger.handlers.copy()
    
    def __enter__(self):
        if self.level is not None:
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)
        return self.logger
    
    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
            if self.close:
                self.handler.close()
        return False  # Don't suppress exceptions