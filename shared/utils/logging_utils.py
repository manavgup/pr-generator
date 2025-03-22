"""
Logging utilities for PR generation.
"""
import functools
import logging
import os
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)

def configure_logging(log_file: str = None, verbose: bool = False):
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to log file (if None, only console logging is enabled)
        verbose: Whether to enable verbose logging
    """
    # Set the root logger level
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create a formatter that includes the logger name
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Configure file handler if a log file is specified
    if log_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries when in verbose mode
    if verbose:
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
    
    logger.info(f"Logging configured with verbose={verbose}, log_file={log_file}")

def log_operation(operation_name: str) -> Callable:
    """
    Decorator for logging operations with timing information.
    
    Args:
        operation_name: Name of the operation to log
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.info(f"Started: {operation_name}")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"Completed: {operation_name} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Failed: {operation_name} after {elapsed:.2f}s - {str(e)}")
                raise
                
        return wrapper
    return decorator

def log_llm_prompt(prompt_name: str, prompt_text: str, verbose: bool = True) -> None:
    """
    Log an LLM prompt if verbose mode is enabled.
    
    Args:
        prompt_name: Name of the prompt
        prompt_text: Text of the prompt
        verbose: Whether to actually log the prompt
    """
    if verbose:
        # Always log a basic message
        logger.info(f"Sending prompt: {prompt_name}")
        
        # Log the full prompt at debug level
        logger.info(f"LLM Prompt [{prompt_name}]:\n{'-'*40}\n{prompt_text}\n{'-'*40}")
    else:
        # Just log that a prompt is being sent without details
        logger.info(f"Sending prompt: {prompt_name}")

def log_llm_response(response_name: str, response_text: str, verbose: bool = False) -> None:
    """
    Log an LLM response if verbose mode is enabled.
    
    Args:
        response_name: Name of the response
        response_text: Text of the response
        verbose: Whether to actually log the response
    """
    if verbose:
        # Always log a basic message
        logger.info(f"Received response: {response_name}")
        
        # Log the full response at debug level
        logger.debug(f"LLM Response [{response_name}]:\n{'-'*40}\n{response_text}\n{'-'*40}")
    else:
        # Just log that a response was received without details
        logger.info(f"Received response: {response_name}")