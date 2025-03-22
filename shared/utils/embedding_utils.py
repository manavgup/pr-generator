"""
Utilities for creating embeddings for file changes in PR generation.
"""
import logging
import os
from typing import List, Dict, Any
import json

from shared.models.pr_models import FileChange
from shared.config.llm_config import LLMProvider

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available for embeddings")


def create_embedding(text: str, api_key: str = None, model: str = "text-embedding-3-small") -> List[float]:
    """
    Create an embedding vector for the given text using OpenAI's API.
    
    Args:
        text: Text to embed
        api_key: OpenAI API key (optional, will use environment variable if not provided)
        model: Embedding model to use
        
    Returns:
        List of floats representing the embedding vector
    """
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI package not installed, cannot create embeddings")
        return []
    
    try:
        # Use provided API key or fall back to environment variable
        client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return []


def create_file_change_embedding(change: FileChange, api_key: str = None) -> List[float]:
    """
    Create an embedding that represents both the file identity and its changes.
    
    Args:
        change: FileChange object to embed
        api_key: OpenAI API key (optional)
        
    Returns:
        List of floats representing the embedding vector
    """
    # Build a representation of the change that captures both identity and diff
    content_parts = [
        f"File: {change.file_path}",
        f"Directory: {change.directory}",
        f"Extension: {change.extension or 'none'}",
        f"Status: {change.status or 'modified'}",
        f"Lines Added: {change.changes.added}",
        f"Lines Deleted: {change.changes.deleted}"
    ]
    
    # Include the full diff for completeness
    if change.diff:
        content_parts.append("Diff:")
        content_parts.append(change.diff)
    
    # Join everything into a single text
    embedding_text = "\n".join(content_parts)
    
    # Create and return the embedding
    return create_embedding(embedding_text, api_key)


def batch_create_embeddings(changes: List[FileChange], api_key: str = None) -> Dict[str, List[float]]:
    """
    Create embeddings for a batch of file changes.
    
    Args:
        changes: List of FileChange objects to embed
        api_key: OpenAI API key (optional)
        
    Returns:
        Dictionary mapping file paths to embedding vectors
    """
    embeddings = {}
    total_changes = len(changes)
    
    logger.info(f"Creating embeddings for {total_changes} file changes")
    
    for i, change in enumerate(changes):
        if i % 20 == 0:  # Log progress every 20 files
            logger.info(f"Processing embedding {i+1}/{total_changes}")
            
        embedding = create_file_change_embedding(change, api_key)
        embeddings[change.file_path] = embedding
    
    logger.info(f"Created {len(embeddings)} embeddings")
    return embeddings


def save_embeddings(embeddings: Dict[str, List[float]], output_path: str) -> None:
    """
    Save embeddings to a file.
    
    Args:
        embeddings: Dictionary mapping file paths to embedding vectors
        output_path: Path to save embeddings to
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(embeddings, f)
        logger.info(f"Saved embeddings to {output_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")


def load_embeddings(input_path: str) -> Dict[str, List[float]]:
    """
    Load embeddings from a file.
    
    Args:
        input_path: Path to load embeddings from
        
    Returns:
        Dictionary mapping file paths to embedding vectors
    """
    try:
        with open(input_path, 'r') as f:
            embeddings = json.load(f)
        logger.info(f"Loaded embeddings for {len(embeddings)} files from {input_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return {}