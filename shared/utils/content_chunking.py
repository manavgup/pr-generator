# --- Text Chunking ---
from typing import List
from logging_utils import get_logger

logger = get_logger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks based on character count.
    A simple chunking strategy. More sophisticated methods (sentence, token)
    could be added later.

    Args:
        text: The input text string.
        chunk_size: The target size of each chunk (in characters).
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If chunk_size is not positive or overlap is negative/too large.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and less than chunk size.")

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        next_start = start + chunk_size - overlap
        if next_start <= start:  # Prevent infinite loop if overlap is too large relative to step
            next_start = start + 1
        start = next_start
        # Ensure we don't go past the end point if the last chunk is smaller
        if start >= len(text):
            break

    logger.debug(f"Chunked text of length {len(text)} into {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks
