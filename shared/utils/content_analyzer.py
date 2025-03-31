# content_analyzer.py
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from utils.content_chunking import chunk_text
from models.git_models import FileChange, LineChanges, FileStatusType
from models.directory_models import DirectorySummary
from utils.logging_utils import get_logger
from utils.cache_utils import summarize_diff, fingerprint_change

logger = get_logger(__name__)

# --- Directory Summarization ---

def summarize_by_directory(changes: List[FileChange]) -> List[DirectorySummary]:
    """
    Groups file changes by directory and calculates summary statistics.

    Args:
        changes: A list of FileChange objects.

    Returns:
        A list of DirectorySummary objects.
    """
    dir_map: Dict[Path, DirectorySummary] = defaultdict(lambda: DirectorySummary(directory_path=Path(), files=[])) # type: ignore # Pydantic complains about lambda

    for change in changes:
        # Use the directory property from the FileChange model
        dir_path = change.directory
        if dir_path not in dir_map:
             # Initialize with the correct path the first time
             dir_map[dir_path] = DirectorySummary(directory_path=dir_path, files=[])

        dir_summary = dir_map[dir_path]
        dir_summary.file_count += 1
        dir_summary.files.append(change.path)
        if change.changes:
            dir_summary.total_added += change.changes.added
            dir_summary.total_deleted += change.changes.deleted

    logger.info(f"Summarized changes into {len(dir_map)} directories.")
    return list(dir_map.values())

# --- Example Usage ---
if __name__ == "__main__":
    # This block is for demonstration; actual usage would involve GitOperations
    # Creating mock FileChange objects
    mock_changes = [
        FileChange(path=Path("src/main.py"), staged_status=FileStatusType.NONE, unstaged_status=FileStatusType.MODIFIED, is_binary=False, changes=LineChanges(added=10, deleted=2)),
        FileChange(path=Path("src/utils/helpers.py"), staged_status=FileStatusType.NONE, unstaged_status=FileStatusType.MODIFIED, is_binary=False, changes=LineChanges(added=5, deleted=0)),
        FileChange(path=Path("tests/test_main.py"), staged_status=FileStatusType.ADDED, unstaged_status=FileStatusType.NONE, is_binary=False, changes=LineChanges(added=25, deleted=0)),
        FileChange(path=Path("README.md"), staged_status=FileStatusType.NONE, unstaged_status=FileStatusType.MODIFIED, is_binary=False, changes=LineChanges(added=3, deleted=3)),
        FileChange(path=Path("src/config.py"), staged_status=FileStatusType.DELETED, unstaged_status=FileStatusType.NONE, is_binary=False, changes=LineChanges(added=0, deleted=50)),
    ]

    # 1. Directory Summary
    dir_summaries = summarize_by_directory(mock_changes)
    print("\n--- Directory Summaries ---")
    for summary in dir_summaries:
        print(summary.model_dump_json(indent=2))

    # 2. Chunking Example
    long_text = "This is a very long string designed to demonstrate the chunking functionality. " * 20
    print("\n--- Text Chunking ---")
    chunks = chunk_text(long_text, chunk_size=100, overlap=20)
    print(f"Original length: {len(long_text)}, Number of chunks: {len(chunks)}")
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1} (length {len(chunk)}):\n{chunk}\n---")

    # 3. Diff Summarization (Placeholder)
    mock_diff = """
diff --git a/src/main.py b/src/main.py
index abcdef1..1234567 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,5 +10,6 @@
 def main():
     print("Hello")
     # Added a new feature
+    new_feature()
     cleanup()
-    old_function() # Removed this
+    # Kept this comment
""" * 10 # Make it longer
    print("\n--- Diff Summarization (Placeholder) ---")
    summary = summarize_diff(mock_diff, max_length=200)
    print(summary)

    # 4. Fingerprinting (Placeholder)
    print("\n--- Content Fingerprinting (Placeholder) ---")
    fp1 = fingerprint_change(mock_diff)
    fp2 = fingerprint_change(mock_diff + " ") # Slightly different diff
    fp3 = fingerprint_change(mock_diff)
    print(f"Fingerprint 1: {fp1}")
    print(f"Fingerprint 2 (different): {fp2}")
    print(f"Fingerprint 3 (same as 1): {fp3}")
    assert fp1 == fp3
    assert fp1 != fp2