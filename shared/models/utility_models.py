import time

from .base_models import BaseModel, ConfigDict

class ProgressReporter:
    """Simple progress reporter for CLI operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self._show_progress()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self._show_progress()
    
    def _show_progress(self):
        """Show progress."""
        if self.total == 0:
            return
        
        percent = min(100, int(100 * self.current / self.total))
        elapsed = time.time() - self.start_time
        
        if self.current > 0 and elapsed > 0:
            items_per_sec = self.current / elapsed
            eta = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0
            eta_str = f"ETA: {int(eta)}s" if eta > 0 else "ETA: done"
        else:
            eta_str = "ETA: calculating..."
        
        print(f"\r{self.description}: {self.current}/{self.total} ({percent}%) {eta_str}", end="", flush=True)
        
        if self.current >= self.total:
            elapsed_str = f"{elapsed:.1f}s"
            print(f"\r{self.description}: {self.current}/{self.total} (100%) completed in {elapsed_str}  ", flush=True)
            print() 
            