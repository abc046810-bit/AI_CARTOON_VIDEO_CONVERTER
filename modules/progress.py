"""Progress tracking and display for video processing."""
import time
from typing import Optional
from datetime import datetime

from .logger import get_logger

logger = get_logger("progress")


class ProgressTracker:
    """Track and display processing progress."""

    def __init__(self, total_chunks: int = 0, total_frames: int = 0):
        self.total_chunks = total_chunks
        self.total_frames = total_frames
        self.current_chunk = 0
        self.current_frame = 0
        self.chunk_frames_processed = 0
        self.chunk_total_frames = 0
        self.start_time = time.time()
        self.chunk_start_time = time.time()

    def start_chunk(self, chunk_idx: int, chunk_frames: int):
        """Start tracking a new chunk."""
        self.current_chunk = chunk_idx
        self.chunk_frames_processed = 0
        self.chunk_total_frames = chunk_frames
        self.chunk_start_time = time.time()

    def update_frame(self, count: int = 1):
        """Update frame progress."""
        self.chunk_frames_processed += count
        self.current_frame += count

    def get_eta(self) -> str:
        """Calculate ETA for current chunk."""
        if self.chunk_frames_processed == 0:
            return "calculating..."

        elapsed = time.time() - self.chunk_start_time
        fps = self.chunk_frames_processed / elapsed
        remaining = self.chunk_total_frames - self.chunk_frames_processed
        eta = remaining / fps if fps > 0 else 0

        minutes = int(eta // 60)
        seconds = int(eta % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_overall_eta(self) -> str:
        """Calculate overall ETA."""
        if self.current_frame == 0:
            return "calculating..."

        elapsed = time.time() - self.start_time
        fps = self.current_frame / elapsed
        remaining = self.total_frames - self.current_frame
        eta = remaining / fps if fps > 0 else 0

        hours = int(eta // 3600)
        minutes = int((eta % 3600) // 60)
        return f"{hours:02d}:{minutes:02d}"

    def print_status(self):
        """Print current processing status."""
        chunk_pct = (self.chunk_frames_processed / self.chunk_total_frames * 100) if self.chunk_total_frames > 0 else 0
        overall_pct = (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0

        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        print(f"\rChunk {self.current_chunk}/{self.total_chunks} | "
              f"Frame {self.chunk_frames_processed}/{self.chunk_total_frames} ({chunk_pct:.1f}%) | "
              f"Overall {overall_pct:.1f}% | ETA: {self.get_eta()} | Elapsed: {elapsed_str}", 
              end='', flush=True)

    def print_summary(self):
        """Print final summary."""
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        print("\n\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total chunks: {self.total_chunks}")
        print(f"Total frames: {self.current_frame}")
        print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        if total_time > 0 and self.current_frame > 0:
            print(f"Average speed: {self.current_frame / total_time:.2f} fps")
        print("=" * 50 + "\n")
