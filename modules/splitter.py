"""Video chunking and management for large video support."""
import os
import glob
import shutil
from typing import List, Tuple
from pathlib import Path

from .logger import get_logger
from .utils import ensure_dir, safe_path
from .video_info import VideoInfo
from .ffmpeg_utils import split_video, get_video_duration

logger = get_logger("splitter")


class VideoSplitter:
    """Split videos into manageable chunks for processing."""
    
    def __init__(self, chunk_duration: int = 300):
        self.chunk_duration = chunk_duration
        self.chunks: List[str] = []
        
    def split(self, input_path: str, output_dir: str) -> List[str]:
        """Split video into chunks.
        
        Args:
            input_path: Source video path
            output_dir: Directory to store chunks
            
        Returns:
            List of chunk file paths (1-based: chunk_001.mp4, chunk_002.mp4...)
        """
        ensure_dir(output_dir)
        
        info = VideoInfo(input_path)
        if not info.valid:
            raise ValueError(f"Cannot split invalid video: {info.error}")
        
        if info.duration <= self.chunk_duration:
            logger.info(f"Video duration ({info.duration:.1f}s) <= chunk size ({self.chunk_duration}s). Using single chunk.")
            chunk_path = os.path.join(output_dir, "chunk_001.mp4")
            shutil.copy2(input_path, chunk_path)
            self.chunks = [chunk_path]
            return self.chunks
        
        logger.info(f"Splitting video into {self.chunk_duration}s chunks...")
        
        # Use temp pattern for FFmpeg (0-based output)
        temp_pattern = os.path.join(output_dir, "chunk_tmp_%03d.mp4")
        temp_chunks = split_video(
            input_path, temp_pattern, self.chunk_duration,
            video_codec="copy", audio_codec="copy"
        )
        
        # Rename to 1-based naming to match checkpoint IDs
        self.chunks = []
        for i, temp_path in enumerate(sorted(temp_chunks), 1):
            new_name = os.path.join(output_dir, f"chunk_{i:03d}.mp4")
            shutil.move(temp_path, new_name)
            self.chunks.append(new_name)
        
        logger.info(f"Created {len(self.chunks)} chunks")
        for i, chunk in enumerate(self.chunks, 1):
            duration = get_video_duration(chunk)
            logger.debug(f"  Chunk {i}: {os.path.basename(chunk)} ({duration:.1f}s)")
        
        return self.chunks
    
    def get_chunk_info(self) -> List[dict]:
        """Get information about all chunks."""
        info_list = []
        for chunk_path in self.chunks:
            info = VideoInfo(chunk_path)
            info_list.append(info.to_dict())
        return info_list
