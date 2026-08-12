"""Chunk merging and final output assembly."""
import os
from typing import List
from pathlib import Path

from .logger import get_logger
from .utils import ensure_dir
from .ffmpeg_utils import merge_videos

logger = get_logger("merger")


class ChunkMerger:
    """Merge processed chunks into final video."""

    def __init__(self, output_dir: str):
        self.output_dir = ensure_dir(output_dir)

    def create_concat_list(self, chunk_paths: List[str], list_path: str) -> str:
        """Create FFmpeg concat demuxer list file.

        Args:
            chunk_paths: List of processed chunk paths
            list_path: Where to save the list file

        Returns:
            Path to list file
        """
        with open(list_path, 'w', encoding='utf-8') as f:
            for chunk_path in chunk_paths:
                # Escape single quotes in path
                escaped = chunk_path.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        logger.debug(f"Concat list created: {list_path}")
        return list_path

    def merge(self, chunk_paths: List[str], output_path: str,
              crf: int = 23, preset: str = "medium") -> str:
        """Merge chunks into final video.

        Args:
            chunk_paths: List of processed chunk paths (must be in order)
            output_path: Final output path
            crf: Video quality
            preset: Encoding preset

        Returns:
            Path to merged video
        """
        if not chunk_paths:
            raise ValueError("No chunks to merge")

        if len(chunk_paths) == 1:
            # Single chunk - just copy
            import shutil
            shutil.copy2(chunk_paths[0], output_path)
            logger.info(f"Single chunk copied to output: {output_path}")
            return output_path

        list_path = os.path.join(self.output_dir, "concat_list.txt")
        self.create_concat_list(chunk_paths, list_path)

        logger.info(f"Merging {len(chunk_paths)} chunks...")
        merge_videos(list_path, output_path, crf=crf, preset=preset)

        # Cleanup list file
        try:
            os.remove(list_path)
        except Exception:
            pass

        logger.info(f"Final video: {output_path}")
        return output_path
