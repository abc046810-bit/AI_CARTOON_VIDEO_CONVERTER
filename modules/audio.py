"""Audio extraction and preservation for video processing."""
import os
import shutil
from typing import Optional
from pathlib import Path

from .logger import get_logger
from .utils import ensure_dir
from .ffmpeg_utils import extract_audio

logger = get_logger("audio")


class AudioManager:
    """Manage audio extraction and reinsertion for video processing."""

    def __init__(self, audio_dir: str):
        self.audio_dir = ensure_dir(audio_dir)
        self.audio_path: Optional[str] = None
        self.has_audio: bool = False

    def extract(self, video_path: str, job_id: str) -> Optional[str]:
        """Extract audio from source video.

        Args:
            video_path: Source video path
            job_id: Job identifier

        Returns:
            Path to extracted audio file or None if no audio
        """
        self.audio_path = os.path.join(self.audio_dir, f"{job_id}_audio.aac")

        try:
            extract_audio(video_path, self.audio_path, codec="aac", bitrate="192k")
            if os.path.exists(self.audio_path) and os.path.getsize(self.audio_path) > 0:
                self.has_audio = True
                logger.info(f"Audio extracted: {self.audio_path}")
                return self.audio_path
            else:
                logger.warning("No audio found in video or extraction failed")
                self.has_audio = False
                return None
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            self.has_audio = False
            return None

    def get_audio_path(self) -> Optional[str]:
        """Get path to extracted audio."""
        if self.has_audio and self.audio_path and os.path.exists(self.audio_path):
            return self.audio_path
        return None

    def cleanup(self):
        """Remove extracted audio file."""
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                os.remove(self.audio_path)
                logger.debug(f"Cleaned up audio: {self.audio_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup audio: {e}")
