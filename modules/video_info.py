"""Video information extraction using FFprobe."""
import os
import json
import subprocess
from typing import Dict, Optional, Tuple
from pathlib import Path

from .logger import get_logger
from .utils import format_bytes, format_duration

logger = get_logger("video_info")


class VideoInfo:
    """Container for video metadata."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.format_name: str = ""
        self.duration: float = 0.0
        self.duration_str: str = "00:00:00"
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0.0
        self.bitrate: int = 0
        self.file_size: int = 0
        self.file_size_str: str = "0 B"
        self.video_codec: str = ""
        self.pixel_format: str = ""
        self.audio_codec: str = ""
        self.audio_channels: int = 0
        self.audio_sample_rate: int = 0
        self.has_audio: bool = False
        self.aspect_ratio: str = ""
        self.total_frames: int = 0
        self.valid: bool = False
        self.error: str = ""

        self._analyze()

    def _analyze(self):
        """Analyze video file using FFprobe."""
        if not os.path.exists(self.filepath):
            self.error = f"File not found: {self.filepath}"
            return

        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                self.filepath
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                self.error = f"FFprobe error: {result.stderr}"
                return

            data = json.loads(result.stdout)
            self._parse_format(data.get('format', {}))
            self._parse_streams(data.get('streams', []))
            self.valid = True

        except subprocess.TimeoutExpired:
            self.error = "FFprobe timed out"
        except json.JSONDecodeError as e:
            self.error = f"Failed to parse FFprobe output: {e}"
        except Exception as e:
            self.error = f"Analysis failed: {e}"

    def _parse_format(self, fmt: Dict):
        """Parse format section."""
        self.format_name = fmt.get('format_name', '')
        self.duration = float(fmt.get('duration', 0))
        self.duration_str = format_duration(self.duration)
        self.bitrate = int(fmt.get('bit_rate', 0))
        self.file_size = int(fmt.get('size', 0))
        self.file_size_str = format_bytes(self.file_size)

    def _parse_streams(self, streams: list):
        """Parse streams section."""
        for stream in streams:
            codec_type = stream.get('codec_type', '')

            if codec_type == 'video':
                self.width = int(stream.get('width', 0))
                self.height = int(stream.get('height', 0))
                self.video_codec = stream.get('codec_name', '')
                self.pixel_format = stream.get('pix_fmt', '')

                # Calculate FPS
                fps_str = stream.get('r_frame_rate', '0/1')
                try:
                    num, den = map(int, fps_str.split('/'))
                    self.fps = round(num / den, 2) if den != 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    self.fps = 0.0

                # Calculate aspect ratio
                if self.width > 0 and self.height > 0:
                    gcd_val = self._gcd(self.width, self.height)
                    self.aspect_ratio = f"{self.width // gcd_val}:{self.height // gcd_val}"
                    self.total_frames = int(self.duration * self.fps) if self.fps > 0 else 0

            elif codec_type == 'audio':
                self.has_audio = True
                self.audio_codec = stream.get('codec_name', '')
                self.audio_channels = int(stream.get('channels', 0))
                self.audio_sample_rate = int(stream.get('sample_rate', 0))

    @staticmethod
    def _gcd(a: int, b: int) -> int:
        """Calculate greatest common divisor."""
        while b:
            a, b = b, a % b
        return a

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "format": self.format_name,
            "duration": self.duration,
            "duration_str": self.duration_str,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bitrate": self.bitrate,
            "file_size": self.file_size,
            "file_size_str": self.file_size_str,
            "video_codec": self.video_codec,
            "pixel_format": self.pixel_format,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "aspect_ratio": self.aspect_ratio,
            "total_frames": self.total_frames,
            "valid": self.valid,
        }

    def print_summary(self):
        """Print formatted video information."""
        print("\n" + "-" * 50)
        print("VIDEO INFORMATION")
        print("-" * 50)
        print(f"Filename:     {self.filename}")
        print(f"Duration:     {self.duration_str} ({self.duration:.2f}s)")
        print(f"Resolution:   {self.width}x{self.height}")
        print(f"FPS:          {self.fps}")
        print(f"Aspect Ratio: {self.aspect_ratio}")
        print(f"Video Codec:  {self.video_codec}")
        print(f"Pixel Format: {self.pixel_format}")
        print(f"Audio:        {'Yes' if self.has_audio else 'No'}")
        if self.has_audio:
            print(f"Audio Codec:  {self.audio_codec}")
            print(f"Audio SR:     {self.audio_sample_rate} Hz")
        print(f"Bitrate:      {self.bitrate // 1000} kbps")
        print(f"File Size:    {self.file_size_str}")
        print(f"Total Frames: {self.total_frames}")
        print("-" * 50 + "\n")


def analyze_video(filepath: str) -> VideoInfo:
    """Analyze a video file and return VideoInfo object.

    Args:
        filepath: Path to video file

    Returns:
        VideoInfo object with metadata
    """
    return VideoInfo(filepath)
