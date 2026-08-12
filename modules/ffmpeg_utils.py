"""FFmpeg utility wrappers for AI Cartoon Video Converter."""
import os
import re
import subprocess
from typing import Optional, List, Tuple
from pathlib import Path

from .logger import get_logger
from .utils import format_bytes, format_duration

logger = get_logger("ffmpeg")


class FFmpegError(Exception):
    """Raised when FFmpeg operation fails."""
    pass


def check_ffmpeg() -> Tuple[bool, str]:
    """Check if FFmpeg is installed and available.

    Returns:
        Tuple of (is_available, version_string)
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        return False, ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, ""


def run_ffmpeg(cmd: List[str], timeout: Optional[int] = None, 
               description: str = "FFmpeg operation") -> None:
    """Run an FFmpeg command with error handling.

    Args:
        cmd: Command list (without 'ffmpeg' prefix)
        timeout: Timeout in seconds
        description: Description for logging

    Raises:
        FFmpegError: If command fails
    """
    full_cmd = ['ffmpeg', '-y'] + cmd
    logger.debug(f"Running: {' '.join(full_cmd)}")

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
            raise FFmpegError(f"{description} failed (code {result.returncode}): {error_msg}")

    except subprocess.TimeoutExpired:
        raise FFmpegError(f"{description} timed out after {timeout}s")
    except FileNotFoundError:
        raise FFmpegError("FFmpeg not found. Please install FFmpeg.")


def extract_audio(input_path: str, output_path: str, 
                  codec: str = "aac", bitrate: str = "192k") -> None:
    """Extract audio track from video.

    Args:
        input_path: Source video path
        output_path: Output audio path
        codec: Audio codec
        bitrate: Audio bitrate
    """
    cmd = [
        '-i', input_path,
        '-vn',                    # No video
        '-acodec', codec,
        '-b:a', bitrate,
        '-ar', '48000',           # Standardize sample rate
        '-ac', '2',               # Stereo
        output_path
    ]
    run_ffmpeg(cmd, timeout=300, description="Audio extraction")
    logger.info(f"Audio extracted: {output_path}")


def split_video(input_path: str, output_pattern: str, 
                chunk_duration: int, video_codec: str = "copy",
                audio_codec: str = "copy") -> List[str]:
    """Split video into chunks using segment muxer.

    Args:
        input_path: Source video path
        output_pattern: Output path pattern with %03d
        chunk_duration: Duration of each chunk in seconds
        video_codec: Video codec (copy or re-encode)
        audio_codec: Audio codec (copy or re-encode)

    Returns:
        List of generated chunk file paths
    """
    cmd = [
        '-i', input_path,
        '-c:v', video_codec,
        '-c:a', audio_codec,
        '-map', '0',
        '-f', 'segment',
        '-segment_time', str(chunk_duration),
        '-segment_format', 'mp4',
        '-reset_timestamps', '1',
        '-avoid_negative_ts', 'make_zero',
        output_pattern
    ]

    run_ffmpeg(cmd, timeout=None, description="Video splitting")

    # Find generated chunks
    chunk_files = []
    output_dir = os.path.dirname(output_pattern)
    base_pattern = os.path.basename(output_pattern).replace('%03d', '')

    if os.path.exists(output_dir):
        for f in sorted(os.listdir(output_dir)):
            if f.startswith(base_pattern.replace('.mp4', '')) and f.endswith('.mp4'):
                chunk_files.append(os.path.join(output_dir, f))

    logger.info(f"Video split into {len(chunk_files)} chunks")
    return chunk_files


def extract_frames(input_path: str, output_pattern: str, 
                   fps: Optional[float] = None,
                   start_time: Optional[float] = None,
                   duration: Optional[float] = None) -> int:
    """Extract frames from video as images.

    Args:
        input_path: Source video path
        output_pattern: Output image pattern (e.g., frame_%04d.png)
        fps: Target FPS (None = original)
        start_time: Start time in seconds
        duration: Duration in seconds

    Returns:
        Number of frames extracted
    """
    cmd = []

    if start_time is not None:
        cmd.extend(['-ss', str(start_time)])

    cmd.extend(['-i', input_path])

    if duration is not None:
        cmd.extend(['-t', str(duration)])

    if fps is not None:
        cmd.extend(['-vf', f'fps={fps}'])

    cmd.extend([
        '-pix_fmt', 'rgb24',
        output_pattern
    ])

    run_ffmpeg(cmd, timeout=None, description="Frame extraction")

    # Count extracted frames
    output_dir = os.path.dirname(output_pattern)
    frame_count = len([f for f in os.listdir(output_dir) 
                       if f.startswith('frame_')])
    logger.info(f"Extracted {frame_count} frames")
    return frame_count


def frames_to_video(frame_pattern: str, output_path: str, 
                    fps: float, audio_path: Optional[str] = None,
                    crf: int = 23, preset: str = "medium",
                    resolution: Optional[Tuple[int, int]] = None) -> None:
    """Assemble frames into video.

    Args:
        frame_pattern: Input frame pattern (e.g., frame_%04d.png)
        output_path: Output video path
        fps: Frame rate
        audio_path: Optional audio file to mux
        crf: Quality (0-51)
        preset: Encoding speed preset
        resolution: Optional (width, height) to resize
    """
    cmd = [
        '-framerate', str(fps),
        '-i', frame_pattern,
    ]

    if audio_path and os.path.exists(audio_path):
        cmd.extend(['-i', audio_path])
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest'
        ])
    else:
        cmd.extend(['-an'])

    vf_filters = []
    if resolution:
        vf_filters.append(f"scale={resolution[0]}:{resolution[1]}")

    # Ensure even dimensions for H.264
    vf_filters.append("format=yuv420p")

    if vf_filters:
        cmd.extend(['-vf', ','.join(vf_filters)])

    cmd.extend([
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', preset,
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        output_path
    ])

    run_ffmpeg(cmd, timeout=None, description="Frame assembly")
    logger.info(f"Video assembled: {output_path}")


def merge_videos(chunk_list_path: str, output_path: str,
                 crf: int = 23, preset: str = "medium") -> None:
    """Merge video chunks using concat demuxer.

    Args:
        chunk_list_path: Path to text file with chunk list
        output_path: Output video path
        crf: Quality
        preset: Encoding preset
    """
    cmd = [
        '-f', 'concat',
        '-safe', '0',
        '-i', chunk_list_path,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', preset,
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    run_ffmpeg(cmd, timeout=None, description="Video merging")
    logger.info(f"Chunks merged: {output_path}")


def generate_thumbnail(video_path: str, output_path: str, 
                       time_offset: Optional[float] = None) -> None:
    """Generate a thumbnail from video.

    Args:
        video_path: Source video path
        output_path: Output thumbnail path
        time_offset: Time in seconds (None = middle of video)
    """
    if time_offset is None:
        # Get duration first
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True, text=True, timeout=30
            )
            duration = float(result.stdout.strip())
            time_offset = duration / 2
        except Exception:
            time_offset = 1.0

    cmd = [
        '-ss', str(time_offset),
        '-i', video_path,
        '-vframes', '1',
        '-q:v', '2',
        '-vf', 'scale=480:-1',
        output_path
    ]

    run_ffmpeg(cmd, timeout=60, description="Thumbnail generation")
    logger.info(f"Thumbnail generated: {output_path}")


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
