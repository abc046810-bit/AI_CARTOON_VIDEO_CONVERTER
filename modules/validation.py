"""Input validation for AI Cartoon Video Converter."""
import os
import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from .utils import is_valid_url, is_google_drive_url, extract_drive_file_id


SUPPORTED_VIDEO_FORMATS = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.m4v', '.3gp')
SUPPORTED_URL_FORMATS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_video_file(filepath: str) -> Tuple[bool, str]:
    """Validate a local video file path.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filepath or not filepath.strip():
        return False, "File path is empty"

    filepath = filepath.strip()

    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"

    if not os.path.isfile(filepath):
        return False, f"Path is not a file: {filepath}"

    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_VIDEO_FORMATS:
        return False, f"Unsupported file format: {ext}. Supported: {SUPPORTED_VIDEO_FORMATS}"

    if os.path.getsize(filepath) == 0:
        return False, "File is empty"

    return True, ""


def validate_video_url(url: str) -> Tuple[bool, str]:
    """Validate a direct video URL.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not url.strip():
        return False, "URL is empty"

    url = url.strip()

    if not is_valid_url(url):
        return False, "Invalid URL format. Must be http:// or https://"

    parsed = urlparse(url)
    path = parsed.path.lower()

    # Check if URL ends with a supported extension
    has_supported_ext = any(path.endswith(ext) for ext in SUPPORTED_URL_FORMATS)

    if not has_supported_ext:
        # Allow URLs without extension but warn
        return True, "WARNING: URL does not have a recognized video extension. Download may fail if server does not provide correct content-type."

    return True, ""


def validate_google_drive_url(url: str) -> Tuple[bool, Optional[str], str]:
    """Validate a Google Drive share link.

    Returns:
        Tuple of (is_valid, file_id, error_message)
    """
    if not url or not url.strip():
        return False, None, "URL is empty"

    url = url.strip()

    if not is_valid_url(url):
        return False, None, "Invalid URL format"

    if not is_google_drive_url(url):
        return False, None, "Not a recognized Google Drive URL"

    file_id = extract_drive_file_id(url)
    if not file_id:
        return False, None, "Could not extract Google Drive file ID from URL"

    return True, file_id, ""


def validate_resolution(resolution: str) -> Tuple[bool, str]:
    """Validate resolution setting."""
    valid = ('original', '480p', '720p', '1080p')
    if resolution not in valid:
        return False, f"Invalid resolution: {resolution}. Must be one of {valid}"
    return True, ""


def validate_fps(fps: str) -> Tuple[bool, str]:
    """Validate FPS setting."""
    valid = ('original', '24', '25', '30', '60')
    if fps not in valid:
        return False, f"Invalid FPS: {fps}. Must be one of {valid}"
    return True, ""


def validate_chunk_duration(duration: int) -> Tuple[bool, str]:
    """Validate chunk duration."""
    valid = (60, 120, 300, 600)
    if duration not in valid:
        return False, f"Invalid chunk duration: {duration}. Must be one of {valid}"
    return True, ""


def validate_model(model: str) -> Tuple[bool, str]:
    """Validate model selection."""
    valid = ('animeganv2', 'whitebox')
    if model not in valid:
        return False, f"Invalid model: {model}. Must be one of {valid}"
    return True, ""


def validate_positive_int(value: str, name: str = "value") -> Tuple[bool, int, str]:
    """Validate that a string is a positive integer."""
    try:
        num = int(value)
        if num <= 0:
            return False, 0, f"{name} must be a positive integer"
        return True, num, ""
    except ValueError:
        return False, 0, f"{name} must be a valid integer"
