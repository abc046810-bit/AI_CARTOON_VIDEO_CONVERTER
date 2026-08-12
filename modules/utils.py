"""Utility functions for AI Cartoon Video Converter."""
import os
import re
import sys
import time
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
from urllib.parse import urlparse

import psutil


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and invalid chars."""
    # Remove path traversal
    filename = os.path.basename(filename)
    # Replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200] + ext
    return filename.strip()


def safe_path(base_dir: str, *paths: str) -> str:
    """Build a safe path inside base_dir, preventing traversal."""
    target = os.path.abspath(os.path.join(base_dir, *paths))
    base = os.path.abspath(base_dir)
    if not target.startswith(base):
        raise ValueError(f"Path traversal detected: {paths}")
    return target


def get_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """Calculate file hash for verification."""
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_bytes(size: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_disk_usage(path: str) -> Tuple[int, int, int]:
    """Get disk usage in bytes: total, used, free."""
    usage = shutil.disk_usage(path)
    return usage.total, usage.used, usage.free


def get_memory_info() -> dict:
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return {
        "total": mem.total,
        "available": mem.available,
        "percent": mem.percent,
        "used": mem.used,
        "free": mem.free,
    }


def ensure_dir(path: str) -> str:
    """Ensure directory exists, return path."""
    os.makedirs(path, exist_ok=True)
    return path


def is_valid_url(url: str) -> bool:
    """Check if string is a valid HTTP/HTTPS URL."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and parsed.netloc
    except Exception:
        return False


def is_google_drive_url(url: str) -> bool:
    """Check if URL is a Google Drive share link."""
    return 'drive.google.com' in url or 'drive.usercontent.google.com' in url


def extract_drive_file_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from URL."""
    patterns = [
        r'/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'file/d/([a-zA-Z0-9_-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def generate_job_id() -> str:
    """Generate a unique job ID based on timestamp."""
    return f"job_{int(time.time() * 1000)}"


def retry_on_error(max_retries: int = 3, delay: float = 5.0, 
                   exceptions: Tuple = (Exception,)):
    """Decorator for retry logic."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exception
        return wrapper
    return decorator


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print application header."""
    print("=" * 50)
    print("     AI CARTOON VIDEO CONVERTER")
    print("=" * 50)
    print()
