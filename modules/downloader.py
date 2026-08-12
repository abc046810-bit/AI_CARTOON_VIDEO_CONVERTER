"""Video download functionality with progress tracking."""
import os
import time
import requests
from typing import Optional, Callable
from urllib.parse import urlparse

from .logger import get_logger
from .utils import sanitize_filename, format_bytes, is_google_drive_url, extract_drive_file_id, retry_on_error
from .validation import validate_video_url

logger = get_logger("downloader")


class DownloadError(Exception):
    """Raised when download fails."""
    pass


def download_file(url: str, output_path: str, 
                  progress_callback: Optional[Callable] = None,
                  timeout: int = 60, max_retries: int = 3) -> str:
    """Download a file from URL with progress tracking.

    Args:
        url: Direct download URL
        output_path: Where to save the file
        progress_callback: Function(current, total, speed, eta)
        timeout: Network timeout
        max_retries: Max retry attempts

    Returns:
        Path to downloaded file

    Raises:
        DownloadError: If download fails after retries
    """
    valid, msg = validate_video_url(url)
    if not valid:
        raise DownloadError(msg)

    @retry_on_error(max_retries=max_retries, delay=5.0, 
                    exceptions=(requests.RequestException,))
    def _download():
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        logger.info(f"Downloading: {url}")
        logger.info(f"Destination: {output_path}")

        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        last_update = start_time

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    current_time = time.time()
                    if current_time - last_update >= 0.5 or downloaded == total_size:
                        elapsed = current_time - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0

                        eta = 0
                        if total_size > 0 and speed > 0:
                            eta = (total_size - downloaded) / speed

                        if progress_callback:
                            progress_callback(downloaded, total_size, speed, eta)

                        last_update = current_time

        logger.info(f"Download complete: {format_bytes(downloaded)}")
        return output_path

    return _download()


def download_with_progress_bar(url: str, output_path: str, 
                                timeout: int = 60, max_retries: int = 3) -> str:
    """Download with a simple terminal progress bar.

    Args:
        url: Direct download URL
        output_path: Output file path
        timeout: Network timeout
        max_retries: Max retries

    Returns:
        Path to downloaded file
    """
    def progress(current, total, speed, eta):
        if total > 0:
            percent = (current / total) * 100
            bar_length = 30
            filled = int(bar_length * current / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r[{bar}] {percent:.1f}% | {format_bytes(current)}/{format_bytes(total)} | "
                  f"{format_bytes(int(speed))}/s | ETA: {int(eta)}s", end='', flush=True)
        else:
            print(f"\rDownloaded: {format_bytes(current)} | {format_bytes(int(speed))}/s", 
                  end='', flush=True)

    try:
        result = download_file(url, output_path, progress, timeout, max_retries)
        print()  # New line after progress
        return result
    except Exception as e:
        print()
        raise DownloadError(f"Download failed: {e}")


def download_from_google_drive(url: str, output_path: str,
                                timeout: int = 60, max_retries: int = 3) -> str:
    """Download a file from Google Drive share link.

    Args:
        url: Google Drive share URL
        output_path: Output file path
        timeout: Network timeout
        max_retries: Max retries

    Returns:
        Path to downloaded file
    """
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise DownloadError("Could not extract Google Drive file ID")

    try:
        import gdown
    except ImportError:
        raise DownloadError("gdown is not installed. Run: pip install gdown")

    logger.info(f"Downloading from Google Drive: {file_id}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    @retry_on_error(max_retries=max_retries, delay=5.0,
                    exceptions=(Exception,))
    def _download():
        gdown.download(id=file_id, output=output_path, quiet=False)
        return output_path

    return _download()


def auto_download(source: str, output_dir: str, 
                  timeout: int = 60, max_retries: int = 3) -> str:
    """Auto-detect source type and download.

    Args:
        source: URL or Google Drive link
        output_dir: Directory to save file
        timeout: Network timeout
        max_retries: Max retries

    Returns:
        Path to downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)

    if is_google_drive_url(source):
        # Try to get filename from URL or use default
        parsed = urlparse(source)
        filename = os.path.basename(parsed.path) or "drive_video.mp4"
        filename = sanitize_filename(filename)
        output_path = os.path.join(output_dir, filename)
        return download_from_google_drive(source, output_path, timeout, max_retries)
    else:
        # Direct URL
        parsed = urlparse(source)
        filename = os.path.basename(parsed.path) or "downloaded_video.mp4"
        filename = sanitize_filename(filename)
        if not filename.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm')):
            filename += '.mp4'
        output_path = os.path.join(output_dir, filename)
        return download_with_progress_bar(source, output_path, timeout, max_retries)
