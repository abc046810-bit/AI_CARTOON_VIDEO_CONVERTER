"""Logging configuration for AI Cartoon Video Converter."""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    COLORS = {
        'DEBUG': '[36m',      # Cyan
        'INFO': '[32m',       # Green
        'WARNING': '[33m',    # Yellow
        'ERROR': '[31m',      # Red
        'CRITICAL': '[35m',   # Magenta
    }
    RESET = '[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str = "ai_cartoon", log_dir: str = "logs", 
                 log_level: str = "INFO", log_to_file: bool = True,
                 console_level: str = "INFO") -> logging.Logger:
    """Setup and return a configured logger.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: File logging level
        log_to_file: Whether to write to file
        console_level: Console logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates on re-import
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Also create a persistent latest.log symlink/copy
        latest_log = os.path.join(log_dir, "latest.log")
        try:
            if os.path.exists(latest_log):
                os.remove(latest_log)
            # On Windows, symlinks may fail; just ignore
            os.symlink(os.path.abspath(log_file), latest_log)
        except (OSError, NotImplementedError):
            pass

    return logger


def get_logger(name: str = "ai_cartoon") -> logging.Logger:
    """Get existing logger or create default."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
