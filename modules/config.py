"""Configuration management for AI Cartoon Video Converter."""
import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path

from .utils import ensure_dir


DEFAULT_CONFIG_PATH = "config.yaml"
LOCAL_CONFIG_PATH = "config.local.yaml"


class Config:
    """Configuration manager with defaults and overrides."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load configuration from YAML files."""
        # Start with built-in defaults
        self._config = self._get_defaults()

        # Load main config if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
                self._deep_update(self._config, user_config)

        # Load local overrides if exists
        if os.path.exists(LOCAL_CONFIG_PATH):
            with open(LOCAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                local_config = yaml.safe_load(f) or {}
                self._deep_update(self._config, local_config)

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "chunk_duration": 300,
            "resolution": "original",
            "fps": "original",
            "output_format": "mp4",
            "video_codec": "libx264",
            "audio_codec": "aac",
            "crf": 23,
            "preset": "medium",
            "model": "animeganv2",
            "model_variant": "paprika",
            "batch_size": 4,
            "mixed_precision": True,
            "output_directory": "outputs",
            "cache_directory": "cache",
            "temp_directory": "temp",
            "log_directory": "logs",
            "jobs_directory": "jobs",
            "models_directory": "models/weights",
            "google_drive_directory": "AI_Cartoon_Output",
            "google_drive_enabled": False,
            "auto_mount_drive": True,
            "max_retries": 3,
            "retry_delay": 5,
            "network_timeout": 60,
            "max_memory_gb": 0,
            "frame_buffer_size": 16,
            "log_level": "INFO",
            "log_to_file": True,
            "console_log_level": "INFO",
            "cleanup_temp_on_success": False,
            "cleanup_chunks_on_success": False,
            "keep_logs": True,
            "keep_checkpoints": True,
            "batch_sleep_interval": 10,
            "ffmpeg_threads": 0,
            "ffmpeg_hwaccel": "",
        }

    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (dot notation supported)."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key (dot notation supported)."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def get_path(self, key: str) -> str:
        """Get a path configuration value, ensuring directory exists."""
        path = self.get(key, key)
        return ensure_dir(path)

    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self._config.copy()

    def save(self, path: Optional[str] = None):
        """Save current configuration to YAML file."""
        save_path = path or LOCAL_CONFIG_PATH
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global configuration instance."""
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance
