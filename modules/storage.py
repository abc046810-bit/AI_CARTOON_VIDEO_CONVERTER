"""Storage management including Google Drive integration."""
import os
import shutil
from typing import Optional
from pathlib import Path

from .logger import get_logger
from .utils import ensure_dir, is_colab

logger = get_logger("storage")

# Google Drive mount point in Colab
COLAB_DRIVE_PATH = "/content/drive"


class StorageManager:
    """Manage output storage locations."""

    def __init__(self, output_dir: str = "outputs", 
                 drive_dir: str = "AI_Cartoon_Output",
                 auto_mount: bool = True):
        self.output_dir = ensure_dir(output_dir)
        self.drive_dir = drive_dir
        self.auto_mount = auto_mount
        self.drive_mounted = False
        self.drive_output_path = None

    def mount_google_drive(self) -> bool:
        """Mount Google Drive if running in Colab.

        Returns:
            True if drive is available
        """
        if not is_colab():
            logger.info("Not running in Colab. Google Drive mount skipped.")
            return False

        if os.path.exists(os.path.join(COLAB_DRIVE_PATH, "MyDrive")):
            logger.info("Google Drive already mounted.")
            self.drive_mounted = True
        elif self.auto_mount:
            try:
                from google.colab import drive
                logger.info("Mounting Google Drive...")
                drive.mount(COLAB_DRIVE_PATH)
                self.drive_mounted = True
                logger.info("Google Drive mounted successfully.")
            except Exception as e:
                logger.error(f"Failed to mount Google Drive: {e}")
                self.drive_mounted = False

        if self.drive_mounted:
            self.drive_output_path = os.path.join(
                COLAB_DRIVE_PATH, "MyDrive", self.drive_dir
            )
            ensure_dir(self.drive_output_path)

        return self.drive_mounted

    def get_output_path(self, job_id: str, use_drive: bool = False) -> str:
        """Get output path for a job.

        Args:
            job_id: Job identifier
            use_drive: Whether to use Google Drive

        Returns:
            Output directory path
        """
        if use_drive and self.drive_mounted and self.drive_output_path:
            path = os.path.join(self.drive_output_path, job_id)
        else:
            path = os.path.join(self.output_dir, job_id)

        return ensure_dir(path)

    def copy_to_drive(self, job_id: str, file_path: str) -> Optional[str]:
        """Copy a file to Google Drive.

        Args:
            job_id: Job ID
            file_path: File to copy

        Returns:
            Path in drive or None
        """
        if not self.drive_mounted or not self.drive_output_path:
            return None

        dest_dir = self.get_output_path(job_id, use_drive=True)
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))

        try:
            shutil.copy2(file_path, dest_path)
            logger.info(f"Copied to Drive: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"Failed to copy to Drive: {e}")
            return None

    def save_job_output(self, job_id: str, source_dir: str, 
                        use_drive: bool = False) -> dict:
        """Save job output to storage.

        Args:
            job_id: Job ID
            source_dir: Source directory with output files
            use_drive: Whether to also save to Drive

        Returns:
            Dictionary with output paths
        """
        local_dir = self.get_output_path(job_id, use_drive=False)

        # Copy files to local output
        copied = []
        for item in os.listdir(source_dir):
            src = os.path.join(source_dir, item)
            dst = os.path.join(local_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied.append(dst)

        result = {
            "local": local_dir,
            "files": copied,
        }

        # Copy to drive if requested
        if use_drive:
            drive_dir = self.get_output_path(job_id, use_drive=True)
            drive_copied = []
            for item in os.listdir(source_dir):
                src = os.path.join(source_dir, item)
                dst = os.path.join(drive_dir, item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    drive_copied.append(dst)
            result["drive"] = drive_dir
            result["drive_files"] = drive_copied

        return result
