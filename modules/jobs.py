"""Job management for video processing."""
import os
import json
import shutil
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

from .logger import get_logger
from .utils import ensure_dir, safe_path, generate_job_id, sanitize_filename
from .config import get_config

logger = get_logger("jobs")


class JobManager:
    """Manage processing jobs with folder structure."""

    JOB_STRUCTURE = {
        "input": "input",
        "chunks": "chunks",
        "processed": "processed",
        "audio": "audio",
        "metadata": "metadata",
        "checkpoints": "checkpoints",
        "logs": "logs",
        "output": "output",
        "frames": "frames",
        "temp": "temp",
    }

    def __init__(self, jobs_dir: str = "jobs"):
        self.jobs_dir = ensure_dir(jobs_dir)
        self.current_job: Optional[str] = None

    def create_job(self, job_id: Optional[str] = None) -> str:
        """Create a new job with folder structure.

        Args:
            job_id: Optional job ID (generated if None)

        Returns:
            Job ID
        """
        job_id = job_id or generate_job_id()
        self.current_job = job_id

        job_dir = os.path.join(self.jobs_dir, job_id)

        for subdir in self.JOB_STRUCTURE.values():
            ensure_dir(os.path.join(job_dir, subdir))

        # Create initial metadata
        metadata = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "progress": {},
        }
        self.save_metadata(metadata)

        logger.info(f"Job created: {job_id}")
        return job_id

    def get_job_dir(self, job_id: Optional[str] = None, subdir: Optional[str] = None) -> str:
        """Get path to job directory or subdirectory.

        Args:
            job_id: Job ID (uses current if None)
            subdir: Optional subdirectory name

        Returns:
            Absolute path
        """
        jid = job_id or self.current_job
        if not jid:
            raise ValueError("No active job")

        path = os.path.join(self.jobs_dir, jid)
        if subdir:
            path = os.path.join(path, self.JOB_STRUCTURE.get(subdir, subdir))

        return ensure_dir(path)

    def list_jobs(self) -> List[str]:
        """List all job IDs."""
        if not os.path.exists(self.jobs_dir):
            return []
        return sorted([d for d in os.listdir(self.jobs_dir) 
                       if os.path.isdir(os.path.join(self.jobs_dir, d))])

    def job_exists(self, job_id: str) -> bool:
        """Check if job exists."""
        return os.path.exists(os.path.join(self.jobs_dir, job_id))

    def save_metadata(self, metadata: Dict, job_id: Optional[str] = None):
        """Save job metadata to JSON."""
        jid = job_id or self.current_job
        meta_dir = self.get_job_dir(jid, "metadata")
        meta_path = os.path.join(meta_dir, "job_metadata.json")

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load_metadata(self, job_id: Optional[str] = None) -> Dict:
        """Load job metadata from JSON."""
        jid = job_id or self.current_job
        meta_path = os.path.join(self.jobs_dir, jid, "metadata", "job_metadata.json")

        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def delete_job(self, job_id: str, keep_output: bool = True):
        """Delete a job and all its data.

        Args:
            job_id: Job to delete
            keep_output: Whether to preserve output directory
        """
        job_dir = os.path.join(self.jobs_dir, job_id)
        if not os.path.exists(job_dir):
            return

        if keep_output:
            # Move output to a safe location first
            output_dir = os.path.join(job_dir, "output")
            if os.path.exists(output_dir):
                preserved = os.path.join(self.jobs_dir, f"{job_id}_output_preserved")
                shutil.move(output_dir, preserved)

        shutil.rmtree(job_dir)
        logger.info(f"Job deleted: {job_id}")

    def get_job_status(self, job_id: Optional[str] = None) -> Dict:
        """Get comprehensive job status."""
        jid = job_id or self.current_job
        metadata = self.load_metadata(jid)

        job_dir = os.path.join(self.jobs_dir, jid)
        status = {
            "job_id": jid,
            "exists": os.path.exists(job_dir),
            "metadata": metadata,
            "folders": {},
        }

        if status["exists"]:
            for key, subdir in self.JOB_STRUCTURE.items():
                path = os.path.join(job_dir, subdir)
                if os.path.exists(path):
                    files = os.listdir(path)
                    status["folders"][key] = {
                        "file_count": len(files),
                        "files": files[:10],  # First 10 files
                    }

        return status
