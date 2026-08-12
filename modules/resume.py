"""Resume/checkpoint system for long video processing."""
import os
import json
from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path

from .logger import get_logger
from .utils import ensure_dir

logger = get_logger("resume")


class CheckpointManager:
    """Manage processing checkpoints for resume capability."""

    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = ensure_dir(checkpoints_dir)
        self.checkpoint_file = os.path.join(checkpoints_dir, "checkpoints.json")
        self._checkpoints: Dict[str, dict] = {}
        self.load()

    def load(self):
        """Load checkpoints from disk."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    self._checkpoints = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoints: {e}")
                self._checkpoints = {}

    def save(self):
        """Save checkpoints to disk."""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self._checkpoints, f, indent=2, ensure_ascii=False)

    def mark_complete(self, chunk_id: str, metadata: Optional[dict] = None):
        """Mark a chunk as completed.

        Args:
            chunk_id: Chunk identifier
            metadata: Optional metadata about the chunk
        """
        self._checkpoints[chunk_id] = {
            "status": "done",
            "completed_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.save()
        logger.debug(f"Checkpoint saved: {chunk_id}")

    def mark_failed(self, chunk_id: str, error: str):
        """Mark a chunk as failed.

        Args:
            chunk_id: Chunk identifier
            error: Error message
        """
        self._checkpoints[chunk_id] = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": error,
        }
        self.save()
        logger.warning(f"Checkpoint marked failed: {chunk_id} - {error}")

    def is_complete(self, chunk_id: str) -> bool:
        """Check if chunk is completed."""
        return chunk_id in self._checkpoints and self._checkpoints[chunk_id].get("status") == "done"

    def is_failed(self, chunk_id: str) -> bool:
        """Check if chunk has failed."""
        return chunk_id in self._checkpoints and self._checkpoints[chunk_id].get("status") == "failed"

    def get_completed(self) -> Set[str]:
        """Get set of completed chunk IDs."""
        return {k for k, v in self._checkpoints.items() if v.get("status") == "done"}

    def get_failed(self) -> Set[str]:
        """Get set of failed chunk IDs."""
        return {k for k, v in self._checkpoints.items() if v.get("status") == "failed"}

    def get_pending(self, all_chunks: List[str]) -> List[str]:
        """Get list of pending chunk IDs from all chunks.

        Args:
            all_chunks: List of all chunk IDs

        Returns:
            List of pending chunk IDs
        """
        completed = self.get_completed()
        return [c for c in all_chunks if c not in completed]

    def reset(self):
        """Reset all checkpoints."""
        self._checkpoints = {}
        self.save()
        logger.info("Checkpoints reset")

    def get_stats(self) -> Dict:
        """Get checkpoint statistics."""
        done = len(self.get_completed())
        failed = len(self.get_failed())
        return {
            "completed": done,
            "failed": failed,
            "total_recorded": len(self._checkpoints),
        }
