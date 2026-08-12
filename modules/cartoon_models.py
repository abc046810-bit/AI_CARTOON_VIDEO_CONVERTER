"""Base cartoon model interface."""
import abc
from typing import Optional
import numpy as np


class BaseCartoonModel(abc.ABC):
    """Abstract base class for cartoonization models."""

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.model = None
        self.loaded = False
        self.model_name = self.__class__.__name__

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights and initialize."""
        pass

    @abc.abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame.

        Args:
            frame: Input frame as BGR numpy array (H, W, 3), uint8

        Returns:
            Processed frame as BGR numpy array (H, W, 3), uint8
        """
        pass

    @abc.abstractmethod
    def process_batch(self, frames: list) -> list:
        """Process a batch of frames.

        Args:
            frames: List of BGR numpy arrays

        Returns:
            List of processed BGR numpy arrays
        """
        pass

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model and free memory."""
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.loaded
