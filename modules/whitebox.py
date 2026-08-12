"""White-box Cartoonization implementation using HuggingFace SavedModel."""
import os
import warnings
from typing import List

import numpy as np
import cv2

from .cartoon_models import BaseCartoonModel
from .logger import get_logger
from .gpu import clear_gpu_cache

logger = get_logger("whitebox")

# HuggingFace model repository
HF_MODEL_ID = "sayakpaul/whitebox-cartoonizer"


class WhiteBoxCartoonizationModel(BaseCartoonModel):
    """White-box Cartoonization model.

    Uses the TensorFlow SavedModel from HuggingFace Hub by sayakpaul.
    Original paper: "Learning to Cartoonize Using White-box Cartoon Representations" (CVPR 2020)
    Original repo: https://github.com/SystemErrorWang/White-box-Cartoonization
    HuggingFace: https://huggingface.co/sayakpaul/whitebox-cartoonizer

    License: See original repository for licensing terms.
    """

    def __init__(self, device: str = "cpu", weights_dir: str = "models/weights"):
        super().__init__(device)
        self.weights_dir = weights_dir
        self.model_path = None
        self.concrete_func = None

    def load(self) -> None:
        """Load White-box Cartoonization model from HuggingFace."""
        if self.loaded:
            return

        logger.info("Loading White-box Cartoonization model...")

        try:
            import tensorflow as tf
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(f"Required packages not installed: {e}. "
                             f"Run: pip install tensorflow huggingface-hub")

        try:
            # Suppress TF warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            tf.get_logger().setLevel('ERROR')

            # Download model from HuggingFace
            cache_dir = os.path.join(self.weights_dir, "whitebox_hf")
            logger.info(f"Downloading model from HuggingFace: {HF_MODEL_ID}")

            self.model_path = snapshot_download(
                repo_id=HF_MODEL_ID,
                cache_dir=cache_dir,
                local_dir=os.path.join(self.weights_dir, "whitebox"),
                local_dir_use_symlinks=False
            )

            logger.info(f"Model downloaded to: {self.model_path}")

            # Load SavedModel
            loaded_model = tf.saved_model.load(self.model_path)
            self.concrete_func = loaded_model.signatures["serving_default"]

            self.loaded = True
            logger.info("White-box Cartoonization loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load White-box Cartoonization: {e}")
            raise RuntimeError(f"White-box Cartoonization load failed: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input.

        The model expects:
        - BGR input (OpenCV default)
        - Resized to multiple of 8
        - Normalized to [-1, 1]
        - Batch dimension added
        """
        # Resize if too large (model works best under 720px)
        h, w = frame.shape[:2]
        if min(h, w) > 720:
            if h > w:
                h, w = int(720 * h / w), 720
            else:
                h, w = 720, int(720 * w / h)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # Crop to multiple of 8
        h, w = (h // 8) * 8, (w // 8) * 8
        frame = frame[:h, :w, :]

        # Normalize to [-1, 1]
        frame = frame.astype(np.float32) / 127.5 - 1.0

        # Add batch dimension and convert to tensor-like (keep numpy for now)
        frame = np.expand_dims(frame, axis=0)

        return frame

    def _postprocess(self, output: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Postprocess model output to BGR uint8."""
        # Remove batch dimension
        output = output[0]

        # Denormalize from [-1, 1] to [0, 255]
        output = (output + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)

        # Resize back to original if dimensions changed
        if output.shape[:2] != original_shape[:2]:
            output = cv2.resize(output, (original_shape[1], original_shape[0]),
                               interpolation=cv2.INTER_LANCZOS4)

        return output

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        if not self.loaded:
            self.load()

        import tensorflow as tf

        original_shape = frame.shape
        preprocessed = self._preprocess(frame)
        input_tensor = tf.constant(preprocessed)

        # Run inference
        result = self.concrete_func(input_tensor)["final_output:0"]
        result = result.numpy()

        return self._postprocess(result, original_shape)

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process a batch of frames (processes one by one for this model)."""
        if not self.loaded:
            self.load()

        results = []
        for frame in frames:
            try:
                result = self.process_frame(frame)
                results.append(result)
            except Exception as e:
                logger.warning(f"Frame processing failed: {e}")
                results.append(frame)  # Fallback

        return results

    def unload(self) -> None:
        """Unload model and free memory."""
        self.concrete_func = None
        self.model_path = None
        self.loaded = False

        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass

        clear_gpu_cache()
        logger.info("White-box Cartoonization unloaded")
