"""AnimeGANv2 implementation using torch.hub."""
import os
import warnings
from typing import List

import numpy as np
import torch
import cv2
from PIL import Image

from .cartoon_models import BaseCartoonModel
from .logger import get_logger
from .gpu import clear_gpu_cache

logger = get_logger("animegan")

# Model variants available via torch.hub
MODEL_VARIANTS = {
    "celeba_distill": "Celeba face style (distilled)",
    "face_paint_512_v1": "Face portrait v1 (512x512)",
    "face_paint_512_v2": "Face portrait v2 (512x512, more robust)",
    "paprika": "Paprika style (landscape/general)",
}

DEFAULT_VARIANT = "paprika"


class AnimeGANv2Model(BaseCartoonModel):
    """AnimeGANv2 cartoonization model.

    Uses the PyTorch implementation by bryandlee via torch.hub.
    Repository: https://github.com/bryandlee/animegan2-pytorch
    Original: https://github.com/TachibanaYoshino/AnimeGANv2

    License: See original repository for licensing terms.
    """

    def __init__(self, device: str = "cpu", variant: str = DEFAULT_VARIANT, 
                 weights_dir: str = "models/weights"):
        super().__init__(device)
        self.variant = variant if variant in MODEL_VARIANTS else DEFAULT_VARIANT
        self.weights_dir = weights_dir
        self.face2paint = None

    def load(self) -> None:
        """Load AnimeGANv2 model via torch.hub."""
        if self.loaded:
            return

        logger.info(f"Loading AnimeGANv2 (variant: {self.variant}) on {self.device}...")

        try:
            # Set torch hub directory to our weights folder
            torch.hub.set_dir(self.weights_dir)

            # Load generator model
            self.model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "generator",
                pretrained=self.variant,
                device=self.device,
                progress=True
            )
            self.model.eval()

            # Try to load face2paint utility (useful for face variants)
            try:
                self.face2paint = torch.hub.load(
                    "bryandlee/animegan2-pytorch:main",
                    "face2paint",
                    size=512,
                    device=self.device
                )
            except Exception:
                self.face2paint = None

            self.loaded = True
            logger.info("AnimeGANv2 loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load AnimeGANv2: {e}")
            raise RuntimeError(f"AnimeGANv2 load failed: {e}")

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy array to model input tensor."""
        # BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To PIL
        pil_img = Image.fromarray(rgb)
        # To tensor [C, H, W], normalized to [0, 1]
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        tensor = transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def _postprocess(self, tensor: torch.Tensor, original_shape: tuple) -> np.ndarray:
        """Convert model output tensor to BGR numpy array."""
        # Denormalize from [-1, 1] to [0, 1]
        output = tensor.squeeze(0).cpu().detach()
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0, 1)

        # To numpy [H, W, C]
        output = output.permute(1, 2, 0).numpy()
        output = (output * 255).astype(np.uint8)

        # RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Resize back to original if needed
        if output.shape[:2] != original_shape[:2]:
            output = cv2.resize(output, (original_shape[1], original_shape[0]), 
                               interpolation=cv2.INTER_LANCZOS4)

        return output

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame."""
        if not self.loaded:
            self.load()

        original_shape = frame.shape

        with torch.no_grad():
            input_tensor = self._preprocess(frame)
            output_tensor = self.model(input_tensor)
            result = self._postprocess(output_tensor, original_shape)

        return result

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Process a batch of frames."""
        if not self.loaded:
            self.load()

        if not frames:
            return []

        original_shapes = [f.shape for f in frames]

        with torch.no_grad():
            # Stack and preprocess
            tensors = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                tensor = transform(pil_img)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors).to(self.device)

            # Inference
            output_tensors = self.model(batch_tensor)

            # Postprocess
            results = []
            for i, output in enumerate(output_tensors):
                result = self._postprocess(output.unsqueeze(0), original_shapes[i])
                results.append(result)

        return results

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            self.model = None
            self.face2paint = None
            self.loaded = False
            if self.device.startswith('cuda'):
                clear_gpu_cache()
            logger.info("AnimeGANv2 unloaded")

    @classmethod
    def list_variants(cls) -> dict:
        """List available model variants."""
        return MODEL_VARIANTS.copy()
