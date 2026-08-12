"""Model manager for loading and switching cartoon models."""
from typing import Optional, Type

from .cartoon_models import BaseCartoonModel
from .animegan import AnimeGANv2Model
from .whitebox import WhiteBoxCartoonizationModel
from .logger import get_logger

logger = get_logger("model_manager")

# Registry of available models
MODEL_REGISTRY = {
    "animeganv2": AnimeGANv2Model,
    "whitebox": WhiteBoxCartoonizationModel,
}


def get_available_models() -> dict:
    """Get dictionary of available models and descriptions."""
    return {
        "animeganv2": {
            "name": "AnimeGANv2",
            "description": "Lightweight GAN for photo animation. Multiple styles available.",
            "variants": AnimeGANv2Model.list_variants(),
            "framework": "PyTorch",
            "source": "https://github.com/bryandlee/animegan2-pytorch",
        },
        "whitebox": {
            "name": "White-box Cartoonization",
            "description": "CVPR 2020 paper implementation. Good for scenery and general images.",
            "variants": {"default": "Scenery/general purpose"},
            "framework": "TensorFlow (SavedModel)",
            "source": "https://huggingface.co/sayakpaul/whitebox-cartoonizer",
        }
    }


def create_model(model_name: str, device: str = "cpu", 
                 variant: Optional[str] = None,
                 weights_dir: str = "models/weights") -> BaseCartoonModel:
    """Create and return a model instance.

    Args:
        model_name: Model identifier
        device: Device to run on ('cpu' or 'cuda:0')
        variant: Model variant (for models that support variants)
        weights_dir: Directory for model weights

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not recognized
    """
    model_name = model_name.lower().strip()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_class = MODEL_REGISTRY[model_name]

    kwargs = {"device": device, "weights_dir": weights_dir}
    if variant and model_name == "animeganv2":
        kwargs["variant"] = variant

    logger.info(f"Creating model: {model_name} on {device}")
    return model_class(**kwargs)


def list_models():
    """Print available models in a formatted way."""
    models = get_available_models()
    print("\nAvailable Models:")
    print("-" * 50)
    for key, info in models.items():
        print(f"\n{key}: {info['name']}")
        print(f"  Framework: {info['framework']}")
        print(f"  Description: {info['description']}")
        print(f"  Source: {info['source']}")
        if info['variants']:
            print(f"  Variants:")
            for vkey, vdesc in info['variants'].items():
                print(f"    - {vkey}: {vdesc}")
    print("-" * 50 + "\n")
