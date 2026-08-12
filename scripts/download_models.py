#!/usr/bin/env python3
"""Pre-download model weights for offline use."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.logger import setup_logger
from modules.model_manager import create_model, get_available_models
from modules.config import get_config

logger = setup_logger("download_models")


def main():
    print("AI Cartoon Video Converter - Model Downloader")
    print("=" * 50)

    config = get_config()
    weights_dir = config.get("models_directory", "models/weights")
    os.makedirs(weights_dir, exist_ok=True)

    models = get_available_models()

    print("\nAvailable models:")
    for key, info in models.items():
        print(f"  - {key}: {info['name']}")

    print("\nThis will download model weights to:", weights_dir)
    confirm = input("Proceed? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Cancelled.")
        return

    for model_name in models.keys():
        print(f"\nDownloading {model_name}...")
        try:
            if model_name == "animeganv2":
                for variant in ["paprika", "celeba_distill", "face_paint_512_v2"]:
                    print(f"  Variant: {variant}")
                    model = create_model(model_name, device="cpu", variant=variant,
                                        weights_dir=weights_dir)
                    model.load()
                    model.unload()
            else:
                model = create_model(model_name, device="cpu", weights_dir=weights_dir)
                model.load()
                model.unload()
            print(f"  ✓ {model_name} ready")
        except Exception as e:
            print(f"  ✗ {model_name} failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
