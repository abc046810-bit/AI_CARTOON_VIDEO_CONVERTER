#!/usr/bin/env python3
"""Batch processing script."""
import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import batch_process
from modules.pipeline import ProcessingPipeline


def main():
    parser = argparse.ArgumentParser(description="Batch process videos")
    parser.add_argument("folder", help="Folder containing videos")
    parser.add_argument("--model", default="animeganv2", choices=["animeganv2", "whitebox"])
    parser.add_argument("--variant", default="paprika")
    parser.add_argument("--resolution", default="original", choices=["original", "480p", "720p", "1080p"])
    parser.add_argument("--fps", default="original")
    parser.add_argument("--chunk", type=int, default=300)
    parser.add_argument("--drive", action="store_true", help="Save to Google Drive")

    args = parser.parse_args()

    pipeline = ProcessingPipeline()
    pipeline.setup(
        model_name=args.model,
        model_variant=args.variant,
        resolution=args.resolution,
        fps=args.fps,
        chunk_duration=args.chunk,
        use_drive=args.drive
    )

    batch_process(pipeline, args.folder)


if __name__ == "__main__":
    main()
