"""GPU detection and management for AI Cartoon Video Converter."""
import os
import sys
import warnings
from typing import Optional, Dict, Tuple

import torch
import psutil

from .logger import get_logger

logger = get_logger("gpu")


def get_gpu_info() -> Dict:
    """Detect and return GPU information.

    Returns:
        Dictionary with GPU details
    """
    info = {
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "recommended_device": "cpu",
        "total_vram_mb": 0,
    }

    if not torch.cuda.is_available():
        logger.info("CUDA not available. Using CPU fallback.")
        return info

    info["cuda_available"] = True
    info["cuda_version"] = torch.version.cuda
    info["device_count"] = torch.cuda.device_count()

    total_vram = 0
    for i in range(info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        vram_mb = props.total_memory // (1024 * 1024)
        total_vram += vram_mb

        device_info = {
            "index": i,
            "name": props.name,
            "vram_mb": vram_mb,
            "vram_gb": round(vram_mb / 1024, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
        info["devices"].append(device_info)
        logger.info(f"GPU {i}: {props.name} | VRAM: {device_info['vram_gb']} GB")

    info["total_vram_mb"] = total_vram
    info["recommended_device"] = "cuda:0"

    logger.info(f"CUDA Version: {info['cuda_version']}")
    logger.info(f"Total VRAM: {round(total_vram / 1024, 2)} GB")

    return info


def get_optimal_batch_size(vram_mb: int, resolution: str = "720p", 
                           model: str = "animeganv2") -> int:
    """Calculate optimal batch size based on available VRAM.

    Args:
        vram_mb: Available VRAM in MB
        resolution: Target resolution
        model: Model name

    Returns:
        Recommended batch size
    """
    # Base memory per frame estimate (MB) at 720p
    base_mem = 150 if model == "animeganv2" else 200

    # Scale by resolution
    resolution_scale = {
        "480p": 0.5,
        "720p": 1.0,
        "1080p": 2.0,
        "original": 1.0,
    }.get(resolution, 1.0)

    mem_per_frame = base_mem * resolution_scale

    # Use 70% of available VRAM for batch processing
    usable_vram = vram_mb * 0.7

    batch_size = max(1, int(usable_vram / mem_per_frame))

    # Cap at reasonable limits
    if batch_size > 32:
        batch_size = 32
    elif batch_size > 16:
        batch_size = 16
    elif batch_size > 8:
        batch_size = 8
    elif batch_size > 4:
        batch_size = 4
    elif batch_size > 2:
        batch_size = 2
    else:
        batch_size = 1

    return batch_size


def auto_select_device(prefer_gpu: bool = True) -> str:
    """Automatically select the best available device.

    Args:
        prefer_gpu: Whether to prefer GPU over CPU

    Returns:
        Device string ('cuda:0', 'cpu', etc.)
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU cache cleared")


def get_system_info() -> Dict:
    """Get comprehensive system information."""
    mem = psutil.virtual_memory()

    info = {
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "ram_percent": mem.percent,
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
    }

    info.update(get_gpu_info())
    return info


def print_system_summary():
    """Print a formatted system summary."""
    info = get_system_info()

    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python: {info['python_version'].split()[0]}")
    print(f"PyTorch: {info['pytorch_version']}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"RAM: {info['ram_total_gb']} GB total, {info['ram_available_gb']} GB available")

    if info['cuda_available']:
        print(f"\nGPU Detected: YES")
        print(f"CUDA Version: {info['cuda_version']}")
        for dev in info['devices']:
            print(f"  [{dev['index']}] {dev['name']}")
            print(f"       VRAM: {dev['vram_gb']} GB")
    else:
        print(f"\nGPU Detected: NO (CPU fallback will be used)")

    print("=" * 50 + "\n")


def handle_oom(func):
    """Decorator to handle CUDA Out of Memory errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA Out of Memory! Clearing cache and retrying with smaller batch...")
                clear_gpu_cache()
                # Reduce batch size if passed as kwarg
                if 'batch_size' in kwargs and kwargs['batch_size'] > 1:
                    kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                    logger.warning(f"Reduced batch size to {kwargs['batch_size']} and retrying...")
                    return func(*args, **kwargs)
                else:
                    logger.error("Batch size is already 1. Cannot reduce further.")
                    raise
            else:
                raise
    return wrapper
