"""Memory-efficient frame processing with batch support."""
import os
import gc
from typing import List, Callable, Optional, Tuple, Generator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    # Fallback tqdm stub
    class tqdm:
        def __init__(self, total=None, desc="", unit=""):
            self.total = total
            self.desc = desc
            self.n = 0
            print(f"{desc}: 0/{total} {unit}")
        def update(self, n=1):
            self.n += n
        def __enter__(self):
            return self
        def __exit__(self, *args):
            print(f"{self.desc}: {self.n}/{self.total} {unit} - done")

from .logger import get_logger
from .utils import ensure_dir, format_bytes
from .gpu import clear_gpu_cache

logger = get_logger("frame_processor")


class FrameProcessor:
    """Process video frames in memory-efficient batches."""
    
    def __init__(self, batch_size: int = 4, device: str = "cpu"):
        self.batch_size = batch_size
        self.device = device
        self.frame_count = 0
        self.processed_count = 0
        
    def extract_frames_to_dir(self, video_path: str, output_dir: str,
                              target_fps: Optional[float] = None) -> int:
        """Extract all frames from video to directory.
        
        Args:
            video_path: Source video
            output_dir: Frame output directory
            target_fps: Optional FPS limit
            
        Returns:
            Number of frames extracted
        """
        ensure_dir(output_dir)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1
        if target_fps and original_fps > target_fps:
            frame_interval = int(round(original_fps / target_fps))
        
        frame_idx = 0
        saved_idx = 0
        
        logger.info(f"Extracting frames from {video_path}...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.png")
                cv2.imwrite(frame_path, frame)
                saved_idx += 1
            
            frame_idx += 1
        
        cap.release()
        self.frame_count = saved_idx
        logger.info(f"Extracted {saved_idx} frames to {output_dir}")
        return saved_idx
    
    def process_frames(self, frame_dir: str, output_dir: str,
                       process_fn: Callable[[np.ndarray], np.ndarray],
                       desc: str = "Processing") -> int:
        """Process frames in batches with progress tracking.
        
        Args:
            frame_dir: Input frames directory
            output_dir: Output frames directory
            process_fn: Function that takes and returns numpy array (BGR)
            desc: Progress bar description
            
        Returns:
            Number of frames processed
        """
        ensure_dir(output_dir)
        
        frame_files = sorted([f for f in os.listdir(frame_dir) 
                              if f.startswith('frame_') and f.endswith('.png')])
        
        if not frame_files:
            logger.warning(f"No frames found in {frame_dir}")
            return 0
        
        total = len(frame_files)
        self.processed_count = 0
        
        logger.info(f"Processing {total} frames with batch_size={self.batch_size}")
        
        with tqdm(total=total, desc=desc, unit="frame") as pbar:
            for i in range(0, total, self.batch_size):
                batch_files = frame_files[i:i + self.batch_size]
                batch_frames = []
                
                # Load batch
                for fname in batch_files:
                    fpath = os.path.join(frame_dir, fname)
                    frame = cv2.imread(fpath)
                    if frame is not None:
                        batch_frames.append((fname, frame))
                
                if not batch_frames:
                    continue
                
                # Process batch
                try:
                    processed = self._process_batch(batch_frames, process_fn)
                    
                    # Save processed frames
                    for fname, frame in processed:
                        out_path = os.path.join(output_dir, fname)
                        cv2.imwrite(out_path, frame)
                    
                    self.processed_count += len(processed)
                    pbar.update(len(processed))
                    
                except Exception as e:
                    logger.error(f"Batch processing failed at frame {i}: {e}")
                    # Save original frames as fallback
                    for fname, frame in batch_frames:
                        out_path = os.path.join(output_dir, fname)
                        cv2.imwrite(out_path, frame)
                    self.processed_count += len(batch_frames)
                    pbar.update(len(batch_frames))
                
                # Memory cleanup
                if self.device.startswith('cuda'):
                    clear_gpu_cache()
                gc.collect()
        
        return self.processed_count
    
    def _process_batch(self, batch: List[Tuple[str, np.ndarray]], 
                       process_fn: Callable) -> List[Tuple[str, np.ndarray]]:
        """Process a single batch of frames.
        
        Args:
            batch: List of (filename, frame) tuples
            process_fn: Processing function
            
        Returns:
            List of (filename, processed_frame) tuples
        """
        results = []
        for fname, frame in batch:
            try:
                processed = process_fn(frame)
                results.append((fname, processed))
            except Exception as e:
                logger.warning(f"Frame processing failed for {fname}: {e}")
                results.append((fname, frame))  # Fallback to original
        return results
    
    def frames_to_video(self, frame_dir: str, output_path: str, 
                        fps: float, audio_path: Optional[str] = None,
                        crf: int = 23, preset: str = "medium") -> str:
        """Assemble frames back into video.
        
        Args:
            frame_dir: Directory containing frame_*.png files
            output_path: Output video path
            fps: Frame rate
            audio_path: Optional audio file
            crf: Quality
            preset: Encoding preset
            
        Returns:
            Output video path
        """
        from .ffmpeg_utils import frames_to_video
        
        frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
        frames_to_video(frame_pattern, output_path, fps, audio_path, crf, preset)
        return output_path
    
    def cleanup_frames(self, frame_dir: str):
        """Remove extracted frame files.
        
        Args:
            frame_dir: Directory with frames to remove
        """
        if not os.path.exists(frame_dir):
            return
        
        count = 0
        for f in os.listdir(frame_dir):
            if f.startswith('frame_') and f.endswith('.png'):
                try:
                    os.remove(os.path.join(frame_dir, f))
                    count += 1
                except Exception:
                    pass
        
        logger.debug(f"Cleaned up {count} frames from {frame_dir}")
