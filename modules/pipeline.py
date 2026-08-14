"""Main processing pipeline for AI Cartoon Video Converter."""
import os
import sys
import gc
import json
import time
import shutil
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

import cv2
import numpy as np

from .logger import get_logger
from .config import get_config
from .utils import (ensure_dir, format_bytes, format_duration, 
                    generate_job_id, sanitize_filename, is_colab)
from .validation import validate_video_file, validate_resolution, validate_fps
from .gpu import (get_gpu_info, get_optimal_batch_size, auto_select_device,
                  clear_gpu_cache, print_system_summary)
from .video_info import VideoInfo, analyze_video
from .downloader import auto_download, DownloadError
from .ffmpeg_utils import (check_ffmpeg, extract_audio, generate_thumbnail,
                           get_video_duration, add_audio_to_video)
from .splitter import VideoSplitter
from .audio import AudioManager
from .frame_processor import FrameProcessor
from .merger import ChunkMerger
from .model_manager import create_model, list_models, get_available_models
from .jobs import JobManager
from .resume import CheckpointManager
from .progress import ProgressTracker
from .storage import StorageManager

logger = get_logger("pipeline")


class ProcessingPipeline:
    """Main video processing pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.device = auto_select_device(prefer_gpu=True)
        self.model = None
        self.job_manager = JobManager(self.config.get("jobs_directory", "jobs"))
        self.storage = StorageManager(
            output_dir=self.config.get("output_directory", "outputs"),
            drive_dir=self.config.get("google_drive_directory", "AI_Cartoon_Output"),
            auto_mount=self.config.get("auto_mount_drive", True)
        )
        self.checkpoint_mgr: Optional[CheckpointManager] = None
        self.audio_mgr: Optional[AudioManager] = None
        self.processing_report: Dict = {}
        
    def setup(self, model_name: Optional[str] = None, 
              model_variant: Optional[str] = None,
              resolution: Optional[str] = None,
              fps: Optional[str] = None,
              chunk_duration: Optional[int] = None,
              batch_size: Optional[int] = None,
              use_drive: bool = False):
        """Setup pipeline with configuration.
        
        Args:
            model_name: Cartoon model to use
            model_variant: Model variant
            resolution: Output resolution
            fps: Output FPS
            chunk_duration: Chunk duration in seconds
            batch_size: Processing batch size
            use_drive: Whether to use Google Drive
        """
        # Check FFmpeg
        ffmpeg_ok, ffmpeg_ver = check_ffmpeg()
        if not ffmpeg_ok:
            logger.error("FFmpeg is not installed or not in PATH")
            raise RuntimeError("FFmpeg is required but not found. Install FFmpeg first.")
        logger.info(f"FFmpeg: {ffmpeg_ver}")
        
        # Print system info
        print_system_summary()
        
        # Setup config
        if model_name:
            self.config.set("model", model_name)
        if model_variant:
            self.config.set("model_variant", model_variant)
        if resolution:
            self.config.set("resolution", resolution)
        if fps:
            self.config.set("fps", fps)
        if chunk_duration:
            self.config.set("chunk_duration", chunk_duration)
        if batch_size:
            self.config.set("batch_size", batch_size)
        
        # Mount drive if requested
        if use_drive or self.config.get("google_drive_enabled", False):
            self.storage.mount_google_drive()
        
        # Determine batch size
        gpu_info = get_gpu_info()
        if batch_size is None and gpu_info["cuda_available"]:
            auto_batch = get_optimal_batch_size(
                gpu_info["total_vram_mb"],
                self.config.get("resolution", "720p"),
                self.config.get("model", "animeganv2")
            )
            self.config.set("batch_size", auto_batch)
            logger.info(f"Auto-selected batch size: {auto_batch}")
        
        logger.info("Pipeline setup complete")
    
    def create_job(self, input_path: str, job_id: Optional[str] = None) -> str:
        """Create a new processing job.
        
        Args:
            input_path: Path to input video
            job_id: Optional job ID
            
        Returns:
            Job ID
        """
        job_id = self.job_manager.create_job(job_id)
        
        # Copy input to job folder
        input_dir = self.job_manager.get_job_dir(job_id, "input")
        input_name = sanitize_filename(os.path.basename(input_path))
        job_input = os.path.join(input_dir, input_name)
        
        if os.path.abspath(input_path) != os.path.abspath(job_input):
            shutil.copy2(input_path, job_input)
        
        # Analyze video
        video_info = analyze_video(job_input)
        if not video_info.valid:
            raise ValueError(f"Invalid video: {video_info.error}")
        
        video_info.print_summary()
        
        # Save metadata
        metadata = {
            "job_id": job_id,
            "input_file": job_input,
            "input_filename": input_name,
            "video_info": video_info.to_dict(),
            "config": self.config.to_dict(),
            "created_at": datetime.now().isoformat(),
            "status": "ready",
        }
        self.job_manager.save_metadata(metadata, job_id)
        
        # Initialize checkpoint manager
        checkpoints_dir = self.job_manager.get_job_dir(job_id, "checkpoints")
        self.checkpoint_mgr = CheckpointManager(checkpoints_dir)
        
        # Initialize audio manager
        audio_dir = self.job_manager.get_job_dir(job_id, "audio")
        self.audio_mgr = AudioManager(audio_dir)
        
        return job_id
    
    def process_job(self, job_id: str) -> str:
        """Process a job from start to finish.
        
        Args:
            job_id: Job ID to process
            
        Returns:
            Path to output video
        """
        metadata = self.job_manager.load_metadata(job_id)
        if not metadata:
            raise ValueError(f"Job not found: {job_id}")
        
        input_file = metadata["input_file"]
        video_info = VideoInfo(input_file)
        
        # Update status
        metadata["status"] = "processing"
        metadata["started_at"] = datetime.now().isoformat()
        self.job_manager.save_metadata(metadata, job_id)
        
        # Load model
        self.load_model()
        
        try:
            # Extract audio
            logger.info("Extracting audio...")
            self.audio_mgr = AudioManager(self.job_manager.get_job_dir(job_id, "audio"))
            audio_path = self.audio_mgr.extract(input_file, job_id)
            
            # Split video
            chunk_duration = self.config.get("chunk_duration", 300)
            chunks_dir = self.job_manager.get_job_dir(job_id, "chunks")
            splitter = VideoSplitter(chunk_duration)
            chunks = splitter.split(input_file, chunks_dir)
            
            # Setup checkpoint manager
            checkpoints_dir = self.job_manager.get_job_dir(job_id, "checkpoints")
            self.checkpoint_mgr = CheckpointManager(checkpoints_dir)
            
            # Determine target FPS and resolution
            target_fps = self._get_target_fps(video_info)
            target_resolution = self._get_target_resolution(video_info)
            
            # Setup frame processor
            batch_size = self.config.get("batch_size", 4)
            frame_processor = FrameProcessor(batch_size=batch_size, device=self.device)
            
            # Process chunks
            processed_chunks = []
            total_frames = sum(self._estimate_chunk_frames(c, target_fps) for c in chunks)
            tracker = ProgressTracker(total_chunks=len(chunks), total_frames=total_frames)
            
            for i, chunk_path in enumerate(chunks, 1):
                chunk_id = f"chunk_{i:03d}"
                
                # Check if already done
                if self.checkpoint_mgr.is_complete(chunk_id):
                    logger.info(f"Skipping completed chunk: {chunk_id}")
                    processed_path = os.path.join(
                        self.job_manager.get_job_dir(job_id, "processed"),
                        f"{chunk_id}_cartoon.mp4"
                    )
                    if os.path.exists(processed_path):
                        processed_chunks.append(processed_path)
                        continue
                
                logger.info(f"\nProcessing {chunk_id}/{len(chunks)}: {os.path.basename(chunk_path)}")
                
                try:
                    processed_path = self._process_chunk(
                        job_id, chunk_path, chunk_id, i, len(chunks),
                        frame_processor, target_fps, target_resolution,
                        tracker
                    )
                    processed_chunks.append(processed_path)
                    self.checkpoint_mgr.mark_complete(chunk_id)
                    
                except Exception as e:
                    logger.error(f"Chunk {chunk_id} failed: {e}")
                    self.checkpoint_mgr.mark_failed(chunk_id, str(e))
                    raise
                
                # Memory cleanup between chunks
                clear_gpu_cache()
                gc.collect()
            
            # Merge chunks
            logger.info("\nMerging processed chunks...")
            output_dir = self.job_manager.get_job_dir(job_id, "output")
            final_output = os.path.join(output_dir, "final_cartoon.mp4")
            
            merger = ChunkMerger(output_dir)
            merger.merge(processed_chunks, final_output,
                        crf=self.config.get("crf", 23),
                        preset=self.config.get("preset", "medium"))
            
            # Add full audio track to final video
            if audio_path and os.path.exists(audio_path):
                logger.info("Adding audio to final video...")
                final_with_audio = os.path.join(output_dir, "final_cartoon_temp.mp4")
                add_audio_to_video(final_output, audio_path, final_with_audio,
                                  crf=self.config.get("crf", 23),
                                  preset=self.config.get("preset", "medium"))
                shutil.move(final_with_audio, final_output)
            
            # Generate thumbnail
            thumbnail_path = os.path.join(output_dir, "thumbnail.jpg")
            try:
                generate_thumbnail(final_output, thumbnail_path)
            except Exception as e:
                logger.warning(f"Thumbnail generation failed: {e}")
            
            # Generate report
            self._generate_report(job_id, metadata, video_info, processed_chunks, final_output)
            
            # Update metadata
            metadata["status"] = "completed"
            metadata["completed_at"] = datetime.now().isoformat()
            metadata["output_file"] = final_output
            self.job_manager.save_metadata(metadata, job_id)
            
            # Save to storage
            use_drive = self.config.get("google_drive_enabled", False)
            storage_result = self.storage.save_job_output(job_id, output_dir, use_drive=use_drive)
            
            tracker.print_summary()
            logger.info(f"Output saved: {storage_result['local']}")
            if "drive" in storage_result:
                logger.info(f"Drive output: {storage_result['drive']}")
            
            return final_output
            
        finally:
            # Always unload model to free GPU memory
            self.unload()
    
    def _process_chunk(self, job_id: str, chunk_path: str, chunk_id: str,
                       chunk_num: int, total_chunks: int,
                       frame_processor: FrameProcessor,
                       target_fps: float, target_resolution: Optional[tuple],
                       tracker: ProgressTracker) -> str:
        """Process a single chunk.
        
        Args:
            job_id: Job ID
            chunk_path: Chunk video path
            chunk_id: Chunk identifier
            chunk_num: Chunk number
            total_chunks: Total chunks
            frame_processor: FrameProcessor instance
            target_fps: Target FPS
            target_resolution: Target resolution (width, height) or None
            tracker: Progress tracker
            
        Returns:
            Path to processed chunk video
        """
        job_dir = os.path.join(self.job_manager.jobs_dir, job_id)
        
        # Directories for this chunk
        chunk_frame_dir = os.path.join(job_dir, "frames", chunk_id)
        chunk_processed_dir = os.path.join(job_dir, "temp", f"{chunk_id}_processed")
        ensure_dir(chunk_frame_dir)
        ensure_dir(chunk_processed_dir)
        
        # Extract frames
        logger.info(f"Extracting frames from {chunk_id}...")
        num_frames = frame_processor.extract_frames_to_dir(
            chunk_path, chunk_frame_dir, target_fps=target_fps
        )
        
        tracker.start_chunk(chunk_num, num_frames)
        
        # Process frames with AI
        logger.info(f"Applying AI cartoon effect to {num_frames} frames...")
        
        def process_fn(frame: np.ndarray) -> np.ndarray:
            """Process single frame through AI model."""
            # Resize if target resolution specified
            if target_resolution:
                frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
            
            # Run through model
            if self.model:
                result = self.model.process_frame(frame)
            else:
                result = frame
            
            tracker.update_frame(1)
            if tracker.chunk_frames_processed % 10 == 0:
                tracker.print_status()
            
            return result
        
        processed_count = frame_processor.process_frames(
            chunk_frame_dir, chunk_processed_dir, process_fn,
            desc=f"Chunk {chunk_num}/{total_chunks}"
        )
        
        # Assemble chunk video
        processed_chunk_path = os.path.join(
            self.job_manager.get_job_dir(job_id, "processed"),
            f"{chunk_id}_cartoon.mp4"
        )
        
        # Extract chunk audio if exists
        chunk_audio = None
        try:
            chunk_audio_dir = os.path.join(job_dir, "temp", "audio")
            ensure_dir(chunk_audio_dir)
            chunk_audio = os.path.join(chunk_audio_dir, f"{chunk_id}_audio.aac")
            from .ffmpeg_utils import extract_audio
            extract_audio(chunk_path, chunk_audio, codec="aac", bitrate="192k")
        except Exception:
            chunk_audio = None
        
        frame_processor.frames_to_video(
            chunk_processed_dir, processed_chunk_path,
            fps=target_fps, audio_path=chunk_audio,
            crf=self.config.get("crf", 23),
            preset=self.config.get("preset", "medium")
        )
        
        # Cleanup frames
        if self.config.get("cleanup_temp_on_success", False):
            frame_processor.cleanup_frames(chunk_frame_dir)
            frame_processor.cleanup_frames(chunk_processed_dir)
        
        logger.info(f"{chunk_id} complete: {processed_chunk_path}")
        return processed_chunk_path
    
    def _get_target_fps(self, video_info: VideoInfo) -> float:
        """Determine target FPS from config."""
        fps_setting = self.config.get("fps", "original")
        if fps_setting == "original" or fps_setting is None:
            return video_info.fps
        return float(fps_setting)
    
    def _get_target_resolution(self, video_info: VideoInfo) -> Optional[tuple]:
        """Determine target resolution from config."""
        res_setting = self.config.get("resolution", "original")
        if res_setting == "original" or res_setting is None:
            return None
        
        # Map resolution to width
        width_map = {
            "480p": 854,
            "720p": 1280,
            "1080p": 1920,
        }
        
        if res_setting not in width_map:
            return None
        
        target_width = width_map[res_setting]
        aspect = video_info.width / video_info.height if video_info.height > 0 else 16/9
        target_height = int(target_width / aspect)
        
        # Ensure even dimensions
        target_width = target_width // 2 * 2
        target_height = target_height // 2 * 2
        
        # Don't upscale
        if target_width > video_info.width:
            logger.info(f"Target resolution {res_setting} larger than original. Using original.")
            return None
        
        return (target_width, target_height)
    
    def _estimate_chunk_frames(self, chunk_path: str, target_fps: float) -> int:
        """Estimate number of frames in a chunk."""
        duration = get_video_duration(chunk_path)
        return int(duration * target_fps)
    
    def _generate_report(self, job_id: str, metadata: dict, 
                         video_info: VideoInfo, processed_chunks: list,
                         output_file: str):
        """Generate processing report JSON."""
        start_time = metadata.get("started_at", "")
        end_time = datetime.now().isoformat()
        
        # Calculate total processing time
        total_seconds = 0
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            total_seconds = (end_dt - start_dt).total_seconds()
        except Exception:
            pass
        
        report = {
            "job_id": job_id,
            "input_file": metadata.get("input_file", ""),
            "input_duration": video_info.duration,
            "input_resolution": f"{video_info.width}x{video_info.height}",
            "input_fps": video_info.fps,
            "model": self.config.get("model", "unknown"),
            "model_variant": self.config.get("model_variant", ""),
            "output_resolution": self.config.get("resolution", "original"),
            "output_fps": self.config.get("fps", "original"),
            "chunk_duration": self.config.get("chunk_duration", 300),
            "number_of_chunks": len(processed_chunks),
            "completed_chunks": len(processed_chunks),
            "processing_start_time": start_time,
            "processing_end_time": end_time,
            "total_processing_time": total_seconds,
            "total_processing_time_formatted": f"{int(total_seconds // 3600):02d}:{int((total_seconds % 3600) // 60):02d}:{int(total_seconds % 60):02d}",
            "GPU_information": get_gpu_info(),
            "errors": [],
            "warnings": [],
            "output_file": output_file,
            "output_size": os.path.getsize(output_file) if os.path.exists(output_file) else 0,
            "output_size_formatted": format_bytes(os.path.getsize(output_file)) if os.path.exists(output_file) else "0 B",
        }
        
        report_path = os.path.join(
            self.job_manager.get_job_dir(job_id, "output"),
            "processing_report.json"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved: {report_path}")
    
    def resume_job(self, job_id: str) -> str:
        """Resume a previously interrupted job.
        
        Args:
            job_id: Job ID to resume
            
        Returns:
            Path to output video
        """
        logger.info(f"Resuming job: {job_id}")
        
        if not self.job_manager.job_exists(job_id):
            raise ValueError(f"Job not found: {job_id}")
        
        metadata = self.job_manager.load_metadata(job_id)
        
        # Restore config from metadata
        if "config" in metadata:
            for key, value in metadata["config"].items():
                self.config.set(key, value)
        
        return self.process_job(job_id)
    
    def cleanup_job(self, job_id: str, keep_output: bool = True):
        """Cleanup temporary files for a job.
        
        Args:
            job_id: Job ID
            keep_output: Whether to preserve output files
        """
        job_dir = os.path.join(self.job_manager.jobs_dir, job_id)
        if not os.path.exists(job_dir):
            return
        
        folders_to_clean = ["chunks", "frames", "temp"]
        
        for folder in folders_to_clean:
            path = os.path.join(job_dir, folder)
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    logger.info(f"Cleaned: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean {path}: {e}")
        
        logger.info(f"Job cleanup complete: {job_id}")
    
    def unload(self):
        """Unload model and free resources."""
        if self.model:
            self.model.unload()
            self.model = None
        clear_gpu_cache()
        gc.collect()
