"""Main application logic for AI Cartoon Video Converter."""
import os
import sys
import argparse
from typing import Optional

from modules.logger import setup_logger
from modules.config import get_config
from modules.utils import clear_screen, print_header, is_colab, is_valid_url, is_google_drive_url
from modules.validation import (validate_video_file, validate_video_url, 
                                 validate_google_drive_url, validate_resolution,
                                 validate_fps, validate_chunk_duration, validate_model)
from modules.pipeline import ProcessingPipeline
from modules.model_manager import list_models
from modules.jobs import JobManager


def interactive_menu():
    """Display interactive CLI menu."""
    clear_screen()
    print_header()
    
    print("Select input source:")
    print("  1. Local video file")
    print("  2. Direct video URL")
    print("  3. Google Drive file/link")
    print("  4. Resume previous job")
    print("  5. Batch process folder")
    print("  6. Settings")
    print("  7. Help")
    print("  8. Exit")
    print()
    
    choice = input("Choose option (1-8): ").strip()
    return choice


def get_local_file() -> str:
    """Get local file path from user."""
    while True:
        path = input("\nEnter video file path: ").strip().strip('"').strip("'")
        valid, msg = validate_video_file(path)
        if valid:
            return path
        print(f"Error: {msg}")


def get_url() -> str:
    """Get direct URL from user."""
    while True:
        url = input("\nEnter direct video URL: ").strip()
        valid, msg = validate_video_url(url)
        if valid:
            if msg.startswith("WARNING"):
                print(msg)
            return url
        print(f"Error: {msg}")


def get_drive_url() -> str:
    """Get Google Drive URL from user."""
    while True:
        url = input("\nEnter Google Drive share link: ").strip()
        valid, file_id, msg = validate_google_drive_url(url)
        if valid:
            return url
        print(f"Error: {msg}")


def select_model() -> tuple:
    """Interactive model selection."""
    print("\nSelect Cartoon Model:")
    print("  1. AnimeGANv2 - Lightweight GAN for photo animation")
    print("     Variants: celeba_distill, face_paint_512_v1, face_paint_512_v2, paprika")
    print("  2. White-box Cartoonization - CVPR 2020 scenery/general cartoonization")
    
    while True:
        choice = input("\nChoose model (1-2): ").strip()
        if choice == "1":
            print("\nSelect variant:")
            print("  1. paprika (default - general/landscape)")
            print("  2. celeba_distill (face style)")
            print("  3. face_paint_512_v1 (face portrait v1)")
            print("  4. face_paint_512_v2 (face portrait v2)")
            vchoice = input("Choose variant (1-4): ").strip()
            variants = {"1": "paprika", "2": "celeba_distill", 
                       "3": "face_paint_512_v1", "4": "face_paint_512_v2"}
            return "animeganv2", variants.get(vchoice, "paprika")
        elif choice == "2":
            return "whitebox", "default"
        print("Invalid choice")


def select_quality() -> tuple:
    """Interactive quality settings."""
    print("\nSelect Resolution:")
    print("  1. Original (no resizing)")
    print("  2. 480p")
    print("  3. 720p")
    print("  4. 1080p")
    
    res_map = {"1": "original", "2": "480p", "3": "720p", "4": "1080p"}
    while True:
        choice = input("Choose resolution (1-4): ").strip()
        if choice in res_map:
            resolution = res_map[choice]
            break
        print("Invalid choice")
    
    print("\nSelect FPS:")
    print("  1. Original")
    print("  2. 24")
    print("  3. 25")
    print("  4. 30")
    print("  5. 60")
    
    fps_map = {"1": "original", "2": "24", "3": "25", "4": "30", "5": "60"}
    while True:
        choice = input("Choose FPS (1-5): ").strip()
        if choice in fps_map:
            fps = fps_map[choice]
            break
        print("Invalid choice")
    
    print("\nSelect chunk duration (for large videos):")
    print("  1. 60 seconds")
    print("  2. 120 seconds")
    print("  3. 300 seconds (default)")
    print("  4. 600 seconds")
    
    chunk_map = {"1": 60, "2": 120, "3": 300, "4": 600}
    while True:
        choice = input("Choose chunk duration (1-4): ").strip()
        if choice in chunk_map:
            chunk = chunk_map[choice]
            break
        print("Invalid choice")
    
    return resolution, fps, chunk


def select_drive_option() -> bool:
    """Ask if user wants Google Drive output."""
    if not is_colab():
        return False
    
    print("\nSave output to Google Drive?")
    print("  1. Yes")
    print("  2. No")
    choice = input("Choose (1-2): ").strip()
    return choice == "1"


def resume_job_interactive(pipeline: ProcessingPipeline):
    """Interactive job resume."""
    job_mgr = JobManager()
    jobs = job_mgr.list_jobs()
    
    if not jobs:
        print("\nNo previous jobs found.")
        input("Press Enter to continue...")
        return
    
    print("\nAvailable jobs:")
    for i, job_id in enumerate(jobs[:10], 1):
        meta = job_mgr.load_metadata(job_id)
        status = meta.get("status", "unknown")
        filename = meta.get("input_filename", "unknown")
        print(f"  {i}. {job_id} - {filename} ({status})")
    
    choice = input("\nSelect job to resume (number): ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(jobs):
            job_id = jobs[idx]
            use_drive = select_drive_option()
            pipeline.setup(use_drive=use_drive)
            output = pipeline.resume_job(job_id)
            print(f"\nOutput saved: {output}")
    except Exception as e:
        print(f"Error: {e}")


def batch_process(pipeline: ProcessingPipeline, folder_path: str):
    """Process all videos in a folder."""
    if not os.path.isdir(folder_path):
        print(f"Error: Not a directory: {folder_path}")
        return
    
    videos = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(('.mp4', '.mkv', '.mov', '.avi', '.webm'))]
    
    if not videos:
        print("No video files found in directory.")
        return
    
    print(f"\nFound {len(videos)} videos to process.")
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    model, variant = select_model()
    resolution, fps, chunk = select_quality()
    use_drive = select_drive_option()
    
    pipeline.setup(model_name=model, model_variant=variant,
                   resolution=resolution, fps=fps, chunk_duration=chunk,
                   use_drive=use_drive)
    
    for i, video in enumerate(videos, 1):
        print(f"\n{'='*50}")
        print(f"Processing {i}/{len(videos)}: {video}")
        print('='*50)
        
        try:
            video_path = os.path.join(folder_path, video)
            job_id = pipeline.create_job(video_path)
            output = pipeline.process_job(job_id)
            print(f"Completed: {output}")
        except Exception as e:
            print(f"Failed: {e}")
        
        if i < len(videos):
            import time
            time.sleep(pipeline.config.get("batch_sleep_interval", 10))


def run_test_mode():
    """Run lightweight test mode."""
    print_header()
    print("RUNNING TEST MODE")
    print("=" * 50)
    
    # Test imports
    print("\n[1/5] Testing imports...")
    try:
        import torch
        import cv2
        import numpy
        import PIL
        import yaml
        print("  ✓ All core imports successful")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Test FFmpeg
    print("\n[2/5] Testing FFmpeg...")
    from modules.ffmpeg_utils import check_ffmpeg
    ok, ver = check_ffmpeg()
    if ok:
        print(f"  ✓ FFmpeg found: {ver[:50]}")
    else:
        print("  ✗ FFmpeg not found")
        return False
    
    # Test GPU
    print("\n[3/5] Testing GPU...")
    from modules.gpu import get_gpu_info
    info = get_gpu_info()
    if info["cuda_available"]:
        print(f"  ✓ GPU detected: {info['devices'][0]['name']}")
    else:
        print("  ⚠ No GPU detected (CPU mode)")
    
    # Test config
    print("\n[4/5] Testing configuration...")
    from modules.config import get_config
    config = get_config()
    print(f"  ✓ Config loaded: model={config.get('model')}")
    
    # Test job manager
    print("\n[5/5] Testing job manager...")
    from modules.jobs import JobManager
    jm = JobManager()
    test_job = jm.create_job()
    print(f"  ✓ Job creation works: {test_job}")
    
    # Cleanup test job
    import shutil
    shutil.rmtree(os.path.join("jobs", test_job), ignore_errors=True)
    
    print("\n" + "=" * 50)
    print("TEST MODE COMPLETE - All checks passed!")
    print("=" * 50)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Cartoon Video Converter - Convert videos to cartoon style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Interactive mode
  python run.py --input video.mp4        # Process local file
  python run.py --url "http://..."       # Process from URL
  python run.py --model animeganv2       # Specify model
  python run.py --resolution 720p        # Specify resolution
  python run.py --resume JOB_ID          # Resume job
  python run.py --batch /path/to/videos  # Batch process
  python run.py --status JOB_ID          # Show job status
  python run.py --cleanup JOB_ID         # Cleanup temp files
  python run.py --test                   # Run tests
        """
    )
    
    parser.add_argument("--input", "-i", help="Local video file path")
    parser.add_argument("--url", "-u", help="Direct video URL")
    parser.add_argument("--drive", "-d", help="Google Drive share link")
    parser.add_argument("--model", "-m", choices=["animeganv2", "whitebox"],
                       help="Cartoon model to use")
    parser.add_argument("--variant", "-v", help="Model variant")
    parser.add_argument("--resolution", "-r", choices=["original", "480p", "720p", "1080p"],
                       help="Output resolution")
    parser.add_argument("--fps", choices=["original", "24", "25", "30", "60"],
                       help="Output FPS")
    parser.add_argument("--chunk", "-c", type=int, choices=[60, 120, 300, 600],
                       help="Chunk duration in seconds")
    parser.add_argument("--batch", "-b", help="Batch process folder")
    parser.add_argument("--resume", help="Resume job ID")
    parser.add_argument("--status", help="Show job status")
    parser.add_argument("--cleanup", help="Cleanup job temp files")
    parser.add_argument("--drive-output", action="store_true",
                       help="Save output to Google Drive")
    parser.add_argument("--test", action="store_true",
                       help="Run test mode")
    parser.add_argument("--config", help="Custom config file path")
    
    args = parser.parse_args()
    
    # Test mode
    if args.test:
        success = run_test_mode()
        sys.exit(0 if success else 1)
    
    # Status check
    if args.status:
        jm = JobManager()
        status = jm.get_job_status(args.status)
        import json
        print(json.dumps(status, indent=2))
        return
    
    # Cleanup
    if args.cleanup:
        jm = JobManager()
        pipeline = ProcessingPipeline()
        pipeline.cleanup_job(args.cleanup)
        return
    
    # Initialize pipeline
    pipeline = ProcessingPipeline(config_path=args.config)
    
    # Batch mode
    if args.batch:
        batch_process(pipeline, args.batch)
        return
    
    # Resume mode
    if args.resume:
        pipeline.setup(use_drive=args.drive_output)
        output = pipeline.resume_job(args.resume)
        print(f"\nOutput: {output}")
        return
    
    # Single file processing (CLI args)
    if args.input or args.url or args.drive:
        # Determine source
        source = args.input or args.url or args.drive
        is_url = args.url is not None
        is_drive = args.drive is not None
        
        # Setup
        pipeline.setup(
            model_name=args.model,
            model_variant=args.variant,
            resolution=args.resolution,
            fps=args.fps,
            chunk_duration=args.chunk,
            use_drive=args.drive_output
        )
        
        # Handle download if needed
        if is_url or is_drive:
            from modules.downloader import auto_download
            print(f"Downloading from {'Google Drive' if is_drive else 'URL'}...")
            temp_dir = os.path.join(pipeline.job_manager.jobs_dir, "temp_downloads")
            os.makedirs(temp_dir, exist_ok=True)
            source = auto_download(source, temp_dir)
            print(f"Downloaded: {source}")
        
        # Validate
        valid, msg = validate_video_file(source)
        if not valid:
            print(f"Error: {msg}")
            sys.exit(1)
        
        # Create job and process
        job_id = pipeline.create_job(source)
        output = pipeline.process_job(job_id)
        
        print(f"\n{'='*50}")
        print(f"PROCESSING COMPLETE")
        print(f"Output: {output}")
        print(f"{'='*50}")
        
        # Ask about cleanup
        cleanup = input("\nDelete temporary files? (y/n): ").strip().lower()
        if cleanup == 'y':
            pipeline.cleanup_job(job_id)
        
        return
    
    # Interactive mode
    while True:
        choice = interactive_menu()
        
        if choice == "8":
            print("\nGoodbye!")
            break
        
        elif choice == "7":
            parser.print_help()
            input("\nPress Enter to continue...")
        
        elif choice == "6":
            print("\nSettings can be modified in config.yaml")
            input("Press Enter to continue...")
        
        elif choice == "5":
            folder = input("Enter folder path: ").strip().strip('"').strip("'")
            batch_process(pipeline, folder)
            input("\nPress Enter to continue...")
        
        elif choice == "4":
            resume_job_interactive(pipeline)
            input("\nPress Enter to continue...")
        
        elif choice in ("1", "2", "3"):
            # Get source
            if choice == "1":
                source = get_local_file()
                is_download = False
            elif choice == "2":
                source = get_url()
                is_download = True
            else:
                source = get_drive_url()
                is_download = True
            
            # Select options
            model, variant = select_model()
            resolution, fps, chunk = select_quality()
            use_drive = select_drive_option()
            
            # Setup and process
            try:
                pipeline.setup(
                    model_name=model, model_variant=variant,
                    resolution=resolution, fps=fps,
                    chunk_duration=chunk, use_drive=use_drive
                )
                
                if is_download:
                    from modules.downloader import auto_download
                    print(f"\nDownloading...")
                    temp_dir = os.path.join(pipeline.job_manager.jobs_dir, "temp_downloads")
                    os.makedirs(temp_dir, exist_ok=True)
                    source = auto_download(source, temp_dir)
                    print(f"Downloaded: {source}")
                
                job_id = pipeline.create_job(source)
                output = pipeline.process_job(job_id)
                
                print(f"\n{'='*50}")
                print(f"COMPLETE! Output: {output}")
                print(f"{'='*50}")
                
                cleanup = input("\nDelete temporary files? (y/n): ").strip().lower()
                if cleanup == 'y':
                    pipeline.cleanup_job(job_id)
                
            except KeyboardInterrupt:
                print("\n\nProcessing interrupted by user.")
                print("Run with --resume to continue.")
            except Exception as e:
                print(f"\nError: {e}")
            
            input("\nPress Enter to continue...")
        
        else:
            print("Invalid choice")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
