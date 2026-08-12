#!/usr/bin/env python3
"""Cleanup utility for jobs and temporary files."""
import os
import sys
import shutil
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.jobs import JobManager
from modules.config import get_config


def main():
    parser = argparse.ArgumentParser(description="Cleanup utility")
    parser.add_argument("--job", help="Clean specific job")
    parser.add_argument("--all-jobs", action="store_true", help="Clean all jobs temp files")
    parser.add_argument("--temp", action="store_true", help="Clean global temp directory")
    parser.add_argument("--cache", action="store_true", help="Clean cache directory")
    parser.add_argument("--logs", action="store_true", help="Clean old logs")
    parser.add_argument("--everything", action="store_true", 
                       help="WARNING: Remove everything including outputs")

    args = parser.parse_args()
    config = get_config()

    if args.everything:
        confirm = input("WARNING: This will delete ALL jobs, outputs, cache, and temp files. Type 'yes' to confirm: ")
        if confirm == "yes":
            for folder in ["jobs", "outputs", "cache", "temp", "logs"]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    print(f"Removed: {folder}")
        return

    if args.job:
        jm = JobManager()
        if jm.job_exists(args.job):
            job_dir = os.path.join(jm.jobs_dir, args.job)
            for subdir in ["chunks", "frames", "temp"]:
                path = os.path.join(job_dir, subdir)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Cleaned: {path}")
        else:
            print(f"Job not found: {args.job}")

    if args.all_jobs:
        jm = JobManager()
        for job_id in jm.list_jobs():
            job_dir = os.path.join(jm.jobs_dir, job_id)
            for subdir in ["chunks", "frames", "temp"]:
                path = os.path.join(job_dir, subdir)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"Cleaned: {path}")

    if args.temp:
        temp_dir = config.get("temp_directory", "temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            print(f"Cleaned: {temp_dir}")

    if args.cache:
        cache_dir = config.get("cache_directory", "cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
            print(f"Cleaned: {cache_dir}")

    if args.logs:
        log_dir = config.get("log_directory", "logs")
        if os.path.exists(log_dir):
            # Keep last 10 logs
            logs = sorted([f for f in os.listdir(log_dir) if f.endswith('.log')])
            for old_log in logs[:-10]:
                os.remove(os.path.join(log_dir, old_log))
                print(f"Removed old log: {old_log}")


if __name__ == "__main__":
    main()
