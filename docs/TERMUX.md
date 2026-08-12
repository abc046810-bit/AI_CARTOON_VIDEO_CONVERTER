# Termux (Android) Guide

## Installation
```bash
pkg update
pkg install python ffmpeg git
git clone https://github.com/YOUR_USERNAME/AI_CARTOON_VIDEO_CONVERTER.git
cd AI_CARTOON_VIDEO_CONVERTER
pip install -r requirements.txt
```

## Usage
```bash
python run.py
```

## Important Notes
- **No GPU acceleration** on Android — processing uses CPU only
- **Speed will be significantly slower** than cloud GPUs
- Use smaller chunk sizes (60s) to prevent memory issues
- Close other apps to free RAM
- Use 480p resolution for faster processing
