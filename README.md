# AI Cartoon Video Converter

Convert any video into cartoon/anime style using state-of-the-art AI models.

## Features
- **AnimeGANv2** — Lightweight GAN with multiple styles (Paprika, Face Portrait, etc.)
- **White-box Cartoonization** — CVPR 2020 scenery/general cartoonization
- **Large Video Support** — Automatic chunking (60s–600s) with memory-efficient processing
- **Resume/Checkpoint System** — Automatically resumes interrupted jobs
- **Google Colab Ready** — One-command setup
- **Termux Compatible** — CPU fallback for Android
- **Google Drive Integration** — Auto-mount and save outputs
- **Batch Processing** — Process entire folders
- **Progress Tracking** — Real-time ETA, FPS, and percentage

## Google Colab Quick Start
```python
!git clone https://github.com/YOUR_USERNAME/AI_CARTOON_VIDEO_CONVERTER.git
%cd AI_CARTOON_VIDEO_CONVERTER
!bash install.sh
!python run.py
```

## Termux Quick Start
```bash
pkg update
pkg install python ffmpeg
pip install -r requirements.txt
python run.py
```

## Usage
```bash
# Interactive mode
python run.py

# CLI mode
python run.py --input video.mp4 --model animeganv2 --resolution 720p
python run.py --url "https://example.com/video.mp4"
python run.py --resume job_1234567890
python run.py --batch /path/to/videos
python run.py --test
```

## Models
| Model | Framework | Source |
|-------|-----------|--------|
| AnimeGANv2 | PyTorch | bryandlee/animegan2-pytorch |
| White-box Cartoonization | TensorFlow SavedModel | sayakpaul/whitebox-cartoonizer |

## Project Structure
See `docs/` for detailed guides:
- `COLAB.md` — Google Colab specific instructions
- `TERMUX.md` — Android/Termux instructions
- `TROUBLESHOOTING.md` — Common errors and fixes
- `MODELS.md` — Model details and licenses

## License
MIT License. See LICENSE for third-party model licenses.
