# Troubleshooting

## FFmpeg not found
Install FFmpeg:
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`
- Windows: Download from ffmpeg.org and add to PATH

## CUDA Out of Memory
The system auto-reduces batch size. If it persists:
- Reduce resolution to 480p
- Reduce chunk duration to 60s
- Close other GPU applications

## Model download fails
Check internet connection. For offline use, run first:
```bash
python scripts/download_models.py
```

## Google Drive mount fails
In Colab, manually mount:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Resume not working
Ensure the `jobs/JOB_ID/` folder still exists. Check with:
```bash
python run.py --status JOB_ID
```

## White-box model errors
Requires TensorFlow 2.x. If import fails:
```bash
pip install tensorflow huggingface-hub
```
