# Google Colab Guide

## Quick Start
1. Open a new Colab notebook
2. Change runtime to GPU: Runtime → Change runtime type → GPU
3. Run:
```python
!git clone https://github.com/YOUR_USERNAME/AI_CARTOON_VIDEO_CONVERTER.git
%cd AI_CARTOON_VIDEO_CONVERTER
!bash install.sh
!python run.py
```

## Google Drive Output
Select option 3 in the interactive menu or use `--drive-output`.
The system will auto-mount `/content/drive` and save to `MyDrive/AI_Cartoon_Output/`.

## Large Videos
Use chunk duration 300s or 600s. If Colab disconnects, simply re-run:
```python
!python run.py --resume JOB_ID
```

## Limitations
- Free Colab has ~12-hour runtime limits
- GPU VRAM may limit batch size (auto-adjusted)
- Storage is limited to your Google Drive + local disk
