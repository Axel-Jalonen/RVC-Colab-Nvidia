# RVC Colab Nvidia

This is a trimmed Nvidia/Google Colab build of RVC focused on two workflows:

- inference with `.pth` models and optional `.index` retrieval files
- training a pitch-guided or non-pitch-guided RVC model, then building the retrieval index

Removed from this build: DirectML/AMD/Intel paths, realtime audio-device UI, UVR vocal separation, ONNX demos, Docker/Windows launchers, multilingual docs, and the large bundled release archive.

## Colab Quick Start

Use a GPU runtime, mount Google Drive first, then run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
git clone https://github.com/Axel-Jalonen/RVC-Colab-Nvidia.git /content/RVC-Colab-Nvidia
cd /content/RVC-Colab-Nvidia
pip install -r requirements-colab.txt
python tools/download_colab_assets.py
python app_colab.py --colab
```

When Drive is mounted, the app automatically uses:

```text
/content/drive/MyDrive/RVC-Colab/models/*.pth
/content/drive/MyDrive/RVC-Colab/indices/*.index
/content/drive/MyDrive/RVC-Colab/datasets/
/content/drive/MyDrive/RVC-Colab/outputs/
/content/drive/MyDrive/RVC-Colab/logs/
```

Set `RVC_DRIVE_ROOT` before launching if you want a different Drive folder. If Drive is not mounted, the app falls back to local VM storage.

## Local Smoke Test

```bash
python -m compileall app_colab.py configs infer tools
python app_colab.py --port 7865 --noautoopen
```

The app is intentionally narrow. If a feature is not needed for Colab Nvidia inference/training, it should stay out of this fork.
