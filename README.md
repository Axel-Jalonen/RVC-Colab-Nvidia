# RVC Colab Nvidia

This is a trimmed Nvidia/Google Colab build of RVC focused on two workflows:

- inference with `.pth` models and optional `.index` retrieval files
- training a pitch-guided or non-pitch-guided RVC model, then building the retrieval index

Removed from this build: DirectML/AMD/Intel paths, realtime audio-device UI, UVR vocal separation, ONNX demos, Docker/Windows launchers, multilingual docs, local VM storage, and the large bundled release archive.

## Use In Colab

Open the notebook and run the cells in order:

[Open RVC_Colab_Nvidia.ipynb in Colab](https://colab.research.google.com/github/Axel-Jalonen/RVC-Colab-Nvidia/blob/main/RVC_Colab_Nvidia.ipynb)

The notebook mounts Google Drive, installs dependencies, downloads the required base assets, and launches the streamlined UI.

## Canonical Drive Layout

This repo intentionally uses one storage location only:

```text
/content/drive/MyDrive/RVC-Colab/
```

Use these folders:

```text
/content/drive/MyDrive/RVC-Colab/models/*.pth
/content/drive/MyDrive/RVC-Colab/indices/*.index
/content/drive/MyDrive/RVC-Colab/datasets/
/content/drive/MyDrive/RVC-Colab/outputs/
/content/drive/MyDrive/RVC-Colab/logs/
/content/drive/MyDrive/RVC-Colab/assets/
```

If Google Drive is not mounted, the app exits instead of falling back to the Colab VM filesystem.

## Local Checks

This repository is built for Colab runtime execution. Local checks are limited to source validation:

```bash
python -m compileall app_colab.py configs infer tools
python -m json.tool RVC_Colab_Nvidia.ipynb
```
