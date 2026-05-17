# RVC Colab Nvidia

This is a trimmed Nvidia/Google Colab build of RVC focused on two workflows:

- inference with `.pth` models and optional `.index` retrieval files
- training a pitch-guided or non-pitch-guided RVC model, then building the retrieval index

Removed from this build: DirectML/AMD/Intel paths, realtime audio-device UI, UVR vocal separation, ONNX demos, Docker/Windows launchers, multilingual docs, local VM storage, and the large bundled release archive.

## Use In Colab

Open the notebook and run the cells in order:

[Open RVC_Colab_Nvidia.ipynb in Colab](https://colab.research.google.com/github/Axel-Jalonen/RVC-Colab-Nvidia/blob/main/RVC_Colab_Nvidia.ipynb)

The notebook mounts Google Drive, installs dependencies, downloads the required base assets, and launches the streamlined UI.

## Google Drive Permission Screen

Colab's Drive mount uses Google's broad Drive authorization flow. If you do not want to grant that access to your main Drive, use a dedicated Google account for this notebook.

When Google asks what Drive access to allow, choose `Select all`. Selecting fewer boxes may produce a `credential propagation error unsuccessful` failure during Drive mount.

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

## License And Liability

This repository contains code derived from RVC-WebUI and preserves its MIT license terms and copyright notices.

Original Colab/Nvidia streamlining, documentation, and wrapper code in this fork are also provided under the MIT License where separable.
