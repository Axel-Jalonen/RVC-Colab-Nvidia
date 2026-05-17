# RVC Colab Nvidia

This is a trimmed Nvidia/Google Colab build of RVC focused on two workflows:

- inference with `.pth` models and optional `.index` retrieval files
- training a pitch-guided or non-pitch-guided RVC model, then building the retrieval index

Everything is driven from the notebook through a small Python API (`rvc_colab`). There is no web UI — Colab's TOS disallows tunneled web apps, and the notebook-native flow plays better with cells and `IPython.display.Audio`.

Removed from this build: DirectML/AMD/Intel paths, realtime audio-device UI, UVR vocal separation, ONNX demos, Docker/Windows launchers, multilingual docs, the Gradio UI, local VM storage, and the large bundled release archive.

## Use In Colab

Open the notebook and run the cells in order:

[Open RVC_Colab_Nvidia.ipynb in Colab](https://colab.research.google.com/github/Axel-Jalonen/RVC-Colab-Nvidia/blob/main/RVC_Colab_Nvidia.ipynb)

The notebook mounts Drive, installs dependencies, downloads the base assets, and then imports `rvc_colab` so you can train and convert directly in cells.

## API At A Glance

```python
import rvc_colab as rvc

rvc.runtime_status()                            # GPU + storage info
rvc.list_models(), rvc.list_indices()           # what's in Drive

model = rvc.load_model('my_voice.pth', index='my_voice.index')
sr, audio = rvc.convert(model, 'input.wav', transpose=0,
                        output_path='/.../outputs/converted.wav')

rvc.train_voice(
    exp_name='my_voice',
    dataset_dir='/content/drive/MyDrive/RVC-Colab/datasets/my_voice',
    sr='40k', total_epoch=100, batch_size=8,
)
```

Training prints subprocess output live into the cell. For finer control the individual steps are also exposed: `preprocess_dataset`, `extract_features`, `train_model`, `build_index`.

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
