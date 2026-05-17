# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Scope

This is a deliberately trimmed Nvidia/Google Colab fork of RVC-WebUI. It supports exactly two workflows: inference with `.pth` voice models (+ optional `.index` retrieval files) and pitch-guided / non-pitch-guided training followed by index building. DirectML/AMD/Intel paths, realtime audio I/O, UVR vocal separation, ONNX demos, Docker/Windows launchers, and multilingual docs have been intentionally removed — do not reintroduce them when modifying code.

## Common Commands

```bash
# Install (Colab; Python 3.12, CUDA Nvidia runtime)
python -m pip install -U pip setuptools wheel
pip install -r requirements-colab.txt

# Download required base assets (HuBERT, RMVPE, pretrained G/D weights) into Drive
python tools/download_colab_assets.py            # add --no-pretrained for inference-only
                                                 # add --overwrite to replace existing files
```

There is no test suite, linter, or formatter. The user-facing entrypoint is the `rvc_colab` Python module, driven from `RVC_Colab_Nvidia.ipynb`. There is **no Gradio UI** — it was removed because Colab disallows tunneled web apps and the notebook-native API is a better fit. Do not reintroduce a web server.

## Architecture

### Drive-first storage contract

`rvc_colab.py` hard-requires `/content/drive/MyDrive` to exist at import time and roots all data under `/content/drive/MyDrive/RVC-Colab/`. The module sets these env vars (consumed throughout the code) before importing `configs.config` or `infer.modules.vc.modules`:

- `weight_root` → `models/`              (user `.pth` models)
- `index_root` → `logs/`                 (per-experiment training output)
- `outside_index_root` → `indices/`      (copied final `.index` files)
- `hubert_root`, `rmvpe_root`, `pretrained_root`, `pretrained_v2_root` → `assets/...`
- `dataset_root` → `datasets/`, `output_root` → `outputs/`

Any new code that needs a path MUST read these env vars, not hardcode Drive paths. The asset downloader (`tools/download_colab_assets.py`) writes into the same layout from the same `STORAGE_ROOT` constant.

### Three-layer code structure

1. **`rvc_colab.py`** — notebook-facing API. Sets up the Drive env vars, instantiates the `Config` singleton + a single `VC`, and exposes module-level functions: `runtime_status`, `list_models`, `list_indices`, `paths`, `load_model` / `convert` / `convert_batch` for inference, and `preprocess_dataset` / `extract_features` / `train_model` / `build_index` (plus the one-shot `train_voice`) for training. Subprocess stdout is streamed live via `_stream()`. Because `Config.arg_parse` reads `sys.argv`, the module stubs argv during the `Config`/`VC` import to survive Jupyter's ipykernel flags — keep that shim if you reorganize imports.

2. **`configs/config.py`** — `Config` singleton that parses CLI args, probes the GPU, decides fp16 vs fp32 (forces fp32 on 16-series / P40 / P10 / 1060–1080 / CPU), tunes pipeline window sizes (`x_pad`, `x_query`, `x_center`, `x_max`) per VRAM, and copies `configs/v{1,2}/*.json` into `configs/inuse/` for in-place editing. `use_fp32_config` rewrites those JSONs by string-replacing `true`→`false`; preserve that behavior if touching it.

3. **`infer/`** — the model code.
   - `infer/modules/vc/{modules.py,pipeline.py}` — inference entrypoints (`VC.get_vc`, `vc_single`, `vc_multi`) and the conversion pipeline.
   - `infer/modules/train/{preprocess.py,extract_feature_print.py,train.py,extract/*}` — training stages invoked as subprocesses by `rvc_colab.py`. Argument order is positional and matches the `_stream` call sites — do not reorder without updating both ends.
   - `infer/lib/infer_pack/` — synthesizer/generator models. `infer/lib/{rmvpe.py,slicer2.py,audio.py}` — F0 extraction, slicing, audio I/O. `infer/lib/train/` — data utils, losses, mel processing, checkpoint utils.

### Training filelist convention

`_write_filelist` in `rvc_colab.py` enforces the `0_gt_wavs` / `2a_f0` / `2b-f0nsf` / `3_feature256|3_feature768` directory layout under `{index_root}/{exp_name}/`. Feature dim is 256 for v1, 768 for v2. The filelist always appends two `logs/mute/...` rows (silence padding) — required by upstream training; keep it.

### Index building

`build_index` uses `faiss.index_factory("IVF{n_ivf},Flat")` with `n_ivf = min(16*sqrt(N), N//39)`. When features exceed 200k vectors it first reduces with `MiniBatchKMeans(n_clusters=10000)`. Two files are written: `trained_IVF*.index` (post-train only) and `added_IVF*.index` (post-add); the added one is copied to `outside_index_root` so `list_indices()` finds it.

## Constraints When Modifying

- Keep dependencies minimal — `requirements-colab.txt` is curated for Colab/Python 3.12 (note `ve-fairseq==0.12.3` which pins `numpy<2`, no `torchcrepe`, no `gradio`). Don't add packages casually. The NumPy<2 pin causes pip warnings about Colab-preinstalled packages (cupy/jax/opencv) wanting NumPy≥2; those are expected and don't affect RVC.
- Do not add CPU/AMD/Intel/DirectML branches or realtime device code — out of scope for this fork.
- Do not reintroduce a Gradio/web UI. Colab disallows tunneled web apps; the API is the entrypoint.
- Code derives from RVC-Project/Retrieval-based-Voice-Conversion-WebUI under MIT; preserve `LICENSE` and `NOTICE.md` references when refactoring.
