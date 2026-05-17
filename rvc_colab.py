"""Notebook-native API for RVC inference and training on Colab.

Designed to be called directly from notebook cells:

    import rvc_colab as rvc

    rvc.runtime_status()
    rvc.list_models()

    model = rvc.load_model("my_voice.pth", index="my_voice.index")
    sr, audio = rvc.convert(model, "input.wav", transpose=0)

    rvc.train_voice(
        exp_name="my_voice",
        dataset_dir="/content/drive/MyDrive/RVC-Colab/datasets/my_voice",
        total_epoch=100,
    )
"""

from __future__ import annotations

import sys as _sys
# Colab's Python 3.12 ships a broken pkg_resources in /usr/lib/python3/dist-packages
# (uses pkgutil.ImpImporter, removed in 3.12). The pip-upgraded setuptools lives in
# /usr/local/... but the Debian path can shadow it. Drop it so fairseq imports work.
_sys.path[:] = [p for p in _sys.path if "/usr/lib/python3/dist-packages" not in p]
_sys.modules.pop("pkg_resources", None)

import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from random import shuffle
from typing import Iterable

import faiss
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT))

load_dotenv()
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

DRIVE_MOUNT = pathlib.Path("/content/drive/MyDrive")
STORAGE_ROOT = DRIVE_MOUNT / "RVC-Colab"

if not DRIVE_MOUNT.exists():
    raise RuntimeError(
        "Google Drive is required. In Colab, run the notebook's Drive mount cell "
        "before importing rvc_colab."
    )

os.environ.setdefault("weight_root", str(STORAGE_ROOT / "models"))
os.environ.setdefault("index_root", str(STORAGE_ROOT / "logs"))
os.environ.setdefault("outside_index_root", str(STORAGE_ROOT / "indices"))
os.environ.setdefault("hubert_root", str(STORAGE_ROOT / "assets" / "hubert"))
os.environ.setdefault("rmvpe_root", str(STORAGE_ROOT / "assets" / "rmvpe"))
os.environ.setdefault("pretrained_root", str(STORAGE_ROOT / "assets" / "pretrained"))
os.environ.setdefault("pretrained_v2_root", str(STORAGE_ROOT / "assets" / "pretrained_v2"))
os.environ.setdefault("dataset_root", str(STORAGE_ROOT / "datasets"))
os.environ.setdefault("output_root", str(STORAGE_ROOT / "outputs"))

for _path in [
    os.environ["weight_root"],
    os.environ["outside_index_root"],
    os.environ["hubert_root"],
    os.environ["rmvpe_root"],
    os.environ["pretrained_root"],
    os.environ["pretrained_v2_root"],
    os.environ["dataset_root"],
    os.environ["output_root"],
    os.environ["index_root"],
    ROOT / "TEMP",
]:
    pathlib.Path(_path).mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(ROOT / "TEMP")
torch.manual_seed(114514)

# Config.arg_parse() reads sys.argv at *instantiation* time; stub argv
# across both the import and the Config()/VC() construction so Jupyter's
# -f kernel.json flag doesn't trip the parser.
_saved_argv = sys.argv
sys.argv = ["rvc_colab"]
try:
    from configs.config import Config
    from infer.modules.vc.modules import VC
    _config = Config()
    _vc = VC(_config)
finally:
    sys.argv = _saved_argv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rvc-colab")
_current_model: str | None = None

SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


@dataclass
class Model:
    """Handle returned by load_model(); passed to convert() / convert_batch()."""

    name: str
    index: str | None = None
    protect: float = 0.33


# ---------------------------------------------------------------------------
# discovery / status
# ---------------------------------------------------------------------------

def runtime_status() -> str:
    lines = [f"Storage: {STORAGE_ROOT}"]
    if torch.cuda.is_available():
        lines.insert(0, f"CUDA: {torch.version.cuda}  device: {torch.cuda.get_device_name(0)}  "
                       f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        lines.insert(0, "No CUDA GPU visible. Set Runtime > Change runtime type > GPU.")
    text = "\n".join(lines)
    print(text)
    return text


def list_models() -> list[str]:
    root = pathlib.Path(os.environ["weight_root"])
    return sorted(p.name for p in root.glob("*.pth"))


def list_indices() -> list[str]:
    found: list[str] = []
    for root in (pathlib.Path(os.environ["index_root"]),
                 pathlib.Path(os.environ["outside_index_root"])):
        if root.exists():
            found.extend(str(p) for p in root.rglob("*.index") if "trained" not in p.name)
    return sorted(dict.fromkeys(found))


def paths() -> dict[str, str]:
    """Return the resolved Drive paths the module is using."""
    return {k: os.environ[k] for k in (
        "weight_root", "index_root", "outside_index_root",
        "hubert_root", "rmvpe_root", "pretrained_root", "pretrained_v2_root",
        "dataset_root", "output_root",
    )}


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def load_model(name: str, index: str | None = None, protect: float = 0.33) -> Model:
    """Load a .pth voice model from {weight_root}. Optional index can be a
    bare filename inside index_root/outside_index_root or an absolute path."""
    global _current_model
    if not name:
        raise ValueError("model name is required")
    if name != _current_model:
        _vc.get_vc(name, protect, protect)
        _current_model = name

    resolved_index = None
    if index:
        p = pathlib.Path(index)
        if not p.is_absolute():
            for root in (pathlib.Path(os.environ["outside_index_root"]),
                         pathlib.Path(os.environ["index_root"])):
                hits = list(root.rglob(index))
                if hits:
                    p = hits[0]
                    break
        resolved_index = str(p)
    return Model(name=name, index=resolved_index, protect=protect)


def convert(
    model: Model,
    input_audio: str,
    *,
    transpose: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float | None = None,
    output_path: str | None = None,
) -> tuple[int, np.ndarray]:
    """Convert one audio file. Returns (sample_rate, audio_ndarray).
    If output_path is given, the audio is also written there (wav/flac/mp3/m4a
    inferred from extension)."""
    protect = model.protect if protect is None else protect
    if model.name != _current_model:
        _vc.get_vc(model.name, protect, protect)
        globals()["_current_model"] = model.name

    info, result = _vc.vc_single(
        0, input_audio, transpose, None, f0_method,
        model.index or "", "",
        index_rate, filter_radius, resample_sr, rms_mix_rate, protect,
    )
    if not info.startswith("Success"):
        raise RuntimeError(info)
    sr, audio = result
    if output_path:
        _write_audio(output_path, audio, sr)
    return sr, audio


def convert_batch(
    model: Model,
    input_dir: str,
    output_dir: str,
    *,
    transpose: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float | None = None,
    output_format: str = "wav",
) -> str:
    """Convert every file in input_dir, writing results to output_dir.
    Streams progress; returns the final log string."""
    protect = model.protect if protect is None else protect
    if model.name != _current_model:
        _vc.get_vc(model.name, protect, protect)
        globals()["_current_model"] = model.name

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    last = ""
    for chunk in _vc.vc_multi(
        0, input_dir, output_dir, [],
        transpose, f0_method, model.index or "", "",
        index_rate, filter_radius, resample_sr, rms_mix_rate, protect,
        output_format,
    ):
        # vc_multi yields the cumulative log; print only the newest line
        new = chunk[len(last):]
        if new:
            print(new, end="" if new.endswith("\n") else "\n", flush=True)
        last = chunk
    return last


def _write_audio(path: str, audio: np.ndarray, sr: int) -> None:
    ext = pathlib.Path(path).suffix.lstrip(".").lower() or "wav"
    if ext in {"wav", "flac"}:
        sf.write(path, audio, sr, format=ext.upper())
        return
    # Fall back via ffmpeg for compressed formats.
    from io import BytesIO
    from infer.lib.audio import wav2
    with BytesIO() as buf:
        sf.write(buf, audio, sr, format="wav")
        buf.seek(0)
        with open(path, "wb") as out:
            wav2(buf, out, ext)


# ---------------------------------------------------------------------------
# training pipeline
# ---------------------------------------------------------------------------

def _stream(cmd: Iterable, log_file: pathlib.Path | None = None) -> None:
    """Run a subprocess, streaming combined stdout/stderr to the notebook."""
    args = [str(part) for part in cmd]
    logger.info("$ %s", " ".join(args))
    proc = subprocess.Popen(
        args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    handle = open(log_file, "a", encoding="utf-8") if log_file else None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
    finally:
        if handle:
            handle.close()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"command exited {code}: {args[1] if len(args) > 1 else args[0]}")


def _exp_dir(exp_name: str) -> pathlib.Path:
    d = pathlib.Path(os.environ["index_root"]) / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gpu_ids_default() -> str:
    if torch.cuda.is_available():
        return "-".join(str(i) for i in range(torch.cuda.device_count()))
    return "0"


def preprocess_dataset(exp_name: str, dataset_dir: str, sr: str = "40k", workers: int = 4) -> None:
    exp_dir = _exp_dir(exp_name)
    log = exp_dir / "preprocess.log"
    log.write_text("", encoding="utf-8")
    _stream([
        _config.python_cmd, "infer/modules/train/preprocess.py",
        dataset_dir, SR_MAP[sr], workers, exp_dir,
        _config.noparallel, _config.preprocess_per,
    ], log)


def extract_features(
    exp_name: str,
    *,
    workers: int = 4,
    f0_method: str = "rmvpe_gpu",
    use_f0: bool = True,
    version: str = "v2",
    gpu_ids: str | None = None,
) -> None:
    exp_dir = _exp_dir(exp_name)
    log = exp_dir / "extract_f0_feature.log"
    log.write_text("", encoding="utf-8")
    gpus = [g.strip() for g in (gpu_ids or _gpu_ids_default()).split("-") if g.strip()]

    if use_f0:
        if f0_method == "rmvpe_gpu":
            for idx, gpu in enumerate(gpus):
                _stream([
                    _config.python_cmd, "infer/modules/train/extract/extract_f0_rmvpe.py",
                    len(gpus), idx, gpu, exp_dir, _config.is_half,
                ], log)
        else:
            _stream([
                _config.python_cmd, "infer/modules/train/extract/extract_f0_print.py",
                exp_dir, workers, f0_method,
            ], log)

    for idx, gpu in enumerate(gpus):
        _stream([
            _config.python_cmd, "infer/modules/train/extract_feature_print.py",
            _config.device, len(gpus), idx, gpu, exp_dir, version, _config.is_half,
        ], log)


def _pretrained_pair(sr: str, use_f0: bool, version: str) -> tuple[str, str]:
    root = pathlib.Path(os.environ["pretrained_root" if version == "v1" else "pretrained_v2_root"])
    prefix = "f0" if use_f0 else ""
    g = root / f"{prefix}G{sr}.pth"
    d = root / f"{prefix}D{sr}.pth"
    return (str(g) if g.exists() else "", str(d) if d.exists() else "")


def _write_filelist(exp_name: str, sr: str, use_f0: bool, speaker_id: int, version: str) -> int:
    exp_dir = _exp_dir(exp_name)
    gt_wavs = exp_dir / "0_gt_wavs"
    feat = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    names = {p.stem for p in gt_wavs.glob("*.wav")} & {p.stem for p in feat.glob("*.npy")}
    if use_f0:
        names &= {p.name.replace(".wav.npy", "") for p in (exp_dir / "2a_f0").glob("*.wav.npy")}
        names &= {p.name.replace(".wav.npy", "") for p in (exp_dir / "2b-f0nsf").glob("*.wav.npy")}

    rows: list[str] = []
    for name in sorted(names):
        if use_f0:
            rows.append(
                f"{gt_wavs / (name + '.wav')}|{feat / (name + '.npy')}|"
                f"{exp_dir / '2a_f0' / (name + '.wav.npy')}|"
                f"{exp_dir / '2b-f0nsf' / (name + '.wav.npy')}|{speaker_id}"
            )
        else:
            rows.append(f"{gt_wavs / (name + '.wav')}|{feat / (name + '.npy')}|{speaker_id}")

    fea_dim = 256 if version == "v1" else 768
    mute = ROOT / "logs" / "mute"
    for _ in range(2):
        if use_f0:
            rows.append(
                f"{mute / '0_gt_wavs' / ('mute' + sr + '.wav')}|"
                f"{mute / ('3_feature' + str(fea_dim)) / 'mute.npy'}|"
                f"{mute / '2a_f0' / 'mute.wav.npy'}|"
                f"{mute / '2b-f0nsf' / 'mute.wav.npy'}|{speaker_id}"
            )
        else:
            rows.append(
                f"{mute / '0_gt_wavs' / ('mute' + sr + '.wav')}|"
                f"{mute / ('3_feature' + str(fea_dim)) / 'mute.npy'}|{speaker_id}"
            )
    shuffle(rows)
    (exp_dir / "filelist.txt").write_text("\n".join(rows), encoding="utf-8")
    return len(names)


def train_model(
    exp_name: str,
    sr: str = "40k",
    *,
    use_f0: bool = True,
    speaker_id: int = 0,
    save_epoch: int = 5,
    total_epoch: int = 100,
    batch_size: int = 8,
    save_latest: bool = True,
    cache_gpu: bool = False,
    save_weights: bool = True,
    version: str = "v2",
    gpu_ids: str | None = None,
) -> None:
    count = _write_filelist(exp_name, sr, use_f0, speaker_id, version)
    if count == 0:
        raise RuntimeError("No prepared training samples found. Run preprocess + extract_features first.")

    exp_dir = _exp_dir(exp_name)
    config_key = f"v1/{sr}.json" if version == "v1" or sr == "40k" else f"v2/{sr}.json"
    if not (exp_dir / "config.json").exists():
        (exp_dir / "config.json").write_text(
            json.dumps(_config.json_config[config_key], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    pg, pd = _pretrained_pair(sr, use_f0, version)
    cmd = [
        _config.python_cmd, "infer/modules/train/train.py",
        "-e", exp_name, "-sr", sr, "-f0", 1 if use_f0 else 0,
        "-bs", batch_size, "-g", gpu_ids or _gpu_ids_default(),
        "-te", total_epoch, "-se", save_epoch,
        "-l", 1 if save_latest else 0,
        "-c", 1 if cache_gpu else 0,
        "-sw", 1 if save_weights else 0,
        "-v", version,
    ]
    if pg:
        cmd.extend(["-pg", pg])
    if pd:
        cmd.extend(["-pd", pd])
    _stream(cmd, exp_dir / "train.log")


def build_index(exp_name: str, version: str = "v2") -> str:
    exp_dir = _exp_dir(exp_name)
    feat = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not feat.exists() or not list(feat.glob("*.npy")):
        raise RuntimeError("Run extract_features before building the index.")

    big = np.concatenate([np.load(p) for p in sorted(feat.glob("*.npy"))], 0)
    np.random.shuffle(big)
    print(f"features: {big.shape}")

    if big.shape[0] > 200000:
        print("Reducing to 10000 centers via MiniBatchKMeans...")
        big = MiniBatchKMeans(
            n_clusters=10000, verbose=True,
            batch_size=256 * _config.n_cpu, compute_labels=False, init="random",
        ).fit(big).cluster_centers_

    np.save(exp_dir / "total_fea.npy", big)
    n_ivf = max(1, min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39))
    dim = 256 if version == "v1" else 768
    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big)
    faiss.write_index(index, str(exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_1_{exp_name}_{version}.index"))
    for start in range(0, big.shape[0], 8192):
        index.add(big[start:start + 8192])
    index_name = f"added_IVF{n_ivf}_Flat_nprobe_1_{exp_name}_{version}.index"
    src = exp_dir / index_name
    faiss.write_index(index, str(src))
    dst = pathlib.Path(os.environ["outside_index_root"]) / f"{exp_name}_{index_name}"
    shutil.copy2(src, dst)
    print(f"index: {src}\ncopy:  {dst}")
    return str(dst)


def train_voice(
    exp_name: str,
    dataset_dir: str,
    *,
    sr: str = "40k",
    version: str = "v2",
    use_f0: bool = True,
    f0_method: str = "rmvpe_gpu",
    workers: int = 4,
    save_epoch: int = 5,
    total_epoch: int = 100,
    batch_size: int = 8,
    cache_gpu: bool = False,
    save_weights: bool = True,
    speaker_id: int = 0,
    gpu_ids: str | None = None,
) -> str:
    """Run preprocess -> extract -> train -> build_index end-to-end.
    Returns the path of the final .index file."""
    print(f"\n=== [1/4] preprocess dataset: {dataset_dir} ===")
    preprocess_dataset(exp_name, dataset_dir, sr=sr, workers=workers)
    print(f"\n=== [2/4] extract f0 + HuBERT features ({f0_method}) ===")
    extract_features(exp_name, workers=workers, f0_method=f0_method,
                     use_f0=use_f0, version=version, gpu_ids=gpu_ids)
    print(f"\n=== [3/4] train model ({total_epoch} epochs) ===")
    train_model(exp_name, sr, use_f0=use_f0, speaker_id=speaker_id,
                save_epoch=save_epoch, total_epoch=total_epoch, batch_size=batch_size,
                cache_gpu=cache_gpu, save_weights=save_weights, version=version,
                gpu_ids=gpu_ids)
    print(f"\n=== [4/4] build retrieval index ===")
    return build_index(exp_name, version=version)
