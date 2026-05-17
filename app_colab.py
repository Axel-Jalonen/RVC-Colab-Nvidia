import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
from random import shuffle

import faiss
import gradio as gr
import numpy as np
import torch
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans

ROOT = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.append(str(ROOT))

load_dotenv()

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("no_proxy", "localhost, 127.0.0.1, ::1")

DRIVE_MOUNT = pathlib.Path("/content/drive/MyDrive")
DRIVE_ROOT = pathlib.Path(
    os.getenv("RVC_DRIVE_ROOT", str(DRIVE_MOUNT / "RVC-Colab"))
).expanduser()
USE_DRIVE = os.getenv("RVC_USE_DRIVE", "auto").lower()


def drive_is_available():
    return DRIVE_MOUNT.exists()


def using_drive():
    if USE_DRIVE in {"1", "true", "yes", "on"}:
        return drive_is_available()
    if USE_DRIVE in {"0", "false", "no", "off"}:
        return False
    return drive_is_available()


STORAGE_ROOT = DRIVE_ROOT if using_drive() else ROOT

os.environ.setdefault("weight_root", str(STORAGE_ROOT / "models"))
os.environ.setdefault("index_root", str(STORAGE_ROOT / "logs"))
os.environ.setdefault("outside_index_root", str(STORAGE_ROOT / "indices"))
os.environ.setdefault("hubert_root", str(STORAGE_ROOT / "assets" / "hubert"))
os.environ.setdefault("rmvpe_root", str(STORAGE_ROOT / "assets" / "rmvpe"))
os.environ.setdefault("pretrained_root", str(STORAGE_ROOT / "assets" / "pretrained"))
os.environ.setdefault(
    "pretrained_v2_root", str(STORAGE_ROOT / "assets" / "pretrained_v2")
)
os.environ.setdefault("dataset_root", str(STORAGE_ROOT / "datasets"))
os.environ.setdefault("output_root", str(STORAGE_ROOT / "outputs"))

from configs.config import Config
from infer.modules.vc.modules import VC

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rvc-colab")

for path in [
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
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(ROOT / "TEMP")
torch.manual_seed(114514)

config = Config()
vc = VC(config)

SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def _gpu_ids():
    if not torch.cuda.is_available():
        return []
    return [str(i) for i in range(torch.cuda.device_count())]


def runtime_status():
    storage = f"Storage: {'Google Drive' if STORAGE_ROOT != ROOT else 'local VM'} ({STORAGE_ROOT})"
    if not torch.cuda.is_available():
        return "\n".join(
            [
                "No CUDA GPU is visible. In Colab, set Runtime > Change runtime type > T4/A100/L4 GPU.",
                storage,
            ]
        )
    lines = [
        f"CUDA ready: {torch.version.cuda}",
        f"Device: {torch.cuda.get_device_name(0)}",
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB",
        storage,
    ]
    return "\n".join(lines)


def list_weights():
    root = pathlib.Path(os.getenv("weight_root", "assets/weights"))
    return sorted(p.name for p in root.glob("*.pth"))


def list_indices():
    roots = [
        pathlib.Path(os.getenv("index_root", "logs")),
        pathlib.Path(os.getenv("outside_index_root", "assets/indices")),
    ]
    found = []
    for root in roots:
        if root.exists():
            found.extend(
                str(p)
                for p in root.rglob("*.index")
                if "trained" not in p.name
            )
    return sorted(dict.fromkeys(found))


def refresh_choices():
    return (
        gr.update(choices=list_weights()),
        gr.update(choices=list_indices()),
        runtime_status(),
    )


def load_model(model_name):
    if not model_name:
        return f"Select a model from {os.environ['weight_root']}."
    vc.get_vc(model_name, 0.5, 0.33)
    return f"Loaded {model_name}."


def infer_one(
    model_name,
    input_audio,
    transpose,
    f0_method,
    index_path,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
):
    if not model_name:
        return "Select a .pth model first.", None
    if input_audio is None:
        return "Upload an input audio file.", None
    vc.get_vc(model_name, protect, protect)
    resolved_index = str(pathlib.Path(index_path)) if index_path else ""
    return vc.vc_single(
        0,
        input_audio,
        transpose,
        None,
        f0_method,
        resolved_index,
        "",
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    )


def infer_batch(
    model_name,
    input_dir,
    output_dir,
    transpose,
    f0_method,
    index_path,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    output_format,
):
    if not model_name:
        return "Select a .pth model first."
    if not input_dir:
        return "Set an input directory."
    output_dir = output_dir or str(pathlib.Path(os.environ["output_root"]) / "batch")
    vc.get_vc(model_name, protect, protect)
    resolved_index = str(pathlib.Path(index_path)) if index_path else ""
    return "\n".join(
        vc.vc_multi(
            0,
            input_dir,
            output_dir,
            [],
            transpose,
            f0_method,
            resolved_index,
            "",
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            output_format,
        )
    )


def _run(cmd, log_file=None):
    logger.info("Execute: %s", " ".join(map(str, cmd)))
    process = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip())
        if log_file:
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(line)
    code = process.wait()
    if code != 0:
        raise RuntimeError("\n".join(lines[-80:]))
    return "\n".join(lines[-80:])


def preprocess_dataset(trainset_dir, exp_name, sr, workers):
    workers = int(workers)
    exp_dir = pathlib.Path(os.environ["index_root"]) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_file = exp_dir / "preprocess.log"
    log_file.write_text("", encoding="utf-8")
    return _run(
        [
            config.python_cmd,
            "infer/modules/train/preprocess.py",
            trainset_dir,
            SR_MAP[sr],
            workers,
            exp_dir,
            config.noparallel,
            config.preprocess_per,
        ],
        log_file,
    )


def extract_features(exp_name, workers, f0_method, use_f0, version, gpu_ids):
    workers = int(workers)
    exp_dir = pathlib.Path(os.environ["index_root"]) / exp_name
    log_file = exp_dir / "extract_f0_feature.log"
    log_file.write_text("", encoding="utf-8")
    gpu_ids = gpu_ids or "0"
    gpus = [gpu.strip() for gpu in gpu_ids.split("-") if gpu.strip()]

    if use_f0:
        if f0_method == "rmvpe_gpu":
            for idx, gpu in enumerate(gpus):
                _run(
                    [
                        config.python_cmd,
                        "infer/modules/train/extract/extract_f0_rmvpe.py",
                        len(gpus),
                        idx,
                        gpu,
                        exp_dir,
                        config.is_half,
                    ],
                    log_file,
                )
        else:
            _run(
                [
                    config.python_cmd,
                    "infer/modules/train/extract/extract_f0_print.py",
                    exp_dir,
                    workers,
                    f0_method,
                ],
                log_file,
            )

    for idx, gpu in enumerate(gpus):
        _run(
            [
                config.python_cmd,
                "infer/modules/train/extract_feature_print.py",
                config.device,
                len(gpus),
                idx,
                gpu,
                exp_dir,
                version,
                config.is_half,
            ],
            log_file,
        )
    return log_file.read_text(encoding="utf-8")[-8000:]


def pretrained_pair(sr, use_f0, version):
    root = (
        pathlib.Path(os.environ["pretrained_root"])
        if version == "v1"
        else pathlib.Path(os.environ["pretrained_v2_root"])
    )
    prefix = "f0" if use_f0 else ""
    g = root / f"{prefix}G{sr}.pth"
    d = root / f"{prefix}D{sr}.pth"
    return str(g) if g.exists() else "", str(d) if d.exists() else ""


def write_filelist(exp_name, sr, use_f0, speaker_id, version):
    exp_dir = pathlib.Path(os.environ["index_root"]) / exp_name
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    names = {p.stem for p in gt_wavs_dir.glob("*.wav")} & {p.stem for p in feature_dir.glob("*.npy")}

    if use_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names &= {p.name.replace(".wav.npy", "") for p in f0_dir.glob("*.wav.npy")}
        names &= {p.name.replace(".wav.npy", "") for p in f0nsf_dir.glob("*.wav.npy")}

    rows = []
    for name in sorted(names):
        if use_f0:
            rows.append(
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|"
                f"{exp_dir / '2a_f0' / (name + '.wav.npy')}|"
                f"{exp_dir / '2b-f0nsf' / (name + '.wav.npy')}|{speaker_id}"
            )
        else:
            rows.append(f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|{speaker_id}")

    fea_dim = 256 if version == "v1" else 768
    mute_root = ROOT / "logs" / "mute"
    for _ in range(2):
        if use_f0:
            rows.append(
                f"{mute_root / '0_gt_wavs' / ('mute' + sr + '.wav')}|"
                f"{mute_root / ('3_feature' + str(fea_dim)) / 'mute.npy'}|"
                f"{mute_root / '2a_f0' / 'mute.wav.npy'}|"
                f"{mute_root / '2b-f0nsf' / 'mute.wav.npy'}|{speaker_id}"
            )
        else:
            rows.append(
                f"{mute_root / '0_gt_wavs' / ('mute' + sr + '.wav')}|"
                f"{mute_root / ('3_feature' + str(fea_dim)) / 'mute.npy'}|{speaker_id}"
            )
    shuffle(rows)
    (exp_dir / "filelist.txt").write_text("\n".join(rows), encoding="utf-8")
    return len(names)


def train_model(exp_name, sr, use_f0, speaker_id, save_epoch, total_epoch, batch_size, save_latest, cache_gpu, save_weights, version, gpu_ids):
    speaker_id = int(speaker_id)
    save_epoch = int(save_epoch)
    total_epoch = int(total_epoch)
    batch_size = int(batch_size)
    count = write_filelist(exp_name, sr, use_f0, speaker_id, version)
    if count == 0:
        raise RuntimeError("No prepared training samples found. Run preprocess and feature extraction first.")

    exp_dir = pathlib.Path(os.environ["index_root"]) / exp_name
    config_key = f"v1/{sr}.json" if version == "v1" or sr == "40k" else f"v2/{sr}.json"
    if not (exp_dir / "config.json").exists():
        (exp_dir / "config.json").write_text(
            json.dumps(config.json_config[config_key], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    pretrained_g, pretrained_d = pretrained_pair(sr, use_f0, version)
    cmd = [
        config.python_cmd,
        "infer/modules/train/train.py",
        "-e",
        exp_name,
        "-sr",
        sr,
        "-f0",
        1 if use_f0 else 0,
        "-bs",
        batch_size,
        "-g",
        gpu_ids or "0",
        "-te",
        total_epoch,
        "-se",
        save_epoch,
        "-l",
        1 if save_latest else 0,
        "-c",
        1 if cache_gpu else 0,
        "-sw",
        1 if save_weights else 0,
        "-v",
        version,
    ]
    if pretrained_g:
        cmd.extend(["-pg", pretrained_g])
    if pretrained_d:
        cmd.extend(["-pd", pretrained_d])
    return _run(cmd, exp_dir / "train.log")


def train_index(exp_name, version):
    exp_dir = pathlib.Path(os.environ["index_root"]) / exp_name
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not feature_dir.exists() or not list(feature_dir.glob("*.npy")):
        return "Run feature extraction before building the index."

    npys = [np.load(path) for path in sorted(feature_dir.glob("*.npy"))]
    big_npy = np.concatenate(npys, 0)
    np.random.shuffle(big_npy)
    infos = [f"features: {big_npy.shape}"]

    if big_npy.shape[0] > 200000:
        infos.append("Reducing feature set to 10000 centers with MiniBatchKMeans.")
        big_npy = MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256 * config.n_cpu,
            compute_labels=False,
            init="random",
        ).fit(big_npy).cluster_centers_

    np.save(exp_dir / "total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(1, n_ivf)
    index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(index, str(exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"))
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start : start + 8192])
    index_name = f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
    index_path = exp_dir / index_name
    faiss.write_index(index, str(index_path))
    outside = pathlib.Path(os.environ["outside_index_root"]) / f"{exp_name}_{index_name}"
    shutil.copy2(index_path, outside)
    infos.append(f"index: {index_path}")
    infos.append(f"copy: {outside}")
    return "\n".join(infos)


def train_one_click(exp_name, dataset_dir, sr, version, use_f0, f0_method, workers, save_epoch, total_epoch, batch_size, cache_gpu, save_weights, gpu_ids):
    try:
        if not exp_name:
            yield "Experiment name is required."
            return
        if not dataset_dir:
            yield "Dataset directory is required."
            return
        gpu_ids = gpu_ids or ("-".join(_gpu_ids()) or "0")
        yield "Step 1/4: preprocessing dataset..."
        preprocess_dataset(dataset_dir, exp_name, sr, workers)
        yield "Step 2/4: extracting f0 and HuBERT features..."
        extract_features(exp_name, workers, f0_method, use_f0, version, gpu_ids)
        yield "Step 3/4: training model..."
        train_model(exp_name, sr, use_f0, 0, save_epoch, total_epoch, batch_size, True, cache_gpu, save_weights, version, gpu_ids)
        yield "Step 4/4: building retrieval index..."
        yield train_index(exp_name, version)
    except Exception:
        yield traceback.format_exc()


def build_ui():
    with gr.Blocks(title="RVC Colab Nvidia") as app:
        gr.Markdown("# RVC Colab Nvidia")
        status = gr.Textbox(label="Runtime", value=runtime_status(), interactive=False, lines=4)
        gr.Markdown(
            f"Drive-first storage root: `{STORAGE_ROOT}`. Put models in `{os.environ['weight_root']}` and datasets in `{os.environ['dataset_root']}`."
        )

        with gr.Row():
            refresh = gr.Button("Refresh models")
            model = gr.Dropdown(label="Model (.pth)", choices=list_weights())
            index = gr.Dropdown(label="Index (.index)", choices=list_indices())
            load = gr.Button("Load model", variant="primary")
        model_status = gr.Textbox(label="Model status", interactive=False)

        with gr.Tab("Inference"):
            with gr.Row():
                audio = gr.Audio(label="Input audio", type="filepath")
                converted = gr.Audio(label="Converted audio")
            with gr.Row():
                transpose = gr.Slider(-24, 24, value=0, step=1, label="Pitch shift")
                f0_method = gr.Radio(["rmvpe", "harvest", "crepe", "pm"], value="rmvpe", label="Pitch extractor")
                index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label="Index strength")
            with gr.Row():
                filter_radius = gr.Slider(0, 7, value=3, step=1, label="F0 median filter")
                resample_sr = gr.Radio([0, 32000, 40000, 48000], value=0, label="Resample")
                rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.05, label="RMS mix")
                protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label="Protect")
            run_infer = gr.Button("Convert", variant="primary")
            infer_log = gr.Textbox(label="Log", lines=6)

            with gr.Accordion("Batch inference", open=False):
                input_dir = gr.Textbox(label="Input directory")
                output_dir = gr.Textbox(
                    label="Output directory",
                    value=str(pathlib.Path(os.environ["output_root"]) / "batch"),
                )
                output_format = gr.Radio(["wav", "flac", "mp3", "m4a"], value="wav", label="Output format")
                run_batch = gr.Button("Convert directory")
                batch_log = gr.Textbox(label="Batch log", lines=10)

        with gr.Tab("Training"):
            with gr.Row():
                exp_name = gr.Textbox(label="Experiment name", value="my_voice")
                dataset_dir = gr.Textbox(
                    label="Dataset directory",
                    value=str(pathlib.Path(os.environ["dataset_root"]) / "my_voice"),
                )
            with gr.Row():
                sr = gr.Radio(["40k", "48k", "32k"], value="40k", label="Sample rate")
                version = gr.Radio(["v2", "v1"], value="v2", label="Model version")
                use_f0 = gr.Checkbox(value=True, label="Pitch-guided model")
                train_f0_method = gr.Radio(["rmvpe_gpu", "rmvpe", "harvest", "dio", "pm"], value="rmvpe_gpu", label="Pitch extractor")
            with gr.Row():
                workers = gr.Slider(1, 16, value=4, step=1, label="CPU workers")
                save_epoch = gr.Slider(1, 50, value=5, step=1, label="Save every N epochs")
                total_epoch = gr.Slider(1, 1000, value=100, step=1, label="Total epochs")
                batch_size = gr.Slider(1, 64, value=8, step=1, label="Batch size")
            with gr.Row():
                gpu_ids = gr.Textbox(label="GPU ids", value="-".join(_gpu_ids()) or "0")
                cache_gpu = gr.Checkbox(value=False, label="Cache dataset in GPU")
                save_weights = gr.Checkbox(value=True, label="Save small model each checkpoint")
            train = gr.Button("Train and build index", variant="primary")
            train_log = gr.Textbox(label="Training log", lines=18)

        refresh.click(refresh_choices, outputs=[model, index, status])
        load.click(load_model, inputs=model, outputs=model_status)
        run_infer.click(
            infer_one,
            inputs=[model, audio, transpose, f0_method, index, index_rate, filter_radius, resample_sr, rms_mix_rate, protect],
            outputs=[infer_log, converted],
        )
        run_batch.click(
            infer_batch,
            inputs=[model, input_dir, output_dir, transpose, f0_method, index, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, output_format],
            outputs=batch_log,
        )
        train.click(
            train_one_click,
            inputs=[exp_name, dataset_dir, sr, version, use_f0, train_f0_method, workers, save_epoch, total_epoch, batch_size, cache_gpu, save_weights, gpu_ids],
            outputs=train_log,
        )
    return app


if __name__ == "__main__":
    build_ui().queue(concurrency_count=2, max_size=16).launch(
        server_name="0.0.0.0",
        server_port=config.listen_port,
        share=config.iscolab,
        inbrowser=False,
    )
