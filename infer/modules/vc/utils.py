import os

import torch

# PyTorch 2.6 flipped torch.load's weights_only default to True, but fairseq's
# load_checkpoint_to_cpu doesn't opt out. The hubert checkpoint is trusted
# (downloaded from lj1995/VoiceConversionWebUI), so restore the old default
# before fairseq is imported.
_torch_load = torch.load
torch.load = lambda *a, **kw: _torch_load(*a, **{"weights_only": False, **kw})

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    hubert_root = os.getenv("hubert_root", "assets/hubert")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(hubert_root, "hubert_base.pt")],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
