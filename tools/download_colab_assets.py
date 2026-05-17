import argparse
import os
from pathlib import Path

import requests

BASE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
ROOT = Path(__file__).resolve().parent.parent
DRIVE_MOUNT = Path("/content/drive/MyDrive")
DEFAULT_DRIVE_ROOT = Path(os.getenv("RVC_DRIVE_ROOT", str(DRIVE_MOUNT / "RVC-Colab")))


PRETRAINED = [
    "D32k.pth",
    "D40k.pth",
    "D48k.pth",
    "G32k.pth",
    "G40k.pth",
    "G48k.pth",
    "f0D32k.pth",
    "f0D40k.pth",
    "f0D48k.pth",
    "f0G32k.pth",
    "f0G40k.pth",
    "f0G48k.pth",
]


def download(url, dest, overwrite=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        print(f"ok: {dest}")
        return
    print(f"download: {url}")
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"saved: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download only the assets needed for Nvidia Colab RVC.")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip training pretrained G/D weights.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files.")
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_DRIVE_ROOT if DRIVE_MOUNT.exists() else ROOT,
        help="Persistent RVC storage root. Defaults to Google Drive when mounted.",
    )
    args = parser.parse_args()

    storage_root = args.storage_root.expanduser()
    print(f"storage: {storage_root}")

    core_assets = [
        ("hubert_base.pt", storage_root / "assets" / "hubert" / "hubert_base.pt"),
        ("rmvpe.pt", storage_root / "assets" / "rmvpe" / "rmvpe.pt"),
    ]

    for remote, dest in core_assets:
        download(f"{BASE_URL}/{remote}", dest, args.overwrite)

    if not args.no_pretrained:
        for model_name in PRETRAINED:
            download(
                f"{BASE_URL}/pretrained/{model_name}",
                storage_root / "assets" / "pretrained" / model_name,
                args.overwrite,
            )
            download(
                f"{BASE_URL}/pretrained_v2/{model_name}",
                storage_root / "assets" / "pretrained_v2" / model_name,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
