import argparse
from pathlib import Path

import requests

BASE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
DRIVE_MOUNT = Path("/content/drive/MyDrive")
STORAGE_ROOT = DRIVE_MOUNT / "RVC-Colab"


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
    args = parser.parse_args()

    if not DRIVE_MOUNT.exists():
        raise RuntimeError(
            "Google Drive is required. Mount Drive before downloading assets."
        )

    print(f"storage: {STORAGE_ROOT}")

    core_assets = [
        ("hubert_base.pt", STORAGE_ROOT / "assets" / "hubert" / "hubert_base.pt"),
        ("rmvpe.pt", STORAGE_ROOT / "assets" / "rmvpe" / "rmvpe.pt"),
    ]

    for remote, dest in core_assets:
        download(f"{BASE_URL}/{remote}", dest, args.overwrite)

    if not args.no_pretrained:
        for model_name in PRETRAINED:
            download(
                f"{BASE_URL}/pretrained/{model_name}",
                STORAGE_ROOT / "assets" / "pretrained" / model_name,
                args.overwrite,
            )
            download(
                f"{BASE_URL}/pretrained_v2/{model_name}",
                STORAGE_ROOT / "assets" / "pretrained_v2" / model_name,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
