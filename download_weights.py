"""
Pull the TRELLIS SS-VAE weights from Hugging Face into ./weights, so the encode
and round-trip scripts load them locally instead of hitting the HF Hub each run.

Run once:

    python download_weights.py

Afterwards mesh_to_ss_latent.py / test_ss_roundtrip.py resolve the checkpoints
from ./weights (via resolve_ckpt) with no network requests.
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "microsoft/TRELLIS-image-large"
CHECKPOINTS = ["ss_enc_conv3d_16l8_fp16", "ss_dec_conv3d_16l8_fp16"]
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for name in CHECKPOINTS:
        for ext in (".json", ".safetensors"):
            src = hf_hub_download(HF_REPO, f"ckpts/{name}{ext}")
            dst = WEIGHTS_DIR / f"{name}{ext}"
            shutil.copyfile(src, dst)
            print(f"{dst.name:40s} {dst.stat().st_size / 1e6:8.1f} MB")
    print(f"weights hosted in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
