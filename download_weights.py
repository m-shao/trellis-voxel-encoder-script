"""
Pull the TRELLIS SS-VAE (stage 1) and SC-VAE (stage 2/3) encoder/decoder weights
from Hugging Face into ./weights, so every script loads them locally instead of
hitting the HF Hub each run.

Run once:

    python download_weights.py

Afterwards mesh_to_ss_latent.py / test_ss_roundtrip.py / mesh_to_slat.py resolve
the checkpoints from ./weights (via resolve_ckpt) with no network requests.
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

# repo -> checkpoint base names (each has a .json + .safetensors under ckpts/)
CHECKPOINTS = {
    "microsoft/TRELLIS-image-large": [        # stage 1: sparse-structure VAE
        "ss_enc_conv3d_16l8_fp16",
        "ss_dec_conv3d_16l8_fp16",
    ],
    "microsoft/TRELLIS.2-4B": [               # stage 2/3: shape + material SC-VAE
        "shape_enc_next_dc_f16c32_fp16",
        "tex_enc_next_dc_f16c32_fp16",
    ],
}
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    for repo, names in CHECKPOINTS.items():
        for name in names:
            for ext in (".json", ".safetensors"):
                src = hf_hub_download(repo, f"ckpts/{name}{ext}")
                dst = WEIGHTS_DIR / f"{name}{ext}"
                shutil.copyfile(src, dst)
                print(f"{dst.name:42s} {dst.stat().st_size / 1e6:8.1f} MB")
    print(f"weights hosted in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
