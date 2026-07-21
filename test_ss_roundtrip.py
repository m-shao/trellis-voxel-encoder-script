"""
Round-trip check for TRELLIS's sparse-structure VAE.

Encodes a mesh's 64^3 occupancy grid to the SS latent, decodes it back with the
matching decoder (occupancy = decoder logits > 0, as in TRELLIS's pipeline), and
compares the recovered grid against the input. Exits non-zero if they differ by
more than --max-mismatch voxels (default 0 = bit-identical).

Usage
-----
Requires a local TRELLIS checkout at ./TRELLIS and Python deps (numpy, torch,
trimesh). SS encoder/decoder weights load from ./weights if present (run
download_weights.py), else are fetched from Hugging Face on first run.

    python test_ss_roundtrip.py model.glb
    python test_ss_roundtrip.py model.obj --max-mismatch 8
    python test_ss_roundtrip.py model.glb --device cpu

Arguments
---------
mesh              Input .glb or .obj file.
--device DEV      cuda or cpu (default: cuda when available).
--max-mismatch N  Voxels allowed to differ before failing (default: 0 = identical).

Exits 0 on PASS, 1 on FAIL.
"""

import argparse
import importlib

import torch

from mesh_to_ss_latent import (
    encode_to_ss_latent,
    load_and_normalize,
    load_ss_encoder,
    resolve_ckpt,
    voxelize_surface,
)

SS_DECODER = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"


def load_ss_decoder(device):
    """Load the SS decoder. `trellis` must already be importable, which
    load_ss_encoder arranges by putting the bundled checkout on sys.path."""
    models = importlib.import_module("trellis.models")
    decoder = models.from_pretrained(resolve_ckpt(SS_DECODER)).eval()
    if device == "cpu":
        decoder.convert_to_fp32()
    return decoder.to(device)


@torch.no_grad()
def decode_to_occupancy(latent, decoder):
    """Decode an SS latent to a 64^3 bool grid (TRELLIS threshold: logits > 0)."""
    logits = decoder(latent)                       # (1, 1, 64, 64, 64)
    return (logits[0, 0] > 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="SS-VAE encode->decode round-trip test.")
    parser.add_argument("mesh", help="Path to a .glb/.obj mesh.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-mismatch", type=int, default=0,
                        help="Voxels allowed to differ before failing (default: 0 = identical).")
    args = parser.parse_args()

    occ = voxelize_surface(load_and_normalize(args.mesh))
    encoder = load_ss_encoder(args.device)
    decoder = load_ss_decoder(args.device)

    latent = encode_to_ss_latent(occ, encoder)
    occ_hat = decode_to_occupancy(latent, decoder)

    mismatch = int((occ != occ_hat).sum())
    union = int((occ | occ_hat).sum())
    iou = int((occ & occ_hat).sum()) / union if union else 1.0
    ok = mismatch <= args.max_mismatch

    print(f"latent           : {tuple(latent.shape)}")
    print(f"input occupied   : {int(occ.sum())}")
    print(f"decoded occupied : {int(occ_hat.sum())}")
    print(f"mismatched voxels: {mismatch} / {occ.size}  (IoU {iou:.6f})")
    print(f"identical        : {mismatch == 0}")
    print(f"result           : {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
