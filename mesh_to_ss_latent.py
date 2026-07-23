"""
Mesh -> 64^3 binary occupancy grid -> TRELLIS sparse-structure (SS) latent.

Produces only the coarse "sparse structure" latent (dense C_s x 16 x 16 x 16),
not the shape/material (SLat) latents.

    mesh (.glb/.obj)
      -> normalize into the [-0.5, 0.5]^3 cube
      -> surface-voxelize into a 64^3 occupancy grid
      -> SS-VAE encoder
      -> latent q in R^{C_s x 16 x 16 x 16}   (C_s = 8 for ss_enc_conv3d_16l8)

Usage
-----
Requires the bundled TRELLIS.2 checkout at ../TRELLIS.2 and Python deps (numpy,
torch, trimesh). SS encoder weights load from ./weights if present (run
download_weights.py), else are fetched from Hugging Face on first run.

    python mesh_to_ss_latent.py model.glb
    python mesh_to_ss_latent.py model.obj --out ss_latent.npy
    python mesh_to_ss_latent.py model.glb --glb
    python mesh_to_ss_latent.py model.glb --device cpu

Arguments
---------
mesh          Input .glb or .obj file.
--out PATH    Output .npy path (default: outputs/<mesh>.npy). Shape: (C_s, 16, 16, 16).
--glb         Also write the 64^3 voxel grid as <out>.voxel.glb
              (default: outputs/<mesh>.voxel.glb).
--device DEV  cuda or cpu (default: cuda when available).
"""

import argparse
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import trimesh

GRID = 64
SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
SS_DECODER = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"

# Paths are env-overridable so this package works both standalone (next to ../TRELLIS.2) and when
# folded into other apps (e.g. on Modal: TRELLIS2_ROOT=/root/TRELLIS.2,
# SS_ENCODE_WEIGHTS_DIR=/vol/pretrained/vae).
TRELLIS2_ROOT = Path(os.environ.get("TRELLIS2_ROOT", Path(__file__).resolve().parent.parent / "TRELLIS.2"))
WEIGHTS_DIR = Path(os.environ.get("SS_ENCODE_WEIGHTS_DIR", Path(__file__).resolve().parent / "weights"))
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# cache loaded VAE modules by (hf_path, device) so repeated calls (notebook / server) don't reload
_MODEL_CACHE: dict = {}


def load_and_normalize(path):
    """Load a mesh and fit it into the canonical [-0.5, 0.5]^3 cube."""
    mesh = trimesh.load(path, force="mesh")
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2.0)
    mesh.apply_scale(0.99 / (hi - lo).max())
    return mesh


def voxelize_surface(mesh, grid=GRID):
    """Voxelize the mesh surface into a dense (grid, grid, grid) bool grid.

    Cell index follows TRELLIS: idx = floor((p + 0.5) * grid), clamped to [0, grid-1].
    """
    pitch = 1.0 / grid
    pts = mesh.voxelized(pitch=pitch).points
    idx = np.clip(np.floor((pts + 0.5) / pitch).astype(np.int64), 0, grid - 1)
    occ = np.zeros((grid, grid, grid), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def voxels_to_mesh(coords, grid=GRID):
    """Build a cube-per-voxel Trimesh from occupied cell indices (K, 3).

    Cube edge = 1/grid; the center of cell i is (i + 0.5)/grid - 0.5, the
    inverse of voxelize_surface's index mapping.
    """
    pitch = 1.0 / grid
    centers = (coords.astype(np.float64) + 0.5) * pitch - 0.5

    unit = trimesh.creation.box(extents=(pitch, pitch, pitch))
    corners = np.asarray(unit.vertices)
    faces = np.asarray(unit.faces)

    n = len(centers)
    verts = (corners[None] + centers[:, None]).reshape(-1, 3)
    tris = (faces[None] + (np.arange(n) * len(corners))[:, None, None]).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=tris, process=False)


def resolve_ckpt(hf_path):
    """Return a local weights/<name> path if present, else the HF hub path.

    Run download_weights.py to populate weights/ and skip per-run HF requests.
    """
    local = WEIGHTS_DIR / hf_path.split("/")[-1]
    if local.with_suffix(".json").is_file() and local.with_suffix(".safetensors").is_file():
        return str(local)
    return hf_path


def import_trellis2_models():
    """Import and return `trellis2.models`, skipping trellis2's package __init__.

    trellis2's top-level __init__ pulls in pipelines/renderers/representations
    (flash_attn, kaolin, nvdiffrast, rembg, ...) that this pure-Conv3d SS-VAE
    doesn't need, so we register a lightweight namespace package for `trellis2`
    and import the `models` subpackage directly, bypassing that heavy __init__.
    """
    if "trellis2" not in sys.modules:
        sys.path.insert(0, str(TRELLIS2_ROOT))
        pkg = types.ModuleType("trellis2")
        pkg.__path__ = [str(TRELLIS2_ROOT / "trellis2")]
        sys.modules["trellis2"] = pkg
    return importlib.import_module("trellis2.models")


def _load_vae(hf_path, device):
    """Load + cache a trellis2 dense Conv3d SS-VAE module (encoder or decoder)."""
    key = (hf_path, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    models = import_trellis2_models()
    net = models.from_pretrained(resolve_ckpt(hf_path)).eval()
    if device != "cuda":
        net.convert_to_fp32()  # fp16 conv torso is CUDA-only
    net = net.to(device)
    _MODEL_CACHE[key] = net
    return net


def load_ss_encoder(device):
    """Load (cached) trellis2's pretrained sparse-structure VAE encoder."""
    return _load_vae(SS_ENCODER, device)


def load_ss_decoder(device):
    """Load (cached) trellis2's pretrained sparse-structure VAE decoder."""
    return _load_vae(SS_DECODER, device)


@torch.no_grad()
def encode_to_ss_latent(occ, encoder):
    """Encode a 64^3 grid into the SS latent (1, C_s, 16, 16, 16) (posterior mean)."""
    x = torch.from_numpy(occ).float().to(encoder.device)[None, None]
    return encoder(x)


@torch.no_grad()
def decode_to_occupancy(latent, decoder):
    """Decode an SS latent (1, C_s, 16, 16, 16) to a 64^3 bool grid (TRELLIS threshold: logits > 0)."""
    if not torch.is_tensor(latent):
        latent = torch.from_numpy(np.asarray(latent))
    latent = latent.float().to(decoder.device)
    if latent.ndim == 4:
        latent = latent[None]
    return (decoder(latent)[0, 0] > 0).cpu().numpy()


def encode_mesh(path, device="cuda"):
    """Full mesh -> (occ 64^3 bool, latent (C_s,16,16,16) float32 numpy). The primary voxel-encode API."""
    occ = voxelize_surface(load_and_normalize(path))
    latent = encode_to_ss_latent(occ, load_ss_encoder(device))[0].float().cpu().numpy()
    return occ, latent


def mesh_to_ss_latent(path, device="cuda", glb_path=None):
    occ = voxelize_surface(load_and_normalize(path))
    print(f"occupied cells: {int(occ.sum())} / {GRID ** 3}")

    if glb_path is not None:
        voxels_to_mesh(np.argwhere(occ)).export(str(glb_path))
        print(f"saved voxels -> {glb_path}")

    encoder = load_ss_encoder(device)
    latent = encode_to_ss_latent(occ, encoder)
    print(f"SS latent shape: {tuple(latent.shape)}")
    return latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a mesh into a TRELLIS SS latent.")
    parser.add_argument("mesh", help="Path to a .glb/.obj mesh.")
    parser.add_argument("--out", help="Output .npy path (default: outputs/<mesh>.npy).")
    parser.add_argument("--glb", action="store_true",
                        help="Also write the 64^3 voxel grid as <out>.voxel.glb.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else OUTPUTS_DIR / f"{Path(args.mesh).stem}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path = out_path.with_name(out_path.stem + ".voxel.glb") if args.glb else None

    latent = mesh_to_ss_latent(args.mesh, device=args.device, glb_path=glb_path)
    np.save(out_path, latent[0].cpu().numpy())  # drop batch -> (C_s, 16, 16, 16)
    print(f"saved -> {out_path}")
