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
Requires a local TRELLIS checkout at ./TRELLIS and Python deps (numpy, torch,
trimesh). SS encoder weights load from ./weights if present (run
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
import sys
import types
from pathlib import Path

import numpy as np
import torch
import trimesh

GRID = 64
SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
TRELLIS_ROOT = Path(__file__).resolve().parent / "TRELLIS"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


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


def load_ss_encoder(device):
    """Load TRELLIS's pretrained sparse-structure VAE encoder.

    TRELLIS's top-level package imports heavy deps (rembg, flash_attn, kaolin, ...)
    that this pure-Conv3d encoder doesn't need, so we import `trellis.models`
    directly and skip that package __init__.
    """
    if "trellis" not in sys.modules:
        sys.path.insert(0, str(TRELLIS_ROOT))
        pkg = types.ModuleType("trellis")
        pkg.__path__ = [str(TRELLIS_ROOT / "trellis")]
        sys.modules["trellis"] = pkg
    models = importlib.import_module("trellis.models")

    encoder = models.from_pretrained(resolve_ckpt(SS_ENCODER)).eval()
    if device == "cpu":
        encoder.convert_to_fp32()  # fp16 convs are CUDA-only
    return encoder.to(device)


@torch.no_grad()
def encode_to_ss_latent(occ, encoder):
    """Encode a 64^3 grid into the SS latent (1, C_s, 16, 16, 16) (posterior mean)."""
    x = torch.from_numpy(occ).float().to(encoder.device)[None, None]
    return encoder(x)


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else OUTPUTS_DIR / f"{Path(args.mesh).stem}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path = out_path.with_name(out_path.stem + ".voxel.glb") if args.glb else None

    latent = mesh_to_ss_latent(args.mesh, device=args.device, glb_path=glb_path)
    np.save(out_path, latent[0].cpu().numpy())  # drop batch -> (C_s, 16, 16, 16)
    print(f"saved -> {out_path}")
