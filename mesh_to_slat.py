"""
GLB -> O-Voxel -> TRELLIS.2 shape latent (stage 2) + material latent (stage 3).

Runs the whole GLB -> latents path in memory, mirroring TRELLIS.2's data_toolkit
(dual_grid.py, encode_shape_latent.py, voxelize_pbr.py, encode_pbr_latent.py) and
the o-voxel README quick-start.

    GLB
      -> normalize into [-0.5, 0.5]^3      (center, scale 0.99999)
      -> mesh_to_flexible_dual_grid         -> shape O-Voxel (dual vertex v, edge flags d)
      -> FlexiDualGridVaeEncoder(v, d)      -> shape latent    (N x 32) on a 64^3 grid
      -> textured_mesh_to_volumetric_attr   -> material O-Voxel (PBR, 6 channels)
      -> SparseUnetVaeEncoder(pbr)          -> material latent (N x 32) on a 64^3 grid

Shape and material voxels are each ordered by the same space-filling curve
(o_voxel.serialize.encode_seq) so their sparse layouts line up, then encoded
independently -- exactly as the toolkit does.

Requirements (a full TRELLIS.2 CUDA install):
  - o_voxel   mesh <-> O-Voxel conversion (also imported by the shape encoder).
  - cumesh    git+https://github.com/JeffreyXiang/CuMesh.git; the shape encoder
              pulls it in transitively via trellis2.representations.Mesh.
  - flexgemm  sparse-conv backend used by the SC-VAE forward pass.
  - a CUDA GPU, plus the TRELLIS.2-4B shape_enc/tex_enc weights in ./weights
    (run download_weights.py; a missing checkpoint raises, no network fallback).
Install with:  ../TRELLIS.2/setup.sh --o-voxel --cumesh --flexgemm   (Linux + CUDA)
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

# TRELLIS.2's sparse convs default to the flex_gemm backend (its triton kernels
# don't JIT cleanly on Windows); use the installed spconv unless the caller overrides.
os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")

RES = 1024                                       # voxel grid; latent grid = RES // 16 = 64
AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]     # canonical cube, fixed by the model
SHAPE_ENC = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TEX_ENC = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
PBR_ATTRS = ["base_color", "metallic", "roughness", "alpha"]   # 3 + 1 + 1 + 1 = 6 channels
# Paths are env-overridable so this package works both standalone (next to ../TRELLIS.2) and when
# folded into other apps (e.g. on Modal: TRELLIS2_ROOT=/root/TRELLIS.2,
# SS_ENCODE_WEIGHTS_DIR=/vol/pretrained/vae).
TRELLIS2_ROOT = Path(os.environ.get("TRELLIS2_ROOT", Path(__file__).resolve().parent.parent / "TRELLIS.2"))
WEIGHTS_DIR = Path(os.environ.get("SS_ENCODE_WEIGHTS_DIR", Path(__file__).resolve().parent / "weights"))
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def resolve_ckpt(name):
    """Return the local weights/<name> checkpoint path, which must exist.

    Weights are always loaded from ./weights; run download_weights.py first.
    """
    local = WEIGHTS_DIR / name.split("/")[-1]
    if not (local.with_suffix(".json").is_file() and local.with_suffix(".safetensors").is_file()):
        raise FileNotFoundError(
            f"missing local weights for '{local.name}' in {WEIGHTS_DIR}; run download_weights.py first"
        )
    return str(local)


def import_trellis2():
    """Import `trellis2.models` + `trellis2.modules.sparse`, skipping TRELLIS.2's
    heavy top-level __init__ (pipelines/renderers pull in deps the encoders don't
    need). Same bypass as the stage-1 script."""
    if "trellis2" not in sys.modules:
        sys.path.insert(0, str(TRELLIS2_ROOT))
        pkg = types.ModuleType("trellis2")
        pkg.__path__ = [str(TRELLIS2_ROOT / "trellis2")]
        sys.modules["trellis2"] = pkg
    models = importlib.import_module("trellis2.models")
    sp = importlib.import_module("trellis2.modules.sparse")
    return models, sp


def _o_voxel():
    try:
        import o_voxel
        return o_voxel
    except ImportError as e:
        raise SystemExit(
            "o_voxel is required for stage 2/3 but is not installed. Build it with a "
            "CUDA toolkit:\n    pip install ../TRELLIS.2/o-voxel --no-build-isolation"
        ) from e


def load_normalized(path):
    """Load a GLB (keeping textures) and fit it into [-0.5, 0.5]^3 with TRELLIS.2's
    exact transform: center on the bbox midpoint, scale by 0.99999 / longest_side."""
    asset = trimesh.load(path)
    lo, hi = asset.bounds
    center = (lo + hi) / 2.0
    scale = 0.99999 / (hi - lo).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    return asset


def _with_batch(voxel_indices):
    """Prepend a zero batch column: (N, 3) -> (N, 4) for a SparseTensor's coords."""
    return torch.cat([torch.zeros_like(voxel_indices[:, :1]), voxel_indices], dim=-1)


@torch.no_grad()
def encode_shape(asset, encoder, sp):
    """mesh -> flexible dual grid -> shape latent (N, 32)."""
    ov = _o_voxel()
    mesh = asset.to_mesh()
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    voxel_indices, dual_vertices, intersected = ov.convert.mesh_to_flexible_dual_grid(
        vertices, faces, grid_size=RES, aabb=AABB,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False,
    )
    order = torch.argsort(ov.serialize.encode_seq(voxel_indices))
    voxel_indices, dual_vertices, intersected = voxel_indices[order], dual_vertices[order], intersected[order]

    # dual vertex -> voxel-local [0, 1], quantized to uint8/255 as in dual_grid.py
    dual_local = torch.clamp(dual_vertices * RES - voxel_indices, 0.0, 1.0)
    dual_local = (dual_local * 255).to(torch.uint8).float() / 255.0

    coords = _with_batch(voxel_indices)
    verts = sp.SparseTensor(dual_local, coords)
    inter = verts.replace(intersected.bool())                  # 3 edge flags, shared coords
    return encoder(verts.cuda(), inter.cuda())


@torch.no_grad()
def encode_material(asset, encoder, sp):
    """mesh + textures -> volumetric PBR -> material latent (N, 32)."""
    ov = _o_voxel()
    voxel_indices, attrs = ov.convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=RES, aabb=AABB, timing=False,
    )
    order = torch.argsort(ov.serialize.encode_seq(voxel_indices))
    voxel_indices = voxel_indices[order]

    # concat PBR channels, uint8 [0, 255] -> [-1, 1], as in encode_pbr_latent.py
    feats = torch.cat([attrs[k][order] for k in PBR_ATTRS], dim=-1).float() / 255.0 * 2 - 1
    x = sp.SparseTensor(feats, _with_batch(voxel_indices))
    return encoder(x.cuda())


def save_latent(z, path):
    """Save a SC-VAE latent as .npz {feats: (N, 32), coords: (N, 3) uint8} -- the
    same format the toolkit stores training latents in."""
    np.savez_compressed(
        path,
        feats=z.feats.cpu().float().numpy(),
        coords=z.coords[:, 1:].cpu().numpy().astype(np.uint8),
    )


def mesh_to_slat(path, shape_out=None, material_out=None):
    models, sp = import_trellis2()
    asset = load_normalized(path)

    z_shape = encode_shape(asset, models.from_pretrained(resolve_ckpt(SHAPE_ENC)).eval().cuda(), sp)
    print(f"shape latent   : {tuple(z_shape.feats.shape)} tokens on "
          f"{z_shape.coords[:, 1:].max(0)[0].tolist()} grid")

    z_tex = encode_material(asset, models.from_pretrained(resolve_ckpt(TEX_ENC)).eval().cuda(), sp)
    print(f"material latent: {tuple(z_tex.feats.shape)} tokens")

    stem = Path(path).stem
    shape_out = Path(shape_out) if shape_out else OUTPUTS_DIR / f"{stem}.shape.npz"
    material_out = Path(material_out) if material_out else OUTPUTS_DIR / f"{stem}.material.npz"
    shape_out.parent.mkdir(parents=True, exist_ok=True)
    save_latent(z_shape, shape_out)
    save_latent(z_tex, material_out)
    print(f"saved -> {shape_out}")
    print(f"saved -> {material_out}")
    return z_shape, z_tex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a mesh into TRELLIS.2 shape + material latents.")
    parser.add_argument("mesh", help="Path to a .glb/.obj mesh (textured, for material).")
    parser.add_argument("--shape-out", help="Shape latent .npz (default: outputs/<mesh>.shape.npz).")
    parser.add_argument("--material-out", help="Material latent .npz (default: outputs/<mesh>.material.npz).")
    parser.add_argument("--res", type=int, default=RES,
                        help=f"Voxel grid resolution (default: {RES}; latent grid = res // 16). "
                             "Lower to 512 if you hit GPU OOM.")
    args = parser.parse_args()
    RES = args.res
    mesh_to_slat(args.mesh, args.shape_out, args.material_out)
