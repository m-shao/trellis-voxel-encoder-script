"""
Mesh -> 64^3 occupancy grid -> .glb of the voxel representation.

Standalone GLB-only companion to mesh_to_ss_latent.py (no encoder / GPU needed).
For the combined workflow, prefer `mesh_to_ss_latent.py --glb`, which writes the
same voxel .glb alongside the latent. Both share voxels_to_mesh, so the exported
cubes are exactly the occupied cells that feed the SS-VAE encoder.
"""

import argparse
from pathlib import Path

import numpy as np

try:  # package or script
    from .mesh_to_ss_latent import GRID, OUTPUTS_DIR, load_and_normalize, voxelize_surface, voxels_to_mesh
except ImportError:
    from mesh_to_ss_latent import GRID, OUTPUTS_DIR, load_and_normalize, voxelize_surface, voxels_to_mesh


def mesh_to_voxel_glb(path):
    occ = voxelize_surface(load_and_normalize(path))
    coords = np.argwhere(occ)
    print(f"occupied cells: {len(coords)} / {GRID ** 3}")
    return voxels_to_mesh(coords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a mesh's 64^3 voxelization as a .glb.")
    parser.add_argument("mesh", help="Path to a .glb/.obj mesh.")
    parser.add_argument("--out", help="Output .glb path (default: outputs/<mesh>.voxel.glb).")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else OUTPUTS_DIR / f"{Path(args.mesh).stem}.voxel.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    voxels = mesh_to_voxel_glb(args.mesh)
    voxels.export(str(out_path))
    print(f"saved -> {out_path}")
