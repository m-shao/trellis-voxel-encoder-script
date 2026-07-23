"""
ss_encode — the project's primary mesh -> voxel encoder.

    mesh (.glb/.obj) -> normalize to [-0.5,0.5]^3 -> 64^3 surface occupancy
                     -> TRELLIS SS-VAE encoder -> latent (8, 16, 16, 16)

Programmatic API (works in notebooks / the Modal Jupyter server):

    import ss_encode
    occ, latent = ss_encode.encode_mesh("scene.glb", device="cuda")   # (64^3 bool, (8,16,16,16))
    fig = ss_encode.visualize("scene.glb")            # interactive Plotly figure -> display inline
    fig = ss_encode.visualize(occ=occ, latent=latent, interactive=False)  # static matplotlib

Paths are env-overridable (TRELLIS2_ROOT, SS_ENCODE_WEIGHTS_DIR) so this package is portable —
the Modal image sets them to the checked-in repo + the prefetched weights on the Volume.
"""
from .mesh_to_ss_latent import (
    GRID,
    SS_ENCODER,
    SS_DECODER,
    load_and_normalize,
    voxelize_surface,
    voxels_to_mesh,
    load_ss_encoder,
    load_ss_decoder,
    encode_to_ss_latent,
    decode_to_occupancy,
    encode_mesh,
    mesh_to_ss_latent,
)

__all__ = [
    "GRID", "SS_ENCODER", "SS_DECODER",
    "load_and_normalize", "voxelize_surface", "voxels_to_mesh",
    "load_ss_encoder", "load_ss_decoder",
    "encode_to_ss_latent", "decode_to_occupancy", "encode_mesh", "mesh_to_ss_latent",
]

# Visualization pulls in matplotlib/plotly; keep it optional so encode-only envs still import.
try:
    from .visualize_ss import (
        visualize, figure_png, figure_plotly, render_png, render_html, compute_grids,
    )
    __all__ += ["visualize", "figure_png", "figure_plotly", "render_png", "render_html", "compute_grids"]
except Exception as _e:  # pragma: no cover
    _VIZ_IMPORT_ERROR = _e
