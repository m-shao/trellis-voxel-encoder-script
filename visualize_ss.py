"""
Visualize the two grids in the mesh -> SS-latent pipeline:

  * the 64^3 binary occupancy grid produced by ``voxelize_surface``, and
  * the C_s x 16^3 sparse-structure latent produced by the SS-VAE encoder
    (C_s = 8 for ss_enc_conv3d_16l8), shown as a per-cell magnitude cloud plus a
    per-channel top-down (x-z) projection montage.

All 3D views use a Y-up convention (X-Z is the ground plane), matching the
glTF/glb meshes, so the third plot axis is the object's height.

Writes a static PNG (matplotlib). With --html it also writes a self-contained,
interactive Plotly page you can rotate/zoom in a browser.

Usage
-----
    python visualize_ss.py assets/bed_bench.raw.glb
    python visualize_ss.py assets/bed_bench.raw.glb --html
    python visualize_ss.py assets/bed_bench.raw.glb --latent outputs/bed_bench.npy
    python visualize_ss.py assets/bed_bench.raw.glb --device cpu

Arguments
---------
mesh          Input .glb/.obj mesh (voxelized to 64^3, then encoded).
--latent PATH Load a precomputed (C_s, 16, 16, 16) .npy instead of running the
              encoder (skips torch / weights; still voxelizes the mesh for the
              occupancy panel).
--out PATH    Output .png path (default: outputs/<mesh>.viz.png).
--html        Also write an interactive Plotly page (default: <out>.html).
--scatter     Render occupancy as a point cloud instead of cubes (faster).
--device DEV  cuda / mps / cpu (default: best available).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to a file, no display needed

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

try:  # works both as a package (ss_encode.visualize_ss) and as a script (python visualize_ss.py)
    from .mesh_to_ss_latent import (
        GRID, OUTPUTS_DIR, encode_mesh, encode_to_ss_latent,
        load_and_normalize, load_ss_encoder, voxelize_surface,
    )
except ImportError:
    from mesh_to_ss_latent import (
        GRID, OUTPUTS_DIR, encode_mesh, encode_to_ss_latent,
        load_and_normalize, load_ss_encoder, voxelize_surface,
    )

OCC_COLOR = (0.30, 0.62, 0.91)  # cornflower blue for occupied voxels
MAX_CUBES = 20000  # above this many voxels, fall back to a point cloud for speed


def compute_grids(mesh_path, device, latent_path=None):
    """Return (occ 64^3 bool, latent (C_s, 16, 16, 16) float32)."""
    occ = voxelize_surface(load_and_normalize(mesh_path))
    if latent_path is not None:
        latent = np.load(latent_path).astype(np.float32)
    else:
        encoder = load_ss_encoder(device)
        latent = encode_to_ss_latent(occ, encoder)[0].float().cpu().numpy()
    return occ, latent


def _signed_maxabs(vol, axis=2):
    """Project a 3D volume along ``axis`` keeping the value of largest magnitude.

    Unlike a plain mean (which cancels the roughly zero-mean latent), this keeps
    each cell's strongest signed activation, so per-channel structure survives.
    """
    idx = np.expand_dims(np.argmax(np.abs(vol), axis=axis), axis=axis)
    return np.take_along_axis(vol, idx, axis=axis).squeeze(axis)


def _style_3d(ax, n, title):
    ax.set_title(title, fontsize=11, pad=0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")  # y is up: X–Z is the ground plane
    ax.view_init(elev=18, azim=-58)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            pane.set_pane_color((0.97, 0.97, 0.99, 1.0))
        except Exception:
            pass


def plot_occupancy(ax, occ, force_scatter=False):
    """Draw the 64^3 occupancy: solid cubes when small, else a point cloud."""
    n_occ = int(occ.sum())
    coords = np.argwhere(occ)  # columns (x, y, z)
    if force_scatter or n_occ > MAX_CUBES:
        ax.scatter(
            coords[:, 0], coords[:, 2], coords[:, 1],  # (x, z, y): y up
            c=[OCC_COLOR], marker="s", s=3, depthshade=True, edgecolors="none",
        )
        mode = "points"
    else:
        occ_v = occ.transpose(0, 2, 1)  # (x, y, z) -> (x, z, y): y on the vertical axis
        facecolors = np.zeros(occ_v.shape + (4,), dtype=float)
        facecolors[occ_v] = (*OCC_COLOR, 0.9)
        ax.voxels(occ_v, facecolors=facecolors, edgecolor=(1, 1, 1, 0.12), linewidth=0.2)
        mode = "cubes"
    _style_3d(ax, GRID, f"64\u00b3 occupancy \u2014 {n_occ} voxels ({mode})")


def plot_latent_magnitude(fig, ax, latent):
    """Scatter the 16^3 grid, colored/sized by per-cell latent magnitude."""
    mag = np.linalg.norm(latent, axis=0)  # L2 over channels -> (16, 16, 16)
    g = mag.shape[0]
    xs, ys, zs = np.mgrid[0:g, 0:g, 0:g]
    flat = mag.ravel()
    norm = Normalize(flat.min(), flat.max())
    unit = norm(flat)  # 0..1 per cell

    emphasis = unit ** 1.5  # push low-energy cells toward invisible, high ones pop
    colors = cm.viridis(unit)
    colors[:, 3] = 0.03 + 0.9 * emphasis
    ax.scatter(
        xs.ravel(), zs.ravel(), ys.ravel(),  # (x, z, y): y up
        c=colors, s=4 + 70 * emphasis, edgecolors="none",
    )
    _style_3d(ax, g, f"latent |z| per cell \u2014 {latent.shape[0]}\u00d7{g}\u00b3")
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="||z|| over channels")


def plot_channels(fig, subspec, latent):
    """Montage of the C_s channels: top-down (x–z) signed projection, scaled per channel."""
    c_s = latent.shape[0]
    ncol = 4
    nrow = int(np.ceil(c_s / ncol))
    grid = subspec.subgridspec(nrow, ncol, hspace=0.4, wspace=0.15)

    im = None
    axes = []
    for i in range(c_s):
        ax = fig.add_subplot(grid[i // ncol, i % ncol])
        proj = _signed_maxabs(latent[i], axis=1)  # project along y (up) -> top-down x–z map
        peak = float(np.abs(proj).max()) or 1.0
        im = ax.imshow(
            proj.T / peak, origin="lower", cmap="RdBu_r",
            vmin=-1, vmax=1, interpolation="nearest",
        )
        ax.set_title(f"ch {i}  (\u00b1{peak:.2f})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
    for j in range(c_s, nrow * ncol):  # blank any unused cells
        fig.add_subplot(grid[j // ncol, j % ncol]).axis("off")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("signed max along y (top-down x\u2013z) \u00f7 channel peak")


def figure_png(occ, latent, mesh_name="mesh", force_scatter=False):
    """Build and RETURN the matplotlib Figure (occupancy + latent-magnitude + per-channel montage).
    Use in notebooks for inline display; ``render_png`` wraps this to save to disk."""
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.12, wspace=0.05)

    plot_occupancy(fig.add_subplot(gs[0, 0], projection="3d"), occ, force_scatter)
    plot_latent_magnitude(fig, fig.add_subplot(gs[0, 1], projection="3d"), latent)
    plot_channels(fig, gs[1, :], latent)

    mag = np.linalg.norm(latent, axis=0)
    fig.suptitle(
        f"{mesh_name}   |   occupancy {int(occ.sum())}/{GRID ** 3}   |   "
        f"latent {tuple(latent.shape)}   "
        f"(|z| max {mag.max():.2f}, mean {mag.mean():.2f})",
        fontsize=13,
    )
    return fig


def render_png(occ, latent, out_path, mesh_name="mesh", force_scatter=False):
    fig = figure_png(occ, latent, mesh_name, force_scatter)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_plotly(occ, latent):
    """Build and RETURN the interactive Plotly Figure (occupancy + latent, side by side, y-up).
    Displays inline in Jupyter; ``render_html`` wraps this to write a self-contained page."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    coords = np.argwhere(occ)
    mag = np.linalg.norm(latent, axis=0)
    g = mag.shape[0]
    xs, ys, zs = np.mgrid[0:g, 0:g, 0:g]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"64\u00b3 occupancy ({int(occ.sum())} voxels)",
            f"latent |z| ({latent.shape[0]}\u00d7{g}\u00b3)",
        ),
    )
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0], y=coords[:, 2], z=coords[:, 1], mode="markers",  # (x, z, y): y up
            marker=dict(size=2.4, color=f"rgb{tuple(int(255 * c) for c in OCC_COLOR)}",
                        symbol="square"),
            name="occupancy",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Volume(
            x=xs.ravel(), y=zs.ravel(), z=ys.ravel(), value=mag.ravel(),  # (x, z, y): y up
            opacity=0.12, surface_count=18, colorscale="Viridis",
            colorbar=dict(title="||z||", x=1.0, len=0.7),
            name="latent",
        ),
        row=1, col=2,
    )
    for scene in ("scene", "scene2"):
        fig.layout[scene].aspectmode = "cube"
    fig.update_scenes(xaxis_title="x", yaxis_title="z", zaxis_title="y")  # y is up
    fig.update_layout(title="SS pipeline: occupancy grid vs. latent grid (y up)", showlegend=False)
    return fig


def render_html(occ, latent, out_path):
    """Self-contained interactive Plotly page: occupancy + latent, side by side."""
    figure_plotly(occ, latent).write_html(out_path, include_plotlyjs=True)


def _best_device():
    import torch
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(mesh=None, *, occ=None, latent=None, device=None, interactive=True,
              mesh_name=None, force_scatter=False):
    """One-call visualization for notebooks / the Jupyter server. Returns a figure to DISPLAY inline.

    - ``interactive=True``  -> Plotly ``go.Figure`` (rotate/zoom in the notebook).
    - ``interactive=False`` -> matplotlib ``Figure`` (static; richer per-channel montage).
    Provide ``mesh`` (path; voxelized + encoded) OR precomputed ``occ`` + ``latent``. If only
    ``latent`` + ``mesh`` are given, the mesh is voxelized for the occupancy panel.
    """
    if occ is None or latent is None:
        assert mesh is not None, "provide `mesh` path, or both `occ` and `latent`."
        if latent is not None:                       # have latent, need occ panel
            occ = voxelize_surface(load_and_normalize(mesh))
        else:                                        # encode from scratch
            occ, latent = encode_mesh(mesh, device=device or _best_device())
    name = mesh_name or (Path(mesh).name if mesh is not None else "mesh")
    return figure_plotly(occ, latent) if interactive else figure_png(occ, latent, name, force_scatter)


def main():
    import torch

    default_device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    parser = argparse.ArgumentParser(description="Visualize the SS occupancy grid and latent.")
    parser.add_argument("mesh", help="Path to a .glb/.obj mesh.")
    parser.add_argument("--latent", help="Precomputed (C_s,16,16,16) .npy; skips the encoder.")
    parser.add_argument("--out", help="Output .png path (default: outputs/<mesh>.viz.png).")
    parser.add_argument("--html", action="store_true", help="Also write an interactive Plotly page.")
    parser.add_argument("--scatter", action="store_true", help="Render occupancy as points, not cubes.")
    parser.add_argument("--device", default=default_device, help="cuda / mps / cpu.")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else OUTPUTS_DIR / f"{Path(args.mesh).stem}.viz.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    occ, latent = compute_grids(args.mesh, args.device, latent_path=args.latent)
    print(f"occupancy: {int(occ.sum())}/{GRID ** 3} voxels   latent: {tuple(latent.shape)}")

    render_png(occ, latent, out_path, Path(args.mesh).name, force_scatter=args.scatter)
    print(f"saved -> {out_path}")

    if args.html:
        html_path = out_path.with_suffix(".html")
        render_html(occ, latent, html_path)
        print(f"saved -> {html_path}")


if __name__ == "__main__":
    main()
