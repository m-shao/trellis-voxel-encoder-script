# trellis-voxel-encoder-script

Encode a mesh (and its textures) into **TRELLIS.2's latent codes** — the same
latents its image-to-3D pipeline generates, run in reverse (asset -> latent):

- **Stage 1 - sparse structure (SS):** a dense `8 x 16 x 16 x 16` occupancy latent (the coarse "where is there stuff").
- **Stage 2 - shape SLat:** `N x 32` sparse tokens encoding precise geometry (a flexible dual grid).
- **Stage 3 - material SLat:** `N x 32` sparse tokens encoding PBR appearance (base color, metallic, roughness, alpha).

Stage 1 is light — a dense `Conv3d` VAE that even runs on CPU. Stages 2/3 need
the **O-Voxel** conversion plus sparse-conv CUDA extensions and a GPU.

## Pipeline

**Stage 1** — `mesh_to_ss_latent.py`:

```
mesh (.glb/.obj)
  -> normalize into the [-0.5, 0.5]^3 cube
  -> surface-voxelize into a 64^3 occupancy grid
  -> SS-VAE encoder            (microsoft/TRELLIS-image-large - ss_enc_conv3d_16l8_fp16)
  -> latent  (8, 16, 16, 16)
```

**Stages 2/3** — `mesh_to_slat.py` (shared normalize, two independent encoders):

```
mesh (+ textures)
  -> normalize into [-0.5, 0.5]^3
  -> o_voxel: mesh -> flexible dual grid -> FlexiDualGridVaeEncoder -> shape SLat    (N, 32)
  -> o_voxel: mesh -> volumetric PBR     -> SparseUnetVaeEncoder    -> material SLat (N, 32)
     (both ordered by the same space-filling curve, so the two latents line up)
```

These invert TRELLIS.2's stage-1/2/3 VAE **decoders**; the generator additionally
runs flow models (noise -> latent) that the encode path never touches.

## Repository layout

| Path | Purpose |
| --- | --- |
| `mesh_to_ss_latent.py` | stage 1: mesh -> SS latent `.npy` (+ optional voxel `.glb`) |
| `mesh_to_slat.py` | stages 2/3: mesh -> shape + material SLat `.npz` |
| `mesh_to_voxel_glb.py` | mesh -> voxel `.glb` only (no model / GPU) |
| `test_ss_roundtrip.py` | stage-1 encode -> decode -> compare occupancy |
| `download_weights.py` | cache all four checkpoints into `weights/` |
| `_env.bat` | Windows build environment (MSVC + CUDA + arch / jobs) |
| `TRELLIS2/` | git submodule — `microsoft/TRELLIS.2`; model code + `o-voxel` + `data_toolkit` |
| `_build/` | local scratch clones of `CuMesh` / `FlexGEMM` built for stages 2/3 |
| `assets/` | sample meshes |
| `weights/` | local checkpoints (git-ignored; populated by `download_weights.py`) |
| `outputs/` | generated latents / voxel meshes (git-ignored) |

## Setup

This checkout is **Python 3.10 + `torch 2.6.0+cu124` (CUDA 12.4)** on Windows,
with the `.venv` managed by [`uv`](https://github.com/astral-sh/uv). Commands
below use `uv`; drop the `uv` prefix if you use plain `pip` / `venv`.

### 1. Clone with submodules

```bash
git clone --recursive https://github.com/m-shao/trellis-voxel-encoder-script.git
# already cloned non-recursively? fetch TRELLIS.2 AND its nested Eigen:
git submodule update --init --recursive TRELLIS2
```

The recursive init matters: `o-voxel` pulls Eigen from its own nested submodule
(`TRELLIS2/o-voxel/third_party/eigen`), which a non-recursive clone skips.

### 2. Python + PyTorch

```bash
uv venv --python 3.10
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Stage-1 dependencies (SS latent + round-trip)

```bash
uv pip install trimesh numpy safetensors huggingface_hub spconv-cu124 cumm-cu124
```

> Stage 1 uses only the dense `Conv3d` SS-VAE, so the heavy TRELLIS extras
> (flash-attn, nvdiffrast, kaolin, rembg, ...) are **not** required. The scripts
> import `trellis2.models` directly and skip TRELLIS.2's top-level `__init__`.
> `spconv` / `cumm` are the sparse-conv backend TRELLIS imports at load time.

This alone is enough for `mesh_to_ss_latent.py`, `mesh_to_voxel_glb.py`, and
`test_ss_roundtrip.py`.

### 4. Stage-2/3 CUDA extensions (shape + material)

Stages 2/3 additionally need three native extensions compiled against your CUDA
toolkit: **`o_voxel`** (mesh <-> O-Voxel), plus **`CuMesh`** and **`FlexGEMM`**
(pulled in transitively when `trellis2.representations.Mesh` is imported by the
shape encoder). Requires a full CUDA Toolkit (`nvcc`), a C++ host compiler, and a
CUDA GPU.

**Linux / CUDA** — use TRELLIS.2's own installer:

```bash
./TRELLIS2/setup.sh --o-voxel --cumesh --flexgemm
```

**Windows** — build manually (what this checkout is configured for). Open a shell
and load the build environment first:

```bat
call _env.bat
```

`_env.bat` loads MSVC (`vcvars64.bat`), points `CUDA_HOME` / `PATH` at the CUDA
12.4 toolkit, and sets `TORCH_CUDA_ARCH_LIST=8.6` (RTX 30-series — set to your
GPU's compute capability) and `MAX_JOBS=16` (lower it if compiling exhausts RAM).
Then build the three extensions:

```bat
uv pip install .\TRELLIS2\o-voxel --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git _build\CuMesh
uv pip install .\_build\CuMesh --no-build-isolation

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git _build\FlexGEMM
uv pip install .\_build\FlexGEMM --no-build-isolation
```

`mesh_to_slat.py` defaults `SPARSE_CONV_BACKEND=spconv` (TRELLIS.2's `flex_gemm`
triton kernels don't JIT cleanly on Windows), so the `spconv` from step 3 is what
actually runs the convolutions; `FlexGEMM` is still imported for its `grid_sample`.

**Windows / MSVC source patches** — already applied to the working tree here.
They live in the pinned submodule / scratch clones, so re-apply on a fresh clone:

- `TRELLIS2/o-voxel/setup.py` — MSVC compile flags (`/O2 /std:c++17 /bigobj`).
- `TRELLIS2/o-voxel/src/convert/flexible_dual_grid.cpp` — drop non-standard `d` double-literal suffixes.
- `TRELLIS2/o-voxel/src/io/{svo,filter_parent,filter_neighbor}.cpp` — `(int64_t)` casts for MSVC narrowing.
- `TRELLIS2/o-voxel/o_voxel/__init__.py` — import only `convert` + `serialize` (skips `cv2` / `nvdiffrast`).
- `_build/FlexGEMM/flex_gemm/kernels/sparse_neighbor_map.cu` — `.template` disambiguator; `kernels/__init__.py` — make the `triton` import optional.

> Linker can't find `cudart.lib`? Add your CUDA `lib\x64` to `LIB`.

### 5. Weights

```bash
python download_weights.py
```

Fetches all four checkpoints (~1.7 GB) into `weights/`: stage-1
`ss_enc` / `ss_dec` from `microsoft/TRELLIS-image-large`, and stage-2/3
`shape_enc` / `tex_enc` from `microsoft/TRELLIS.2-4B`. Every script then loads
only from `weights/` and raises a clear error if a checkpoint is missing — there
is no network fallback.

## Usage

Activate the venv first (or prefix each command with `.venv\Scripts\python.exe`).

**Stage 1 — SS latent** (writes `outputs/<mesh>.npy`):

```bash
python mesh_to_ss_latent.py assets/bed_bench.raw.glb
python mesh_to_ss_latent.py assets/bed_bench.raw.glb --glb          # also writes outputs/<mesh>.voxel.glb
python mesh_to_ss_latent.py assets/bed_bench.raw.glb --out latent.npy --device cpu
```

**Voxel `.glb` only** (no encoder / GPU):

```bash
python mesh_to_voxel_glb.py assets/bed_bench.raw.glb
```

**Round-trip check** (encode -> decode -> compare; exits non-zero on mismatch):

```bash
python test_ss_roundtrip.py assets/bed_bench.raw.glb
python test_ss_roundtrip.py assets/dining_serving_bowl.raw.glb --max-mismatch 4
```

**Stages 2/3 — shape + material SLat** (writes `outputs/<mesh>.shape.npz` and `<mesh>.material.npz`):

```bash
python mesh_to_slat.py assets/bed_bench.raw.glb
python mesh_to_slat.py assets/bed_bench.raw.glb --res 512           # lower voxel res if the GPU OOMs
```

## Notes

- **Outputs.** Everything defaults to `outputs/`, named after the input mesh:
  stage 1 `<mesh>.npy` shape `(8, 16, 16, 16)` (+ `<mesh>.voxel.glb` with `--glb`);
  stages 2/3 `<mesh>.shape.npz` / `<mesh>.material.npz`, each holding
  `{feats: (N, 32), coords: (N, 3) uint8}` — the layout TRELLIS.2 stores training
  latents in. `--out` overrides the stage-1 `.npy`; the voxel `.glb` tracks it.
- **O-Voxel & resolution.** `mesh_to_slat.py` builds the dual grid / PBR voxels at
  `--res` (default 1024); the SC-VAE downsamples 16x to an `N x 32` latent on a
  `64^3` grid. `--res 512` roughly quarters the token count if you hit GPU OOM.
- **Determinism vs. losslessness.** Every encoder returns the posterior mean (no
  sampling), so runs are reproducible. The VAEs are *lossy*: stage-1
  encode -> decode is usually bit-identical but can differ by a voxel or two on
  some shapes — `--max-mismatch` sets the tolerance.
- **Weights.** `resolve_ckpt` loads `weights/<name>.{json,safetensors}` and raises
  if a checkpoint is missing — runs never contact Hugging Face.
- **CUDA / fp16.** The `*_fp16` checkpoints run their torso in fp16 (CUDA-only).
  Stage 1 `--device cpu` folds it to fp32; stages 2/3 are GPU-only.
