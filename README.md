# THz Sub-Surface Ice Pebbles

Analysis scripts for extracting surfaces, subsurface pebble volumes, B-scan
layers, noise thresholds, and transmission properties from THz measurements.

## Setup

### 1. Clone the repository and initialize submodules

For a new clone:

```bash
git clone --recurse-submodules https://github.com/unibe-icelab/thz_sub_surface_ice_pebbles thz_sub_surface_ice_pebbles
cd thz_sub_surface_ice_pebbles
```

For an existing clone whose submodules have not yet been initialized:

```bash
git submodule update --init --recursive
```

The required `pyhelper` submodule provides shared plotting colors and small
helper utilities and is imported directly from the checkout.

### 2. Create and synchronize the Python environment

The project requires Python 3.14 or newer and uses UV for dependency and
environment management.

```bash
uv sync --locked
```

Run a script through the synchronized environment with:

```bash
uv run python full_3d_analysis.py
```

`pyproject.toml` declares the direct dependencies. `uv.lock` pins the complete
resolved environment, including transitive dependencies.

### 3. Download the measurement data

Download the data archive from the external storage location:

**[Download the THz analysis data](https://zenodo.org/records/18155021)**

[10.5281/zenodo.18155020](https://doi.org/10.5281/zenodo.18155020)

Extract the archive into the repository root. The resulting layout must begin
with a `data/` directory:

```text
thz_sub_surface_ice_pebbles/
├── data/
│   ├── envelope/
│   ├── estimate_noise/
│   ├── full_3d_analysis/
│   ├── layer_detection/
│   ├── psf/
│   ├── sub_surface_extraction/
│   ├── surface_extraction/
│   └── transmission_pebble/
├── pyproject.toml
└── uv.lock
```

The scripts use project-relative paths and do not require the data to remain in
the original acquisition locations.

## Configuration

Most analysis scripts define a `MEASUREMENTS` mapping near the top of the file.
Select an input by changing its `ACTIVE_MEASUREMENT` value. Transmission
analysis also has an `ACTIVE_REFERENCE` selector.

All PNG and PDF backgrounds are controlled centrally in `plot_style.py`:

```python
PLOT_BACKGROUND = "white"  # or "black"
```

Generated figures and other analysis artifacts are written below `results/`.

## Active Python files

| File | Purpose |
| --- | --- |
| `envelope.py` | Computes Hilbert and Gaussian-convolution signal envelopes. |
| `estimate_noise.py` | Fits mixture models to signal amplitudes and estimates noise thresholds. |
| `full_3d_analysis.py` | Combines surface extraction and subsurface segmentation into a rendered 3D measurement view. |
| `layer_detection.py` | Produces annotated B-scan layer figures for COC and ice measurements. |
| `plot_style.py` | Defines the shared white/black plot theme, export colors, and save helpers. |
| `scicolorscales.py` | Supplies scientific color scales used by Plotly figures. |
| `sub_surface_extraction.py` | Deconvolves THz image data and segments subsurface pebble volumes. |
| `surface_extraction.py` | Detects, smooths, caches, and optionally exports measured surfaces. |
| `transmission_pebble.py` | Calculates frequency-dependent refractive-index maps from sample and reference measurements. |

Historical scripts removed from the active analysis are retained, untracked, in
the ignored `legacy/` directory.
