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

The project requires Python 3.12 or newer and uses UV for dependency and
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
│   ├── pebbles/
│   │   ├── 3d_files/
│   │   ├── measurements/
│   │   ├── plots/
│   │   ├── psf/
│   │   └── resolution_target_sample.thzimg
│   └── spectroscopy/
│       ├── metadata/
│       │   ├── calculated_data.json
│       │   ├── pellets_meta_analysis.json
│       │   └── sub_pellets_meta_analysis.json
│       ├── raw/
│       ├── plots/
│       └── *_data.npy
├── pyproject.toml
└── uv.lock
```

The scripts use project-relative paths and do not require the data to remain in
the original acquisition locations.

## Code Ocean

Place all tracked repository files, including the initialized `pyhelper`
submodule, in `/code`. Upload the **contents** of the local `data/` directory
to `/data`, producing `/data/pebbles/` and `/data/spectroscopy/`.

Configure the capsule with Python 3.12 and install the locked environment
during its build step:

```bash
uv sync --locked
```

The Linux image also needs the OpenGL runtime libraries required by
VTK/PyVista (`libgl1`, `libxrender1`, `libxext6`, and `libsm6`). Then set
`/code/run.sh` as the executable run entry point. It runs headlessly, writes
temporary caches to `/scratch`, and writes generated artifacts to `/results`.

The Python files contain ordinary repository-relative defaults, just as they
do for local runs. Each executable analysis script also accepts an optional
positional input path. `run.sh` passes the Code Ocean paths explicitly and
creates `/code/data` and `/code/results` symlinks so shared PSF inputs and
normal output paths continue to resolve without Code Ocean-specific Python
code.

The spectroscopy input is copied to `/scratch/spectroscopy`
because that workflow writes processed `.npy` arrays and the Code Ocean
`/data` mount is treated as read-only.

`run.sh` deliberately contains direct, explicit invocations such as
`uv run python full_3d_analysis.py /data/pebbles/measurements/.../data_image_rfl`.
Therefore the uploaded archive must be extracted with `pebbles/` and
`spectroscopy/` directly below `/data`; an archive such as `/data/data.zip`
is not expanded during the reproducible run.

## Configuration

Most analysis scripts define a `MEASUREMENTS` mapping near the top of the file.
Select the no-argument default by changing `ACTIVE_MEASUREMENT`, or pass an
input path without modifying the source. For example:

```bash
uv run python full_3d_analysis.py \
  data/pebbles/measurements/ICE_PEBBLE_SPHERE_CoDA_T3_dust_high_res_vac_9_fast_focus_on_sohere_top/data_image_rfl
```

For image analyses, the argument is the `data_image_rfl` or `data_image`
directory. `envelope.py` takes the `.thzimg` file itself. The three
material-property scripts take the `spectroscopy` directory containing
`metadata/`, `raw/`, and the processed arrays. Omitting the argument always
uses the path configured in that script.

All PNG and PDF backgrounds are controlled centrally in `plot_style.py`:

```python
PLOT_BACKGROUND = "white"  # or "black"
```

Generated figures and other analysis artifacts are written below `results/`.

### Material-property workflow

Rebuild the processed material arrays from the raw pellet measurements and the
metadata in `data/spectroscopy/metadata/` with:

```bash
uv run python analyse_material_properties.py
```

The script validates every raw path before starting and lists all missing
measurements if the downloaded data archive is incomplete. It writes processed
arrays to `data/spectroscopy/` and analysis plots to
`results/material_properties/raw_analysis/`.

Generate the publication-style selected-material plots from those processed
arrays with:

```bash
uv run python plot_material_properties.py
```

Solve the two-component SiO2/CI Bruggeman mixture against measured CoDA-T2
with:

```bash
uv run python compare_dust_hesso_pellets_emt.py
```

This reproduces the historical HESSO pellet comparison, evaluates the mixture
at the original selected frequency index, and writes figures plus
`sio2_ci_coda_t2_solution.json` to
`results/material_properties/mixture_analysis/`.

The material-property data files are:

| Path | Contents |
| --- | --- |
| `data/spectroscopy/raw/` | Complete raw measurement directories, including THz, CSV, PNG, and PDF acquisition files. |
| `data/spectroscopy/metadata/pellets_meta_analysis.json` | Host, clear-space, thickness, volume, density, and raw-path metadata for PTFE and the original dust pellets. |
| `data/spectroscopy/metadata/sub_pellets_meta_analysis.json` | Equivalent metadata for T0, T3, juniper charcoal, and related sub-pellets. |
| `data/spectroscopy/metadata/calculated_data.json` | Derived thickness, sample-volume, and density values used by the historical mixture script. |
| `data/spectroscopy/*_data.npy` | Five-column processed caches used by `plot_material_properties.py`. |
| `results/material_properties/raw_analysis/` | Figures regenerated directly from raw measurements and metadata. |
| `results/material_properties/mixture_analysis/` | HESSO comparison figures, SiO2/CI fraction scan, and solved mixture JSON. |
| `results/material_properties/` | Publication-style material-selection PDFs and PNGs. |

## Active Python files

| File | Purpose |
| --- | --- |
| `envelope.py` | Computes Hilbert and Gaussian-convolution signal envelopes. |
| `estimate_noise.py` | Fits mixture models to signal amplitudes and estimates noise thresholds. |
| `full_3d_analysis.py` | Combines surface extraction and subsurface segmentation into a rendered 3D measurement view. |
| `layer_detection.py` | Produces annotated B-scan layer figures for COC and ice measurements. |
| `analyse_material_properties.py` | Rebuilds material-property arrays and plots from raw pellet measurements and local metadata JSON files. |
| `compare_dust_hesso_pellets_emt.py` | Reproduces the HESSO effective-medium comparisons and solves the SiO2/CI mixture against CoDA-T2. |
| `emt.py` | Implements the effective-medium model used by the raw material-property analysis. |
| `plot_material_properties.py` | Recreates the selected cometary-dust analogue optical-property plots from processed arrays in `data/spectroscopy/`. |
| `plot_style.py` | Defines the shared white/black plot theme, export colors, and save helpers. |
| `scicolorscales.py` | Supplies scientific color scales used by Plotly figures. |
| `sub_surface_extraction.py` | Deconvolves THz image data and segments subsurface pebble volumes. |
| `surface_extraction.py` | Detects, smooths, caches, and optionally exports measured surfaces. |
| `transmission_pebble.py` | Calculates frequency-dependent refractive-index maps from sample and reference measurements. |

Historical scripts removed from the active analysis are retained, untracked, in
the ignored `legacy/` directory.
