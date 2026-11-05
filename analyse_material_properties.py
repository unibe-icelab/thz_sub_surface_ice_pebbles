import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydotthz import DotthzFile
from shapely import Polygon, Point
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab

import re

from emt import compute_bruggeman_n_eff
from plot_style import material_color, save_figure
from pyhelper import unibe

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "material_properties"
METADATA_DIR = DATA_DIR / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results" / "material_properties" / "raw_analysis"
RAW_DATA_DIR = DATA_DIR / "raw"


def project_path(value: str | Path) -> Path:
    """Resolve absolute, project-relative, or raw-directory metadata paths."""
    path = Path(value)
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return RAW_DATA_DIR / path
    return PROJECT_ROOT / path


def save_result(filename: str) -> None:
    """Save the active Matplotlib figure below the local results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(RESULTS_DIR / filename)


def missing_raw_inputs(config_path: Path) -> list[Path]:
    """Return every raw measurement referenced by a config but not available."""
    with config_path.open("r") as config_file:
        config = json.load(config_file)

    missing = []
    for entry in config.values():
        if not isinstance(entry, dict) or "path" not in entry or entry["path"] == ".":
            continue
        input_path = project_path(entry["path"])
        if not input_path.exists():
            missing.append(input_path)
    return missing

def add_wavelength_axis(ax, freq, unit="µm"):
    """
    Add a wavelength axis on top of the given axis in the specified unit.
    Supported units: "cm", "µm", "nm".
    """
    c_in_units = {
        "cm": scipy.constants.c * 1e2,  # Speed of light in cm/s
        "µm": scipy.constants.c * 1e6,  # Speed of light in µm/s
        "nm": scipy.constants.c * 1e9,  # Speed of light in nm/s
    }

    if unit not in c_in_units:
        raise ValueError(f"Unsupported unit '{unit}'. Choose from 'cm', 'µm', or 'nm'.")

    c_unit_per_s = c_in_units[unit]
    wavelengths = c_unit_per_s / (freq * 1e12)  # Convert frequency (THz) to wavelength in the chosen unit

    def wavelength_formatter(x, _):
        return f"{x:.1f}"

    ax_top = ax.twiny()
    ax_top.set_xlim(wavelengths[0], wavelengths[-1])
    ax_top.set_xlabel(f"Wavelength [{unit}]")
    ax_top.xaxis.set_major_formatter(plt.FuncFormatter(wavelength_formatter))
    return ax_top



def emt(n_m, n_eff, m_i, m_h, rho_i, rho_h, vol_frac=None, method="bruggeman"):
    epsilon_h = n_m ** 2
    epsilon_eff = n_eff ** 2

    v_i = m_i / rho_i / (m_i / rho_i + m_h / rho_h)
    v_h = m_h / rho_h / (m_i / rho_i + m_h / rho_h)

    if vol_frac:
        print(f"v_i = {v_i}, vol_frac = {vol_frac}")
        v_i = vol_frac
        v_h = 1 - v_i

    if method == "bruggeman":
        denom = epsilon_eff * (v_h - 2 * v_i) - epsilon_h * (v_h + v_i)
        epsilon_incl = -epsilon_eff * (2 * epsilon_eff * (v_h + v_i) + epsilon_h * (v_i - 2 * v_h)) / denom
    elif method == "maxwell-garnett":
        denom = 2 * epsilon_h * v_i + epsilon_h + epsilon_eff * (v_i - 1)
        epsilon_incl = epsilon_h * (2 * epsilon_h * (v_i - 1) + epsilon_eff * (v_i + 2)) / denom
    elif method == "LLL":
        epsilon_incl = (1 / v_i * (epsilon_eff ** (1 / 3) - v_h * epsilon_h ** (1 / 3))) ** 3
    else:
        raise ValueError(f"Unknown EMT method: {method}")

    return np.sqrt(epsilon_incl)


def find_meta_entry(meta: dict, name: str, desired_d: Optional[float] = None):
    """
    Find the metadata entry for a given sample name and desired thickness (mm).
    If an exact key `name` exists that is returned first. Otherwise keys of the form `name_<thickness>`
    are parsed and the entry with thickness closest to desired_d is returned.
    """
    # exact name match
    if name in meta:
        return name, meta[name]

    candidates = []
    for k, v in meta.items():
        if not k.startswith(name + "_"):
            continue
        try:
            th = float(k.split("_")[-1])
        except Exception:
            continue
        candidates.append((th, k, v))

    if not candidates:
        raise KeyError(f"No metadata entries found for sample name `{name}` in metadata keys")

    if desired_d is None:
        # choose the candidate with median or smallest thickness
        chosen = min(candidates, key=lambda x: x[0])
    else:
        chosen = min(candidates, key=lambda x: abs(x[0] - desired_d))

    return chosen[1], chosen[2]


def extract_rois(path: Path, measurement_key: Optional[str] = None, ax: Optional[Axes] = None):
    """
    Extracts Regions of Interest (ROIs) from a DotthzFile and returns the pixel coordinates inside each ROI.

    Parameters:
    path (Path): Path to the DotthzFile containing the image data.
    measurement_key (Optional[str]): The key for selecting a specific measurement. Defaults to the first measurement.
    ax (Optional[Axes]): Matplotlib Axes object for optional visualization. If provided, overlays ROI outlines on the image.

    Returns:
    dict: A dictionary mapping ROI labels to lists of pixel coordinates within each ROI.
    """
    rois = {}
    with DotthzFile(path, "r") as image_file:
        # Read the measurement
        if measurement_key is None:
            # Just get the first one
            measurement_key = list(image_file.keys())[0]
        datasets = image_file[measurement_key].datasets
        metadata = image_file[measurement_key].metadata

        # Extract image data
        image = np.array(datasets["dataset"])
        image_to_plot = np.sum(image ** 2, axis=2)  # Aggregate along 3rd dimension

        # Get image dimensions
        height, width = int(float(metadata["height"])), int(float(metadata["width"]))

        if ax:
            # Plot the image
            ax.imshow(image_to_plot, cmap="gray")

            ax.legend()

            # Extract relevant metadata values
            # Extract relevant metadata values
            x_min = float(metadata['x_min [mm]'])
            x_max = float(metadata['x_max [mm]'])
            y_min = float(metadata['y_min [mm]'])
            y_max = float(metadata['y_max [mm]'])
            width = int(float(metadata['width']))  # Assumes width is given as a string
            height = int(float(metadata['height']))  # Assumes height is given as a string

            # Number of ticks we want (including both start and end)
            num_ticks_x = (width // 11) + 1  # Adjusted to include the first and last tick
            num_ticks_y = (height // 11) + 1  # Adjusted to include the first and last tick

            # Create evenly spaced tick values for x and y axes in mm using linspace
            x_ticks = np.linspace(x_min, x_max, num_ticks_x, endpoint=True)
            y_ticks = np.linspace(y_min, y_max, num_ticks_y, endpoint=True)

            # Calculate the corresponding pixel positions for the ticks
            x_tick_positions = np.linspace(0, width - 1, num_ticks_x, dtype=int)
            y_tick_positions = np.linspace(0, height - 1, num_ticks_y, dtype=int)

            # Set the x and y ticks
            ax.set_xticks(x_tick_positions)  # Set ticks at the pixel positions
            ax.set_xticklabels(np.round(x_ticks, 2))  # Convert pixel to mm and label

            ax.set_yticks(y_tick_positions)  # Set ticks at the pixel positions
            ax.set_yticklabels(np.round(y_ticks, 2))  # Convert pixel to mm and label
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        # Extract and parse ROI polygon from metadata
        for (index, roi_label) in enumerate(metadata["ROI Labels"].split(",")):
            roi_raw = metadata[f"ROI {index}"]  # Assuming ROI is stored as a string like "[[x1,y1],[x2,y2],...]"
            roi_points = eval(roi_raw) if isinstance(roi_raw, str) else roi_raw  # Convert to list of lists
            roi_points = list(roi_points)  # Ensure it's a list

            # Convert to image coordinate system (flip Y and swap x and y)
            roi_points_corrected = [(width - 1 - y, x) for x, y in roi_points]
            polygon = Polygon(roi_points_corrected)

            # Find all pixels inside the ROI
            pixels_inside_roi = [
                (x, y) for y in range(height) for x in range(width) if polygon.contains(Point(x, y))
            ]
            rois[roi_label] = pixels_inside_roi

            if ax:
                # Overlay ROI polygon (convert to list before concatenation)
                roi_y, roi_x = zip(*(roi_points_corrected + [roi_points_corrected[0]]))  # Close the polygon
                ax.plot(roi_x, roi_y, linewidth=2, label=roi_label)  # Red outline

    if len(rois) == 0:
        raise Exception(f"No ROIs found in {path}")

    return rois  # Returns list of pixel coordinates


def read_thz_file(path: Path, roi=None, single_pixel=False):
    print(path)

    if not path.is_file():
        for file in os.listdir(path):
            if not file.endswith(".thz"):
                continue
            path = path.joinpath(file)
            break

    roi_pixels = None
    if roi and not single_pixel:
        rois = extract_rois(path)
        roi_pixels = rois[roi]

    with DotthzFile(path, "r") as image_file:
        if single_pixel:
            try:
                t = np.array(image_file["Reference"]["Reference"][:, 0])
                p = np.array(image_file["Reference"]["Reference"][:, 1])
                return t, p
            except:
                pass
            t = np.array(image_file["Single Pixel 0"]["Sample"][:, 0])
            p = np.array(image_file["Single Pixel 0"]["Sample"][:, 1])
            return t, p,

        key = list(image_file.keys())[0]

        # read the first group/measurement
        datasets = image_file[key].datasets
        # from the first dataset, extract the image:

        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    if roi_pixels:
        print(f"Extracting ROI '{roi}' with {len(roi_pixels)} pixels")
        roi_values = np.array([traces[x, y, :] for x, y in roi_pixels])
        return times, roi_values

    return times, traces


def load_reference_signal(reference_path_clear_space: Path, roi: str, time_cutoff: float):
    """
    Load and prepare the clear-space (vacuum/air) reference signal used as the measurement reference.
    Returns t_ref (1D) and data_ref (paired [t_ref, signal]).
    """

    if Path(reference_path_clear_space).is_file():
        t_ref, data_ref = read_thz_file(Path(reference_path_clear_space), roi=roi, single_pixel=True)
        data_ref = data_ref[t_ref < time_cutoff]
        t_ref = t_ref[t_ref < time_cutoff]
    else:
        t_ref, data_ref = read_thz_file(Path(reference_path_clear_space).joinpath("data_image"), roi=roi)
        data_ref = data_ref[:, t_ref < time_cutoff]
        t_ref = t_ref[t_ref < time_cutoff]
        data_ref = np.mean(data_ref, axis=0)
    data_ref -= data_ref[0]
    return np.array([t_ref, data_ref])


def process_host_references(reference_paths: list[Tuple[Path, str]],
                            data_ref: np.ndarray,
                            meta: dict,
                            time_cutoff: float,
                            window_half_width: float,
                            win_func: str,
                            min_frequency: float,
                            max_frequency: float,
                            roi: str,
                            avg_in_fourier_domain: bool = False):
    """
    Process a list of host reference files (e.g. PTFE) and return averaged complex refractive index and thickness.
    Returns: (freq, complex_n_host (array), thickness_mm)
    """
    complex_list = []
    thicknesses = []
    freq = None

    for path, name in reference_paths:
        try:
            _, meta_entry = find_meta_entry(meta, name)
            d_meta = meta_entry.get("average_thickness [mm]", None)
            d = float(d_meta) if d_meta is not None else None
        except KeyError:
            d = None

        if d is None:
            raise ValueError(f"Missing thickness in metadata for host `{name}`")

        buffer = compute_buffer_from_trace(Path(path), data_ref, d,
                                           time_cutoff, window_half_width, win_func,
                                           min_frequency, max_frequency, roi, avg_in_fourier_domain)
        complex_n = buffer["refractive_index"]
        freq = buffer["frequency"]
        complex_list.append(np.array(complex_n))
        thicknesses.append(d)

    complex_mean = np.mean(np.array(complex_list), axis=0)
    thickness_mean = float(np.mean(thicknesses))
    return freq, complex_mean, thickness_mean


def process_pellet(file_path: Path, name: str, mass_frac_sample: float, data_ref: np.ndarray,
                   meta: dict, n_host_complex: np.ndarray, rho_host: Optional[float],
                   time_cutoff: float, window_half_width: float, win_func: str,
                   min_frequency: float, max_frequency: float, roi: str, avg_in_fourier_domain: bool = False):
    """
    Process a single pellet, compute its complex refractive index and apply EMT with the host.
    Returns a dict with results.
    """
    try:
        _, meta_entry = find_meta_entry(meta, name)
        rho = meta_entry.get("density [g/cm3]", None)
        mass = meta_entry.get("nass [g]", None)
        d_meta = meta_entry.get("average_thickness [mm]", None)
        sample_vol = meta_entry.get("sample_volume [cm3]", None)
        d = float(d_meta) if d_meta is not None else None
    except KeyError:
        rho = None
        sample_vol = None
        d = None

    if d is None:
        raise ValueError(f"Missing thickness metadata for pellet `{name}`")

    buffer = compute_buffer_from_trace(Path(file_path), data_ref, d,
                                       time_cutoff, window_half_width, win_func,
                                       min_frequency, max_frequency, roi, avg_in_fourier_domain)
    complex_n_sample = buffer["refractive_index"]
    freq = buffer["frequency"]

    # Set defaults for densities if unavailable
    rho_sample = rho if rho is not None else 1.0
    rho_host_used = rho_host if rho_host is not None else 0.95

    mass_frac_host = 1.0 - mass_frac_sample

    m_host = mass_frac_host * mass

    v_host = m_host/ rho_host

    v_i = (sample_vol - v_host ) / sample_vol

    # apply EMT to get effective inclusion refractive index (as in original flow)
    n_eff = emt(n_host_complex, complex_n_sample, mass_frac_sample, mass_frac_host, rho_sample, rho_host_used,
                vol_frac=v_i,
                method="bruggeman")

    refractive_index_eff = np.real(n_eff)
    absorption_coefficient = 4 * np.pi * np.imag(n_eff) * buffer["frequency"] * 1e12 / scipy.constants.c * 1e-2

    return {
        "name": name,
        "d_mm": d,
        "frequency": freq,
        "complex_n_sample": complex_n_sample,
        "n_eff": n_eff,
        "refractive_index_eff": refractive_index_eff,
        "absorption_coefficient": absorption_coefficient,
        "buffer": buffer,
    }


def parse_config(cfg: dict):
    """
    Parse JSON config with keys:
    - 'host': { 'material': ..., 'density [g/cm3]': ..., ...}
    - 'clear_space': { 'path': ... }
    - 'pellet_*': per-pellet dicts
    Returns:
      - clear_space_path (Path)
      - host_material (str)
      - host_density (float|None)
      - host_pellets (list of pellet dicts with valid paths)
      - sample_pellets (list of pellet dicts)
    """
    clear_space_path = project_path(cfg.get("clear_space", {}).get("path", "."))
    host = cfg.get("host", {})
    host_material = host.get("material", "PTFE")
    host_density = host.get("density [g/cm3]", None)

    host_pellets = []
    sample_pellets = []

    for k, v in cfg.items():
        if k.startswith("clear_space"):
            continue
        pellet = dict(v)  # copy
        pellet.setdefault("path", ".")
        if pellet["path"] != ".":
            pellet["path"] = str(project_path(pellet["path"]))
        pellet.setdefault("material", "Unknown")
        # normalize keys existence
        pellet.setdefault("average_thickness [mm]", None)
        pellet.setdefault("sample_content", 0.1)
        pellet.setdefault("sample_volume [cm3]", None)
        pellet.setdefault("density [g/cm3]", None)

        if pellet["material"] == host_material and pellet["path"] != ".":
            host_pellets.append(pellet)
        # keep all pellets in sample list (we may skip ones without data later)
        sample_pellets.append(pellet)

    return clear_space_path, host_material, host_density, host_pellets, sample_pellets


def compute_buffer_from_trace(path: Path, data_ref: np.ndarray, d_mm: float,
                              time_cutoff: float, window_half_width: float, win_func: str,
                              min_frequency: float, max_frequency: float, roi: str,
                              avg_in_fourier_domain: bool = False):
    """
    Read a pellet file and compute the uniform_slab buffer dictionary.
    Returns the buffer dict from thzpy.transferfunctions.uniform_slab and
    adds keys:
      - 'refractive_index_std' (real-part std across ROI pixels)
      - 'absorption_coefficient_std' (std across ROI pixels)
    If avg_in_fourier_domain is True we compute buffer per pixel in the optical domain,
    otherwise we compute a buffer per pixel from each ROI trace and then compute mean/std.
    """
    t_sample, data_sample = read_thz_file(path.joinpath("data_image"), roi=roi)
    data_sample = data_sample[:, t_sample < time_cutoff]
    t_sample = t_sample[t_sample < time_cutoff]

    # collect per-pixel buffers
    refr_list = []
    abs_list = []
    freq = None

    # if single pixel array, ensure shape (n_pixels, n_times)
    if data_sample.ndim == 1:
        data_sample = data_sample[np.newaxis, :]

    n_pixels = data_sample.shape[0]

    for pixel in range(n_pixels):
        p = data_sample[pixel, :].copy()
        p -= p[0]
        p_pair = np.array([t_sample, p])
        sample, reference = common_window([p_pair, data_ref],
                                          half_width=window_half_width, win_func=win_func)
        buffer_pixel = uniform_slab(d_mm,
                                    sample, reference, n_med=1,
                                    upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                    all_optical_constants=True)
        freq = buffer_pixel["frequency"]
        refr_list.append(np.array(buffer_pixel["refractive_index"]).astype(complex))
        abs_list.append(np.array(buffer_pixel["absorption_coefficient"]).astype(float))

        # if user chose avg_in_fourier_domain, we still compute per-pixel and will average later
        if not avg_in_fourier_domain:
            # continue collecting per-pixel results (already done)
            pass

    # stack and compute mean/std
    refr_stack = np.vstack(refr_list)  # shape (n_pixels, n_freq)
    abs_stack = np.vstack(abs_list)  # shape (n_pixels, n_freq)

    refr_mean = np.mean(refr_stack, axis=0)
    refr_std_real = np.std(np.real(refr_stack), axis=0)
    abs_mean = np.mean(abs_stack, axis=0)
    abs_std = np.std(abs_stack, axis=0)

    # Build a representative buffer from last pixel to preserve other keys, then overwrite values
    buffer = {
        "frequency": freq,
        "refractive_index": refr_mean,
        "refractive_index_std": refr_std_real,
        "absorption_coefficient": abs_mean,
        "absorption_coefficient_std": abs_std,
    }

    return buffer


def _compute_n_eff_and_alpha_for_d(path: Path, d_val: float, data_ref: np.ndarray,
                                   n_host_complex: np.ndarray, mass_frac_sample: float, mass_frac_host: float,
                                   rho_sample: float, rho_host_used: float, pellet_radius_mm: float,
                                   sample_vol, time_cutoff, window_half_width, win_func,
                                   min_frequency, max_frequency, roi: str, avg_in_fourier_domain: bool):
    """
    Helper: compute n_eff and absorption coefficient for a given thickness d_val.
    """
    buffer_d = compute_buffer_from_trace(path, data_ref, d_val,
                                         time_cutoff, window_half_width, win_func,
                                         min_frequency, max_frequency, roi, avg_in_fourier_domain)
    complex_n_sample_d = buffer_d["refractive_index"]
    freq = buffer_d["frequency"]

    vol_frac = None
    if sample_vol:
        pellet_area_cm2 = np.pi * (pellet_radius_mm ** 2) / 100.0
        full_volume_cm3 = pellet_area_cm2 * (d_val / 10.0)
        vol_frac = sample_vol / full_volume_cm3 if full_volume_cm3 != 0 else None

    n_eff_d = emt(n_host_complex, complex_n_sample_d, mass_frac_sample, mass_frac_host, rho_sample, rho_host_used,
                  vol_frac=vol_frac, method="bruggeman")
    refr_real_d = np.real(n_eff_d)
    absorption_d = 4 * np.pi * np.imag(n_eff_d) * freq * 1e12 / scipy.constants.c * 1e-2
    return refr_real_d, absorption_d


def process_pellet_from_config(pellet: dict,
                               data_ref: np.ndarray,
                               n_host_complex: np.ndarray,
                               rho_host: Optional[float],
                               time_cutoff: float,
                               window_half_width: float,
                               win_func: str,
                               min_frequency: float,
                               max_frequency: float,
                               roi: str,
                               avg_in_fourier_domain: bool = False):
    """
    Process a single pellet dict from config and apply EMT.
    Now includes per-pellet ROI std and optional thickness-propagated std in the returned dict:
      - 'absorption_coefficient_std' (from ROI)
      - 'refractive_index_std' (from ROI)
      - 'absorption_coefficient_thickness_std' (from d uncertainty)
      - 'refractive_index_thickness_std' (from d uncertainty)
    Expects optional pellet key `average_thickness_std [mm]`.
    """
    path_str = pellet.get("path", ".")
    if path_str == "." or not Path(path_str).exists():
        raise FileNotFoundError(f"No data path for pellet `{pellet.get('material')}` at `{path_str}`")

    d_meta = pellet.get("average_thickness [mm]", None)
    if d_meta is None:
        raise ValueError(f"Pellet `{path_str}` missing average_thickness [mm]")

    d = float(d_meta)
    thickness_std = float(
        pellet.get(
            "average_thickness_std [mm]",
            pellet.get("thickness_error [mm]", 0.0),
        )
    )

    pellet_radius_mm = pellet.get(
        "pellet_radius [mm]",
        pellet.get("radius [mm]", 14.85 / 2),
    )
    sample_vol = pellet.get("sample_volume [cm3]", None)
    rho_sample = pellet.get("density [g/cm3]", None) or 1.0
    mass_frac_sample = float(pellet.get("sample_content", 0.1))
    mass_frac_host = 1.0 - mass_frac_sample
    rho_host_used = rho_host if rho_host is not None else 0.95

    # main buffer at nominal d (this already provides ROI-based stds)
    buffer = compute_buffer_from_trace(Path(path_str), data_ref, d,
                                       time_cutoff, window_half_width, win_func,
                                       min_frequency, max_frequency, roi, avg_in_fourier_domain)
    complex_n_sample = buffer["refractive_index"]
    freq = buffer["frequency"]

    # main EMT result
    vol_frac_nom = None
    if sample_vol:
        pellet_area_cm2 = np.pi * (pellet_radius_mm ** 2) / 100.0
        full_volume_cm3 = pellet_area_cm2 * (d / 10.0)
        vol_frac_nom = sample_vol / full_volume_cm3 if full_volume_cm3 != 0 else None

    n_eff = emt(n_host_complex, complex_n_sample, mass_frac_sample, mass_frac_host, rho_sample, rho_host_used,
                vol_frac=vol_frac_nom, method="bruggeman")

    refractive_index_eff = np.real(n_eff)
    absorption_coefficient = 4 * np.pi * np.imag(n_eff) * freq * 1e12 / scipy.constants.c * 1e-2

    # thickness-propagated uncertainty (Gaussian propagation via finite difference)
    refr_thickness_std = np.zeros_like(refractive_index_eff)
    abs_thickness_std = np.zeros_like(absorption_coefficient)
    if thickness_std and thickness_std > 0:
        # ensure positive thickness for minus side
        delta = thickness_std
        d_minus = max(1e-6, d - delta)
        d_plus = d + delta

        refr_plus, abs_plus = _compute_n_eff_and_alpha_for_d(Path(path_str), d_plus, data_ref,
                                                             n_host_complex, mass_frac_sample, mass_frac_host,
                                                             rho_sample, rho_host_used, pellet_radius_mm,
                                                             sample_vol, time_cutoff, window_half_width, win_func,
                                                             min_frequency, max_frequency, roi, avg_in_fourier_domain)

        refr_minus, abs_minus = _compute_n_eff_and_alpha_for_d(Path(path_str), d_minus, data_ref,
                                                               n_host_complex, mass_frac_sample, mass_frac_host,
                                                               rho_sample, rho_host_used, pellet_radius_mm,
                                                               sample_vol, time_cutoff, window_half_width, win_func,
                                                               min_frequency, max_frequency, roi, avg_in_fourier_domain)

        # propagated sigma = |f(d+sd) - f(d-sd)| / 2  (equivalent to |df/dd| * sd)
        refr_thickness_std = np.abs(refr_plus - refr_minus) / 2.0
        abs_thickness_std = np.abs(abs_plus - abs_minus) / 2.0

    return {
        "name": pellet.get("material", "unknown"),
        "path": path_str,
        "d_mm": d,
        "frequency": freq,
        "complex_n_sample": complex_n_sample,
        "n_eff": n_eff,
        "refractive_index_eff": refractive_index_eff,
        "absorption_coefficient": absorption_coefficient,
        "absorption_coefficient_std": buffer.get("absorption_coefficient_std"),
        "refractive_index_std": buffer.get("refractive_index_std"),
        "absorption_coefficient_thickness_std": abs_thickness_std,
        "refractive_index_thickness_std": refr_thickness_std,
        "buffer": buffer,
    }


# python
def plot_material_averages(results: list, label: str, files: Optional[list] = None, npoints: int = 500,
                           errors: bool = False):
    color_map = {}
    if files:
        for f in files:
            try:
                color_map[f[1]] = f[2]
            except Exception:
                continue

    materials = {}
    for r in results:
        name = r.get("name", "unknown")
        freq = np.array(r["frequency"])
        abs_val = np.array(r["absorption_coefficient"])

        abs_std_roi = np.array(r.get("absorption_coefficient_std", np.zeros_like(abs_val)))
        abs_std_d = np.array(r.get("absorption_coefficient_thickness_std", np.zeros_like(abs_val)))
        abs_std_total_per_sample = np.sqrt(np.array(abs_std_roi) ** 2 + np.array(abs_std_d) ** 2)

        if "n_eff" in r and r["n_eff"] is not None:
            n_complex = np.array(r["n_eff"])
        elif "complex_n_sample" in r and r["complex_n_sample"] is not None:
            n_complex = np.array(r["complex_n_sample"])
        else:
            if "refractive_index_eff" in r:
                n_complex = np.array(r["refractive_index_eff"]).astype(complex)
            else:
                continue

        n_real_std_roi = np.array(r.get("refractive_index_std", np.zeros_like(np.real(n_complex))))
        n_real_std_d = np.array(r.get("refractive_index_thickness_std", np.zeros_like(np.real(n_complex))))
        n_real_std_total = np.sqrt(n_real_std_roi ** 2 + n_real_std_d ** 2)

        materials.setdefault(name, []).append({
            "f": freq,
            "abs": abs_val,
            "abs_std": abs_std_total_per_sample,
            "n": n_complex,
            "n_real_std": n_real_std_total
        })

    cmap = plt.get_cmap("tab10")
    aggregated = {}
    coda_t3 = None
    for idx, (name, entries) in enumerate(materials.items()):
        f_mins = [np.min(e["f"]) for e in entries]
        f_maxs = [np.max(e["f"]) for e in entries]
        f_min = max(f_mins)
        f_max = min(f_maxs)
        if f_max <= f_min:
            f_min = min(f_mins)
            f_max = max(f_maxs)
        f_common = np.linspace(f_min, f_max, npoints)

        abs_interp = []
        abs_std_interp = []
        n_interp = []
        n_real_std_interp = []

        for e in entries:
            abs_i = np.interp(f_common, e["f"], e["abs"])
            abs_std_i = np.interp(f_common, e["f"], e["abs_std"])
            n_real_i = np.interp(f_common, e["f"], np.real(e["n"]))
            n_imag_i = np.interp(f_common, e["f"], np.imag(e["n"]))
            n_std_i = np.interp(f_common, e["f"], e["n_real_std"])
            abs_interp.append(abs_i)
            abs_std_interp.append(abs_std_i)
            n_interp.append(n_real_i + 1j * n_imag_i)
            n_real_std_interp.append(n_std_i)

        abs_stack = np.vstack(abs_interp) if abs_interp else np.empty((0, len(f_common)))
        abs_std_stack = np.vstack(abs_std_interp) if abs_std_interp else np.empty((0, len(f_common)))
        n_stack = np.vstack(n_interp) if n_interp else np.empty((0, len(f_common)), dtype=complex)
        n_real_std_stack = np.vstack(n_real_std_interp) if n_real_std_interp else np.empty((0, len(f_common)))

        abs_mean = np.mean(abs_stack, axis=0) if abs_stack.size else np.zeros_like(f_common)
        between_abs_std = np.std(abs_stack, axis=0) if abs_stack.size else np.zeros_like(f_common)
        mean_abs_var = np.mean(abs_std_stack ** 2, axis=0) if abs_std_stack.size else np.zeros_like(f_common)
        total_abs_std = np.sqrt(between_abs_std ** 2 + mean_abs_var)

        n_mean = np.mean(n_stack, axis=0) if n_stack.size else np.zeros_like(f_common, dtype=complex)
        between_n_real_std = np.std(np.real(n_stack), axis=0) if n_stack.size else np.zeros_like(f_common)
        mean_n_var = np.mean(n_real_std_stack ** 2, axis=0) if n_real_std_stack.size else np.zeros_like(f_common)
        total_n_real_std = np.sqrt(between_n_real_std ** 2 + mean_n_var)

        aggregated[name] = {
            "frequency": f_common,
            "absorption_mean": abs_mean,
            "absorption_std": total_abs_std,
            "n_mean": n_mean,
            "n_real_std": total_n_real_std,
            "color": material_color(name, color_map.get(name, cmap(idx % cmap.N)))
        }

    # --- compute CoDA-T3 composite (40% SiO2 / 60% CI) if both materials present ---
    def _find_key_ci_sio2(agg):
        key_sio2 = None
        key_ci = None
        for k in agg.keys():
            kl = k.lower()
            if "sio2" in kl or "silica" in kl or "silicon dioxide" in kl:
                key_sio2 = k
            if kl in ("ci", "carbon", "ink", "ci-ink", "ci ink") or "ci" == kl:
                key_ci = k
        return key_sio2, key_ci

    k_sio2, k_ci = _find_key_ci_sio2(aggregated)
    if k_sio2 and k_ci:
        f1 = aggregated[k_sio2]["frequency"]
        f2 = aggregated[k_ci]["frequency"]
        f_min = max(np.min(f1), np.min(f2))
        f_max = min(np.max(f1), np.max(f2))
        if f_max > f_min:
            f_common = np.linspace(f_min, f_max, npoints)
            # interpolate complex n for both materials
            n1_real = np.interp(f_common, f1, np.real(aggregated[k_sio2]["n_mean"]))
            n1_imag = np.interp(f_common, f1, np.imag(aggregated[k_sio2]["n_mean"]))
            n1 = n1_real + 1j * n1_imag
            n2_real = np.interp(f_common, f2, np.real(aggregated[k_ci]["n_mean"]))
            n2_imag = np.interp(f_common, f2, np.imag(aggregated[k_ci]["n_mean"]))
            n2 = n2_real + 1j * n2_imag

            # mass fractions: 66% SiO2 (inclusion), 34% CI (host)
            m_sio2 = 0.66
            m_ci = 0.34
            # assumed densities (g/cm^3) -- adjust if you have measured values
            rho_sio2 = 2.41
            rho_ci = 2.65

            try:
                n_eff_coda = compute_bruggeman_n_eff(n1, n2, m_sio2, m_ci, rho_sio2, rho_ci, vol_frac=None)
                alpha_coda = 4 * np.pi * np.imag(n_eff_coda) * f_common * 1e12 / scipy.constants.c * 1e-2
                coda_t3 = {
                    "frequency": f_common,
                    "n_real": np.real(n_eff_coda),
                    "alpha": alpha_coda
                }
            except Exception:
                coda_t3 = None

    # plotting
    fig, axes = plt.subplots(nrows=2, figsize=(12, 10))
    for name, d in aggregated.items():
        color = d["color"]
        f = d["frequency"]
        absorption_per_cm = d["absorption_mean"]
        absorption_std = d["absorption_std"]
        n_real = np.real(d["n_mean"])
        n_real_std = d["n_real_std"]

        axes[1].plot(f, absorption_per_cm, color=color, label=name)
        if errors:
            axes[1].fill_between(f, absorption_per_cm - absorption_std, absorption_per_cm + absorption_std,
                                 color=color, alpha=0.2)
        axes[0].plot(f, n_real, color=color, label=name)
        if errors:
            axes[0].fill_between(f, n_real - n_real_std, n_real + n_real_std, color=color, alpha=0.2)

    # plot CoDA-T3 composite if available
    # if coda_t3 is not None:
    #     axes[0].plot(coda_t3["frequency"], coda_t3["n_real"], color="black", linestyle="--",
    #                  label="CoDA-T3 calculated")
    #     axes[1].plot(coda_t3["frequency"], coda_t3["alpha"], color="black", linestyle="--", label="CoDA-T3 calculated")
    #
    #     aggregated["CoDA-T3"] = {
    #         "frequency": coda_t3["frequency"],
    #         "absorption_mean": coda_t3["alpha"],
    #         "absorption_std": 0,
    #         "n_mean": coda_t3["n_real"],
    #         "n_real_std": 0,
    #         "color": "black"
    #     }

    axes[0].set_xlabel("Frequency [THz]")
    axes[1].set_xlabel("Frequency [THz]")
    axes[1].set_ylabel(r"Absorption coefficient $\alpha$ [cm$^{-1}$]")
    axes[0].set_ylabel(r"Refractive index $n$ [-]")

    add_wavelength_axis(axes[0], next(iter(aggregated.values()))["frequency"])
    add_wavelength_axis(axes[1], next(iter(aggregated.values()))["frequency"])

    fig.subplots_adjust(hspace=0.5, bottom=0.18, top=0.9)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=4)
    save_result(f"combined_properties_{label}.png")
    save_result(f"combined_properties_{label}.pdf")
    plt.close(fig)

    # individual per-material averaged plots (unchanged)
    for name, d in aggregated.items():
        f = d["frequency"]
        absorption_per_cm = d["absorption_mean"]
        absorption_std = d["absorption_std"]
        n_real = np.real(d["n_mean"])
        n_real_std = d["n_real_std"]

        fig, ax1 = plt.subplots(figsize=(8, 6))

        color_abs = unibe.BLUE
        color_n = unibe.RED

        l1 = ax1.plot(f, absorption_per_cm, color=color_abs, label=r"Absorption coefficient $\alpha$")
        if errors:
            ax1.fill_between(f, absorption_per_cm - absorption_std, absorption_per_cm + absorption_std,
                             color=color_abs, alpha=0.2)
        ax1.set_xlabel("Frequency [THz]")
        ax1.set_ylabel(r"Absorption coefficient $\alpha$ [cm$^{-1}$]", color=color_abs)
        ax1.tick_params(axis="y", labelcolor=color_abs)
        ax1.tick_params(axis="x")

        safe_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', name)
        data_to_save = np.column_stack((
            d["frequency"],
            d["absorption_mean"],
            d["absorption_std"],
            np.real(d["n_mean"]),
            d["n_real_std"]
        ))

        # Save as .npy file
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(DATA_DIR / f"{safe_name}_data.npy", data_to_save)

        ax2 = ax1.twinx()
        l2 = ax2.plot(f, n_real, color=color_n, label=r"Refractive index $n$")
        if errors:
            ax2.fill_between(f, n_real - n_real_std, n_real + n_real_std, color=color_n, alpha=0.2)
        ax2.set_ylabel(r"Refractive index $n$ [-]", color=color_n)
        ax2.tick_params(axis="y", labelcolor=color_n)

        n_min = np.min(n_real)
        n_max = np.max(n_real)
        n_range = (n_max - n_min) if (n_max - n_min) != 0 else max(0.1, abs(n_max) * 0.1)
        ax2.set_ylim([round(n_min - n_range, 3), round(n_max + n_range, 3)])

        lines = l1 + l2
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="best")

        add_wavelength_axis(ax1, f)
        add_wavelength_axis(ax2, f)

        plt.tight_layout()
        safe_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', name)
        save_result(f"{safe_name}_properties.png")
        save_result(f"{safe_name}_properties.pdf")
        plt.close(fig)



def process_host_references_from_config(host_pellets: list[dict],
                                        data_ref: np.ndarray,
                                        time_cutoff: float,
                                        window_half_width: float,
                                        win_func: str,
                                        min_frequency: float,
                                        max_frequency: float,
                                        roi: str,
                                        avg_in_fourier_domain: bool = False):
    """
    Process host pellets provided as dicts (from config). Returns (freq, n_host_complex, thickness_mean).
    """
    if not host_pellets:
        raise ValueError("No host pellets available for reference (check config).")

    complex_list = []
    thicknesses = []
    freq = None

    for pellet in host_pellets:
        path = Path(pellet["path"])
        d_meta = pellet.get("average_thickness [mm]", None)
        if d_meta is None:
            raise ValueError(f"Host pellet at {path} missing 'average_thickness [mm]'")
        d = float(d_meta)
        # compute buffer using existing helper
        buffer = compute_buffer_from_trace(path, data_ref, d,
                                           time_cutoff, window_half_width, win_func,
                                           min_frequency, max_frequency, roi, avg_in_fourier_domain)
        complex_n = buffer["refractive_index"]
        freq = buffer["frequency"]
        complex_list.append(np.array(complex_n))
        thicknesses.append(d)

    complex_mean = np.mean(np.array(complex_list), axis=0)
    thickness_mean = float(np.mean(thicknesses))
    return freq, complex_mean, thickness_mean


def analyse_pellets_from_config(config_path: Path,
                                avg_in_fourier_domain: bool = False,
                                window_half_width: float = 20.0,
                                win_func: str = "blackman",
                                min_frequency: float = 0.25,
                                max_frequency: float = 3.5,
                                time_cutoff: float = 1960.0,
                                roi="ROI 1",
                                errors: bool = False):
    """
    Main entry: reads a JSON config file and runs the analysis.
    """
    with open(config_path, "r") as fh:
        cfg = json.load(fh)

    clear_space_path, host_material, host_density, host_pellets, sample_pellets = parse_config(cfg)

    # load clear-space reference
    data_ref = load_reference_signal(Path(clear_space_path), roi=roi, time_cutoff=time_cutoff)

    # process host references (PTFE) collected among pellets in config
    if host_pellets:
        freq_host, n_host_complex, thickness_host = process_host_references_from_config(
            host_pellets, data_ref, time_cutoff, window_half_width, win_func, min_frequency, max_frequency, roi,
            avg_in_fourier_domain)
    else:
        raise ValueError("No host pellet measurements found in config for host material.")

    # host density from config (preferred) or None
    rho_host = host_density

    results = []
    for pellet in sample_pellets:
        # skip host entries used for references if you don't want to treat them as samples
        if pellet["material"] == host_material:
            # optionally still include host pellets in results (comment/uncomment as desired)
            try:
                res = process_pellet_from_config(pellet, data_ref, n_host_complex, rho_host,
                                                 time_cutoff, window_half_width, win_func,
                                                 min_frequency, max_frequency, roi, avg_in_fourier_domain)
                results.append(res)
            except FileNotFoundError:
                # host pellet without path -> skip
                continue
        else:
            try:
                res = process_pellet_from_config(pellet, data_ref, n_host_complex, rho_host,
                                                 time_cutoff, window_half_width, win_func,
                                                 min_frequency, max_frequency, roi, avg_in_fourier_domain)
                results.append(res)
            except FileNotFoundError:
                # skip pellets without data (path == .)
                continue

    # plotting (same layout as before)
    fig, axes = plt.subplots(nrows=2, figsize=(12, 10))
    axes[0].plot(freq_host, np.real(n_host_complex), label=f"Host ({host_material}) averaged")
    axes[1].plot(freq_host,
                 -4 * np.pi * np.imag(np.sqrt(n_host_complex ** 2)) * freq_host * 1e12 / scipy.constants.c * 1e-2,
                 label="Host (avg)")

    for r in results:
        axes[0].plot(r["frequency"], r["refractive_index_eff"], label=f"{r['name']}, {r['d_mm']:.2f} mm")
        axes[1].plot(r["frequency"], r["absorption_coefficient"], label=f"{r['name']}, {r['d_mm']:.2f} mm")

    axes[0].set_xlabel("Frequency [THz]")
    axes[0].set_ylabel("Refractive Index [-]")
    axes[1].set_xlabel("Frequency [THz]")
    axes[1].set_ylabel("Absorption [cm$^{-1}$]")

    add_wavelength_axis(axes[0], freq_host)
    add_wavelength_axis(axes[1], freq_host)

    axes[0].legend(loc="lower right")
    plt.tight_layout()
    save_result(f"combined_pellet_emt_{config_path.stem}.png")
    plt.close(fig)

    plot_material_averages(results, config_path.stem, files=None, errors=errors)

    return results


if __name__ == "__main__":
    config_files = (
        METADATA_DIR / "pellets_meta_analysis.json",
        METADATA_DIR / "sub_pellets_meta_analysis.json",
    )
    missing_inputs = [
        input_path
        for config_file in config_files
        for input_path in missing_raw_inputs(config_file)
    ]
    if missing_inputs:
        missing_list = "\n".join(f"- {path}" for path in missing_inputs)
        raise FileNotFoundError(
            "Raw material-property measurements are missing:\n" + missing_list
        )

    for config_file in config_files:
        analyse_pellets_from_config(
            config_file,
            avg_in_fourier_domain=True,
            errors=True,
            roi="ROI 1",
            max_frequency=4,
        )
