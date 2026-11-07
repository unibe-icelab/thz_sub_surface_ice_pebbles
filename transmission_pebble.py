import argparse
from pathlib import Path

from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from plotly.subplots import make_subplots
from pydotthz import DotthzFile
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab

from scicolorscales import vik
from sub_surface_extraction import extract_pebble
import plotly.graph_objects as go

from surface_extraction import get_thz_file_from_path
from surface_extraction import extract_surface
import pyvista as pv
import numpy as np
from skimage.measure import marching_cubes
from plot_style import FOREGROUND_COLOR, apply_plot_style, save_figure
ROOT_DIR = Path(__file__).resolve().parent
MEASUREMENTS_DIR = ROOT_DIR / "data" / "pebbles" / "measurements"
MEASUREMENTS = {
    "solid_ice_pebble": {
        "material": "solid",
        "directory": "ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_5",
        "reference": "comet_dust_image_mean",
        "d_mm": 10.0,
        "mask_radius": 25,
        "mask_center_x": -3,
        "mask_center_y": -6,
        "vmin": 1.0,
        "vmax": 2.0,
    },
    "spipa_b_pebble": {
        "material": "SPIPA-B",
        "directory": "ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_4",
        "reference": "comet_dust_image_mean",
        "d_mm": 10.0,
        "mask_radius": 25,
        "mask_center_x": -3,
        "mask_center_y": -6,
        "vmin": 1.0,
        "vmax": 2.0,
    },
    "frost_hr": {
        "material": "Frost HR",
        "directory": "porosity_august2_2026_hr_focused_frost_10.0mm_7.2g",
        "reference": "frost_single_pixel",
        "d_mm": 10.0,
        "mask_radius": 60,
        "mask_center_x": -3,
        "mask_center_y": -6,
        "vmin": 1.0,
        "vmax": 1.5,
    },
}

REFERENCES = {
    "comet_dust_image_mean": (
        "image_mean",
        MEASUREMENTS_DIR / "comet_dust_pellet_HESSO_silicon_ref_vac_3" / "data_image",
    ),
    "frost_single_pixel": (
        "single_pixel",
        MEASUREMENTS_DIR
        / "porosity_august2_2026_focused_silicon_reference"
        / "data"
        / "trans"
        / "single_pixel"
        / "1786972921.4414234_sp_data.thz",
    ),
}

ACTIVE_MEASUREMENT = "solid_ice_pebble"
CONFIG = MEASUREMENTS[ACTIVE_MEASUREMENT]
MATERIAL = CONFIG["material"]
DATA_DIR = MEASUREMENTS_DIR / CONFIG["directory"] / "data_image"
REFERENCE_MODE, REFERENCE_PATH = REFERENCES[CONFIG["reference"]]
RESULTS_DIR = ROOT_DIR / "results" / "transmission_pebble"

params = {"ytick.color":
              "black",

          "xtick.color":
              "black",

          "axes.labelcolor":
              "black",

          "axes.edgecolor":
              "black",

          "mathtext.fontset":
              "cm",

          "mathtext.rm":
              "Times New Roman",

          "font.family":
              "Arial",

          "font.size":
              18}

plt.rcParams.update(params)
apply_plot_style()


def get_refraction_index(time, traces, t_ref, p_ref, window_half_width=15, win_func="hanning",
                         min_frequency=0.2, max_frequency=3, d_mm=1.0, mask_radius=None, mask_center_x=None, mask_center_y=None):
    """
    Compute refractive index map for all pixels in `traces`.
    This function first determines the frequency axis from the first valid pixel,
    then preallocates arrays with a homogeneous shape (nx, ny, n_freq) and fills
    them. Failed pixels are filled with NaN to avoid inhomogeneous nested lists.
    """
    nx, ny = traces.shape[0], traces.shape[1]
    data_ref = np.array([t_ref, p_ref])

    # Determine frequency axis from the first pixel that yields a valid result
    freq = None
    freq_len = None
    for x in range(nx):
        for y in range(ny):
            trace = traces[x, y, :]
            p_pair = np.array([time, trace])
            try:
                sample, reference = common_window([p_pair, data_ref],
                                                  half_width=window_half_width, win_func=win_func)
                buffer_pixel = uniform_slab(d_mm,
                                            sample, reference, n_med=1,
                                            upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                            all_optical_constants=True)
                freq = np.array(buffer_pixel["frequency"])
                freq_len = len(freq)
                break
            except Exception:
                # try next pixel
                continue
        if freq is not None:
            break

    if freq is None:
        raise RuntimeError("Unable to compute frequency axis from any pixel. Check input traces/ref.")

    # Preallocate arrays with homogeneous shape and fill with NaN for failures
    refractive_index = np.full((nx, ny, freq_len), np.nan, dtype=float)
    absorption_coefficient = np.full((nx, ny, freq_len), np.nan, dtype=float)
    complex_refractive_index = np.full((nx, ny, freq_len), np.nan + 0j, dtype=complex)

    for x in range(nx):
        for y in range(ny):

            if mask_radius:
                if not mask_center_y:
                    mask_center_y = 0
                if not mask_center_x:
                    mask_center_x = 0

                mask_radius_mm = mask_radius * 1 / 0.5

                if ((x - mask_center_x - nx // 2) ** 2 + (y - mask_center_y - ny // 2) ** 2) > mask_radius_mm ** 2:
                    continue
            trace = traces[x, y, :]
            p_pair = np.array([time, trace])
            try:

                sample, reference = common_window([p_pair, data_ref],
                                                  half_width=window_half_width, win_func=win_func)
                buffer_pixel = uniform_slab(d_mm,
                                            sample, reference, n_med=1,
                                            upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                            all_optical_constants=True)
                freq_pix = np.array(buffer_pixel["frequency"])
                n = np.array(buffer_pixel["refractive_index"]).astype(complex)
                alpha = np.array(buffer_pixel["absorption_coefficient"]).astype(float)
                # Only accept if frequency axis matches expected length; otherwise fill NaNs
                if len(freq_pix) == freq_len:
                    complex_refractive_index[x, y, :] = n
                    refractive_index[x, y, :] = n.real
                    absorption_coefficient[x, y, :] = alpha
                else:
                    # leave as NaN if mismatch
                    continue
            except Exception:
                # leave as NaN on any error
                continue

    return freq, refractive_index


def load_reference(mode, reference_path):
    """Load either a spatially averaged image reference or a single-pixel reference."""
    if mode == "image_mean":
        path = get_thz_file_from_path(reference_path)
        if path is None:
            raise FileNotFoundError(f"No *_data.thz reference found in {reference_path}")

        with DotthzFile(path) as reference_file:
            key = list(reference_file.keys())[0]
            datasets = reference_file[key].datasets
            reference_time = np.array(datasets["time"])
            reference_traces = np.array(datasets["dataset"])

        reference_pulse = np.mean(reference_traces, axis=(0, 1))
        return reference_time, reference_pulse

    if mode == "single_pixel":
        with DotthzFile(reference_path) as reference_file:
            sample = reference_file["Single Pixel 0"].datasets["Sample"][:]
        return sample[:, 0], sample[:, 1]

    raise ValueError(f"Unknown reference mode: {mode}")


def configuration_for_path(path: Path) -> dict:
    """Use the known configuration associated with an input measurement."""
    measurement_name = path.parent.name if path.name == "data_image" else path.name
    for candidate in MEASUREMENTS.values():
        if candidate["directory"] == measurement_name:
            return candidate
    return CONFIG


def reference_for_configuration(config: dict, input_path: Path) -> tuple[str, Path]:
    """Resolve a reference stored beside the selected transmission measurement."""
    measurements_dir = input_path.parent.parent if input_path.name == "data_image" else input_path.parent
    if config["reference"] == "comet_dust_image_mean":
        return (
            "image_mean",
            measurements_dir / "comet_dust_pellet_HESSO_silicon_ref_vac_3" / "data_image",
        )
    return (
        "single_pixel",
        measurements_dir / "porosity_august2_2026_focused_silicon_reference" /
        "data" / "trans" / "single_pixel" / "1786972921.4414234_sp_data.thz",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot transmission through an ice pebble.")
    parser.add_argument("path", nargs="?", type=Path, default=DATA_DIR,
                        help="Measurement data_image directory.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = configuration_for_path(args.path)
    reference_mode, reference_path = reference_for_configuration(config, args.path)
    t_ref, p_ref = load_reference(reference_mode, reference_path)

    material = config["material"]
    path = get_thz_file_from_path(args.path)

    with DotthzFile(path) as psf_data:
        key = list(psf_data.keys())[0]
        dx = float(psf_data[key].metadata['dx [mm]'])
        dy = float(psf_data[key].metadata['dy [mm]'])
        x_min = float(psf_data[key].metadata['x_min [mm]'])
        x_max = float(psf_data[key].metadata['x_max [mm]'])
        y_min = float(psf_data[key].metadata['y_min [mm]'])
        y_max = float(psf_data[key].metadata['y_max [mm]'])
        width = int(float(psf_data[key].metadata['width']))
        height = int(float(psf_data[key].metadata['height']))
        datasets = psf_data[key].datasets

        print(psf_data[key].metadata["description"])

        # from the first dataset, extract the image:
        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    # traces = traces[::, ::, times < 1960]
    # times = times[times < 1960]

    # plt.plot(times, traces[width // 2, height // 2, :])
    # plt.plot(times, traces[width // 4, height // 4, :])
    # #plt.plot(times, traces[48, 68, :])
    # #plt.plot(times, traces[42, 60, :])
    # plt.plot(t_ref, p_ref, color=FOREGROUND_COLOR)
    # plt.title("Example Trace Before Surface Extraction")
    # plt.xlabel("Time (ps)")
    # plt.ylabel("Signal (a.u.)")
    # plt.tight_layout()
    # plt.show()

    freqs, refractive_index = get_refraction_index(times, traces, t_ref, p_ref,
                                                   window_half_width=25, win_func="hanning",
                                                   min_frequency=0.5, max_frequency=3,
                                                   d_mm=config["d_mm"],
                                                   mask_radius=config["mask_radius"],
                                                   mask_center_x=config["mask_center_x"],
                                                   mask_center_y=config["mask_center_y"])

    # plt.plot(freqs, refractive_index[width // 2, height // 2, :])
    # plt.tight_layout()
    # plt.show()

    image = np.nanmean(refractive_index, axis=2)
    cutoff = 2.1
    image[image > cutoff] = np.nan
    # image = refractive_index[:, :, np.round(freqs,2) == 0.5]
    norm = Normalize(vmin=config["vmin"], vmax=config["vmax"])
    plt.imshow(image, cmap=cm.managua, norm=norm, aspect="equal", extent=(0, x_max - x_min, 0, y_max - y_min))
    plt.colorbar(label='Refractive Index')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.xticks(range(0,int(x_max - x_min + 1),10))
    plt.yticks(range(0,int(y_max - y_min + 1),10))

    plt.tight_layout()
    # plt.title('Average Refractive Index Map')
    save_figure(RESULTS_DIR / f'pebble_transmission_{material}_inv.png', dpi=300)
    save_figure(RESULTS_DIR / f'pebble_transmission_{material}_inv.pdf')
    plt.show()
