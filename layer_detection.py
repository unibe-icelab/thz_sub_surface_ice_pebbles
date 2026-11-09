from pathlib import Path

from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pydotthz import DotthzFile
from thz_deconvolution import Deconvolution
from tqdm import tqdm

from envelope import gaussian_kernel1d, convolve1d
from plot_style import FOREGROUND_COLOR, apply_plot_style, save_figure
from pyhelper import unibe
from surface_extraction import get_thz_file_from_path, extract_surface, get_thickness


ROOT_DIR = Path(__file__).resolve().parent
MEASUREMENTS_DIR = ROOT_DIR / "data" / "layer_detection" / "measurements"
MEASUREMENTS = {
    "coc": ("COC_PEBBLE_CoDA_T3_dust_low_res_vac_4", "COC", False),
    "coc_tilted": ("COC_PEBBLE_CoDA_T3_dust_low_res_vac_5", "COC", True),
    "ice": ("ICE_PEBBLE_CoDA_T3_dust_low_res_vac_5", "ICE", False),
    "ice_tilted": ("ICE_PEBBLE_CoDA_T3_dust_low_res_vac_6_small_fov_slow", "ICE", True),
}

# Change this key to select one of the four locally copied measurements above.
ACTIVE_MEASUREMENT = "coc"
measurement_directory, MATERIAL, TILTED = MEASUREMENTS[ACTIVE_MEASUREMENT]
DATA_DIR = MEASUREMENTS_DIR / measurement_directory / "data_image_rfl"
PSF_DATA_DIR = ROOT_DIR / "data" / "psf" / "example_beam_width"
RESULTS_DIR = ROOT_DIR / "results" / "layer_detection"


params = {
    "font.size": 24,
}
plt.rcParams.update(params)
apply_plot_style()


def percentile_scale(cube, percentile=97.0):
    """
    Scale a 3D time-domain cube symmetrically around zero.

    Returns the clipped float cube, an 8-bit grayscale cube, and the
    percentile amplitude used for clipping.
    """
    if not isinstance(cube, np.ndarray) or cube.ndim != 3:
        raise ValueError("cube must be a 3D numpy array (nx, ny, nt)")

    percentile_amplitude = np.percentile(np.abs(cube), percentile)

    if percentile_amplitude == 0:
        scaled_float = np.zeros_like(cube, dtype=np.float32)
        scaled_u8 = np.full_like(cube, 128, dtype=np.uint8)
        return scaled_float, scaled_u8, percentile_amplitude

    clipped = np.clip(cube, -percentile_amplitude, percentile_amplitude)
    scaled_float = (clipped / percentile_amplitude).astype(np.float32)
    scaled_u8 = np.round((scaled_float + 1.0) * 0.5 * 255.0).astype(np.uint8)
    return scaled_float, scaled_u8, percentile_amplitude


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = get_thz_file_from_path(DATA_DIR)
    material = MATERIAL
    tilted = TILTED

    with DotthzFile(path) as measurement:
        key = list(measurement.keys())[0]
        dx = float(measurement[key].metadata["dx [mm]"])
        dy = float(measurement[key].metadata["dy [mm]"])
        x_min = float(measurement[key].metadata["x_min [mm]"])
        x_max = float(measurement[key].metadata["x_max [mm]"])
        y_min = float(measurement[key].metadata["y_min [mm]"])
        y_max = float(measurement[key].metadata["y_max [mm]"])
        scan_width = int(float(measurement[key].metadata["width"]))
        scan_height = int(float(measurement[key].metadata["height"]))
        datasets = measurement[key].datasets
        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    surface = extract_surface(
        times,
        traces,
        x_min,
        x_max,
        scan_width,
        y_min,
        y_max,
        scan_height,
        t_cutoff=None,
        smooth=True,
        smooth_median_size=1,
        smooth_threshold=2.0,
        smooth_iterations=3,
        smooth_gaussian_sigma=0.5,
        smooth_preserve_nan=True,
    )

    deconvolution = Deconvolution(
        knife_edge_x_path=(PSF_DATA_DIR / "measurement_x" / "data" /
                           "1750085285.8557956_data.thz"),
        knife_edge_y_path=(PSF_DATA_DIR / "measurement_y" / "data" /
                           "1750163177.929295_data.thz"),
        low_cut=0.1,
        high_cut=1.0,
        start_freq=0.1,
        end_freq=1.0,
    )
    traces = deconvolution.apply_deconvolution(
        traces,
        x_min,
        x_max,
        scan_width,
        y_min,
        y_max,
        scan_height,
        max_iter=100,
    )

    width, _, _ = traces.shape
    b_scan = traces[:, width // 2, :]
    b_scan_convolved = np.zeros_like(b_scan)

    sigma = 10
    kernel = gaussian_kernel1d(sigma=sigma, radius=sigma * 3)
    for x_index in tqdm(range(traces.shape[0])):
        b_scan_convolved[x_index] = convolve1d(np.abs(b_scan[x_index, :]), kernel)

    gate = 5
    b_scan_convolved[b_scan_convolved > gate] = gate

    spacing_x = dx
    refractive_index = 1.0
    spacing_z = -(get_thickness(times[0], times[1]) * 1000) / refractive_index

    image = b_scan_convolved.T
    surface_b_scan = surface[:, scan_width // 2 + 4]

    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [0, b_scan.shape[0] * spacing_x, b_scan.shape[1] * spacing_z, 0]
    image_artist = ax.imshow(image, extent=extent, aspect=1, norm="log", cmap=cm.vik)

    x_axis = np.linspace(0, b_scan.shape[0] * spacing_x, b_scan.shape[0])
    ax.plot(x_axis, surface_b_scan, color=FOREGROUND_COLOR, linewidth=4)
    ax.set_xlabel("Width [mm]")
    ax.set_ylabel("OPL (Depth) [mm]")
    ax.set_title("B-Scan")

    divider = make_axes_locatable(ax)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.5)
    colorbar = fig.colorbar(image_artist, cax=colorbar_axis)
    colorbar.set_label("Intensity [a.u.]")

    rim_width = 5
    height_mm = b_scan.shape[1] * spacing_z
    width_mm = b_scan.shape[0] * spacing_x

    if not tilted:
        layer_rectangle_width = 18
        layer_rectangle_height = 5
        if "ICE" in material:
            cube_offset = 4
            cube_height = 9
        else:
            cube_offset = 0
            cube_height = 8

        top_surface_start = (
            width_mm / 2 - layer_rectangle_width / 2,
            height_mm / 2 - cube_offset - layer_rectangle_height / 2 - cube_height,
        )
        ax.add_patch(Rectangle(
            top_surface_start,
            layer_rectangle_width,
            layer_rectangle_height,
            edgecolor=unibe.RED,
            linewidth=4,
            facecolor="none",
            zorder=2,
        ))

        bottom_surface_start = (
            width_mm / 2 - layer_rectangle_width / 2,
            height_mm / 2 - cube_offset - layer_rectangle_height / 2 + cube_height,
        )
        ax.add_patch(Rectangle(
            bottom_surface_start,
            layer_rectangle_width,
            layer_rectangle_height,
            edgecolor=unibe.BLUE,
            linewidth=4,
            facecolor="none",
            zorder=2,
        ))

    if not ("ICE" in material and tilted):
        ax.add_patch(Rectangle(
            (0, 0), rim_width, height_mm,
            color=FOREGROUND_COLOR, zorder=2, alpha=0.7,
        ))
        ax.add_patch(Rectangle(
            (b_scan.shape[0] * spacing_x - rim_width, 0),
            rim_width,
            height_mm,
            color=FOREGROUND_COLOR,
            zorder=2,
            alpha=0.7,
        ))

    tilt_suffix = "_45" if tilted else ""
    output_stem = f"{material}_B-Scan-Layers{tilt_suffix}"
    save_figure(RESULTS_DIR / f"{output_stem}.png", dpi=300)
    save_figure(RESULTS_DIR / f"{output_stem}.pdf")
    plt.show()
