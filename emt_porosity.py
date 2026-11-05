# python
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
              "Helvetica Neue",

          "font.size":
              18}

plt.rcParams.update(params)


def bruggeman_vi(epsilon_eff, epsilon_i):
    epsilon_h = 1.0  # vacuum

    # (1 - v_i) * LHS = -v_i * RHS
    # LHS - v_i * LHS = -v_i * RHS
    # LHS = v_i * (LHS - RHS)
    LHS = (epsilon_h - epsilon_eff) / (epsilon_h + 2 * epsilon_eff)
    RHS = (epsilon_i - epsilon_eff) / (epsilon_i + 2 * epsilon_eff)
    v_i = LHS / (LHS - RHS)
    return v_i


def get_refraction_index(time, traces, t_ref, p_ref, window_half_width=15, win_func="hanning",
                         min_frequency=0.2, max_frequency=3, d_mm=1.0, mask_radius=None):
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

                mask_radius_mm = mask_radius * 1 / 0.5

                if ((x - nx // 2) ** 2 + (y - ny // 2) ** 2) > mask_radius_mm ** 2:
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

    return freq, refractive_index, complex_refractive_index


def extract_complex_refractive_index_image(path: Path):
    ref_path = Path("/Users/linus/Documents/comet_dust_pellet_HESSO_silicon_ref_vac_3/data_image")

    ref_path = get_thz_file_from_path(ref_path)

    with DotthzFile(ref_path) as psf_data:
        key = list(psf_data.keys())[0]
        datasets = psf_data[key].datasets

        # from the first dataset, extract the image:
        t_ref = np.array(datasets["time"])
        traces_ref = np.array(datasets["dataset"])

    p_ref = np.mean(traces_ref, axis=(0, 1))

    path = get_thz_file_from_path(path)

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

    freqs, refractive_index, complex_refractive_index = get_refraction_index(times, traces, t_ref, p_ref,
                                                                             window_half_width=25, win_func="hanning",
                                                                             min_frequency=0.5, max_frequency=3,
                                                                             d_mm=12.0,
                                                                             mask_radius=5)

    return freqs, refractive_index, complex_refractive_index, x_max, x_min, y_min, y_max


if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

    # SPIPA-B ICE, low res, 38 deg phase angle
    material, path = ("SPIPA-B", Path("/Users/linus/Documents/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_4/data_image"))
    freqs, refractive_index_spipa, complex_refractive_index_spipa, x_max, x_min, y_min, y_max = extract_complex_refractive_index_image(
        path)

    image = np.nanmean(refractive_index_spipa, axis=2)
    cutoff = 2.1
    image[image > cutoff] = np.nan
    norm = Normalize(vmin=1, vmax=2)
    im0 = axes[0].imshow(image, cmap=cm.managua, norm=norm, aspect="equal",
                         extent=(0, x_max - x_min, 0, y_max - y_min))
    axes[0].set_xlabel('X [mm]')
    axes[0].set_ylabel('Y [mm]')
    axes[0].set_xticks(range(0, int(x_max - x_min + 1), 10))
    axes[0].set_yticks(range(0, int(y_max - y_min + 1), 10))

    # solid ICE, low res, 38 deg phase angle
    material, path = ("solid", Path("/Users/linus/Documents/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_trans_5/data_image"))
    freqs, refractive_index_solid, complex_refractive_index_solid, x_max, x_min, y_min, y_max = extract_complex_refractive_index_image(
        path)

    image = np.nanmean(refractive_index_solid, axis=2)
    cutoff = 2.1
    image[image > cutoff] = np.nan
    norm = Normalize(vmin=1, vmax=2)
    im1 = axes[1].imshow(image, cmap=cm.managua, norm=norm, aspect="equal",
                         extent=(0, x_max - x_min, 0, y_max - y_min))

    # reserve space on the right for a single colorbar for both subplots
    plt.tight_layout(rect=(0, 0, 0.88, 1))

    # create one colorbar to the right for both subplots
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.05, pad=0.03)
    cbar.set_label('Refractive Index')

    axes[1].set_xlabel('X [mm]')
    axes[1].set_ylabel('Y [mm]')
    axes[1].set_xticks(range(0, int(x_max - x_min + 1), 10))
    axes[1].set_yticks(range(0, int(y_max - y_min + 1), 10))

    plt.show()

    neff = np.nanmean(complex_refractive_index_spipa, axis=(0, 1))
    epsilon_eff = neff ** 2

    n_ice = np.nanmean(complex_refractive_index_solid, axis=(0, 1))
    epsilon_ice = n_ice ** 2

    v_i = bruggeman_vi(epsilon_eff, epsilon_ice)

    print(v_i.real)

    plt.plot(freqs, v_i.real)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Porosity (Volume Fraction)")
    plt.title("Porosity of SPIPA-B Ice Pebble Sample")
    plt.tight_layout()
    plt.savefig("emt_spipa_b_porosity.png")
    plt.show()
