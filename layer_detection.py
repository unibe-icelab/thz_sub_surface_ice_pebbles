from pathlib import Path

from matplotlib.patches import Rectangle
from pydotthz import DotthzFile
import matplotlib.pyplot as plt
import numpy as np
from thz_deconvolution import Deconvolution

from pyhelper import unibe
from surface_extraction import get_thz_file_from_path, extract_surface
from tqdm import tqdm
from envelope import gaussian_kernel1d, convolve1d
from cmcrameri import cm
from surface_extraction import get_thickness
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
              24}

plt.rcParams.update(params)


def percentile_scale(cube, percentile=97.0):
    """
    cube: 3D ndarray (nx, ny, nt) with bipolar time-domain signals.
    Returns:
      scaled_float: same shape, clipped symmetrically to ±p97 and scaled to [-1, 1]
      scaled_u8: same shape, mapped to 8-bit grayscale [0, 255] (white=positive, black=negative)
      p97: the symmetric clip value used
    """
    if not isinstance(cube, np.ndarray) or cube.ndim != 3:
        raise ValueError("cube must be a 3D numpy array (nx, ny, nt)")

    # robust symmetric limit from absolute amplitudes
    p = np.percentile(np.abs(cube), percentile)

    # avoid divide-by-zero if the cube is all zeros
    if p == 0:
        scaled_float = np.zeros_like(cube, dtype=np.float32)
        scaled_u8 = np.full_like(cube, 128, dtype=np.uint8)  # mid-gray
        return scaled_float, scaled_u8, p

    # clip symmetrically and scale to [-1, 1]
    clipped = np.clip(cube, -p, p)
    scaled_float = (clipped / p).astype(np.float32)

    # map [-1, 1] -> [0, 255] (0=black negative, 128=mid-gray zero, 255=white positive)
    scaled_u8 = np.round((scaled_float + 1.0) * 0.5 * 255.0).astype(np.uint8)

    return scaled_float, scaled_u8, p


if __name__ == "__main__":
    # COC solid, low res, 38 deg phase angle
    path, material, tilted = (Path("/Users/linus/Documents/COC_PEBBLE_CoDA_T3_dust_low_res_vac_4/data_image_rfl"),
                              "COC", False)

    # COC solid, low res, 38 deg phase angle, tilted
    # path, material, tilted = (Path("/Users/linus/Documents/COC_PEBBLE_CoDA_T3_dust_low_res_vac_5/data_image_rfl"),
    #                          "COC", True)

    # solid, low res, 38 deg phase angle
    # path, material, tilted = (Path("/Users/linus/Documents/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_5/data_image_rfl"),
    #                           "ICE", False)

    # solid, low res, 38 deg phase angle, tilted
    # path, material, tilted = (
    #     Path("/Users/linus/Documents/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_6_small_fov_slow/data_image_rfl"), "ICE", True)

    path = get_thz_file_from_path(path)

    with DotthzFile(path) as psf_data:
        key = list(psf_data.keys())[0]
        dx = float(psf_data[key].metadata['dx [mm]'])
        dy = float(psf_data[key].metadata['dy [mm]'])
        x_min = float(psf_data[key].metadata['x_min [mm]'])
        x_max = float(psf_data[key].metadata['x_max [mm]'])
        y_min = float(psf_data[key].metadata['y_min [mm]'])
        y_max = float(psf_data[key].metadata['y_max [mm]'])
        rectangle_width = int(float(psf_data[key].metadata['width']))
        rectangle_height = int(float(psf_data[key].metadata['height']))
        datasets = psf_data[key].datasets

        # from the first dataset, extract the image:
        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    traces = traces[::, ::, :]
    surface = extract_surface(times, traces, x_min, x_max, rectangle_width, y_min, y_max, rectangle_height,
                              t_cutoff=None, smooth=True,
                              smooth_median_size=1,
                              smooth_threshold=2.0,
                              smooth_iterations=3,
                              smooth_gaussian_sigma=0.5,
                              smooth_preserve_nan=True)

    # Decon = Deconvolution(psf_path=Path("psf_data/psf.npz"), low_cut=0.1, high_cut=5.0)
    Decon = Deconvolution(
        knife_edge_x_path=Path("psf_data/example_beam_width/measurement_x/data/1750085285.8557956_data.thz"),
        knife_edge_y_path=Path("psf_data/example_beam_width/measurement_y/data/1750163177.929295_data.thz"),
        low_cut=0.1, high_cut=1.0, start_freq=0.1, end_freq=1.0)

    traces_old = traces
    deconvolved_traces = Decon.apply_deconvolution(traces, x_min, x_max, rectangle_width, y_min, y_max,
                                                   rectangle_height, max_iter=100)
    traces = deconvolved_traces

    # p = traces[::, ::, times < 1912]
    # t = times[times < 1912]

    traces = traces[::, ::, :]
    times = times[:]

    w, h, d = traces.shape
    b_scan = traces[:, w // 2, :]
    # b_scan = traces[w // 2, :, :]

    b_scan_convolved = np.zeros_like(b_scan)

    sigma = 10
    kernel = gaussian_kernel1d(sigma=sigma, radius=sigma * 3)

    for x in tqdm(range(traces.shape[0])):
        convolved_data = convolve1d(np.abs(b_scan[x, :]), kernel)
        b_scan_convolved[x] = convolved_data

    gate = 5
    b_scan_convolved[b_scan_convolved > gate] = gate

    spacing_x = dx
    spacing_y = dy

    n_COC = 1.0

    spacing_z = -(get_thickness(times[0], times[1]) * 1000) / n_COC

    image = b_scan_convolved
    image = image.T

    surface_b_scan = surface[:, rectangle_width // 2 + 4]
    print(surface_b_scan)

    fig, ax = plt.subplots(figsize=(12, 8))
    extent = [0, b_scan.shape[0] * spacing_x, b_scan.shape[1] * spacing_z, 0]
    im = ax.imshow(image, extent=extent, aspect=1, norm="log", cmap=cm.vik)

    x = np.linspace(0, b_scan.shape[0] * spacing_x, b_scan.shape[0])
    ax.plot(x, surface_b_scan, color="black", linewidth=4)

    ax.set_xlabel("Width [mm]")
    ax.set_ylabel("OPL (Depth) [mm]")
    ax.set_title("B-Scan")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Intensity [a.u.]")

    # Overlay white rectangles on left and right 10mm
    rim_width = 5  # mm
    height_mm = b_scan.shape[1] * spacing_z
    width_mm = b_scan.shape[0] * spacing_x

    if not tilted:
        # highlight top surface
        rectangle_width = 18
        rectangle_height = 5
        if "ICE" in material:
            cube_offset = 4
            cube_height = 9
        else:
            cube_offset = 0
            cube_height = 8
        top_surface_start = (width_mm / 2 - rectangle_width / 2,
                             height_mm / 2 - cube_offset - rectangle_height / 2 - cube_height)
        ax.add_patch(Rectangle(top_surface_start, rectangle_width, rectangle_height, edgecolor=unibe.RED, linewidth=4,
                               facecolor="none", zorder=2, alpha=1))

        bottom_surface_start = (width_mm / 2 - rectangle_width / 2,
                                height_mm / 2 - cube_offset - rectangle_height / 2 + cube_height)
        ax.add_patch(
            Rectangle(bottom_surface_start, rectangle_width, rectangle_height, edgecolor=unibe.BLUE, linewidth=4,
                      facecolor="none", zorder=2, alpha=1))

    if "ICE" in material and tilted:
        pass
    else:
        # Left rectangle
        ax.add_patch(Rectangle((0, 0), rim_width, height_mm, color='white', zorder=2, alpha=0.7))
        # Right rectangle
        ax.add_patch(
            Rectangle((b_scan.shape[0] * spacing_x - rim_width, 0), rim_width, height_mm, color='white', zorder=2,
                      alpha=0.7))

    if tilted:
        tilted = "_45"
    else:
        tilted = ""
    plt.savefig(f"{material}_B-Scan-Layers{tilted}.png")
    plt.savefig(f"{material}_B-Scan-Layers{tilted}.pdf")

    plt.show()
