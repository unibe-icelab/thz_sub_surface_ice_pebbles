from pathlib import Path
from pydotthz import DotthzFile
from scipy.ndimage import gaussian_filter1d
import pyvista as pv
import os
import scipy.constants
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_opening, generate_binary_structure
import numpy as np
from thz_deconvolution import Deconvolution

from surface_extraction import get_thz_file_from_path
from tqdm import tqdm
from envelope import gaussian_kernel1d, convolve1d
from cmcrameri import cm
from surface_extraction import get_thickness


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
    path = Path("/Users/linus/Documents/COC_PEBBLE_CoDA_T3_dust_low_res_vac_4/data_image_rfl")

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

        # from the first dataset, extract the image:
        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    traces = traces[::, ::, :]

    # Decon = Deconvolution(psf_path=Path("psf_data/psf.npz"), low_cut=0.1, high_cut=5.0)
    Decon = Deconvolution(
        knife_edge_x_path=Path("psf_data/example_beam_width/measurement_x/data/1750085285.8557956_data.thz"),
        knife_edge_y_path=Path("psf_data/example_beam_width/measurement_y/data/1750163177.929295_data.thz"),
        low_cut=0.1, high_cut=0.3, start_freq=0.1, end_freq=0.3)

    traces_old = traces
    deconvolved_traces = Decon.apply_deconvolution(traces, x_min, x_max, width, y_min, y_max, height, max_iter=100)
    traces = deconvolved_traces

    # p = traces[::, ::, times < 1912]
    # t = times[times < 1912]

    traces = traces[::, ::, :]
    times = times[:]

    w, h, d = traces.shape
    b_scan = traces[:, w // 2, :]

    b_scan_convolved = np.zeros_like(b_scan)

    sigma = 10
    kernel = gaussian_kernel1d(sigma=sigma, radius=sigma * 3)

    for x in tqdm(range(traces.shape[0])):
        convolved_data = convolve1d(np.abs(b_scan[x, :]), kernel)
        b_scan_convolved[x] = convolved_data

    spacing_x = dx
    spacing_y = dy

    n_COC = 1.5

    spacing_z = (get_thickness(times[0], times[1]) * 1000) / n_COC

    iamge = b_scan_convolved
    image = iamge.T

    # Plot using imshow with the calculated extent
    extent = [0, b_scan.shape[0] * spacing_x, b_scan.shape[1] * spacing_z,0]
    plt.imshow(image, extent=extent, aspect='1',norm="log", cmap=cm.berlin)
    plt.xlabel("Width [mm]")
    plt.ylabel("Depth [mm]")
    plt.title("B-Scan")
    plt.colorbar(label="Intensity [a.u.]")
    plt.show()