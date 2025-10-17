from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pydotthz import DotthzFile
from math_utils import get_fft, get_ifft
from scipy.signal import find_peaks
import scipy.constants

from thz_deconvolution import Deconvolution
from envelope import gaussian_kernel1d, convolve1d, envelope_hilbert

if __name__ == '__main__':

    path1 = Path("/Users/linus/Documents/COC_terrain/data_image_rfl/1758834352.9015865_data.thz")
    path2 = Path("/Users/linus/Documents/COC_SPHERES_fine_scan/data_image_rfl/1759959272.6226816_data.thz")
    path3 = Path("/Users/linus/Documents/COC_FULL_SPHERES_fine_scan/data_image_rfl/1760023882.972197_data.thz")
    path4 = Path("/Users/linus/Documents/resolution_scan_A_high_res_6/data_image_rfl/1760606720.3549016_data.thz")
    path5 = Path("/Users/linus/Documents/COC_terrain_res_target_2/data_image_rfl/1759921401.9838753_data.thz")

    paths = [path1, path2, path3, path4, path5]

    fig, axs = plt.subplots(nrows=3, figsize=[12, 9])

    for path in paths:
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

        pulse = traces[width // 2, height // 2, :]
        name = path.parent.parent.name
        distance = 1000 * (times - times[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12

        # axs[0].plot(distance, pulse / np.max(pulse),ls="--", label=f"{name} raw")

        f, a, arg = get_fft(times, pulse )

        a[f < 1.0] = 0
        a[f > 5.0] = 0

        times, pulse = get_ifft(f, a, arg, t0=times[0])
        distance = 1000 * (times - times[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12

        kernel = gaussian_kernel1d(sigma=6.0, radius=12)
        convolved_data = convolve1d(pulse ** 2, kernel)

        axs[0].plot(distance, convolved_data / np.max(convolved_data), label=f"{name}")
        axs[0].plot(distance, envelope_hilbert(pulse),ls="--", label=f"{name}")

        axs[1].hist(convolved_data[convolved_data < 20], bins=100, label=f"{name}", alpha=0.5)

        # axs[1].set_xlim(-0.1,0.1)

        axs[0].legend()

    plt.show()
