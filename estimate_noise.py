import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pydotthz import DotthzFile

from thz_deconvolution import Deconvolution

from envelope import convolve1d, gaussian_kernel1d
from sklearn.mixture import GaussianMixture
from pyhelper import unibe

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
              16}

plt.rcParams.update(params)


def get_thz_file_from_path(path: Path):
    for file in os.listdir(path):
        if file.endswith("_data.thz"):
            return path.joinpath(file)


class NoiseEstimator:

    def __init__(self, n_components=3, ):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, verbose=False)

    def fit_gmm_and_get_noise_threshold(self, data, std_multiplier=5, plot=False, combined_plot=False):
        data = np.asarray(data)
        data = data[np.isfinite(data) & (data > 0)]  # only positive, finite values
        data_reshaped = data.reshape(-1, 1)  # GMM expects 2D input

        # Fit GMM
        self.gmm.fit(data_reshaped)

        # Extract parameters
        means = self.gmm.means_.flatten()
        stds = np.sqrt(np.array([np.diag(cov) for cov in self.gmm.covariances_])).flatten()
        weights = self.gmm.weights_

        # Sort components by mean
        sorted_indices = np.argsort(means)
        means = means[sorted_indices]
        stds = stds[sorted_indices]
        weights = weights[sorted_indices]

        # Assume first component is noise
        mu_noise = means[0]
        sigma_noise = stds[0]
        threshold = mu_noise + std_multiplier * sigma_noise

        if plot:
            # Plot histogram
            counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.2, color='black', edgecolor='black')

            # Plot GMM components
            x = np.linspace(0, np.max(data), 1000).reshape(-1, 1)
            for i in range(self.n_components):
                mean = means[i]
                std = stds[i]
                weight = weights[i]
                y = weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x.flatten() - mean) ** 2 / (2 * std ** 2))
                if i == 0:
                    plt.plot(x, y, color=unibe.BLUE, ls="-", label=f'Primary Gaussian')
                    plt.axvline(x=mean, color=unibe.ORANGE, linestyle='--', label="Noise Level $L$")
                else:
                    plt.plot(x, y, color=unibe.BLUE, ls="--")

            plt.axvline(threshold, color=unibe.RED, linestyle='--', label=f'Threshold $I_\text{th}$')
            plt.xlabel('Signal [nA]')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig('noise_GMM_fit.pdf')
            plt.savefig('noise_GMM_fit.png')
            plt.show()

        if combined_plot:

            fig, axes = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]}, sharey=True)

            axes[0].plot(time, pulse, color="black", alpha=0.25, label="Pulse, Band-Pass")
            axes[0].plot(time, convolved_data, color=unibe.BLUE, label="Gaussian Convolution")
            axes[0].axhline(threshold, color=unibe.RED, ls='--', label=r'Threshold $I_\text{th}$')

            # Plot histogram
            counts, bins, _ = axes[1].hist(data, orientation="horizontal", bins=50, density=True, alpha=0.2,
                                           color='black', edgecolor='black')
            axes[1].set_xlim(axes[1].get_xlim()[::-1])

            # Plot GMM components
            x = np.linspace(0, np.max(data), 1000).reshape(-1, 1)
            for i in range(self.n_components):
                mean = means[i]
                std = stds[i]
                weight = weights[i]
                y = weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x.flatten() - mean) ** 2 / (2 * std ** 2))
                if i == 0:
                    axes[1].plot(y, x, color=unibe.BLUE, ls="-", label=f'Primary')
                    axes[1].axhline(threshold, color=unibe.RED, linestyle='--', label=r'$I_\text{th}$')
                    axes[1].axhline(mean, color=unibe.ORANGE, linestyle='--', label="$L$")
                    axes[0].axhline(mean, color=unibe.ORANGE, linestyle='--', label="Noise Level $L$")
                else:
                    axes[1].plot(y, x, color=unibe.BLUE, ls="--")


            axes[0].legend()
            axes[0].set_xlabel("Time [ps]")
            axes[0].set_ylabel("Signal [nA]")

            axes[1].set_xlabel('Density')
            axes[1].legend()

            plt.tight_layout()
            plt.savefig('combined_noise_GMM_fit.pdf')
            plt.savefig('combined_noise_GMM_fit.png')
            plt.show()

        return threshold, (means, stds, weights)


if __name__ == "__main__":
    path = Path("/Users/linus/Documents/COC_PEBBLE_CoDA_T1a_2/data_image_rfl")
    # path = Path("/Users/linus/Documents/COC_SPHERES_fine_scan_3/data_image_rfl")
    path = Path("/Users/linus/Documents/DUST_surface_res_target_3d_4/data_image_rfl")

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
        low_cut=1.0, high_cut=5.0)

    traces_old = traces
    deconvolved_traces = Decon.apply_deconvolution(traces, x_min, x_max, width, y_min, y_max, height, max_iter=100)
    traces = deconvolved_traces

    p = traces[::, ::, times < 1912]
    t = times[times < 1912]
    x = 60
    y = 66
    time, pulse = t, p[x, y, :]

    kernel = gaussian_kernel1d(sigma=6.0, radius=18)

    convolved_data = convolve1d(np.abs(pulse), kernel)

    Gmm = NoiseEstimator(n_components=3)
    threshold, components = Gmm.fit_gmm_and_get_noise_threshold(convolved_data, std_multiplier=6, combined_plot=True)

    # plt.plot(time, pulse, color="black", alpha=0.25, label="Pulse, Band-Pass")
    # plt.plot(time, convolved_data, color=unibe.BLUE, label="Gaussian Convolution")
    # plt.axhline(threshold, color=unibe.RED, ls='--', label=f'Threshold $I_\text{th}$')
    #
    # plt.legend()
    # plt.xlabel("Time [ps]")
    # plt.ylabel("Signal [nA]")
    # plt.tight_layout()
    # plt.savefig('noise_threshold.pdf')
    # plt.savefig('noise_threshold.png')
    # plt.show()
