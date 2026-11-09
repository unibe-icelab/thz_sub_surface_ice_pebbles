import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pydotthz import DotthzFile

from thz_deconvolution import Deconvolution

from envelope import convolve1d, gaussian_kernel1d
from sklearn.mixture import GaussianMixture
from pyhelper import unibe
from plot_style import FOREGROUND_COLOR, apply_plot_style, save_figure

ROOT_DIR = Path(__file__).resolve().parent
MEASUREMENTS_DIR = ROOT_DIR / "data" / "pebbles" / "measurements"
MEASUREMENTS = {
    "coc_spheres_fine_scan_3": MEASUREMENTS_DIR / "COC_SPHERES_fine_scan_3",
    "dust_surface_resolution_target_3d_4": (
        MEASUREMENTS_DIR / "DUST_surface_res_target_3d_4"
    ),
}

ACTIVE_MEASUREMENT = "dust_surface_resolution_target_3d_4"
DATA_DIR = MEASUREMENTS[ACTIVE_MEASUREMENT] / "data_image_rfl"
PSF_DATA_DIR = ROOT_DIR / "data" / "pebbles" / "psf"
RESULTS_DIR = ROOT_DIR / "results" / "estimate_noise"

params = {"ytick.color":
              "black",

          "xtick.color":
              "black",

          "axes.labelcolor":
              "black",

          "axes.edgecolor":
              "black",

          "font.size":
              16}

plt.rcParams.update(params)
apply_plot_style()


def get_thz_file_from_path(path: Path):
    for file in os.listdir(path):
        if file.endswith("_data.thz"):
            return path.joinpath(file)


class NoiseEstimator:

    def __init__(self, n_components=3, ):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, verbose=False)

    def fit_lnmm_and_get_noise_threshold(
            self, data, time=None, pulse=None,
            std_multiplier=3, plot=False, combined_plot=False
    ):
        data = np.asarray(data)
        convolved_data = data
        data = data[np.isfinite(data) & (data > 0)]

        # Log-transform
        log_data = np.log(data).reshape(-1, 1)
        self.gmm.fit(log_data)

        # Extract log-space Gaussian parameters
        log_means = self.gmm.means_.flatten()
        log_stds = np.sqrt(np.array([np.diag(cov) for cov in self.gmm.covariances_])).flatten()
        weights = self.gmm.weights_

        # Sort by mean (first = noise)
        idx = np.argsort(log_means)
        log_means, log_stds, weights = log_means[idx], log_stds[idx], weights[idx]

        mu_n, sigma_n = log_means[0], log_stds[0]
        log_thresh = mu_n + std_multiplier * sigma_n
        # threshold = np.exp(log_thresh)
        # threshold = (np.exp(log_means[0]) + np.exp(log_means[1])) / 2
        # threshold = np.exp(log_means[1] + log_stds[1])
        threshold = np.exp(log_means[1] + 0.5 * log_stds[1] ** 2)

        # Optional: Simple plotting
        if plot:
            x = np.linspace(0, np.max(data), 1000)
            log_x = np.log(x + 1e-12)

            plt.hist(data, bins=60, density=True, alpha=0.25, edgecolor=FOREGROUND_COLOR, label="Data")

            for i in range(self.n_components):
                mu, sigma, w = log_means[i], log_stds[i], weights[i]
                y = w * (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(log_x - mu) ** 2 / (2 * sigma ** 2))
                plt.plot(x, y, '--' if i > 0 else '-', label=f'LN Comp. {i + 1}')

            plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
            plt.xlabel("Signal Amplitude")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Combined plot view: pulse + histogram + LNMM model
        if combined_plot and time is not None and pulse is not None and convolved_data is not None:

            mu_1 = np.exp(log_means[1])

            fig, axes = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1.1]}, sharey=True)

            # Time-domain signal view
            axes[0].plot(time, pulse, color=FOREGROUND_COLOR, alpha=0.25, label="Pulse (band-pass)")
            axes[0].plot(time, convolved_data, color=unibe.BLUE, label="Envelope")
            axes[0].axhline(threshold, color=unibe.RED, ls='--', label='Threshold')
            axes[0].set_xlabel("Time [ps]")
            axes[0].set_ylabel("Amplitude")
            axes[0].legend()

            # Histogram view
            counts, bins, _ = axes[1].hist(
                data, orientation="horizontal", bins=50,
                density=True, alpha=0.2, edgecolor=FOREGROUND_COLOR
            )

            # axes[1].set_xlim(axes[1].get_xlim()[::-1])

            x = np.linspace(0, np.max(data), 1000)
            log_x = np.log(x + 1e-12)

            for i in range(self.n_components):
                mu, sigma, w = log_means[i], log_stds[i], weights[i]
                y = w * (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(log_x - mu) ** 2 / (2 * sigma ** 2))
                axes[1].plot(y, x, '--' if i > 0 else '-', color=unibe.BLUE, label=f'LN Comp. {i + 1}')

            axes[1].axhline(threshold, color=unibe.RED, linestyle='--', label="Threshold")
            axes[1].set_xlabel("Density")
            axes[1].legend()

            plt.tight_layout()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            save_figure(RESULTS_DIR / 'noise_LNM_fit.pdf')
            save_figure(RESULTS_DIR / 'noise_LNM_fit.png')
            plt.show()

        return threshold, (log_means, log_stds, weights)

    def fit_gmm_and_get_noise_threshold(self, data, std_multiplier=1, dynamic_range_offset=0.1, plot=False,
                                        combined_plot=False):
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

        threshold += dynamic_range_offset * (np.max(data) - np.min(data))

        # threshold = (means[0] + means[1]) / 2

        if plot:
            # Plot histogram
            counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.2,
                                       color=FOREGROUND_COLOR, edgecolor=FOREGROUND_COLOR)

            # Plot GMM components
            x = np.linspace(0, np.max(data), 1000).reshape(-1, 1)
            for i in range(self.n_components):
                mean = means[i]
                std = stds[i]
                weight = weights[i]
                y = weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x.flatten() - mean) ** 2 / (2 * std ** 2))
                if i == 0:
                    plt.plot(x, y, color=unibe.BLUE, ls="-", label=f'Primary Gaussian')
                    plt.axvline(x=mean, color=unibe.ORANGE, linestyle='--', label=r"$\mu_1$")
                else:
                    plt.plot(x, y, color=unibe.BLUE, ls="--")

            plt.axvline(threshold, color=unibe.RED, linestyle='--', label=r'$I_\text{th}$')
            plt.xlabel('Signal [nA]')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            save_figure(RESULTS_DIR / 'noise_GMM_fit.pdf')
            save_figure(RESULTS_DIR / 'noise_GMM_fit.png')
            plt.show()

        if combined_plot:

            fig, axes = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]}, sharey=True)

            axes[0].plot(time, pulse, color=FOREGROUND_COLOR, alpha=0.25, label="Pulse, Band-Pass")
            axes[0].plot(time, convolved_data, color=unibe.BLUE, label="Gaussian Convolution")
            axes[0].axhline(threshold, color=unibe.RED, ls='--', label=r'Threshold $I_\text{th}$')

            # Plot histogram
            counts, bins, _ = axes[1].hist(data, orientation="horizontal", bins=50, density=True, alpha=0.2,
                                           color=FOREGROUND_COLOR, edgecolor=FOREGROUND_COLOR)
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
                else:
                    axes[1].plot(y, x, color=unibe.BLUE, ls="--", label=f'Secondary')

            for i in range(self.n_components):
                mean = means[i]
                std = stds[i]
                weight = weights[i]
                y = weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x.flatten() - mean) ** 2 / (2 * std ** 2))
                if i == 0:
                    axes[1].axhline(threshold, color=unibe.RED, linestyle='--', label=r'$I_\text{th}$')
                    axes[1].axhline(mean, color=unibe.ORANGE, linestyle='--', label=r"$\mu_1$")
                    axes[0].axhline(mean, color=unibe.ORANGE, linestyle='--', label=r"$\mu_1$")

            axes[0].legend()
            axes[0].set_xlabel("Time [ps]")
            axes[0].set_ylabel("Signal [nA]")

            axes[1].set_xlabel('Density')
            axes[1].legend()

            plt.tight_layout()
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            save_figure(RESULTS_DIR / 'combined_noise_GMM_fit.pdf')
            save_figure(RESULTS_DIR / 'combined_noise_GMM_fit.png')
            plt.show()

        return threshold, (means, stds, weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate noise in a THz image measurement.")
    parser.add_argument("path", nargs="?", type=Path, default=DATA_DIR,
                        help="Measurement data_image_rfl directory.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

        # from the first dataset, extract the image:
        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    traces = traces[::, ::, :]

    Decon = Deconvolution(
        knife_edge_x_path=(PSF_DATA_DIR / "example_beam_width" / "measurement_x" / "data" /
                           "1750085285.8557956_data.thz"),
        knife_edge_y_path=(PSF_DATA_DIR / "example_beam_width" / "measurement_y" / "data" /
                           "1750163177.929295_data.thz"),
        low_cut=1.0, high_cut=5.0)

    traces_old = traces
    deconvolved_traces = Decon.apply_deconvolution(traces, x_min, x_max, width, y_min, y_max, height, max_iter=100)
    traces = deconvolved_traces

    p = traces[::, ::, times < 1912]
    t = times[times < 1912]

    p = traces[::, ::, :]
    t = times[:]
    x = 50
    y = 66
    time, pulse = t, p[x, y, :]

    kernel = gaussian_kernel1d(sigma=6.0, radius=18)

    convolved_data = convolve1d(np.abs(pulse), kernel)

    Gmm = NoiseEstimator(n_components=2)
    threshold, components = Gmm.fit_lnmm_and_get_noise_threshold(convolved_data, time=time, pulse=pulse,
                                                                 std_multiplier=1, combined_plot=True)

    # plt.plot(time, pulse, color=FOREGROUND_COLOR, alpha=0.25, label="Pulse, Band-Pass")
    # plt.plot(time, convolved_data, color=unibe.BLUE, label="Gaussian Convolution")
    # plt.axhline(threshold, color=unibe.RED, ls='--', label=f'Threshold $I_\text{th}$')
    #
    # plt.legend()
    # plt.xlabel("Time [ps]")
    # plt.ylabel("Signal [nA]")
    # plt.tight_layout()
    # save_figure(RESULTS_DIR / 'noise_threshold.pdf')
    # save_figure(RESULTS_DIR / 'noise_threshold.png')
    # plt.show()
