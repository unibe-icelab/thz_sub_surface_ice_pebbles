import os
from pathlib import Path

import numpy as np
import scipy
from matplotlib import pyplot as plt
from pydotthz import DotthzFile

from math_utils import get_fft, get_ifft
from scipy.optimize import curve_fit, OptimizeWarning
from thz_deconvolution import Deconvolution

from envelope import convolve1d, gaussian_kernel1d


def get_thz_file_from_path(path: Path):
    for file in os.listdir(path):
        if file.endswith("_data.thz"):
            return path.joinpath(file)


def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset


def get_threshold_from_noise(convolved_data, nbins=500, plot=False, debug=False):
    convolved_data = convolved_data[convolved_data < 1.0]
    # Compute histogram
    counts, bin_edges = np.histogram(convolved_data, bins=nbins, density=True)

    # Compute bin centers (midpoints)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Initial guesses (important for good fitting)
    p0 = [np.max(counts), bin_edges[np.where(counts==np.max(counts))[0][0]], 0.05, 0]
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=p0, maxfev=2000)
        amp, mu, sigma, offset = popt

    except (RuntimeError, OptimizeWarning):
        print("Error - curve_fit failed")
        mu = np.mean(convolved_data)
        sigma = np.std(convolved_data)
        popt = p0
        if debug:

            # Fit a normal distribution to the data
            mu, std = scipy.stats.norm.fit(convolved_data)

            # Plot the fitted Gaussian curve
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = scipy.stats.norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
        else:
            return None

    threshold = mu + 6 * sigma

    if plot:
        plt.hist(convolved_data, bins=nbins, density=True, alpha=0.6, color='g')
        x = np.linspace(0, 1, 1000)
        plt.plot(x, gaussian(x, *popt), 'r-', linewidth=2)

        plt.axvline(threshold, color='red', ls='--')

        plt.title(f"Gaussian fit: μ={mu:.2f}, σ={sigma:.2f}")
        plt.show()

    return threshold


from sklearn.mixture import GaussianMixture

def fit_gmm_and_get_noise_threshold(data, n_components=3, std_multiplier=5, plot=False):
    data = np.asarray(data)
    data = data[np.isfinite(data) & (data > 0)]  # only positive, finite values
    data_reshaped = data.reshape(-1, 1)  # GMM expects 2D input

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(data_reshaped)

    # Extract parameters
    means = gmm.means_.flatten()
    stds = np.sqrt(np.array([np.diag(cov) for cov in gmm.covariances_])).flatten()
    weights = gmm.weights_

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
        counts, bins, _ = plt.hist(data, bins=100, density=True, alpha=0.5, color='green')

        # Plot GMM components
        x = np.linspace(0, np.max(data), 1000).reshape(-1, 1)
        for i in range(n_components):
            mean = means[i]
            std = stds[i]
            weight = weights[i]
            y = weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x.flatten() - mean) ** 2 / (2 * std ** 2))
            plt.plot(x, y, label=f'Component {i+1}: μ={mean:.2f}, σ={std:.2f}')

        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
        plt.title('Gaussian Mixture Model Fit')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    return threshold, (means, stds, weights)

def fit_best_gmm(data, max_components=6, criterion='bic', std_multiplier=5, plot=True):
    """
    Fit Gaussian Mixture Model to multimodal data and pick best number of components based on BIC/AIC.

    Args:
        data: 1D array-like
        max_components: max number of Gaussian components to try
        criterion: 'bic' or 'aic'
        std_multiplier: threshold = mu + N * sigma (default 5)
        plot: show histogram and fits

    Returns:
        threshold: float
        model: fitted GMM object
        best_n: number of components used
        sorted_params: (means, stds, weights) sorted by mean
    """
    data = np.asarray(data)
    data = data[np.isfinite(data) & (data > 0)]  # remove invalid
    X = data.reshape(-1, 1)

    lowest_score = np.inf
    best_gmm = None
    best_n = None

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(X)
        score = gmm.bic(X) if criterion == 'bic' else gmm.aic(X)

        if score < lowest_score:
            lowest_score = score
            best_gmm = gmm
            best_n = n

    # Extract parameters from best model
    means = best_gmm.means_.flatten()
    stds = np.sqrt([np.diag(cov)[0] for cov in best_gmm.covariances_])
    weights = best_gmm.weights_

    # Sort by mean (lowest = noise peak)
    sorted_idx = np.argsort(means)
    means = means[sorted_idx]
    stds = stds[sorted_idx]
    weights = weights[sorted_idx]

    # Threshold from first (leftmost) Gaussian
    mu_noise = means[0]
    sigma_noise = stds[0]
    threshold = mu_noise + std_multiplier * sigma_noise

    if plot:
        # Plot histogram
        counts, bins, _ = plt.hist(data, bins=100, density=True, alpha=0.5, color='green')

        # Plot GMM components
        x = np.linspace(0, np.max(data), 1000).reshape(-1, 1)
        for i in range(best_n):
            mean = means[i]
            std = stds[i]
            weight = weights[i]
            y = weight * (1 / (std * np.sqrt(2 * np.pi))) * \
                np.exp(-(x.flatten() - mean) ** 2 / (2 * std ** 2))
            plt.plot(x, y, label=f'μ={mean:.2f}, σ={std:.2f}, w={weight:.2f}')

        plt.axvline(threshold, color='red', ls='--', label=f'Threshold = {threshold:.2f}')
        plt.title(f"GMM fit (selected {best_n} components using {criterion.upper()})")
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    return threshold, best_gmm, best_n, (means, stds, weights)

if __name__ == "__main__":
    path = Path("/Users/linus/Documents/COC_PEBBLE_CoDA_T1a_2/data_image_rfl")
    # path = Path("/Users/linus/Documents/COC_SPHERES_fine_scan_3/data_image_rfl")

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
    x = 30
    y = 30
    time, pulse = t, p[x, y, :]

    kernel = gaussian_kernel1d(sigma=6.0, radius=18)

    convolved_data = convolve1d(np.abs(pulse), kernel)

    # threshold = get_threshold_from_noise(convolved_data, nbins=500, plot=True, debug=True)
    threshold, components = fit_gmm_and_get_noise_threshold(convolved_data, n_components=2, plot=True)
    # threshold, gmm_model, n_components, params = fit_best_gmm(
    #     convolved_data, max_components=6, criterion='bic', std_multiplier=5, plot=True
    # )

    plt.plot(time, pulse)
    plt.plot(time, convolved_data)
    if threshold:
        plt.axhline(threshold, color='red', ls='--')
    plt.show()
