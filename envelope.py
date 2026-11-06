import numpy as np
from pathlib import Path
from pydotthz import DotthzFile
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from plot_style import FOREGROUND_COLOR, save_figure
from capsule_paths import DATA_ROOT, RESULTS_ROOT

DATA_FILE = (DATA_ROOT / "envelope" / "measurements" / "sample_data" /
             "resolution_target_sample.thzimg")
RESULTS_DIR = RESULTS_ROOT / "envelope"

def envelope_hilbert(x: np.ndarray, smooth_sigma: float | None = None) -> np.ndarray:
    """
    Return the analytic envelope of `x` via the Hilbert transform.
    If `smooth_sigma` is given, apply Gaussian smoothing to the envelope.
    """
    analytic = hilbert(x)
    env = np.abs(analytic)
    if smooth_sigma is not None and smooth_sigma > 0:
        env = gaussian_filter1d(env, sigma=smooth_sigma, mode="reflect")
    return env / np.max(env)

# def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
#     size = 2 * radius + 1
#     x = np.arange(size) - radius
#     kernel = np.exp(-x ** 2 / (2 * sigma * sigma))
#     kernel /= kernel.sum()
#     return kernel
#
#
# def convolve1d(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
#     radius = len(kernel) // 2
#     output = np.zeros_like(data)
#     for i in range(len(data)):
#         acc = 0.0
#         for k, coeff in enumerate(kernel):
#             j = i + k - radius
#             if 0 <= j < len(data):
#                 acc += data[j] * coeff
#         output[i] = acc
#     return output

from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve

def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    """
    Fast Gaussian kernel creation using numpy's vectorized operations.
    Drop-in replacement for the original manual implementation.
    """
    size = 2 * radius + 1
    x = np.arange(size) - radius
    kernel = np.exp(-x ** 2 / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel

def convolve1d(
    data: np.ndarray,
    kernel: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """
    Fast convolution using scipy's optimized implementation.
    Set ``normalize=True`` to scale the result to a maximum of one. Analysis
    callers retain absolute amplitudes by default.
    """
    output = convolve(data, kernel, mode='same')
    if normalize:
        maximum = np.max(output)
        if maximum > 0:
            output = output / maximum
    return output



if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with DotthzFile(DATA_FILE) as f:
        data = np.array(f["Image"]["dataset"])
        time = np.array(f["Image"]["time"])
        print(data.shape, time.shape)

        dt = time[1] - time[0]
        pulse_width = 1 # ps
        sigma = int(pulse_width / 3 / dt)
        print(sigma)
        kernel = gaussian_kernel1d(sigma=sigma, radius=sigma*3)
        convolved_data = convolve1d(np.abs(data[30, 30, :]), kernel, normalize=True)
        convolved_data2 = convolve1d(np.abs(data[30, 30, :])**2, kernel, normalize=True)

        fig, ax1 = plt.subplots()
        ax1.plot(time, data[30, 30, :], color=FOREGROUND_COLOR, label="Original Signal")
        ax1.set_ylabel("Original Signal [nA]")
        ax1.set_xlabel("Time [ps]")
        ax1.set_ylim(-1000, 1000)
        ax1.set_xlim(1875, 1920)

        ax2 = ax1.twinx()
        ax2.plot(time, convolved_data , color="red", ls="--", label="Convolved Signal")
        ax2.plot(time, convolved_data2 , color="green", ls="--", label="Convolved Signal")
        ax2.plot(time, envelope_hilbert(data[30, 30, :]), color="pink", ls="--", label="Hilbert")
        ax2.axhline(0.1, color="orange", ls="--", label="Threshold (0.1)")
        ax2.set_ylabel("Convolved Signal (0-1)")
        ax2.set_ylim(-1.1, 1.1)

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        plt.tight_layout()

        save_figure(RESULTS_DIR / "convolution_example.png", bbox_inches="tight")
        save_figure(RESULTS_DIR / "convolution_example.pdf", bbox_inches="tight")
        plt.show()
