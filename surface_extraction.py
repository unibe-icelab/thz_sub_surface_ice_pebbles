import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from pydotthz import DotthzFile
from scipy.signal import find_peaks
import scipy.constants
from tqdm import tqdm
from scipy.ndimage import median_filter, gaussian_filter

from thz_deconvolution import Deconvolution
from envelope import gaussian_kernel1d, convolve1d, envelope_hilbert
from estimate_noise import NoiseEstimator

ROOT_DIR = Path(__file__).resolve().parent
PSF_DATA_DIR = ROOT_DIR / "data" / "psf"
MEASUREMENTS_DIR = ROOT_DIR / "data" / "surface_extraction" / "measurements"
MEASUREMENTS = {
    "coc_spheres_fine_scan_3": "COC_SPHERES_fine_scan_3",
}

ACTIVE_MEASUREMENT = "coc_spheres_fine_scan_3"
EXAMPLE_DATA_DIR = MEASUREMENTS_DIR / MEASUREMENTS[ACTIVE_MEASUREMENT] / "data_image_rfl"
EXAMPLE_RESULTS_DIR = ROOT_DIR / "results" / "surface_extraction"


def get_dynamic_range(data, percentile=0.999):
    """
    Calculate dynamic range for image data based on a given percentile.
    """
    abs_data = np.abs(data)
    max_val = np.percentile(abs_data, percentile * 100)
    min_val = np.percentile(abs_data, (1 - percentile) * 100)
    return max_val - min_val
    return np.max(abs_data) - np.min(abs_data)


def detect_first_two_peaks_with_zero_crossings(t, signal, height=2, distance=None):
    """
    Detect first two peaks and their zero crossings

    Returns:
        list of tuples: [(peak_time1, zero_crossing_time1), (peak_time2, zero_crossing_time2)]
    """
    if distance is None:
        distance = len(signal) // 50  # Minimum distance between peaks

    peaks, _ = find_peaks(signal, height=height, distance=distance)

    if len(peaks) == 0:
        return []

    # Take first two peaks
    first_peaks = peaks[:2]
    results = []

    for peak_idx in first_peaks:
        peak_time = t[peak_idx]

        # Find zero crossing after this peak
        zero_crossing_time = None
        for i in range(peak_idx, len(signal) - 1):
            if signal[i] == 0:
                zero_crossing_time = t[i]
                break
            elif signal[i] * signal[i + 1] < 0:
                t0, t1 = t[i], t[i + 1]
                y0, y1 = signal[i], signal[i + 1]
                zero_crossing_time = t0 - y0 * (t1 - t0) / (y1 - y0)
                break

        if zero_crossing_time is not None:
            results.append((peak_time, zero_crossing_time))

    return results


def smooth_peak_selection(surface_candidates, neighborhood_size=3, max_deviation_factor=2.0):
    """
    Select the most consistent peak based on neighborhood analysis

    Args:
        surface_candidates: 3D array [x, y, num_candidates] with peak times
        neighborhood_size: size of neighborhood window (default: 3x3)
        max_deviation_factor: maximum allowed deviation from neighborhood median

    Returns:
        2D array with selected peak times
    """
    nx, ny, num_candidates = surface_candidates.shape
    selected_surface = np.zeros((nx, ny))

    half_window = neighborhood_size // 2

    for x in range(nx):
        for y in range(ny):
            # Get valid candidates for this pixel
            candidates = surface_candidates[x, y, :]
            valid_candidates = candidates[candidates > 0]  # Remove zeros/invalid

            if len(valid_candidates) == 0:
                selected_surface[x, y] = 0
                continue
            elif len(valid_candidates) == 1:
                selected_surface[x, y] = valid_candidates[0]
                continue

            # Define neighborhood bounds
            x_start = max(0, x - half_window)
            x_end = min(nx, x + half_window + 1)
            y_start = max(0, y - half_window)
            y_end = min(ny, y + half_window + 1)

            # Collect all valid peaks from neighborhood
            neighborhood_peaks = []
            for nx_i in range(x_start, x_end):
                for ny_i in range(y_start, y_end):
                    neighbor_candidates = surface_candidates[nx_i, ny_i, :]
                    valid_neighbor_candidates = neighbor_candidates[neighbor_candidates > 0]
                    neighborhood_peaks.extend(valid_neighbor_candidates)

            if len(neighborhood_peaks) == 0:
                selected_surface[x, y] = valid_candidates[0]
                continue

            # Calculate neighborhood statistics
            neighborhood_median = np.median(neighborhood_peaks)
            neighborhood_std = np.std(neighborhood_peaks)

            # Find candidate closest to neighborhood median
            deviations = np.abs(valid_candidates - neighborhood_median)
            best_candidate_idx = np.argmin(deviations)
            best_candidate = valid_candidates[best_candidate_idx]

            # Check if best candidate is within acceptable deviation
            if deviations[best_candidate_idx] <= max_deviation_factor * neighborhood_std:
                selected_surface[x, y] = best_candidate
            else:
                # Fallback to first candidate if none are consistent with neighborhood
                selected_surface[x, y] = valid_candidates[0]

    return selected_surface


def detect_first_peak_with_zero_crossing(t, signal, snr_threshold: float = 3.0, min_prominence=None,
                                         noise_method='mad'):
    """
    Detect first peak with adaptive threshold based on SNR and return zero crossing

    Args:
        t: time array
        signal: signal array
        snr_threshold: minimum SNR for peak detection (default: 3.0)
        min_prominence: minimum prominence relative to noise (optional)
        noise_method: method to estimate noise level (default: 'mad', median absolute deviation)

    Returns:
        tuple: (peak_time, zero_crossing_time, noise_std, height_threshold)
    """
    # Estimate noise level
    noise_std = estimate_noise_level(signal, method=noise_method)

    # Set height threshold based on SNR
    height_threshold = snr_threshold * noise_std

    # Set prominence threshold if not provided
    if min_prominence is None:
        min_prominence = 2 * noise_std

    # Find peaks with adaptive thresholds
    peaks, properties = find_peaks(
        signal,
        height=height_threshold,
        prominence=min_prominence,
        distance=len(signal) // 20  # Minimum distance between peaks
    )

    if len(peaks) == 0:
        return None, None, noise_std, height_threshold

    first_peak_index = peaks[0]
    peak_time = t[first_peak_index]

    # Find zero crossing after the peak
    zero_crossing_time = None
    for i in range(first_peak_index, len(signal) - 1):
        if signal[i] == 0:
            zero_crossing_time = t[i]
            break
        elif signal[i] * signal[i + 1] < 0:
            # Linear interpolation for better accuracy
            t0, t1 = t[i], t[i + 1]
            y0, y1 = signal[i], signal[i + 1]
            zero_crossing_time = t0 - y0 * (t1 - t0) / (y1 - y0)
            break

    return peak_time, zero_crossing_time, noise_std, height_threshold


def estimate_noise_level(signal, method='mad', noise_fraction=0.1):
    """
    Estimate noise level using different methods

    Args:
        signal: signal array
        method: 'mad' (median absolute deviation), 'std_baseline', or 'std_full'
        noise_fraction: fraction of signal start to use for baseline method

    Returns:
        noise_std: estimated noise standard deviation
    """
    if method == 'mad':
        # Median Absolute Deviation - robust to outliers/peaks
        median_val = np.median(signal)
        mad = np.median(np.abs(signal - median_val))
        # Convert MAD to equivalent standard deviation (assuming normal distribution)
        noise_std = mad  # * 1.4826

    elif method == 'std_baseline':
        # Original method - std of early signal portion
        noise_samples = int(len(signal) * noise_fraction)
        noise_std = np.std(signal[:noise_samples])

    elif method == 'std_full':
        # Standard deviation of full signal
        noise_std = np.std(signal)

    else:
        raise ValueError("Method must be 'mad', 'std_baseline', or 'std_full'")

    return noise_std


def detect_first_zero_crossing_after_peak(t, signal, height=5):
    # Step 1: Find the first positive peak
    peaks, _ = find_peaks(signal, height=height)
    if len(peaks) == 0:
        return None  # No peaks found

    first_peak_index = peaks[0]

    # Step 2: Search for the first zero crossing after the peak
    for i in range(first_peak_index, len(signal) - 1):
        if signal[i] == 0:
            return t[i]
        elif signal[i] * signal[i + 1] < 0:
            # Linear interpolation for better accuracy
            t0, t1 = t[i], t[i + 1]
            y0, y1 = signal[i], signal[i + 1]
            zero_crossing_time = t0 - y0 * (t1 - t0) / (y1 - y0)
            return zero_crossing_time

    return None  # No zero crossing found


def detect_first_peak(t, signal, height=5):
    # Step 1: Find the first positive peak
    peaks, _ = find_peaks(signal, height=height)
    if len(peaks) == 0:
        return None  # No peaks found

    first_peak_index = peaks[0]

    return t[first_peak_index]


def get_thickness(t1, t2, angle=19):
    l = 6 * 25.4 / 1000
    delta_t = np.abs(t2 - t1) * 1e-12
    alpha = np.radians(angle)
    # kosinussatz
    h = (2 * scipy.constants.c * l * delta_t - scipy.constants.c ** 2 * delta_t ** 2) / (
            2 * (l * np.cos(2 * alpha) - scipy.constants.c
                 * delta_t + l))
    thickness = h * np.cos(alpha)

    # but the actual shortest path is
    thickness = delta_t * scipy.constants.c / np.cos(np.radians(angle)) / 2
    return thickness


def get_time_diff(d, angle=19):
    delta_t = d * 1e12 / scipy.constants.c * 2 * np.cos(np.radians(angle))
    return delta_t


def save_surface_as_point_cloud(surface, x_max, x_min, y_max, y_min, filename, format='txt'):
    """
    Save 2D surface data as a point cloud file.

    Args:
        surface: 2D numpy array with height values
        x_max, x_min, y_max, y_min: Physical dimensions in mm
        filename: Output filename
        format: 'txt', 'csv', 'xyz', or 'ply'
    """
    nx, ny = surface.shape

    surface -= np.min(surface)

    # Create coordinate arrays (matching your plotting code)
    x_mm = np.linspace(y_min, y_max, ny)
    y_mm = np.linspace(-x_min, -x_max, nx)
    X, Y = np.meshgrid(x_mm, y_mm, indexing='xy')

    # Flatten arrays to create point cloud
    x_points = X.flatten()
    y_points = Y.flatten()
    z_points = surface.flatten()

    # Remove any NaN or invalid points
    valid_mask = np.isfinite(z_points)
    x_points = x_points[valid_mask]
    y_points = y_points[valid_mask]
    z_points = z_points[valid_mask]

    print(f"Saving {len(x_points)} points to {filename}")
    print(f"Point cloud bounds: X[{x_points.min():.2f}, {x_points.max():.2f}], "
          f"Y[{y_points.min():.2f}, {y_points.max():.2f}], "
          f"Z[{z_points.min():.2f}, {z_points.max():.2f}]")

    if format.lower() == 'txt':
        # Save as space-separated text file
        np.savetxt(filename, np.column_stack([x_points, y_points, z_points]),
                   fmt='%.6f', delimiter=' ', header='X Y Z', comments='# ')

    elif format.lower() == 'csv':
        # Save as comma-separated values
        np.savetxt(filename, np.column_stack([x_points, y_points, z_points]),
                   fmt='%.6f', delimiter=',', header='X,Y,Z', comments='')

    elif format.lower() == 'xyz':
        # Save as XYZ format (space-separated, no header)
        np.savetxt(filename, np.column_stack([x_points, y_points, z_points]),
                   fmt='%.6f', delimiter=' ')

    elif format.lower() == 'ply':
        # Save as PLY format
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(x_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            for i in range(len(x_points)):
                f.write(f"{x_points[i]:.6f} {y_points[i]:.6f} {z_points[i]:.6f}\n")

    else:
        raise ValueError("Format must be 'txt', 'csv', 'xyz', or 'ply'")

    print(f"Point cloud saved successfully!")

    return x_points, y_points, z_points


def get_thz_file_from_path(path: Path):
    for file in os.listdir(path):
        if file.endswith("_data.thz"):
            return path.joinpath(file)


def _create_surface_hash(times, traces, x_min, x_max, width, y_min, y_max, height, t_cutoff, smooth, smooth_median_size,
                         smooth_threshold, smooth_iterations, smooth_gaussian_sigma, smooth_preserve_nan):
    scan_hash = hashlib.md5(traces.tobytes()).hexdigest()
    times_hash = hashlib.md5(times.tobytes()).hexdigest()
    config = (scan_hash, times_hash, x_min, x_max, width, y_min, y_max, height, t_cutoff, smooth, smooth_median_size,
              smooth_threshold, smooth_iterations, smooth_gaussian_sigma, smooth_preserve_nan)
    config_str = str(config)
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_surface_cache_path(config_hash):
    temp_dir = Path(tempfile.gettempdir()) / "surface_extraction_cache"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir / f"surface_{config_hash}.pkl"


def despike_and_smooth(surface,
                       median_size=3,
                       threshold=5.0,
                       iterations=2,
                       gaussian_sigma=1.0,
                       preserve_nan=True):
    """
    Despike and smooth a 2D surface.

    Parameters:
    - surface: 2D numpy array
    - median_size: window size for median filter (odd integer)
    - threshold: multiplier of local MAD to classify a point as spike
    - iterations: number of despike iterations
    - gaussian_sigma: sigma for final Gaussian smoothing (0 = no smoothing)
    - preserve_nan: keep NaNs in the output at the same positions

    Returns:
    - cleaned: new 2D numpy array (doesn't modify input)
    """
    s = np.array(surface, dtype=float, copy=True)

    # preserve NaN positions
    if preserve_nan:
        nan_mask = ~np.isfinite(s)
        # temporarily fill NaNs with global median for filtering
        if np.all(nan_mask):
            return s  # all NaNs, return as is
        global_median = np.nanmedian(s)
        s[nan_mask] = global_median
    else:
        nan_mask = None

    for _ in range(max(1, int(iterations))):
        med = median_filter(s, size=median_size, mode='reflect')
        diff = s - med

        # robust scale estimate (MAD). fallback to std if MAD==0
        abs_dev = np.abs(diff - np.median(diff))
        mad = np.median(abs_dev)
        scale = mad if mad > 0 else np.std(diff) if np.std(diff) > 0 else 1.0

        outliers = np.abs(diff) > (threshold * scale)
        if not np.any(outliers):
            break
        s[outliers] = med[outliers]

    if gaussian_sigma and gaussian_sigma > 0:
        s = gaussian_filter(s, sigma=gaussian_sigma, mode='reflect')

    if preserve_nan:
        s[nan_mask] = np.nan

    return s


def extract_surface(times, traces, x_min, x_max, width, y_min, y_max, height, path: Optional[Path] = None,
                    t_cutoff=1912,
                    low_cut=1.0,
                    high_cut=5.0,
                    start_freq=1.0,
                    end_freq=5.0,
                    win_width=0.5,
                    n_filters=20,
                    w_max=30,
                    smooth=False, smooth_median_size=1,
                    smooth_threshold=5.0,
                    smooth_iterations=2,
                    smooth_gaussian_sigma=1.0,
                    smooth_preserve_nan=True):
    config_hash = _create_surface_hash(times, traces, x_min, x_max, width, y_min, y_max, height, t_cutoff, smooth,
                                       smooth_median_size, smooth_threshold, smooth_iterations, smooth_gaussian_sigma,
                                       smooth_preserve_nan)
    cache_path = _get_surface_cache_path(config_hash)

    # Try loading from cache
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                print(f"Loading cached surface (hash: {config_hash[:8]}...)")
                return pickle.load(f)
        except Exception:
            cache_path.unlink(missing_ok=True)

    traces = traces[::, ::, :]
    Decon = Deconvolution(
        knife_edge_x_path=(PSF_DATA_DIR / "example_beam_width" / "measurement_x" / "data" /
                           "1750085285.8557956_data.thz"),
        knife_edge_y_path=(PSF_DATA_DIR / "example_beam_width" / "measurement_y" / "data" /
                           "1750163177.929295_data.thz"),
        low_cut=low_cut,
        high_cut=high_cut,
        start_freq=start_freq,
        end_freq=end_freq,
        win_width=win_width,
        n_filters=n_filters,
        w_max=w_max
    )

    deconvolved_traces = Decon.apply_deconvolution(traces, x_min, x_max, width, y_min, y_max, height, max_iter=100)
    traces = deconvolved_traces

    if t_cutoff:
        p = traces[::, ::, times < t_cutoff]
        t = times[times < t_cutoff]
    else:
        p = traces[::, ::, :]
        t = times[:]

    surface = np.zeros(p.shape[:2])
    kernel = gaussian_kernel1d(sigma=6.0, radius=18)
    Gmm = NoiseEstimator(n_components=2)

    for x in tqdm(range(p.shape[0])):
        for y in range(p.shape[1]):
            time, pulse = t, p[x, y, :]
            convolved_data_h = envelope_hilbert(pulse)
            convolved_data = convolve1d(np.abs(pulse), kernel)
            noise_mean = np.mean(np.abs(convolved_data[:60]))
            noise_std = np.std(np.abs(convolved_data[:60]))
            smooth_threshold, components = Gmm.fit_gmm_and_get_noise_threshold(convolved_data, plot=False)
            t1 = detect_first_peak(time, convolved_data, height=smooth_threshold)
            if t1 is None:
                t1 = t[-1]
            surface[x, y] = - get_thickness(t[0], t1) * 1000  # in mm

    if smooth:
        surface = despike_and_smooth(surface,
                                     median_size=smooth_median_size,
                                     threshold=smooth_threshold,
                                     iterations=smooth_iterations,
                                     gaussian_sigma=smooth_gaussian_sigma,
                                     preserve_nan=smooth_preserve_nan)
    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(surface, f)
    print(f"Surface cached (hash: {config_hash[:8]}...)")

    if path is not None:
        save_surface_as_point_cloud(surface, x_max, x_min, y_max, y_min,
                                    path.with_suffix(".ply"), format='ply')

    return surface


if __name__ == '__main__':
    EXAMPLE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = get_thz_file_from_path(EXAMPLE_DATA_DIR)

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

    # p = traces[::, ::, :]
    # t = times[:]

    # p = p[p.shape[0]//2-10:p.shape[0]//2+10, ::2, :]
    # x_min = -5
    # x_max = 5
    surface = np.zeros(p.shape[:2])
    kernel = gaussian_kernel1d(sigma=6.0, radius=18)

    Gmm = NoiseEstimator(n_components=2)

    for x in tqdm(range(p.shape[0])):
        for y in range(p.shape[1]):

            time, pulse = t, p[x, y, :]

            # t1 = detect_first_zero_crossing_after_peak(time, pulse, height=1.2)

            convolved_data_h = envelope_hilbert(pulse)
            convolved_data = convolve1d(np.abs(pulse), kernel)

            noise_mean = np.mean(np.abs(convolved_data[:60]))
            noise_std = np.std(np.abs(convolved_data[:60]))

            threshold = noise_mean + 10.0 * noise_std
            # threshold = get_threshold_from_noise(convolved_data)
            threshold, components = Gmm.fit_gmm_and_get_noise_threshold(convolved_data, plot=False)

            # threshold = 0.2 + convolved_data[0]

            t1 = detect_first_peak(time, convolved_data, height=threshold)
            t1h = detect_first_peak(time, convolved_data_h, height=threshold)

            # threshold = 0.08
            #
            # if x == 0 and y == 0:
            #     plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #              convolved_data )
            # if x == 21 and y == 22:
            #     plt.title(f"Pixel ({x}, {y})")
            #     plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #              pulse)
            #     plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #              convolved_data, ls="--")
            #     # plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #     #          convolved_data_h, ls="--")
            #     plt.axhline(threshold, color='blue', label='threshold')
            #     plt.axhline(noise_mean, color='red', label='noise mean')
            #     plt.axvline(1000 * (time[60] - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #                 color='black', label='noise gate')
            #     plt.axvline(1000 * (t1 - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #                 color='orange', label='t1')
            #     # plt.axvline(1000 * (t1h - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #     #             color='green', ls="--", label="Hilbert peak")
            #     plt.legend()
            #     plt.show()

            if t1 is None:
                t1 = t[-1]
                # plt.title(f"Pixel ({x}, {y})")
                # plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
                #          pulse / np.max(pulse))
                # plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
                #          convolved_data, ls="--")
                # plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
                #          convolved_data_h, ls="--")
                # plt.axhline(threshold, color='k')
                # plt.axhline(noise_mean, color='red')
                # plt.axvline(1000 * (time[60] - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
                #             color='black')
                #
                # plt.show()
                # if y > 0:
                #     surface[x, y] = surface[x, y - 1]
                # else:
                #     surface[x, y] = surface[x - 1, y]
                surface[x, y] = - get_thickness(t[0], t1) * 1000  # in mm

            else:
                surface[x, y] = - get_thickness(t[0], t1) * 1000  # in mm

    save_surface_as_point_cloud(surface, x_max, x_min, y_max, y_min,
                                EXAMPLE_RESULTS_DIR / f"{path.stem}.ply", format='ply')

    # Plot the surface as a 3D mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define physical dimensions
    nx, ny = surface.shape

    # flip because of orientation
    x_mm = np.linspace(0, y_max - y_min, ny)
    y_mm = np.linspace(0, x_max - x_min, nx)
    X, Y = np.meshgrid(x_mm, y_mm, indexing='xy')

    print(X.shape, surface.shape)

    ax.plot_surface(X, Y, surface, rcount=nx, ccount=ny, cmap='jet')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    x_range = x_mm.max() - x_mm.min()
    y_range = y_mm.max() - y_mm.min()
    z_range = surface.max() - surface.min()

    ax.set_box_aspect([x_range, y_range, z_range])  # Match data aspect

    plt.show()
