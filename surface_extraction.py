from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pydotthz import DotthzFile
from math_utils import get_fft, get_ifft
from scipy.signal import find_peaks
import scipy.constants
from tqdm import tqdm

from thz_deconvolution import Deconvolution
from envelope import gaussian_kernel1d, convolve1d, envelope_hilbert


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


if __name__ == '__main__':

    path = Path("/Users/linus/Documents/COC_terrain/data_image_rfl/1758834352.9015865_data.thz")
    # path = Path("/Users/linus/Documents/COC_SPHERES_fine_scan/data_image_rfl/1759959272.6226816_data.thz")
    # path = Path("/Users/linus/Documents/COC_SPHERES_fine_scan_3/data_image_rfl/1760820898.0132043_data.thz")
    # path = Path("/Users/linus/Documents/COC_FULL_SPHERES_fine_scan/data_image_rfl/1760023882.972197_data.thz")
    # path = Path("/Users/linus/Documents/resolution_scan_A_high_res_6/data_image_rfl/1760606720.3549016_data.thz")
    # path = Path("/Users/linus/Documents/COC_SPHERES_ext_rng/data_image_rfl/1744835132.148543_data.thz")
    path = Path("/Users/linus/Documents/COC_terrain_res_target_2/data_image_rfl/1759921401.9838753_data.thz")
    path = Path("/Users/linus/Documents/DUST_surface_res_target_3d_3/data_image_rfl/1760633536.387254_data.thz")

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

    #p = traces[::, ::, :]
    #t = times[:]

    # p = p[p.shape[0]//2-10:p.shape[0]//2+10, ::2, :]
    # x_min = -5
    # x_max = 5
    surface = np.zeros(p.shape[:2])
    kernel = gaussian_kernel1d(sigma=6.0, radius=18)
    for x in tqdm(range(p.shape[0])):
        for y in range(p.shape[1]):

            # f, a, arg = get_fft(t, p[x, y, :])
            #
            # a[f < 1.0] = 0
            # a[f > 5.0] = 0
            #
            # time, pulse = get_ifft(f, a, arg, t0=t[0])

            time, pulse = t, p[x, y, :]

            # t1 = detect_first_zero_crossing_after_peak(time, pulse, height=1.2)

            convolved_data_h = envelope_hilbert(pulse)
            convolved_data = convolve1d(np.abs(pulse), kernel)

            noise_mean = np.mean(np.abs(convolved_data[:60]))
            noise_std = np.std(np.abs(convolved_data[:60]))

            threshold = noise_mean + 15.0 * noise_std
            threshold = 0.4 + convolved_data[0]

            t1 = detect_first_peak(time, convolved_data, height=threshold)
            t1h = detect_first_peak(time, convolved_data_h, height=threshold)

            # threshold = 0.08
            #
            # if x == 0 and y == 0:
            #     plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
            #              convolved_data )
            if x == 21 and y == 138:
                plt.title(f"Pixel ({x}, {y})")
                plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
                         pulse/np.max(pulse))
                plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
                         convolved_data ,ls="--")
                plt.plot(1000 * (time - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12,
                         convolved_data_h,ls="--")
                plt.axhline(threshold, color='blue', label='threshold')
                plt.axhline(noise_mean, color='red', label='noise mean')
                plt.axvline(1000 * (time[60] - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12, color='black',label='noise gate')
                plt.axvline(1000 * (t1 - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12, color='orange',label='t1')
                plt.axvline(1000 * (t1h - time[0]) * scipy.constants.c / np.cos(np.radians(19)) / 2 * 1e-12, color='green',ls="--",label="Hilbert peak")
                plt.legend()
                plt.show()

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

    # # Initialize arrays to store multiple peak candidates
    # max_candidates = 2
    # surface_candidates = np.zeros((p.shape[0], p.shape[1], max_candidates))
    #
    # for x in range(p.shape[0]):
    #     for y in range(p.shape[1]):
    #         f, a, arg = get_fft(t, p[x, y, :])
    #         a[f < 0.1] = 0
    #         time, pulse = get_ifft(f, a, arg, t0=t[0])
    #
    #         # Detect first two peaks with their zero crossings
    #         peak_results = detect_first_two_peaks_with_zero_crossings(time, pulse, height=2)
    #
    #         # Store candidates (use zero crossing times)
    #         for i, (peak_time, zero_crossing_time) in enumerate(peak_results):
    #             if i < max_candidates:
    #                 surface_candidates[x, y, i] = zero_crossing_time
    #
    # # Apply neighborhood-based smoothing to select best peaks
    # surface_raw = smooth_peak_selection(surface_candidates, neighborhood_size=3, max_deviation_factor=1)
    #
    # # Convert to thickness
    # surface = np.zeros_like(surface_raw)
    # for x in range(surface_raw.shape[0]):
    #     for y in range(surface_raw.shape[1]):
    #         t1 = surface_raw[x, y]
    #         if t1 == 0:
    #             surface[x, y] = 0
    #         else:
    #             surface[x, y] = get_thickness(t1, 0) * 1000  # in mm

    save_surface_as_point_cloud(surface, x_max, x_min, y_max, y_min,
                                "extracted_surface_dust_res_target_pointcloud.ply", format='ply')

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
