import argparse
import json
import os
from pathlib import Path
from typing import Optional

import scipy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import Polygon, Point
from thzpy.timedomain import common_window
from thzpy.transferfunctions import uniform_slab
from pydotthz import DotthzFile
import numpy as np

from emt import bruggeman_eps_eff, solve_v_i_two_component
from plot_style import FOREGROUND_COLOR, material_color, save_figure
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "spectroscopy"
RAW_DATA_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "metadata" / "calculated_data.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "material_properties" / "mixture_analysis"


def save_result(filename: str) -> None:
    """Save the active figure below the local material-analysis results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(RESULTS_DIR / filename)


def extract_rois(path: Path, measurement_key: Optional[str] = None, ax: Optional[Axes] = None):
    """
    Extracts Regions of Interest (ROIs) from a DotthzFile and returns the pixel coordinates inside each ROI.

    Parameters:
    path (Path): Path to the DotthzFile containing the image data.
    measurement_key (Optional[str]): The key for selecting a specific measurement. Defaults to the first measurement.
    ax (Optional[Axes]): Matplotlib Axes object for optional visualization. If provided, overlays ROI outlines on the image.

    Returns:
    dict: A dictionary mapping ROI labels to lists of pixel coordinates within each ROI.
    """
    rois = {}
    with DotthzFile(path, "r") as image_file:
        # Read the measurement
        if measurement_key is None:
            # Just get the first one
            measurement_key = list(image_file.keys())[0]
        datasets = image_file[measurement_key].datasets
        metadata = image_file[measurement_key].metadata

        # Extract image data
        image = np.array(datasets["dataset"])
        image_to_plot = np.sum(image ** 2, axis=2)  # Aggregate along 3rd dimension

        # Get image dimensions
        height, width = int(float(metadata["height"])), int(float(metadata["width"]))

        if ax:
            # Plot the image
            ax.imshow(image_to_plot, cmap="gray")

            ax.legend()

            # Extract relevant metadata values
            # Extract relevant metadata values
            x_min = float(metadata['x_min [mm]'])
            x_max = float(metadata['x_max [mm]'])
            y_min = float(metadata['y_min [mm]'])
            y_max = float(metadata['y_max [mm]'])
            width = int(float(metadata['width']))  # Assumes width is given as a string
            height = int(float(metadata['height']))  # Assumes height is given as a string

            # Number of ticks we want (including both start and end)
            num_ticks_x = (width // 11) + 1  # Adjusted to include the first and last tick
            num_ticks_y = (height // 11) + 1  # Adjusted to include the first and last tick

            # Create evenly spaced tick values for x and y axes in mm using linspace
            x_ticks = np.linspace(x_min, x_max, num_ticks_x, endpoint=True)
            y_ticks = np.linspace(y_min, y_max, num_ticks_y, endpoint=True)

            # Calculate the corresponding pixel positions for the ticks
            x_tick_positions = np.linspace(0, width - 1, num_ticks_x, dtype=int)
            y_tick_positions = np.linspace(0, height - 1, num_ticks_y, dtype=int)

            # Set the x and y ticks
            ax.set_xticks(x_tick_positions)  # Set ticks at the pixel positions
            ax.set_xticklabels(np.round(x_ticks, 2))  # Convert pixel to mm and label

            ax.set_yticks(y_tick_positions)  # Set ticks at the pixel positions
            ax.set_yticklabels(np.round(y_ticks, 2))  # Convert pixel to mm and label
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        # Extract and parse ROI polygon from metadata
        for (index, roi_label) in enumerate(metadata["ROI Labels"].split(",")):
            roi_raw = metadata[f"ROI {index}"]  # Assuming ROI is stored as a string like "[[x1,y1],[x2,y2],...]"
            roi_points = eval(roi_raw) if isinstance(roi_raw, str) else roi_raw  # Convert to list of lists
            roi_points = list(roi_points)  # Ensure it's a list

            # Convert to image coordinate system (flip Y and swap x and y)
            roi_points_corrected = [(width - 1 - y, x) for x, y in roi_points]
            polygon = Polygon(roi_points_corrected)

            # Find all pixels inside the ROI
            pixels_inside_roi = [
                (x, y) for y in range(height) for x in range(width) if polygon.contains(Point(x, y))
            ]
            rois[roi_label] = pixels_inside_roi

            if ax:
                # Overlay ROI polygon (convert to list before concatenation)
                roi_y, roi_x = zip(*(roi_points_corrected + [roi_points_corrected[0]]))  # Close the polygon
                ax.plot(roi_x, roi_y, linewidth=2, label=roi_label)  # Red outline

    if len(rois) == 0:
        raise Exception(f"No ROIs found in {path}")

    return rois  # Returns list of pixel coordinates


def read_thz_file(path: Path, roi=None, single_pixel=False):
    print(path)

    if not path.is_file():
        for file in os.listdir(path):
            if not file.endswith(".thz"):
                continue
            path = path.joinpath(file)
            break

    roi_pixels = None
    if roi:
        rois = extract_rois(path)
        roi_pixels = rois[roi]

    with DotthzFile(path, "r") as image_file:
        if single_pixel:
            t = np.array(image_file["Single Pixel 0"]["Sample"][:, 0])
            p = np.array(image_file["Single Pixel 0"]["Sample"][:, 1])
            try:
                t_ref = np.array(image_file["Single Pixel 0"]["Ref"][:, 0])
                p_ref = np.array(image_file["Single Pixel 0"]["Ref"][:, 1])
                return t, p, t_ref, p_ref
            except:
                pass
            return t, p, None, None

        key = list(image_file.keys())[0]

        # read the first group/measurement
        datasets = image_file[key].datasets
        # from the first dataset, extract the image:

        times = np.array(datasets["time"])
        traces = np.array(datasets["dataset"])

    if roi_pixels:
        print(f"Extracting ROI '{roi}' with {len(roi_pixels)} pixels")
        roi_values = np.array([traces[x, y, :] for x, y in roi_pixels])
        return times, roi_values

    return times, traces


def emt(n_m, n_eff, m_i, m_h, rho_i, rho_h, vol_frac=None, method="bruggeman"):
    epsilon_h = n_m ** 2
    epsilon_eff = n_eff ** 2

    v_i = m_i / rho_i / (m_i / rho_i + m_h / rho_h)
    v_h = m_h / rho_h / (m_i / rho_i + m_h / rho_h)

    if vol_frac:
        print(f"v_i = {v_i}, vol_frac = {vol_frac}")
        v_i = vol_frac
        v_h = 1 - v_i

    # if method == "bruggeman":
    #     epsilon_incl = epsilon_h * ((1 + v_h) * epsilon_eff + v_i * epsilon_h) / (
    #             v_i * epsilon_h - (1 + v_h) * epsilon_eff)
    # elif method == "maxwell-garnett":
    #     epsilon_incl = epsilon_h * ((2 + v_i) * epsilon_h + (1 - v_i) * epsilon_eff) / (
    #             epsilon_h * (1 - v_i) + (2 + v_i) * epsilon_eff)
    if method == "bruggeman":
        denom = epsilon_eff * (v_h - 2 * v_i) - epsilon_h * (v_h + v_i)
        epsilon_incl = -epsilon_eff * (2 * epsilon_eff * (v_h + v_i) + epsilon_h * (v_i - 2 * v_h)) / denom
    elif method == "maxwell-garnett":
        denom = 2 * epsilon_h * v_i + epsilon_h + epsilon_eff * (v_i - 1)
        epsilon_incl = epsilon_h * (2 * epsilon_h * (v_i - 1) + epsilon_eff * (v_i + 2)) / denom
    elif method == "LLL":
        epsilon_incl = (1 / v_i * (epsilon_eff ** (1 / 3) - v_h * epsilon_h ** (1 / 3))) ** 3
    else:
        raise ValueError(f"Unknown EMT method: {method}")

    return np.sqrt(epsilon_incl)


reference_path = RAW_DATA_DIR / "comet_dust_pellet_HESSO_silicon_ref_vac_3"

# path and thickness in mm
files = [
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_PTFE_1.8_vac_3", "PTFE", material_color("PTFE"), 1.78, 2.2),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_PTFE_2.1_vac_3", "PTFE", material_color("PTFE"), 2.12, 2.2),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_SiO2_1.8_vac_3", "SiO2", material_color("SiO2"), 1.8, 2.65),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_SiO2_2.1_vac_3", "SiO2", material_color("SiO2"), 2.13, 2.65),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_Charcoal_1.8_vac_3", "Charcoal", material_color("Charcoal"), 1.85, 2.19),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_Charcoal_2.1_vac_3", "Charcoal", material_color("Charcoal"), 2.13, 2.19),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_CoDA-T1_1.8_vac_3", "CoDA-T1", material_color("CoDA-T1"), 1.8, 2.19),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_CoDA-T1_2.1_vac_3", "CoDA-T1", material_color("CoDA-T1"), 2.06, 2.19),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_CoDA-T2_1.8_vac_3", "CoDA-T2", material_color("CoDA-T2"), 1.8, 2.19),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_CoDA-T2_2.1_vac_3", "CoDA-T2", material_color("CoDA-T2"), 2.08, 2.19),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_CI_1.8_vac_3", "CI", material_color("CI"), 1.8, 2.45),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_CI_2.1_vac_3", "CI", material_color("CI"), 2.1, 2.45),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_Forsterite_1.8_vac_inv_3", "Forsterite", material_color("Forsterite"), 1.8, 3.27),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_Forsterite_2.1_vac_3", "Forsterite", material_color("Forsterite"), 2.1, 3.27),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_Fayalite_1.8_vac_3", "Fayalite", material_color("Fayalite"), 1.76, 4.39),
    (RAW_DATA_DIR / "comet_dust_pellet_HESSO_Fayalite_2.1_vac_3", "Fayalite", material_color("Fayalite"), 2.1, 4.39),
]

avg_in_fourier_domain = False
half_width = 20
win_func = "blackman"
min_frequency = 0.2  # THz
max_frequency = 4.0  # THz

cutoff = 1960
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare HESSO dust pellets with EMT mixtures.")
    parser.add_argument("path", nargs="?", type=Path, default=DATA_DIR,
                        help="Material-properties data directory containing metadata/ and raw/.")
    args = parser.parse_args()

    DATA_DIR = args.path
    RAW_DATA_DIR = DATA_DIR / "raw"
    METADATA_PATH = DATA_DIR / "metadata" / "calculated_data.json"
    reference_path = RAW_DATA_DIR / reference_path.name
    files = [
        (RAW_DATA_DIR / Path(file_path).name, name, color, thickness, density)
        for file_path, name, color, thickness, density in files
    ]

    t_ref, data_ref = read_thz_file(reference_path / "data_image", roi="ROI 1")

    data_ref = data_ref[:, t_ref < cutoff]
    t_ref = t_ref[t_ref < cutoff]

    data_ref = np.mean(data_ref, axis=0)
    data_ref -= data_ref[0]

    data_ref = np.array([t_ref, data_ref])

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 13))

    dataset = {}
    dataset_ref = {}

    with METADATA_PATH.open('r') as f:
        meta_data_pellets = json.load(f)

    rho_ptfe = meta_data_pellets["PTFE"]["density [g/cm3]"]

    r = 14.85 / 2  # mm
    for file, name, color, d_0, rho in files:
        d = d_0

        if d <= 2.0:
            ls = "--"
            rho = meta_data_pellets[f"{name}_1.8"]["density [g/cm3]"]
            d = meta_data_pellets[f"{name}_1.8"]["average_thickness [mm]"]
            sample_vol = meta_data_pellets[f"{name}_1.8"]["sample_volume [cm3]"]
        else:
            ls = "-"
            rho = meta_data_pellets[f"{name}_2.1"]["density [g/cm3]"]
            d = meta_data_pellets[f"{name}_2.1"]["average_thickness [mm]"]
            sample_vol = meta_data_pellets[f"{name}_2.1"]["sample_volume [cm3]"]

        full_volume = np.pi * r ** 2 * d / 1000  # in cm3

        try:
            t_sample, data_sample = read_thz_file(Path(file).joinpath("data_image"), roi="ROI 1")
        except FileNotFoundError:
            continue

        data_sample = data_sample[:, t_sample < cutoff]
        t_sample = t_sample[t_sample < cutoff]

        if "PTFE" not in name:
            if d < 2.0:
                data_ref = dataset_ref["1.8"]
            else:
                data_ref = dataset_ref["2.1"]
        else:
            n_ptfe = 1  # will be overwritten for the next runs

        if avg_in_fourier_domain:

            p = np.array([t_sample, data_sample[0]])

            sample, reference = common_window([p, data_ref],
                                              half_width=half_width, win_func=win_func)

            # Calculate buffer material properties.
            buffer = uniform_slab(d,
                                  sample, reference, n_med=n_ptfe,
                                  upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                  all_optical_constants=True)

            refractive_index = np.zeros_like(buffer["refractive_index"])
            absorption_coefficient = np.zeros_like(buffer["absorption_coefficient"])
            complex_n = np.zeros_like(buffer["refractive_index"])
            n = 0

            for pixel in range(data_sample.shape[0]):
                p = data_sample[pixel, :]

                # axes[2].plot(t_sample, p, label=f"{name}, {d:.2f} mm", color=color, ls=ls)

                p -= p[0]

                p = np.array([t_sample, p])

                # Apply a window function to all datasets with a half-width of 15 ps.
                sample, reference = common_window([p, data_ref],
                                                  half_width=half_width, win_func=win_func)

                # Calculate buffer material properties.
                buffer = uniform_slab(d,
                                      sample, reference, n_med=n_ptfe,
                                      upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                      all_optical_constants=True)
                refractive_index += np.real(buffer["refractive_index"])
                absorption_coefficient += np.real(buffer["absorption_coefficient"])
                complex_n += buffer["refractive_index"]
                n += 1

            absorption_coefficient /= n
            refractive_index /= n
            complex_n /= n
        else:
            p = np.mean(data_sample, axis=0)

            p -= p[0]

            # axes[2].plot(t_sample, p, label=f"{name}, {d:.2f} mm", color=color, ls=ls)

            p = np.array([t_sample, p])

            sample, reference = common_window([p, data_ref],
                                              half_width=half_width, win_func=win_func)

            # Calculate buffer material properties.
            buffer = uniform_slab(d,
                                  sample, reference, n_med=n_ptfe,
                                  upsampling=1, min_frequency=min_frequency, max_frequency=max_frequency,
                                  all_optical_constants=True)

            refractive_index = np.real(buffer["refractive_index"])
            absorption_coefficient = np.real(buffer["absorption_coefficient"])
            complex_n = buffer["refractive_index"]

        if "PTFE" in name:
            n_ptfe = complex_n
            if d < 2.0:
                dataset_ref["1.8"] = p
            else:
                dataset_ref["2.1"] = p

        if "PTFE" not in name:
            n_sample = complex_n

            mass_frac_sample = 0.10
            mass_frac_ptfe = 0.90

            n_eff = emt(n_ptfe, n_sample, mass_frac_sample, mass_frac_ptfe, rho, rho_ptfe,
                        vol_frac=sample_vol / full_volume, method="bruggeman")

            refractive_index = np.real(n_eff)

            absorption_coefficient = 4 * np.pi * np.imag(n_eff) * buffer["frequency"] * 1e12 / scipy.constants.c * 1e-2

            # v_sample = mass_frac_sample / rho
            # v_ptfe = mass_frac_ptfe / rho_ptfe
            # v_total = v_sample + v_ptfe
            # phi_sample = v_sample / v_total  # volume fraction of dust
            # absorption_coefficient = absorption_coefficient / phi_sample
            # refractive_index = (refractive_index - (1 - phi_sample) * n_ptfe) / phi_sample

        axes[0].plot(buffer["frequency"],
                     refractive_index,
                     label=f"{name}, {d:.2f} mm", color=color, ls=ls)

        axes[1].plot(buffer["frequency"],
                     absorption_coefficient, label=f"{name}, {d:.2f} mm", color=color, ls=ls)

        extinction_coefficient = absorption_coefficient * scipy.constants.c / (
                4 * np.pi * buffer["frequency"] * 1e12) * 1e2

        epsilon = (refractive_index - 1j * extinction_coefficient) ** 2

        if name not in dataset:
            dataset[name] = {
                "frequency": [buffer["frequency"]],
                "refractive_index": [refractive_index],
                "absorption_coefficient": [absorption_coefficient],
                "epsilon": [epsilon],
            }
        else:
            dataset[name]["frequency"].append(buffer["frequency"])
            dataset[name]["refractive_index"].append(refractive_index)
            dataset[name]["absorption_coefficient"].append(absorption_coefficient)
            dataset[name]["epsilon"].append(epsilon)

    axes[0].set_xlabel("Frequency [THz]")
    axes[0].set_ylabel("Refractive Index [-]")

    axes[1].set_xlabel("Frequency [THz]")
    axes[1].set_ylabel("Absorption Coefficient [cm$^{-1}$]")

    # axes[2].set_xlabel("Time (ps)")
    # axes[2].set_ylabel("Signal (nA)")

    # Retrieve the target absorption coefficient

    d = dataset["PTFE"]
    frequency_ptfe = np.mean(np.array(d["frequency"]), axis=0)
    absorption_coefficient_ptfe = np.mean(np.array(d["absorption_coefficient"]), axis=0)
    epsilon_ptfe = np.mean(np.array(d["epsilon"]), axis=0)
    axes[1].plot(frequency_ptfe,
                 absorption_coefficient_ptfe,
                 label="PTFE", color=FOREGROUND_COLOR, ls=":")

    d = dataset["CoDA-T2"]
    frequency_target = np.mean(np.array(d["frequency"]), axis=0)
    absorption_coefficient_target = np.mean(np.array(d["absorption_coefficient"]), axis=0)
    epsilon_target = np.mean(np.array(d["epsilon"]), axis=0)
    axes[1].plot(frequency_target,
                 absorption_coefficient_target,
                 label="CoDA-T2", color=FOREGROUND_COLOR, ls=":")

    # Retrieve the absorption coefficients for Charcoal, SiO2, and CI
    d = dataset["Charcoal"]
    frequency_charcoal = np.mean(np.array(d["frequency"]), axis=0)
    absorption_coefficient_charcoal = np.mean(np.array(d["absorption_coefficient"]), axis=0)
    epsilon_charcoal = np.mean(np.array(d["epsilon"]), axis=0)
    axes[1].plot(frequency_target,
                 absorption_coefficient_charcoal,
                 label="Charcoal", color=FOREGROUND_COLOR, ls=":")

    d = dataset["SiO2"]
    frequency_sio2 = np.mean(np.array(d["frequency"]), axis=0)
    absorption_coefficient_sio2 = np.mean(np.array(d["absorption_coefficient"]), axis=0)
    epsilon_sio2 = np.mean(np.array(d["epsilon"]), axis=0)
    axes[1].plot(frequency_target,
                 absorption_coefficient_sio2,
                 label="SiO2", color=FOREGROUND_COLOR, ls=":")

    d = dataset["CI"]
    frequency_ci = np.mean(np.array(d["frequency"]), axis=0)
    absorption_coefficient_ci = np.mean(np.array(d["absorption_coefficient"]), axis=0)
    epsilon_ci = np.mean(np.array(d["epsilon"]), axis=0)
    axes[1].plot(frequency_target,
                 absorption_coefficient_ci,
                 label="CI", color=FOREGROUND_COLOR, ls=":")

    d = dataset["Forsterite"]
    frequency_forsterite = np.mean(np.array(d["frequency"]), axis=0)
    absorption_coefficient_ci = np.mean(np.array(d["absorption_coefficient"]), axis=0)
    epsilon_forsterite = np.mean(np.array(d["epsilon"]), axis=0)
    axes[1].plot(frequency_target,
                 absorption_coefficient_ci,
                 label="Forsterite", color=FOREGROUND_COLOR, ls=":")

    m_i = 0.52
    rho_i = 2.41
    m_h = 1 - m_i
    rho_h = 0.95
    v_i = m_i / rho_i / (m_i / rho_i + m_h / rho_h)
    v_h = 1 - v_i
    print(f"{v_i=}")
    coda_t1_target_epsilon = bruggeman_eps_eff([epsilon_sio2, epsilon_charcoal], [v_i, v_h])
    coda_t1_target_absorption = -4 * np.pi * np.imag(
        np.sqrt(coda_t1_target_epsilon)) * frequency_target * 1e12 / scipy.constants.c * 1e-2
    axes[1].plot(frequency_target,
                 coda_t1_target_absorption,
                 label="CoDA-T1 calculated", color="red", ls=":")
    axes[0].plot(frequency_target,
                 np.real(np.sqrt(coda_t1_target_epsilon)),
                 label="CoDA-T1 calculated", color="red", ls=":")

    m_sio2 = 0.13
    rho_sio2 = 2.41
    m_charcoal = 0.37
    rho_charcoal = 0.95
    m_forsterite = 0.5
    rho_forsterite = 1.95
    v_sio2 = m_sio2 / rho_sio2 / (m_sio2 / rho_sio2 + m_charcoal / rho_charcoal + m_forsterite / rho_forsterite)
    v_charcoal = m_charcoal / rho_charcoal / (
            m_sio2 / rho_sio2 + m_charcoal / rho_charcoal + m_forsterite / rho_forsterite)
    v_forsterite = 1 - v_sio2 - v_charcoal

    coda_t2_target_epsilon = bruggeman_eps_eff([epsilon_sio2, epsilon_charcoal, epsilon_forsterite],
                                               [v_sio2, v_charcoal, v_forsterite])
    coda_t2_target_absorption = -4 * np.pi * np.imag(
        np.sqrt(coda_t2_target_epsilon)) * frequency_target * 1e12 / scipy.constants.c * 1e-2
    axes[1].plot(frequency_target,
                 coda_t2_target_absorption,
                 label="CoDA-T2 calculated", color="blue", ls=":")
    axes[0].plot(frequency_target,
                 np.real(np.sqrt(coda_t2_target_epsilon)),
                 label="CoDA-T2 calculated", color="blue", ls=":")

    v_sio2 = 0.68082862
    v_ci = 1 - v_sio2

    coda_t3_target_epsilon = bruggeman_eps_eff([epsilon_sio2, epsilon_ci],
                                               [v_sio2, v_ci])
    coda_t3_target_absorption = -4 * np.pi * np.imag(
        np.sqrt(coda_t3_target_epsilon)) * frequency_target * 1e12 / scipy.constants.c * 1e-2
    axes[1].plot(frequency_target,
                 coda_t3_target_absorption,
                 label="CoDA-T3 calculated", color="orange", ls=":")
    axes[0].plot(frequency_target,
                 np.real(np.sqrt(coda_t3_target_epsilon)),
                 label="CoDA-T3 calculated", color="orange", ls=":")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4)
    fig.subplots_adjust(bottom=0.2, hspace=0.25)
    save_result("dust_hesso_pellets_emt.png")
    save_result("dust_hesso_pellets_emt.pdf")
    plt.close(fig)

    index = 200
    print(frequency_target[index])

    epsilons = np.array([epsilon_sio2[index], epsilon_ci[index]])
    epsilon_eff = epsilon_target[index]

    v_i = solve_v_i_two_component(epsilon_sio2[index], epsilon_ci[index], epsilon_eff)
    print(f"Solved volume fraction v_i: {v_i}")
    print(f"Sum of volume fractions: {np.sum(v_i)}")

    rho_ci = 2.65  # g/cm^3

    m_i_sio2 = v_i[0] * rho_sio2
    m_i_ci = (1 - v_i[0]) * rho_ci

    m_tot = m_i_sio2 + m_i_ci
    m_i_sio2 /= m_tot
    m_i_ci /= m_tot

    print("10 g total:")
    print("si_o2:", 0.68 * 10, "g")
    print("ci:", 0.68 * 10 / m_i_sio2 * m_i_ci, "g")
    print(f"Sum of volume fractions: {0.68 * 10 / m_i_sio2 * m_i_ci + 0.68 * 10}")

    print(f"Solved mass fraction m_i SiO2: {m_i_sio2}")
    print(f"Solved mass fraction m_i CI: {m_i_ci}")
    print(f"Sum of volume fractions: {m_i_ci + m_i_sio2}")

    print(f"target epsilon: {epsilon_eff}")
    calculated_epsilon_eff = bruggeman_eps_eff(epsilons, v_i)
    print(f"calculated epsilon: {calculated_epsilon_eff}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    solution = {
        "frequency [THz]": float(frequency_target[index]),
        "SiO2 volume fraction": float(v_i[0]),
        "CI volume fraction": float(v_i[1]),
        "SiO2 mass fraction": float(m_i_sio2),
        "CI mass fraction": float(m_i_ci),
        "target epsilon real": float(np.real(epsilon_eff)),
        "target epsilon imaginary": float(np.imag(epsilon_eff)),
        "calculated epsilon real": float(np.real(calculated_epsilon_eff)),
        "calculated epsilon imaginary": float(np.imag(calculated_epsilon_eff)),
    }
    with (RESULTS_DIR / "sio2_ci_coda_t2_solution.json").open("w") as solution_file:
        json.dump(solution, solution_file, indent=2)

    v_i = np.linspace(0.1, 0.9, 10)
    effs = []
    for vi in v_i:
        effs.append(np.sqrt(bruggeman_eps_eff(epsilons, [vi, 1 - vi])).imag)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(v_i, effs)
    ax.axhline(np.sqrt(epsilon_eff).imag)
    ax.set_xlabel("SiO2 volume fraction [-]")
    ax.set_ylabel(r"Imaginary refractive index $\operatorname{Im}(n)$ [-]")
    fig.tight_layout()
    save_result("sio2_ci_coda_t2_fraction_scan.png")
    save_result("sio2_ci_coda_t2_fraction_scan.pdf")
    plt.close(fig)

    # combined plots

    fig, axes = plt.subplots(nrows =2, figsize=(12, 10))

    for name, d in dataset.items():

        for file in files:
            if file[1] == name:
                color = file[2]
                break

        f = np.mean(np.array(d["frequency"]), axis=0)
        absorption_per_cm = np.mean(np.array(d["absorption_coefficient"]), axis=0)
        n = np.real(np.mean(np.array(d["refractive_index"]), axis=0))

        l1 = axes[1].plot(f, absorption_per_cm, color=color, label=name)
        l2 = axes[0].plot(f, n, color=color, label=name)


    axes[1].set_xlabel("Frequency [THz]")
    axes[1].set_ylabel(r"Absorption coefficient $\alpha$ [cm$^{-1}$]")
    axes[0].set_ylabel(r"Refractive index $n$ [-]")

    fig.subplots_adjust(hspace=0.4, bottom=0.18)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=False, ncol=4)

    save_result("combined_dust_properties.png")
    save_result("combined_dust_properties.pdf")
    plt.close(fig)

    # individual plots
    for name, d in dataset.items():

        fig, ax1 = plt.subplots(figsize=(8, 6))
        color_abs = "tab:orange"
        color_n = "tab:blue"

        f = np.mean(np.array(d["frequency"]), axis=0)
        absorption_per_cm = np.mean(np.array(d["absorption_coefficient"]), axis=0)
        n = np.real(np.mean(np.array(d["refractive_index"]), axis=0))

        l1 = ax1.plot(f, absorption_per_cm, color=color_abs, label=r"Absorption coefficient $\alpha$")
        ax1.set_xlabel("Frequency [THz]")
        ax1.set_ylabel(r"Absorption coefficient $\alpha$ [cm$^{-1}$]", color=color_abs)
        ax1.tick_params(axis="y", labelcolor=color_abs)
        # ax1.set_ylim([-1, 90])
        ax1.tick_params(axis="x")

        ax2 = ax1.twinx()
        l2 = ax2.plot(f, n, color=color_n, label=r"Refractive index $n$")
        ax2.set_ylabel(r"Refractive index $n$ [-]", color=color_n)
        ax2.tick_params(axis="y", labelcolor=color_n)

        n_min = np.min(n)
        n_max = np.max(n)
        n_range = (n_max - n_min)
        ax2.set_ylim([round(n_min - n_range, 3), round(n_max + n_range, 3)])

        # index = 100
        # center_a = (f[index], absorption_per_cm[index] * 2.0)
        # index = -100
        # center_n = (f[index], n[index] * 0.999)
        #
        # dual_axis_arrow(ax1, center=center_a, target_x=np.min(f), radius=None, color=color_abs)
        # dual_axis_arrow(ax2, center=center_n, target_x=np.max(f), radius=None, color=color_n)

        # Combined legend
        lines = l1 + l2
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="best")

        #plt.title(name)

        plt.tight_layout()
        save_result(f"{name}_properties.png")
        save_result(f"{name}_properties.pdf")
        plt.close(fig)

