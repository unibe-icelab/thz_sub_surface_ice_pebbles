"""Plot optical properties for the cometary-dust analogue materials.

The input arrays contain five columns: frequency, absorption coefficient,
absorption standard deviation, real refractive index, and refractive-index
standard deviation.  Change ``PLOT_BACKGROUND`` in :mod:`plot_style` to
switch every export between the white and black themes.
"""

from pathlib import Path

import numpy as np
import scipy.constants
from matplotlib import pyplot as plt

from plot_style import save_figure, selection_material_color
from pyhelper import unibe
from capsule_paths import MATERIAL_CACHE_ROOT, RESULTS_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = MATERIAL_CACHE_ROOT
RESULTS_DIR = RESULTS_ROOT / "material_properties"

# Display name, input stem, and requested output stem.
INDIVIDUAL_MATERIALS = (
    ("SiO2", "SiO2", "SiO2"),
    ("CI", "CI", "CI"),
    ("Forsterite", "Forsterite", "Forsterite"),
    ("Fayalite", "Fayalite", "Fayalite"),
    ("Charcoal", "Charcoal", "Charcoal"),
    ("Juniper charcoal", "Juniper_charcoal", "Juniper_charcoal"),
    ("CoDA-T0", "CoDA-T0", "T0"),
    ("CoDA-T1", "CoDA-T1", "T1"),
    ("CoDA-T2", "CoDA-T2", "T2"),
    ("CoDA-T3", "CoDA-T3", "T3"),
)

# This is the material selection used by the original combined figure.
COMBINED_MATERIALS = (
    ("Forsterite", "Forsterite"),
    ("Charcoal", "Charcoal"),
    ("CI", "CI"),
    ("SiO2", "SiO2"),
    ("CoDA-T1", "CoDA-T1"),
    ("CoDA-T0", "CoDA-T0"),
    ("PTFE", "PTFE"),
    ("Juniper charcoal", "Juniper_charcoal"),
    ("CoDA-T2", "CoDA-T2"),
    ("CoDA-T3", "CoDA-T3"),
)

NOISE_FLOORS_THZ = {
    "PTFE": 4.0,
    "Forsterite": 2.55,
    "CoDA-T0": 3.3,
    "CoDA-T1": 3.0,
    "CoDA-T2": 2.7,
    "CoDA-T3": 2.7,
    "Charcoal": 3.0,
    "Juniper charcoal": 2.15,
    "SiO2": 3.3,
    "CI": 2.15,
}


def load_material(input_stem: str) -> np.ndarray:
    """Load and validate one processed material-property array."""
    path = DATA_DIR / f"{input_stem}_data.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing material-property input: {path}")

    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(
            f"Expected {path} to have five columns, found shape {data.shape}"
        )
    return data


def add_wavelength_axis(ax, frequency_thz: np.ndarray, unit: str = "µm"):
    """Add the wavelength scale used by the original material plots."""
    scale = {
        "cm": 1e2,
        "µm": 1e6,
        "nm": 1e9,
    }
    if unit not in scale:
        raise ValueError(f"Unsupported wavelength unit: {unit}")

    wavelengths = scipy.constants.c * scale[unit] / (frequency_thz * 1e12)
    wavelength_axis = ax.twiny()
    wavelength_axis.set_xlim(wavelengths[0], wavelengths[-1])
    wavelength_axis.set_xlabel(f"Wavelength [{unit}]")
    wavelength_axis.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.1f}")
    )
    return wavelength_axis


def save_current_figure(fig, *filenames: str) -> None:
    """Save one figure in each requested format and close it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(fig.number)
    for filename in filenames:
        save_figure(RESULTS_DIR / filename)
    plt.close(fig)


def plot_individual_properties() -> None:
    """Generate the ten requested per-material PDF and PNG figures."""
    for _display_name, input_stem, output_stem in INDIVIDUAL_MATERIALS:
        data = load_material(input_stem)
        frequency, absorption, absorption_std, refractive_index, index_std = data.T

        fig, absorption_axis = plt.subplots(figsize=(8, 6))
        absorption_line = absorption_axis.plot(
            frequency,
            absorption,
            color=unibe.BLUE,
            label=r"Absorption coefficient $\alpha$",
        )
        absorption_axis.fill_between(
            frequency,
            absorption - absorption_std,
            absorption + absorption_std,
            color=unibe.BLUE,
            alpha=0.2,
        )
        absorption_axis.set_xlabel("Frequency [THz]")
        absorption_axis.set_ylabel(
            r"Absorption coefficient $\alpha$ [cm$^{-1}$]", color=unibe.BLUE
        )
        absorption_axis.tick_params(axis="y", labelcolor=unibe.BLUE)

        index_axis = absorption_axis.twinx()
        index_line = index_axis.plot(
            frequency,
            refractive_index,
            color=unibe.RED,
            label=r"Refractive index $n$",
        )
        index_axis.fill_between(
            frequency,
            refractive_index - index_std,
            refractive_index + index_std,
            color=unibe.RED,
            alpha=0.2,
        )
        index_axis.set_ylabel(r"Refractive index $n$ [-]", color=unibe.RED)
        index_axis.tick_params(axis="y", labelcolor=unibe.RED)

        index_min = np.min(refractive_index)
        index_max = np.max(refractive_index)
        index_range = index_max - index_min
        if index_range == 0:
            index_range = max(0.1, abs(index_max) * 0.1)
        index_axis.set_ylim(round(index_min - index_range, 3), round(index_max + index_range, 3))

        lines = absorption_line + index_line
        absorption_axis.legend(lines, [line.get_label() for line in lines], loc="best")
        add_wavelength_axis(absorption_axis, frequency)
        add_wavelength_axis(index_axis, frequency)
        fig.tight_layout()

        save_current_figure(
            fig,
            f"{output_stem}_properties.pdf",
            f"{output_stem}_properties.png",
        )


def plot_combined_properties() -> None:
    """Generate the requested combined material-selection figure."""
    fig, axes = plt.subplots(nrows=2, figsize=(12, 10))
    color_map = plt.get_cmap("tab10")
    last_frequency = None

    for index, (display_name, input_stem) in enumerate(COMBINED_MATERIALS, start=1):
        data = load_material(input_stem)
        frequency, absorption, absorption_std, refractive_index, index_std = data.T
        noise_floor = NOISE_FLOORS_THZ[display_name]
        reliable = frequency <= noise_floor
        color = selection_material_color(display_name, color_map(index % color_map.N))

        axes[1].plot(frequency[reliable], absorption[reliable], color=color, label=display_name)
        axes[1].plot(frequency[~reliable], absorption[~reliable], color=color, linestyle="--")
        axes[1].fill_between(
            frequency, absorption - absorption_std, absorption + absorption_std,
            color=color, alpha=0.2,
        )
        axes[0].plot(
            frequency[reliable], refractive_index[reliable], color=color, label=display_name
        )
        axes[0].plot(
            frequency[~reliable], refractive_index[~reliable], color=color, linestyle="--"
        )
        axes[0].fill_between(
            frequency, refractive_index - index_std, refractive_index + index_std,
            color=color, alpha=0.2,
        )
        last_frequency = frequency

    axes[0].set_xlabel("Frequency [THz]")
    axes[1].set_xlabel("Frequency [THz]")
    axes[1].set_ylabel(r"Absorption coefficient $\alpha$ [cm$^{-1}$]")
    axes[0].set_ylabel(r"Refractive index $n$ [-]")
    add_wavelength_axis(axes[0], last_frequency)
    add_wavelength_axis(axes[1], last_frequency)

    fig.subplots_adjust(hspace=0.5, bottom=0.18, top=0.9)
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=False, ncol=4
    )
    save_current_figure(
        fig,
        "combined_properties_selection.pdf",
        "combined_properties_selection.png",
    )


def main() -> None:
    plot_individual_properties()
    plot_combined_properties()


if __name__ == "__main__":
    main()
