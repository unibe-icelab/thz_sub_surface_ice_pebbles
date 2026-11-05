"""Shared colors and export helpers for every generated plot."""

from matplotlib import pyplot as plt

from pyhelper import unibe

# Change this one value to "white" or "black" for every PNG/PDF export.
PLOT_BACKGROUND = "white"

if PLOT_BACKGROUND not in {"white", "black"}:
    raise ValueError('PLOT_BACKGROUND must be either "white" or "black"')

BACKGROUND_COLOR = PLOT_BACKGROUND
FOREGROUND_COLOR = "white" if PLOT_BACKGROUND == "black" else "black"
PEBBLE_COLOR = "white" if PLOT_BACKGROUND == "black" else unibe.BLUE
GRID_COLOR = "#777777" if PLOT_BACKGROUND == "black" else "#d3d3d3"

# Material colors used by the original HESSO pellet comparison plots. CI uses
# the foreground color so the same curve remains visible in both themes.
MATERIAL_COLORS = {
    "PTFE": "gray",
    "SiO2": "cyan",
    "Charcoal": "violet",
    "CoDA-T0": "blue",
    "CoDA-T1": "pink",
    "CoDA-T2": "yellow",
    "CoDA-T3": "orange",
    "CI": FOREGROUND_COLOR,
    "Forsterite": "green",
    "Fayalite": "red",
    "Juniper charcoal": "brown",
    "Juniper_charcoal": "brown",
}

# Exact mapping used by the historical combined_properties_selection figure.
# These are Matplotlib's tab10 colors, pinned by material instead of depending
# on the order returned by os.listdir().
SELECTION_MATERIAL_COLORS = {
    "Forsterite": "#ff7f0e",
    "Charcoal": "#2ca02c",
    "CI": "#d62728",
    "SiO2": "#9467bd",
    "CoDA-T1": "#8c564b",
    "CoDA-T0": "#e377c2",
    "PTFE": "#7f7f7f",
    "Juniper charcoal": "#bcbd22",
    "Juniper_charcoal": "#bcbd22",
    "CoDA-T2": "#17becf",
    "CoDA-T3": "#1f77b4",
}


def material_color(name, fallback=None):
    """Return the stable legacy color assigned to a measured material."""
    return MATERIAL_COLORS.get(name, fallback)


def selection_material_color(name, fallback=None):
    """Return the color used by the historical combined-selection figure."""
    return SELECTION_MATERIAL_COLORS.get(name, fallback)


def apply_plot_style():
    """Apply the selected background and readable foreground to Matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND_COLOR,
        "axes.facecolor": BACKGROUND_COLOR,
        "savefig.facecolor": BACKGROUND_COLOR,
        "savefig.edgecolor": BACKGROUND_COLOR,
        "text.color": FOREGROUND_COLOR,
        "axes.labelcolor": FOREGROUND_COLOR,
        "axes.edgecolor": FOREGROUND_COLOR,
        "axes.titlecolor": FOREGROUND_COLOR,
        "xtick.color": FOREGROUND_COLOR,
        "ytick.color": FOREGROUND_COLOR,
        "legend.facecolor": BACKGROUND_COLOR,
        "legend.edgecolor": FOREGROUND_COLOR,
        "legend.labelcolor": FOREGROUND_COLOR,
    })


def save_figure(path, **kwargs):
    """Save a Matplotlib figure using the selected opaque background."""
    plt.savefig(
        path,
        facecolor=BACKGROUND_COLOR,
        edgecolor=BACKGROUND_COLOR,
        transparent=False,
        **kwargs,
    )


def apply_plotly_style(fig):
    """Apply the selected background and foreground to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font_color=FOREGROUND_COLOR,
    )
    fig.update_scenes(
        bgcolor=BACKGROUND_COLOR,
        xaxis_color=FOREGROUND_COLOR,
        yaxis_color=FOREGROUND_COLOR,
        zaxis_color=FOREGROUND_COLOR,
    )
    return fig


apply_plot_style()
