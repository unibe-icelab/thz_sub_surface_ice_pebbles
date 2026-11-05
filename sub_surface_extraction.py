import os
from pathlib import Path

import numpy as np
from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pydotthz import DotthzFile
from scipy.ndimage import zoom, gaussian_filter
from scipy.signal import fftconvolve
from skimage.morphology import ball
from skimage.restoration import ball_kernel
from tqdm import tqdm
import pyvista as pv

from envelope import gaussian_kernel1d, convolve1d
from surface_extraction import get_thickness
from thz_deconvolution import Deconvolution
from surface_extraction import get_thz_file_from_path

import plotly.graph_objects as go
from plot_style import (
    BACKGROUND_COLOR,
    FOREGROUND_COLOR,
    GRID_COLOR,
    apply_plot_style,
    apply_plotly_style,
    save_figure,
)

ROOT_DIR = Path(__file__).resolve().parent
PSF_DATA_DIR = ROOT_DIR / "data" / "psf"
MEASUREMENTS_DIR = ROOT_DIR / "data" / "sub_surface_extraction" / "measurements"
MEASUREMENTS = {
    "solid_low_res_34deg": "ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_9_fast_focus_on_sohere_top",
    "spipa_b_high_res_34deg": "ICE_PEBBLE_SPHERE_CoDA_T3_dust_high_res_vac_10_fast_focus_on_sphere_top",
    "spipa_b_low_res_34deg": "ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_11_fast_focus_on_sphere_top",
    "empty_control_low_res_34deg": "CONTROL_CoDA_T3_dust_low_res_vac_11_fast_focus_on_sphere_top",
}

ACTIVE_MEASUREMENT = "empty_control_low_res_34deg"
EXAMPLE_DATA_DIR = MEASUREMENTS_DIR / MEASUREMENTS[ACTIVE_MEASUREMENT] / "data_image_rfl"
EXAMPLE_RESULTS_DIR = ROOT_DIR / "results" / "sub_surface_extraction"

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
              24}

plt.rcParams.update(params)
apply_plot_style()


def get_scaled_ball_kernel(radius, downsample_x, downsample_y, downsample_z):
    downsample_factor = max(downsample_x, downsample_y, downsample_z)
    scaled_radius = max(1, int(round(radius / downsample_factor)))
    kernel = ball(scaled_radius).astype(float)
    return kernel, kernel.sum()


def extract_pebble(times, traces, dx, dy, x_min, x_max, width, y_min, y_max, height, output_path: Path,
                   filename: str,
                   n_dust: float = 1.0,
                   low_cut=0.2,
                   high_cut=0.5,
                   start_freq=0.2,
                   end_freq=0.5,
                   win_width=0.5,
                   n_filters=20,
                   w_max=30,
                   z_range=[5, 20],
                   sigma=6,
                   power=1.0,
                   ball_kernel_radius=6,
                   threshold=0.15 ** 1,
                   sigma_smooth=2.0,
                   mask_threshold=0.5,
                   plot_b_scan: bool = True,
                   surface=None,
                   rim_width_l=5,  # mm
                   rim_width_r=5  # mm
                   ):
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
        w_max=w_max)

    os.makedirs(output_path, exist_ok=True)

    w, h, d = traces.shape
    d_scan = traces[:, w // 2 - 10, :]
    b_scan = traces[h // 2, :, :]

    spacing_x = dx
    spacing_y = dy

    spacing_z = -(get_thickness(times[0], times[1]) * 1000) / n_dust

    # Plot using imshow with the calculated extent
    image = np.abs(b_scan)
    image = image.T

    extent = [0, b_scan.shape[0] * spacing_x, b_scan.shape[1] * spacing_z, 0]

    if plot_b_scan:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylabel("OPL (Depth) [mm]")
        im = ax.imshow(image, extent=extent, aspect='1', norm="log", cmap=cm.vik)
        ax.set_xlabel("Width [mm]")
        # ax.set_title("Raw B-Scan")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Intensity [a.u.]")

        save_figure(output_path.joinpath(f"{filename}_raw_b_scan.png"), dpi=300)
        save_figure(output_path.joinpath(f"{filename}_raw_b_scan.pdf"))
        plt.show()

    deconvolved_traces = Decon.apply_deconvolution(traces, x_min, x_max, width, y_min, y_max, height, max_iter=100)
    traces = deconvolved_traces

    traces = traces[::, ::, :]
    times = times[:]

    surface_b_scan = surface[h // 2, :]


    b_scan_convolved = np.zeros_like(b_scan)

    kernel = gaussian_kernel1d(sigma=sigma, radius=sigma * 3)

    for x in tqdm(range(traces.shape[0])):
        convolved_data = convolve1d(np.abs(b_scan[x, :]), kernel)
        b_scan_convolved[x] = convolved_data
        # b_scan_convolved[x] = scipy.signal.hilbert(np.abs(b_scan[x])**0.5).imag


    # -------------
    # 3D Processing
    # -------------

    # First, apply 1D Gaussian convolution along the z-axis for each (x, y) position

    image_convolved_3d = np.zeros_like(traces)
    kernel = gaussian_kernel1d(sigma=sigma, radius=sigma * 3)

    for x in tqdm(range(traces.shape[0])):
        for y in range(traces.shape[1]):
            convolved_data = convolve1d(np.abs(traces[x, y, :]) ** power, kernel)
            image_convolved_3d[x, y, :] = convolved_data

    # Calculate downsampling factors to make dimensions more cubic
    target_spacing = max(spacing_x, spacing_y, spacing_z)
    downsample_x = max(1, int(np.round(target_spacing / spacing_x)))
    downsample_y = max(1, int(np.round(target_spacing / spacing_y)))
    downsample_z = max(1, int(np.round(target_spacing / np.abs(spacing_z))))

    print(f"Original shape: {image_convolved_3d.shape}")
    print(f"Original spacing: x={spacing_x:.3f}, y={spacing_y:.3f}, z={spacing_z:.3f} mm")
    print(f"Downsampling factors: x={downsample_x}, y={downsample_y}, z={downsample_z}")

    # Downsample using averaging (zoom with order=1 for linear interpolation, or use block averaging)
    zoom_factors = (1 / downsample_x, 1 / downsample_y, 1 / downsample_z)
    image_downsampled = zoom(image_convolved_3d, zoom_factors, order=1)  # Linear interpolation

    # Update spacing after downsampling
    spacing_x_down = spacing_x * downsample_x
    spacing_y_down = spacing_y * downsample_y
    spacing_z_down = spacing_z * downsample_z

    print(f"Downsampled shape: {image_downsampled.shape}")
    print(f"Downsampled spacing: x={spacing_x_down:.3f}, y={spacing_y_down:.3f}, z={spacing_z_down:.3f} mm")

    # Visualize final mask slice
    b_scan_downsampled = image_downsampled[:, w // 2 - 10, :]
    extent = [0, b_scan_downsampled.shape[0] * spacing_x_down, b_scan_downsampled.shape[1] * spacing_z_down, 0]

    # Overlay white rectangles on left and right 10mm
    height_mm = b_scan.shape[1] * spacing_z

    if plot_b_scan:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylabel("OPL (Depth) [mm]")
        im = ax.imshow(b_scan_downsampled.T, extent=extent, aspect='1', norm="log", cmap=cm.vik)
        ax.set_xlabel("Width [mm]")
        # ax.set_title("Filtered B-Scan")

        if surface is not None:
            x = np.linspace(0, b_scan.shape[0] * spacing_x, b_scan.shape[0])
            ax.plot(x, surface_b_scan, color=FOREGROUND_COLOR, linewidth=4)

        # Left rectangle
        ax.add_patch(Rectangle((0, 0), rim_width_l, height_mm, color='white', zorder=2, alpha=0.7))
        # Right rectangle
        ax.add_patch(
            Rectangle((b_scan.shape[0] * spacing_x - rim_width_r, 0), rim_width_r, height_mm, color='white', zorder=2,
                      alpha=0.7))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.5)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Intensity [a.u.]")

        save_figure(output_path.joinpath(f"{filename}_filtered_b_scan.png"), dpi=300)
        save_figure(output_path.joinpath(f"{filename}_filtered_b_scan.pdf"))
        plt.show()

    # Apply 3D Gaussian smoothing to the downsampled data
    image_smoothed = gaussian_filter(image_downsampled, sigma=sigma_smooth)

    # Apply threshold to smoothed image
    # threshold = 0.15 ** 1
    solid_mask = (image_smoothed >= threshold).astype(np.uint8)

    # Use kernel radius
    R_small = max(1, ball_kernel_radius // max(downsample_x, downsample_y, downsample_z))
    kernel = ball(R_small).astype(float)
    kernel_size = kernel.sum()

    print(f"Using ball kernel with radius: {R_small}")

    # Apply 3D convolution
    image_filtered = fftconvolve(solid_mask.astype(float), kernel, mode='same') / kernel_size

    # Visualize local density slice
    w, h, d = image_filtered.shape
    b_scan_filtered = image_filtered[:, w // 2 - 10, :]
    extent = [0, b_scan_filtered.shape[0] * spacing_x_down, b_scan_filtered.shape[1] * spacing_z_down, 0]

    if plot_b_scan:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylabel("OPL (Depth) [mm]")
        ax.imshow(b_scan_filtered.T, extent=extent, aspect='1', cmap=cm.vik)
        ax.set_xlabel("Width [mm]")
        # ax.set_title("Masked B-Scan")

        if surface is not None:
            x = np.linspace(0, b_scan.shape[0] * spacing_x, b_scan.shape[0])
            ax.plot(x, surface_b_scan, color=FOREGROUND_COLOR, linewidth=4)

        # Left rectangle
        ax.add_patch(Rectangle((0, 0), rim_width_l, height_mm, color='white', zorder=2, alpha=0.7))
        # Right rectangle
        ax.add_patch(
            Rectangle((b_scan.shape[0] * spacing_x - rim_width_r, 0), rim_width_r, height_mm, color='white', zorder=2,
                      alpha=0.7))


        save_figure(output_path.joinpath(f"{filename}_masked_b_scan.png"), dpi=300)
        save_figure(output_path.joinpath(f"{filename}_masked_b_scan.pdf"))
        plt.show()

    z_min_mm, z_max_mm = z_range
    z_min_idx = int(z_min_mm / spacing_z_down)
    z_max_idx = int(z_max_mm / spacing_z_down)

    mask = (image_filtered < mask_threshold).astype(np.float32)

    # Explicitly zero out voxels outside z-range
    mask[:, :, :z_min_idx] = 0.0
    mask[:, :, z_max_idx:] = 0.0

    return mask, (spacing_x_down, spacing_y_down, np.abs(spacing_z_down))


if __name__ == "__main__":
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

    mask_volume, spacing = extract_pebble(times, traces, dx, dy, x_min, x_max, width, y_min, y_max, height,
                                          output_path=EXAMPLE_RESULTS_DIR,
                                          filename=path.stem,
                                          low_cut=0.2,
                                          high_cut=0.5,
                                          z_range=[-7, -25],
                                          n_dust=1.5,
                                          sigma=6,
                                          power=1.0,
                                          ball_kernel_radius=6,
                                          threshold=0.1,
                                          sigma_smooth=2.0,
                                          mask_threshold=0.5,
                                          plot_b_scan=True
                                          )

    # mask_volume_flipped = mask_volume[:, :, ::-1]
    #
    # # Create PyVista ImageData grid
    # grid = pv.ImageData()
    # z_max = mask_volume_flipped.shape[2] * spacing[2]
    # grid.dimensions = np.array(mask_volume_flipped.shape) + 1
    # grid.spacing = spacing
    # grid.origin = (0, 0, -z_max)
    #
    # # Use cell_data instead of point_data
    # grid.cell_data["mask"] = mask_volume_flipped.flatten(order="F")
    #
    # # Create plotter and add volume with opacity mapping
    # plotter = pv.Plotter()
    #
    # plotter.add_volume(grid, scalars="mask",
    #                    cmap=cm.vik,
    #                    name="mask_volume")
    #
    # plotter.add_axes()
    # plotter.show_grid()
    # plotter.camera.zoom(1.2)
    # plotter.show()

    # Flip and normalize
    volume_data = mask_volume.astype(float)
    vmin, vmax = volume_data.min(), volume_data.max()
    if vmax > vmin:
        volume_data = (volume_data - vmin) / (vmax - vmin)
    else:
        volume_data = np.zeros_like(volume_data)

    # Assume volume_data is (nx, ny, nz) and mask_volume_flipped is already prepared
    nx, ny, nz = volume_data.shape
    x_mm = np.linspace(0, spacing[0] * (nx - 1), nx)
    y_mm = np.linspace(0, spacing[1] * (ny - 1), ny)
    z_mm = np.linspace(0, -spacing[2] * (nz - 1), nz)

    # Create meshgrid and flatten
    X, Y, Z = np.meshgrid(x_mm, y_mm, z_mm, indexing='ij')
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    value_flat = volume_data.flatten()

    # After all processing, before visualization
    binary_mask = (volume_data >= 0.5).astype(float)

    from skimage.measure import marching_cubes
    import plotly.graph_objects as go

    # Extract surface mesh from binary mask
    verts, faces, _, _ = marching_cubes(binary_mask, level=0.5)

    # Convert voxel coordinates to physical units
    verts_mm = np.zeros_like(verts)
    verts_mm[:, 0] = verts[:, 0] * spacing[0]
    verts_mm[:, 1] = verts[:, 1] * spacing[1]
    verts_mm[:, 2] = -verts[:, 2] * spacing[2]  # negative for z

    # Prepare face indices for Plotly
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts_mm[:, 0],
            y=verts_mm[:, 1],
            z=verts_mm[:, 2],
            i=i, j=j, k=k,
            color='red',
            opacity=1.0,
            name="mask_surface"
        )
    ])

    fig.update_layout(
        height=600,
        width=1400,
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='data',

            xaxis=dict(
                showbackground=False,  # no shaded plane
                showgrid=True,  # keep grid
                gridcolor=GRID_COLOR,
                gridwidth=1,
                showline=True,
                linecolor=FOREGROUND_COLOR,
                linewidth=2,
                zeroline=False,
                ticks='outside',
                tickcolor=FOREGROUND_COLOR,
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=True,
                gridcolor=GRID_COLOR,
                gridwidth=1,
                showline=True,
                linecolor=FOREGROUND_COLOR,
                linewidth=2,
                zeroline=False,
                ticks='outside',
                tickcolor=FOREGROUND_COLOR,
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=True,
                gridcolor=GRID_COLOR,
                gridwidth=1,
                showline=True,
                linecolor=FOREGROUND_COLOR,
                linewidth=2,
                zeroline=False,
                ticks='outside',
                tickcolor=FOREGROUND_COLOR,
            ),
        ),
        scene_camera=dict(
            projection=dict(type='orthographic')
        ),

        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
    )

    apply_plotly_style(fig)
    fig.show()
