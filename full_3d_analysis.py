from pathlib import Path

from cmcrameri import cm
from plotly.subplots import make_subplots
from pydotthz import DotthzFile

from scicolorscales import vik
from sub_surface_extraction import extract_pebble
import plotly.graph_objects as go

from surface_extraction import get_thz_file_from_path
from surface_extraction import extract_surface
import pyvista as pv
import numpy as np
from skimage.measure import marching_cubes

if __name__ == "__main__":
    # COC solid, low res, 38 deg phase angle
    path = Path("/Users/linus/Documents/COC_PEBBLE_CoDA_T3_dust_low_res_vac_4/data_image_rfl")

    # solid, low res, 38 deg phase angle, tilted
    path = Path("/Users/linus/Documents/ICE_PEBBLE_CoDA_T3_dust_low_res_vac_6_small_fov_slow/data_image_rfl")

    # solid, low res, 38 deg phase angle
    path = Path(
        "/Users/linus/Documents/ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_7_fast_focus_on_sohere_top/data_image_rfl")

    # solid, low res. 34 deg phase angle
    path = Path(
        "/Users/linus/Documents/ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_9_fast_focus_on_sohere_top/data_image_rfl")

    # solid, high res 34 deg phase angle
    path = Path(
        "/Users/linus/Documents/ICE_PEBBLE_SPHERE_CoDA_T3_dust_high_res_vac_9_fast_focus_on_sohere_top/data_image_rfl")

    # SPIPA B, low res, 34 deg phase angle
    # path = Path(
    #     "/Users/linus/Documents/ICE_PEBBLE_SPHERE_CoDA_T3_dust_low_res_vac_10_fast_focus_on_sphere_top/data_image_rfl")

    # SPIPA B, high res, 34 deg phase angle
    path = Path(
        "/Users/linus/Documents/ICE_PEBBLE_SPHERE_CoDA_T3_dust_high_res_vac_10_fast_focus_on_sphere_top/data_image_rfl")

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

    mask_volume, spacing = extract_pebble(times, traces, dx, dy, x_min, x_max, width, y_min, y_max, height,
                                          output_path=Path("./pebble_extraction"),
                                          filename=path.stem,
                                          z_range=[-11, -25],
                                          n_dust=1.0,
                                          sigma=6,
                                          ball_kernel_radius=20,
                                          threshold=0.1,
                                          sigma_smooth=2.0,
                                          mask_threshold=0.5,
                                          plot_a_scan=False,
                                          plot_b_scan=True
                                          )

    surface = extract_surface(times, traces, x_min, x_max, width, y_min, y_max, height)

    # --- Surface setup ---
    nx, ny = surface.shape
    x_mm = np.linspace(0, y_max - y_min, ny)
    y_mm = np.linspace(0, x_max - x_min, nx)
    X, Y = np.meshgrid(x_mm, y_mm, indexing='xy')
    surface_grid = pv.StructuredGrid(X, Y, surface)

    nx, ny = surface.shape

    # flip because of orientation
    x_mm = np.linspace(0, y_max - y_min, ny)
    y_mm = np.linspace(0, x_max - x_min, nx)
    X, Y = np.meshgrid(x_mm, y_mm, indexing='xy')

    # Create subplots
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Original Surface'),
        specs=[[{'type': 'surface'}]]
    )

    # Original surface
    fig.add_trace(
        go.Surface(x=X, y=Y, z=surface, colorscale=vik, showscale=False),
        row=1, col=1
    )

    # Update layout
    fig.update_layout(
        height=600,
        width=1400,
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='data'
        ),
    )

    # Flip x and y axes to match surface orientation
    mask_volume_flipped = np.transpose(mask_volume, (1, 0, 2))

    # Continue with normalization
    volume_data = mask_volume_flipped.astype(float)
    vmin, vmax = volume_data.min(), volume_data.max()
    if vmax > vmin:
        volume_data = (volume_data - vmin) / (vmax - vmin)
    else:
        volume_data = np.zeros_like(volume_data)

    binary_mask = (volume_data >= 0.5).astype(float)

    # Mesh extraction and visualization as before
    verts, faces, _, _ = marching_cubes(binary_mask, level=0.5)
    verts_mm = np.zeros_like(verts)
    verts_mm[:, 0] = verts[:, 0] * spacing[0]
    verts_mm[:, 1] = verts[:, 1] * spacing[1]
    verts_mm[:, 2] = -verts[:, 2] * spacing[2]

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    fig.add_trace(
        go.Mesh3d(
            x=verts_mm[:, 0],
            y=verts_mm[:, 1],
            z=verts_mm[:, 2],
            i=i, j=j, k=k,
            color='red',
            opacity=1.0,
            name="mask_surface"
        ),
        row=1, col=1
    )

    fig.show()
