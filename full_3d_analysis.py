from pathlib import Path

from cmcrameri import cm
from plotly.subplots import make_subplots
from pydotthz import DotthzFile

from pyhelper import unibe
from scicolorscales import vik
from sub_surface_extraction import extract_pebble
import plotly.graph_objects as go

from surface_extraction import get_thz_file_from_path
from surface_extraction import extract_surface
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
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

    # # SPIPA B, high res, 34 deg phase angle
    path = Path(
         "/Users/linus/Documents/ICE_PEBBLE_SPHERE_CoDA_T3_dust_high_res_vac_10_fast_focus_on_sphere_top/data_image_rfl")

    # # empty control, low res, 34 deg phase angle
    # path = Path(
    #      "/Users/linus/Documents/CONTROL_CoDA_T3_dust_low_res_vac_11_fast_focus_on_sphere_top/data_image_rfl")

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


    surface = extract_surface(times, traces, x_min, x_max, width, y_min, y_max, height,
                              low_cut=1.0,
                              high_cut=5.0,
                              start_freq=1.2,
                              end_freq=4.0,
                              )

    mask_volume, spacing = extract_pebble(times, traces, dx, dy, x_min, x_max, width, y_min, y_max, height,
                                          output_path=Path("./pebble_extraction"),
                                          filename=path.stem,
                                          low_cut=0.2,
                                          high_cut=0.5,
                                          start_freq=0.2,
                                          end_freq=0.5,
                                          z_range=[-11, -30],
                                          n_dust=1.0,
                                          power=1.0,
                                          sigma=6,
                                          ball_kernel_radius=20,
                                          threshold=0.08,
                                          sigma_smooth=2.0,
                                          mask_threshold=0.5,
                                          plot_b_scan=True,
                                          surface=surface,
                                          rim_width_r=7.5,
                                          )

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

    plot_rendered = "matplotlib"

    if plot_rendered == "plotly":

        # Create subplots
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'surface'}]]
        )

        # Original surface
        fig.add_trace(
            go.Surface(x=X, y=Y, z=surface, colorscale=vik, showscale=False),
            row=1, col=1
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
                    gridcolor="lightgray",  # typical mpl look
                    gridwidth=1,
                    showline=True,
                    linecolor='black',
                    linewidth=2,
                    zeroline=False,
                    ticks='outside',
                    tickcolor='black',
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=1,
                    showline=True,
                    linecolor='black',
                    linewidth=2,
                    zeroline=False,
                    ticks='outside',
                    tickcolor='black',
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=1,
                    showline=True,
                    linecolor='black',
                    linewidth=2,
                    zeroline=False,
                    ticks='outside',
                    tickcolor='black',
                ),
            ),
            scene_camera=dict(
                projection=dict(type='orthographic'),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1, y=-1.25, z=0.45)
            ),

            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        fig.write_image(Path("./pebble_extraction").joinpath(f"{path.stem}_3D.pdf"))
        fig.show()
    else:

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
                      18}

        plt.rcParams.update(params)

        # Flip x and y axes to match surface orientation
        mask_volume_flipped = np.transpose(mask_volume, (1, 0, 2))

        # Normalize
        volume_data = mask_volume_flipped.astype(float)
        vmin, vmax = volume_data.min(), volume_data.max()
        if vmax > vmin:
            volume_data = (volume_data - vmin) / (vmax - vmin)
        else:
            volume_data = np.zeros_like(volume_data)

        binary_mask = (volume_data >= 0.5).astype(float)

        # --- Matplotlib 3D plot ---
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Surface plot
        ax.plot_surface(X, Y, surface, cmap=cm.vik, edgecolor='none', alpha=1.0)

        try:
            # --- Marching cubes mesh ---
            verts, faces, _, _ = marching_cubes(binary_mask, level=0.5)
            verts_mm = np.zeros_like(verts)
            verts_mm[:, 0] = verts[:, 0] * spacing[0]
            verts_mm[:, 1] = verts[:, 1] * spacing[1]
            verts_mm[:, 2] = -verts[:, 2] * spacing[2]

            # Triangles
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

            # Mask surface
            tris = ax.plot_trisurf(
                verts_mm[:, 0], verts_mm[:, 1], verts_mm[:, 2],
                triangles=faces,
                color=unibe.BLUE,
                alpha=1.0,
                linewidth=0.1
            )
            # Ensure equal axis scaling in mm
            x_range_mm = np.ptp(verts_mm[:, 0])
            y_range_mm = np.ptp(verts_mm[:, 1])
            z_range_mm = np.ptp(verts_mm[:, 2])
        except ValueError:
            ax.set_zticks([0, -10])
            x_range_mm = np.ptp(x_mm)
            y_range_mm = np.ptp(y_mm)
            z_range_mm = np.ptp(surface)
            pass

        # --- Axis styling ---
        ax.set_xlabel('X [mm]', labelpad=20)
        ax.set_ylabel('Y [mm]', labelpad=20)
        ax.set_zlabel('Z [mm]', labelpad=8)

        ax.tick_params(colors='black', direction='out', length=5)

        ax.grid(True, color='lightgray', linewidth=0.5)


        ax.set_box_aspect([x_range_mm, y_range_mm, z_range_mm])

        # Optional: Orthographic-ish look
        ax.view_init(elev=18, azim=-30)

        # Save and show
        plt.savefig(Path("./pebble_extraction").joinpath(f"{path.stem}_3D.pdf"), bbox_inches='tight', pad_inches=0.55,
                    # Optional: reduce padding
                    )
        plt.show()
