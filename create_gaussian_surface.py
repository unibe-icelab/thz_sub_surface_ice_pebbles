import numpy as np
import trimesh
from scipy import ndimage
import matplotlib.pyplot as plt


def create_gaussian_surface_cylinder(feature_size: float = 3, plot=False):
    # Parameters
    outer_diameter = 59.0  # mm
    thickness = feature_size / 2.0 + 1   # mm
    outer_radius = outer_diameter / 2.0
    surface_amplitude = feature_size  # mm - actual peak-to-valley amplitude

    # Grid resolution for the surface
    resolution = 150

    # Create coordinate grids
    x = np.linspace(-outer_radius, outer_radius, resolution)
    y = np.linspace(-outer_radius, outer_radius, resolution)
    X, Y = np.meshgrid(x, y)

    # Create circular mask
    R = np.sqrt(X ** 2 + Y ** 2)
    mask = R <= outer_radius

    # Create flat quarter mask (first quadrant: 0° to 90°)
    angle = np.arctan2(Y, X)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    flat_quarter_mask = (angle >= 0) & (angle <= np.pi / 2)

    # Apply Gaussian surface only outside the flat quarter
    gaussian_mask = mask & ~flat_quarter_mask

    # Generate Gaussian random surface
    np.random.seed(42)
    gaussian_noise = np.random.normal(0, 1.0, (resolution, resolution))

    # Calculate sigma for desired feature size
    pixel_size = (2 * outer_radius) / resolution
    sigma_pixels = (feature_size / 2.35) / pixel_size
    gaussian_surface = ndimage.gaussian_filter(gaussian_noise, sigma=sigma_pixels)

    # Scale the Gaussian surface to desired amplitude, centered around 0
    if np.any(gaussian_mask):
        surface_valid = gaussian_surface[gaussian_mask]
        surface_mean = np.mean(surface_valid)
        surface_valid = surface_valid - surface_mean

        surface_std = np.std(surface_valid)
        if surface_std > 0:
            scale_factor = (surface_amplitude / 2) / (3 * surface_std)
            surface_valid = surface_valid * scale_factor

        gaussian_surface[gaussian_mask] = surface_valid

    # Create final surface: Gaussian where needed, zero elsewhere
    final_surface = np.zeros_like(X)
    final_surface[gaussian_mask] = gaussian_surface[gaussian_mask]
    final_surface[~mask] = np.nan  # Outside circle is NaN for plotting

    if plot:
        # Plot the surface for visualization
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(final_surface, extent=[-outer_radius, outer_radius, -outer_radius, outer_radius],
                   cmap='viridis', origin='lower')
        plt.colorbar(label='Height (mm)')
        plt.title('Gaussian Surface (Top View)')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')

        plt.subplot(1, 2, 2)
        from mpl_toolkits.mplot3d import Axes3D
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X_plot = X[mask]
        Y_plot = Y[mask]
        Z_plot = final_surface[mask]
        ax.scatter(X_plot, Y_plot, Z_plot, c=Z_plot, cmap='viridis', s=1)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Height (mm)')
        ax.set_title('Gaussian Surface (3D View)')

        plt.tight_layout()
        plt.show()

    # Now create the cylinder mesh with this surface as the top
    vertices = []
    faces = []

    # Create top surface vertices using the Gaussian surface
    top_vertex_map = np.full((resolution, resolution), -1, dtype=int)
    vertex_count = 0

    for i in range(resolution):
        for j in range(resolution):
            if mask[i, j]:
                z_height = final_surface[i, j]
                vertices.append([X[i, j], Y[i, j], z_height])
                top_vertex_map[i, j] = vertex_count
                vertex_count += 1

    # Create top surface triangular faces
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            if (mask[i, j] and mask[i + 1, j] and
                    mask[i, j + 1] and mask[i + 1, j + 1]):

                idx1 = top_vertex_map[i, j]
                idx2 = top_vertex_map[i + 1, j]
                idx3 = top_vertex_map[i, j + 1]
                idx4 = top_vertex_map[i + 1, j + 1]

                if idx1 >= 0 and idx2 >= 0 and idx3 >= 0 and idx4 >= 0:
                    faces.append([idx1, idx2, idx3])
                    faces.append([idx2, idx4, idx3])

    # Create bottom surface (flat circular base)
    n_bottom_points = 128
    angles = np.linspace(0, 2 * np.pi, n_bottom_points, endpoint=False)

    # Bottom center
    vertices.append([0, 0, -thickness])
    bottom_center_idx = len(vertices) - 1

    # Bottom perimeter
    bottom_perimeter_indices = []
    for angle in angles:
        x_bottom = outer_radius * np.cos(angle)
        y_bottom = outer_radius * np.sin(angle)
        vertices.append([x_bottom, y_bottom, -thickness])
        bottom_perimeter_indices.append(len(vertices) - 1)

    # Bottom faces
    for i in range(n_bottom_points):
        next_i = (i + 1) % n_bottom_points
        faces.append([bottom_center_idx, bottom_perimeter_indices[next_i], bottom_perimeter_indices[i]])

    # Create side walls connecting top surface boundary to bottom perimeter
    surface_boundary_indices = []
    for angle in angles:
        target_x = outer_radius * np.cos(angle)
        target_y = outer_radius * np.sin(angle)

        # Find closest surface point on the actual boundary
        distances = np.sqrt((X - target_x) ** 2 + (Y - target_y) ** 2)
        valid_candidates = mask & (top_vertex_map >= 0)

        if np.any(valid_candidates):
            distances[~valid_candidates] = np.inf
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            i_min, j_min = min_idx
            surface_idx = top_vertex_map[i_min, j_min]
            surface_boundary_indices.append(surface_idx)

    # Ensure we have the same number of boundary points
    assert len(surface_boundary_indices) == len(bottom_perimeter_indices)

    # Side wall faces with correct winding order
    for i in range(len(surface_boundary_indices)):
        next_i = (i + 1) % len(surface_boundary_indices)
        top1 = surface_boundary_indices[i]
        top2 = surface_boundary_indices[next_i]
        bottom1 = bottom_perimeter_indices[i]
        bottom2 = bottom_perimeter_indices[next_i]

        # Ensure consistent face normals pointing outward
        faces.append([top1, top2, bottom1])
        faces.append([bottom1, top2, bottom2])

    # Create and export mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Clean up mesh
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    # Calculate actual amplitude
    if np.any(gaussian_mask):
        gaussian_z_values = final_surface[gaussian_mask]
        actual_min = np.min(gaussian_z_values)
        actual_max = np.max(gaussian_z_values)
        actual_amplitude = actual_max - actual_min
    else:
        actual_amplitude = 0

    # Create and export mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Fix mesh issues for 3D printing
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.fill_holes()
    mesh.fix_normals()

    # Verify manifold
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight!")

    if not mesh.is_winding_consistent:
        print("Warning: Inconsistent winding order!")

    mesh.export(f'gaussian_cylinder_{feature_size}.stl')

    print(f"STL file created successfully!")
    print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
    print(f"Actual amplitude: {actual_amplitude:.2f}mm")
    print(f"Flat quarter: 0° to 90° (positive x, positive y quadrant)")

    return mesh


if __name__ == "__main__":
    for d in [1.0,1.5,2.0,2.5,5.0]:
        create_gaussian_surface_cylinder(d)
