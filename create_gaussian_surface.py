import numpy as np
import trimesh
from scipy import ndimage


def create_gaussian_surface_cylinder(feature_size: float = 2.5):
    # Parameters
    outer_diameter = 59.0  # mm
    thickness = feature_size / 2.0  # mm
    outer_radius = outer_diameter / 2.0
    feature_size = feature_size  # mm - actual characteristic wavelength of surface features
    surface_amplitude = feature_size  # mm - actual peak-to-valley amplitude

    # Grid resolution for the surface
    resolution = 150

    # Create coordinate grids
    x = np.linspace(-outer_radius, outer_radius, resolution)
    y = np.linspace(-outer_radius, outer_radius, resolution)
    X, Y = np.meshgrid(x, y)

    # Create circular mask - use full outer radius for the surface
    R = np.sqrt(X ** 2 + Y ** 2)
    mask = R <= outer_radius

    # Generate Gaussian random surface
    np.random.seed(42)
    gaussian_noise = np.random.normal(0, 1.0, (resolution, resolution))

    # Calculate sigma based on feature_size - corrected relationship
    pixel_size = (2 * outer_radius) / resolution  # mm per pixel
    # For Gaussian filter: feature_size ≈ 2.35 * sigma (FWHM relationship)
    # So sigma = feature_size / 2.35
    sigma_pixels = (feature_size / 2.35) / pixel_size  # Convert to pixels

    gaussian_surface = ndimage.gaussian_filter(gaussian_noise, sigma=sigma_pixels)

    # Apply mask first
    gaussian_surface[~mask] = 0

    # Normalize to achieve exact amplitude
    surface_valid = gaussian_surface[mask]
    surface_min = np.min(surface_valid)
    surface_max = np.max(surface_valid)
    current_range = surface_max - surface_min

    # Scale to desired peak-to-valley amplitude
    if current_range > 0:
        gaussian_surface[mask] = (surface_valid - surface_min) / current_range * surface_amplitude - surface_amplitude/2

    # Create vertices and faces lists
    vertices = []
    faces = []

    # Create top surface vertices - Gaussian surface goes all the way to edge
    top_vertex_map = np.full((resolution, resolution), -1, dtype=int)
    vertex_count = 0

    for i in range(resolution):
        for j in range(resolution):
            if mask[i, j]:
                z_height = gaussian_surface[i, j]
                vertices.append([X[i, j], Y[i, j], z_height])
                top_vertex_map[i, j] = vertex_count
                vertex_count += 1

    # Store the number of top vertices for amplitude calculation
    num_top_vertices = vertex_count

    # Create top surface faces
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

    # Create bottom surface vertices (flat circle)
    n_bottom_points = 128
    angles = np.linspace(0, 2 * np.pi, n_bottom_points, endpoint=False)

    # Add center point for bottom
    vertices.append([0, 0, -thickness])
    bottom_center_idx = len(vertices) - 1

    # Add bottom perimeter points
    bottom_perimeter_indices = []
    for angle in angles:
        x_bottom = outer_radius * np.cos(angle)
        y_bottom = outer_radius * np.sin(angle)
        vertices.append([x_bottom, y_bottom, -thickness])
        bottom_perimeter_indices.append(len(vertices) - 1)

    # Create bottom surface faces
    for i in range(n_bottom_points):
        next_i = (i + 1) % n_bottom_points
        faces.append([bottom_center_idx, bottom_perimeter_indices[next_i], bottom_perimeter_indices[i]])

    # Find boundary points of the Gaussian surface for side walls
    surface_boundary_indices = []

    for angle in angles:
        target_x = outer_radius * np.cos(angle)
        target_y = outer_radius * np.sin(angle)

        # Find closest surface point to the target boundary point
        distances = np.sqrt((X - target_x) ** 2 + (Y - target_y) ** 2)
        valid_candidates = mask & (top_vertex_map >= 0)

        if np.any(valid_candidates):
            distances[~valid_candidates] = np.inf
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            i_min, j_min = min_idx

            surface_idx = top_vertex_map[i_min, j_min]
            surface_boundary_indices.append(surface_idx)

    # Create side wall faces - connect surface boundary directly to bottom perimeter
    for i in range(len(surface_boundary_indices)):
        next_i = (i + 1) % len(surface_boundary_indices)

        top1 = surface_boundary_indices[i]
        top2 = surface_boundary_indices[next_i]
        bottom1 = bottom_perimeter_indices[i]
        bottom2 = bottom_perimeter_indices[next_i]

        # Two triangles for each side face
        faces.append([top1, bottom1, top2])
        faces.append([bottom1, bottom2, top2])

    # Create mesh
    vertices = np.array(vertices)
    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Remove degenerate faces and fix mesh (updated methods)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    # Calculate actual amplitude for verification - only use top surface vertices
    top_z_values = vertices[:num_top_vertices, 2]
    actual_min = np.min(top_z_values)
    actual_max = np.max(top_z_values)
    actual_amplitude = actual_max - actual_min

    # Export to STL
    mesh.export('gaussian_cylinder.stl')
    print(f"STL file 'gaussian_cylinder.stl' created successfully!")
    print(f"Mesh has {len(vertices)} vertices and {len(faces)} faces")
    print(f"Feature size: {feature_size}mm (target characteristic wavelength)")
    print(f"Sigma used: {sigma_pixels:.2f} pixels ({sigma_pixels * pixel_size:.2f}mm)")
    print(f"Surface amplitude: {surface_amplitude}mm (target peak-to-valley)")
    print(f"Actual amplitude: {actual_amplitude:.2f}mm (measured peak-to-valley)")


if __name__ == "__main__":
    create_gaussian_surface_cylinder()