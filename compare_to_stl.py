import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh


def plot_stl_file(stl_filename):
    """
    Load and plot an STL file using matplotlib.

    Args:
        stl_filename (str): Path to the STL file
    """
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(stl_filename)

    # Create a new plot
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection='3d')

    # Plot the mesh
    axes.add_collection3d(Poly3DCollection(your_mesh.vectors))

    # Auto scale to the mesh size
    axes.auto_scale_xyz(your_mesh.x.flatten(),
                        your_mesh.y.flatten(),
                        your_mesh.z.flatten())

    # Set labels
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_title('STL File Visualization')

    plt.show()


def plot_stl_wireframe(stl_filename):
    """
    Alternative wireframe visualization of STL file.

    Args:
        stl_filename (str): Path to the STL file
    """
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(stl_filename)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each triangle as wireframe
    for triangle in your_mesh.vectors:
        # Close the triangle by adding the first point at the end
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], triangle_closed[:, 2], 'b-')

    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('STL Wireframe Visualization')

    plt.show()


def convert_2d_surface_to_mesh_vectors(surface, x_max, x_min, y_max, y_min):
    """
    Convert a 2D surface array to mesh vectors (triangular mesh format like STL).

    Args:
        surface: 2D numpy array representing height values
        x_max, x_min, y_max, y_min: Physical dimensions of the surface

    Returns:
        mesh_vectors: Array of triangular faces in the same format as STL mesh.vectors
    """
    nx, ny = surface.shape

    # Create coordinate arrays (matching your plotting code)
    x_mm = np.linspace(y_min, y_max, ny)
    y_mm = np.linspace(x_min, x_max, nx)
    X, Y = np.meshgrid(x_mm, y_mm, indexing='xy')

    # Create triangular mesh from grid
    triangles = []

    # For each grid cell, create two triangles
    for i in range(nx - 1):
        for j in range(ny - 1):
            # Get the four corners of the current grid cell
            # Bottom-left triangle
            v1 = [X[i, j], Y[i, j], surface[i, j]]
            v2 = [X[i + 1, j], Y[i + 1, j], surface[i + 1, j]]
            v3 = [X[i, j + 1], Y[i, j + 1], surface[i, j + 1]]
            triangles.append([v1, v2, v3])

            # Top-right triangle
            v1 = [X[i + 1, j], Y[i + 1, j], surface[i + 1, j]]
            v2 = [X[i + 1, j + 1], Y[i + 1, j + 1], surface[i + 1, j + 1]]
            v3 = [X[i, j + 1], Y[i, j + 1], surface[i, j + 1]]
            triangles.append([v1, v2, v3])

    return np.array(triangles)


def compare_stl_to_3d_grid(grid_data, x_max, x_min, y_max, y_min, z_max, z_min, stl_mesh):
    """
    Compare STL mesh to a 3D voxel grid.

    Args:
        grid_data: 3D numpy array (nx, ny, nz) representing voxel values
        x_max, x_min, y_max, y_min, z_max, z_min: Physical dimensions
        stl_mesh: STL mesh object

    Returns:
        Comparison results and visualizations
    """
    from scipy.spatial import cKDTree

    nx, ny, nz = grid_data.shape

    # Create 3D coordinate arrays
    x_coords = np.linspace(x_min, x_max, nx)
    y_coords = np.linspace(y_min, y_max, ny)
    z_coords = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Create 3D points from grid
    grid_points_3d = np.column_stack([
        X.ravel(),
        Y.ravel(),
        Z.ravel()
    ])
    grid_values = grid_data.ravel()

    print(f"3D Grid: {nx}x{ny}x{nz} = {len(grid_points_3d)} voxels")

    # Get STL vertices and triangle centroids for better coverage
    stl_vertices = stl_mesh.vectors.reshape(-1, 3)
    stl_centroids = np.mean(stl_mesh.vectors, axis=1)
    stl_points = np.vstack([stl_vertices, stl_centroids])

    # Remove duplicates
    stl_points_unique = np.unique(stl_points, axis=0)

    print(f"STL: {len(stl_mesh.vectors)} triangles, {len(stl_points_unique)} unique points")

    # Find overlapping 3D region
    x_min_common = max(x_coords.min(), stl_points_unique[:, 0].min())
    x_max_common = min(x_coords.max(), stl_points_unique[:, 0].max())
    y_min_common = max(y_coords.min(), stl_points_unique[:, 1].min())
    y_max_common = min(y_coords.max(), stl_points_unique[:, 1].max())
    z_min_common = max(z_coords.min(), stl_points_unique[:, 2].min())
    z_max_common = min(z_coords.max(), stl_points_unique[:, 2].max())

    print(f"Common region - X: [{x_min_common:.3f}, {x_max_common:.3f}]")
    print(f"                Y: [{y_min_common:.3f}, {y_max_common:.3f}]")
    print(f"                Z: [{z_min_common:.3f}, {z_max_common:.3f}]")

    # Filter grid points to overlapping region
    grid_mask = ((grid_points_3d[:, 0] >= x_min_common) & (grid_points_3d[:, 0] <= x_max_common) &
                 (grid_points_3d[:, 1] >= y_min_common) & (grid_points_3d[:, 1] <= y_max_common) &
                 (grid_points_3d[:, 2] >= z_min_common) & (grid_points_3d[:, 2] <= z_max_common))

    valid_grid_points = grid_points_3d[grid_mask]
    valid_grid_values = grid_values[grid_mask]

    print(f"Valid grid points in overlap: {len(valid_grid_points)}")

    # Filter STL points to overlapping region
    stl_mask = ((stl_points_unique[:, 0] >= x_min_common) & (stl_points_unique[:, 0] <= x_max_common) &
                (stl_points_unique[:, 1] >= y_min_common) & (stl_points_unique[:, 1] <= y_max_common) &
                (stl_points_unique[:, 2] >= z_min_common) & (stl_points_unique[:, 2] <= z_max_common))

    stl_points_filtered = stl_points_unique[stl_mask]

    print(f"STL points in overlap: {len(stl_points_filtered)}")

    if len(stl_points_filtered) == 0:
        print("No STL points in overlapping region!")
        return None

    # Build KDTree for 3D nearest neighbor search
    tree = cKDTree(stl_points_filtered)

    # Find distances from grid points to nearest STL points
    distances, indices = tree.query(valid_grid_points)

    # Classify grid points based on distance to STL
    distance_threshold = 1.0  # mm - adjust based on your requirements
    grid_near_stl = distances <= distance_threshold

    # Visualization
    fig = plt.figure(figsize=(20, 15))

    # Plot 1: STL mesh
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(stl_points_filtered[:, 0], stl_points_filtered[:, 1], stl_points_filtered[:, 2],
                c='red', s=1, alpha=0.6, label='STL Points')
    ax1.set_title('STL Points in Overlap Region')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot 2: Grid points near STL
    ax2 = fig.add_subplot(232, projection='3d')
    if np.any(grid_near_stl):
        ax2.scatter(valid_grid_points[grid_near_stl, 0],
                    valid_grid_points[grid_near_stl, 1],
                    valid_grid_points[grid_near_stl, 2],
                    c=valid_grid_values[grid_near_stl], cmap='Blues', s=1, alpha=0.6)
    ax2.set_title(f'Grid Points Near STL (≤{distance_threshold}mm)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Plot 3: Both overlaid
    ax3 = fig.add_subplot(233, projection='3d')
    ax3.scatter(stl_points_filtered[:, 0], stl_points_filtered[:, 1], stl_points_filtered[:, 2],
                c='red', s=1, alpha=0.4, label='STL')
    if np.any(grid_near_stl):
        ax3.scatter(valid_grid_points[grid_near_stl, 0],
                    valid_grid_points[grid_near_stl, 1],
                    valid_grid_points[grid_near_stl, 2],
                    c='blue', s=0.5, alpha=0.6, label='Grid')
    ax3.set_title('STL + Grid Overlay')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()

    # Plot 4: Distance histogram
    ax4 = fig.add_subplot(234)
    ax4.hist(distances, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(distance_threshold, color='red', linestyle='--', label=f'Threshold: {distance_threshold}mm')
    ax4.set_xlabel('Distance to Nearest STL Point (mm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Cross-section view (XY plane at middle Z)
    ax5 = fig.add_subplot(235)
    mid_z = (z_min_common + z_max_common) / 2
    z_tolerance = (z_max_common - z_min_common) / nz * 2  # 2 voxel thickness

    # STL points near middle Z
    stl_mid_z = stl_points_filtered[np.abs(stl_points_filtered[:, 2] - mid_z) <= z_tolerance]
    grid_mid_z = valid_grid_points[np.abs(valid_grid_points[:, 2] - mid_z) <= z_tolerance]

    if len(stl_mid_z) > 0:
        ax5.scatter(stl_mid_z[:, 0], stl_mid_z[:, 1], c='red', s=10, alpha=0.6, label='STL')
    if len(grid_mid_z) > 0:
        ax5.scatter(grid_mid_z[:, 0], grid_mid_z[:, 1], c='blue', s=5, alpha=0.6, label='Grid')
    ax5.set_title(f'XY Cross-section at Z≈{mid_z:.1f}mm')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Statistics summary
    ax6 = fig.add_subplot(236)
    ax6.axis('off')

    stats_text = f"""
    Comparison Statistics:

    Total grid points: {len(grid_points_3d):,}
    Grid points in overlap: {len(valid_grid_points):,}
    STL points in overlap: {len(stl_points_filtered):,}

    Distance Analysis:
    Mean distance: {np.mean(distances):.3f} mm
    Std distance: {np.std(distances):.3f} mm
    Min distance: {np.min(distances):.3f} mm
    Max distance: {np.max(distances):.3f} mm

    Threshold Analysis (≤{distance_threshold}mm):
    Points near STL: {np.sum(grid_near_stl):,}
    Percentage: {100 * np.sum(grid_near_stl) / len(valid_grid_points):.1f}%
    """

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print(f"\nDetailed Statistics:")
    print(f"Total grid voxels: {len(grid_points_3d):,}")
    print(f"Valid grid points in overlap: {len(valid_grid_points):,}")
    print(f"STL points in overlap: {len(stl_points_filtered):,}")
    print(f"Mean distance to STL: {np.mean(distances):.3f} mm")
    print(
        f"Points within {distance_threshold}mm: {np.sum(grid_near_stl):,} ({100 * np.sum(grid_near_stl) / len(valid_grid_points):.1f}%)")

    return {
        'distances': distances,
        'grid_points': valid_grid_points,
        'stl_points': stl_points_filtered,
        'near_stl_mask': grid_near_stl,
        'grid_values': valid_grid_values
    }


def rotate_stl(stl_filename, rotate_x=0, rotate_y=0, rotate_z=0, save_rotated=False, output_filename=None):
    """
    Rotate an STL file around X, Y, and/or Z axes.

    Args:
        stl_filename (str): Path to the input STL file
        rotate_x (float): Rotation angle in degrees around X-axis
        rotate_y (float): Rotation angle in degrees around Y-axis
        rotate_z (float): Rotation angle in degrees around Z-axis
        save_rotated (bool): Whether to save the rotated STL to a new file
        output_filename (str): Output filename if saving (optional)

    Returns:
        mesh.Mesh: The rotated mesh object
    """
    # Load the STL file
    your_mesh = mesh.Mesh.from_file(stl_filename)

    # Apply X-axis rotation if specified
    if rotate_x != 0:
        angle_rad = np.radians(rotate_x)
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
        for i in range(len(your_mesh.vectors)):
            for j in range(3):
                your_mesh.vectors[i][j] = np.dot(rotation_matrix_x, your_mesh.vectors[i][j])

    # Apply Y-axis rotation if specified
    if rotate_y != 0:
        angle_rad = np.radians(rotate_y)
        rotation_matrix_y = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
        for i in range(len(your_mesh.vectors)):
            for j in range(3):
                your_mesh.vectors[i][j] = np.dot(rotation_matrix_y, your_mesh.vectors[i][j])

    # Apply Z-axis rotation if specified
    if rotate_z != 0:
        angle_rad = np.radians(rotate_z)
        rotation_matrix_z = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        for i in range(len(your_mesh.vectors)):
            for j in range(3):
                your_mesh.vectors[i][j] = np.dot(rotation_matrix_z, your_mesh.vectors[i][j])

    # Save rotated STL if requested
    if save_rotated:
        if output_filename is None:
            # Generate default output filename
            base_name = stl_filename.rsplit('.', 1)[0]
            rotation_suffix = f"_x{rotate_x}_y{rotate_y}_z{rotate_z}"
            output_filename = f"{base_name}_rotated{rotation_suffix}.stl"

        your_mesh.save(output_filename)
        print(f"Rotated STL saved as: {output_filename}")

    print(f"STL rotated - X: {rotate_x}°, Y: {rotate_y}°, Z: {rotate_z}°")
    return your_mesh



if __name__ == "__main__":
    stl_file = "3d_files/COC_3D_surface_resolution_target.stl"

    # Rotate STL 90 degrees around X-axis
    rotated_mesh = rotate_stl(stl_file, rotate_x=90, save_rotated=False)

    # Create 3D grid
    grid = np.zeros((120, 120, 2000))
    x_max, x_min = 30, -30
    y_max, y_min = 30, -30
    z_max, z_min = 10, 0

    # Compare STL to 3D grid
    results = compare_stl_to_3d_grid(
        grid, x_max, x_min, y_max, y_min, z_max, z_min, rotated_mesh
    )
