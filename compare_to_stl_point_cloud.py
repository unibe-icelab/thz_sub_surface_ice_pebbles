import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from scipy.spatial import cKDTree


def load_point_cloud(filename, delimiter=None, skip_header=0):
    """
    Load point cloud from various file formats.

    Args:
        filename: Path to point cloud file (.txt, .csv, .xyz, .ply, etc.)
        delimiter: Column delimiter (None for auto-detect)
        skip_header: Number of header lines to skip

    Returns:
        numpy array of shape (N, 3) with X, Y, Z coordinates
    """
    try:
        # Try to load as simple text file with coordinates
        if filename.lower().endswith('.ply'):
            # Basic PLY loader (ASCII format)
            with open(filename, 'r') as f:
                lines = f.readlines()

            # Find vertex count and data start
            vertex_count = 0
            data_start = 0
            for i, line in enumerate(lines):
                if 'element vertex' in line:
                    vertex_count = int(line.split()[-1])
                elif 'end_header' in line:
                    data_start = i + 1
                    break

            # Read vertex data
            points = []
            for i in range(data_start, data_start + vertex_count):
                coords = lines[i].strip().split()
                points.append([float(coords[0]), float(coords[1]), float(coords[2])])

            point_cloud = np.array(points)

        else:
            # Load as text file (CSV, TXT, XYZ)
            point_cloud = np.loadtxt(filename, delimiter=delimiter, skiprows=skip_header)

            # Ensure we have at least 3 columns (X, Y, Z)
            if point_cloud.shape[1] < 3:
                raise ValueError(f"Point cloud must have at least 3 columns (X,Y,Z), got {point_cloud.shape[1]}")

            # Take only first 3 columns as coordinates
            point_cloud = point_cloud[:, :3]

        print(f"Loaded point cloud: {filename}")
        print(f"  Points: {len(point_cloud)}")
        print(f"  Bounds: X[{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}]")
        print(f"          Y[{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}]")
        print(f"          Z[{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}]")

        return point_cloud

    except Exception as e:
        print(f"Error loading point cloud {filename}: {e}")
        return None


def load_stl(filename):
    """Load STL file and return mesh object."""
    try:
        stl_mesh = mesh.Mesh.from_file(filename)
        print(f"Loaded STL: {filename}")
        print(f"  Triangles: {len(stl_mesh.vectors)}")
        return stl_mesh
    except Exception as e:
        print(f"Error loading STL {filename}: {e}")
        return None


def rotate_stl(stl_mesh, rotate_x=0, rotate_y=0, rotate_z=0):
    """Rotate STL mesh by specified angles in degrees."""
    if stl_mesh is None:
        return None

    # Create a copy to avoid modifying original
    rotated_mesh = mesh.Mesh(stl_mesh.data.copy())

    # Convert degrees to radians
    rx = np.radians(rotate_x)
    ry = np.radians(rotate_y)
    rz = np.radians(rotate_z)

    # Apply rotations
    if rotate_x != 0:
        rotated_mesh.rotate([1, 0, 0], rx)
    if rotate_y != 0:
        rotated_mesh.rotate([0, 1, 0], ry)
    if rotate_z != 0:
        rotated_mesh.rotate([0, 0, 1], rz)

    print(f"Rotated STL by X:{rotate_x}°, Y:{rotate_y}°, Z:{rotate_z}°")
    return rotated_mesh


def subsample_point_cloud(point_cloud, max_points=50000):
    """Subsample point cloud if it's too large for efficient processing."""
    if len(point_cloud) <= max_points:
        return point_cloud

    # Random sampling
    indices = np.random.choice(len(point_cloud), max_points, replace=False)
    subsampled = point_cloud[indices]

    print(f"Subsampled point cloud from {len(point_cloud)} to {len(subsampled)} points")
    return subsampled


def find_best_alignment(point_cloud, stl_points, search_range=50, step_size=2):
    """
    Find the best translation to align point cloud with STL points.

    Args:
        point_cloud: Point cloud array (N, 3)
        stl_points: STL points array (M, 3)
        search_range: Range to search in each direction (mm)
        step_size: Step size for search grid (mm)

    Returns:
        Best translation (tx, ty, tz) and overlap count
    """
    print("Searching for best alignment...")

    # Create search grid
    search_steps = np.arange(-search_range, search_range + step_size, step_size)
    best_overlap = 0
    best_translation = (0, 0, 0)

    # Get bounds for faster overlap checking
    pc_size = np.array([
        point_cloud[:, 0].max() - point_cloud[:, 0].min(),
        point_cloud[:, 1].max() - point_cloud[:, 1].min(),
        point_cloud[:, 2].max() - point_cloud[:, 2].min()
    ])

    stl_size = np.array([
        stl_points[:, 0].max() - stl_points[:, 0].min(),
        stl_points[:, 1].max() - stl_points[:, 1].min(),
        stl_points[:, 2].max() - stl_points[:, 2].min()
    ])

    total_searches = len(search_steps) ** 3
    search_count = 0

    for tx in search_steps:
        for ty in search_steps:
            for tz in search_steps:
                search_count += 1
                if search_count % 1000 == 0:
                    print(
                        f"  Search progress: {search_count}/{total_searches} ({100 * search_count / total_searches:.1f}%)")

                # Apply translation to point cloud
                translated_pc = point_cloud + np.array([tx, ty, tz])

                # Check overlap using bounding boxes
                pc_min = translated_pc.min(axis=0)
                pc_max = translated_pc.max(axis=0)
                stl_min = stl_points.min(axis=0)
                stl_max = stl_points.max(axis=0)

                # Calculate overlap volume
                overlap_min = np.maximum(pc_min, stl_min)
                overlap_max = np.minimum(pc_max, stl_max)

                # Check if there's any overlap
                if np.all(overlap_max > overlap_min):
                    overlap_volume = np.prod(overlap_max - overlap_min)

                    if overlap_volume > best_overlap:
                        best_overlap = overlap_volume
                        best_translation = (tx, ty, tz)

    print(f"Best translation found: {best_translation}")
    print(f"Overlap volume: {best_overlap:.2f}")

    return best_translation, best_overlap


def apply_translation(points, translation):
    """Apply translation to points."""
    return points + np.array(translation)


def compare_point_cloud_to_stl(point_cloud_file, stl_file,
                               stl_rotation=(0, 0, 0),
                               pc_translation=(0, 0, 0),
                               auto_align=True,
                               max_points=50000,
                               distance_threshold=1.0,
                               point_cloud_delimiter=None,
                               apply_mask=True,
                               mask_radius=20.0):
    """
    Compare a point cloud to an STL file with automatic alignment.

    Args:
        point_cloud_file: Path to point cloud file
        stl_file: Path to STL file
        stl_rotation: Rotation angles (x,y,z) for STL in degrees
        pc_translation: Manual translation for point cloud (x,y,z) in mm
        auto_align: Whether to automatically find best alignment
        max_points: Maximum points to use (for performance)
        distance_threshold: Distance threshold for visualization (mm)
        point_cloud_delimiter: Delimiter for point cloud file
        apply_mask: Whether to apply circular mask to point cloud
        mask_radius: Radius for circular mask in mm

    Returns:
        Dictionary with comparison results
    """

    # Load point cloud
    point_cloud = load_point_cloud(point_cloud_file, delimiter=point_cloud_delimiter)
    if point_cloud is None:
        return None

    # Apply circular mask if requested
    if apply_mask:
        point_cloud = apply_circular_mask(point_cloud, radius=mask_radius)
        if len(point_cloud) == 0:
            print("No points left after applying mask!")
            return None

    # Load STL
    stl_mesh = load_stl(stl_file)
    if stl_mesh is None:
        return None

    # Apply rotation to STL if needed
    if stl_rotation != (0, 0, 0):
        stl_mesh = rotate_stl(stl_mesh, *stl_rotation)

    # Subsample point cloud if needed
    if len(point_cloud) > max_points:
        point_cloud = subsample_point_cloud(point_cloud, max_points)

    # Get STL points (vertices + centroids for better coverage)
    stl_vertices = stl_mesh.vectors.reshape(-1, 3)
    stl_centroids = np.mean(stl_mesh.vectors, axis=1)
    stl_points = np.vstack([stl_vertices, stl_centroids])

    # Remove duplicate STL points
    stl_points_unique = np.unique(stl_points, axis=0)

    print(f"Point cloud: {len(point_cloud)} points")
    print(f"STL: {len(stl_mesh.vectors)} triangles, {len(stl_points_unique)} unique points")

    # Apply manual translation first
    if pc_translation != (0, 0, 0):
        point_cloud = apply_translation(point_cloud, pc_translation)
        print(f"Applied manual translation: {pc_translation}")

    # Auto-align if requested
    if auto_align:
        best_translation, overlap_volume = find_best_alignment(point_cloud, stl_points_unique)
        if overlap_volume > 0:
            point_cloud = apply_translation(point_cloud, best_translation)
            print(f"Applied auto-alignment translation: {best_translation}")
        else:
            print("No good alignment found, proceeding with current positions")

    # Find overlapping region
    pc_bounds = {
        'x_min': point_cloud[:, 0].min(), 'x_max': point_cloud[:, 0].max(),
        'y_min': point_cloud[:, 1].min(), 'y_max': point_cloud[:, 1].max(),
        'z_min': point_cloud[:, 2].min(), 'z_max': point_cloud[:, 2].max()
    }

    stl_bounds = {
        'x_min': stl_points_unique[:, 0].min(), 'x_max': stl_points_unique[:, 0].max(),
        'y_min': stl_points_unique[:, 1].min(), 'y_max': stl_points_unique[:, 1].max(),
        'z_min': stl_points_unique[:, 2].min(), 'z_max': stl_points_unique[:, 2].max()
    }

    # Common region
    x_min_common = max(pc_bounds['x_min'], stl_bounds['x_min'])
    x_max_common = min(pc_bounds['x_max'], stl_bounds['x_max'])
    y_min_common = max(pc_bounds['y_min'], stl_bounds['y_min'])
    y_max_common = min(pc_bounds['y_max'], stl_bounds['y_max'])
    z_min_common = max(pc_bounds['z_min'], stl_bounds['z_min'])
    z_max_common = min(pc_bounds['z_max'], stl_bounds['z_max'])

    print(f"Point cloud bounds: X[{pc_bounds['x_min']:.2f}, {pc_bounds['x_max']:.2f}], "
          f"Y[{pc_bounds['y_min']:.2f}, {pc_bounds['y_max']:.2f}], "
          f"Z[{pc_bounds['z_min']:.2f}, {pc_bounds['z_max']:.2f}]")

    print(f"STL bounds: X[{stl_bounds['x_min']:.2f}, {stl_bounds['x_max']:.2f}], "
          f"Y[{stl_bounds['y_min']:.2f}, {stl_bounds['y_max']:.2f}], "
          f"Z[{stl_bounds['z_min']:.2f}, {stl_bounds['z_max']:.2f}]")

    print(f"Overlap region: X[{x_min_common:.2f}, {x_max_common:.2f}], "
          f"Y[{y_min_common:.2f}, {y_max_common:.2f}], "
          f"Z[{z_min_common:.2f}, {z_max_common:.2f}]")

    # Check if overlap exists
    if (x_max_common <= x_min_common or
            y_max_common <= y_min_common or
            z_max_common <= z_min_common):
        print("No overlapping region found even after alignment!")

        # Show suggestions for manual alignment
        center_pc = np.mean(point_cloud, axis=0)
        center_stl = np.mean(stl_points_unique, axis=0)
        suggested_translation = center_stl - center_pc

        print(f"Suggested manual translation: {suggested_translation}")
        print("Try running again with pc_translation parameter or adjust stl_rotation")
        return None

    # Filter points to overlapping region
    pc_mask = ((point_cloud[:, 0] >= x_min_common) & (point_cloud[:, 0] <= x_max_common) &
               (point_cloud[:, 1] >= y_min_common) & (point_cloud[:, 1] <= y_max_common) &
               (point_cloud[:, 2] >= z_min_common) & (point_cloud[:, 2] <= z_max_common))

    stl_mask = ((stl_points_unique[:, 0] >= x_min_common) & (stl_points_unique[:, 0] <= x_max_common) &
                (stl_points_unique[:, 1] >= y_min_common) & (stl_points_unique[:, 1] <= y_max_common) &
                (stl_points_unique[:, 2] >= z_min_common) & (stl_points_unique[:, 2] <= z_max_common))

    pc_filtered = point_cloud[pc_mask]
    stl_filtered = stl_points_unique[stl_mask]

    print(f"Points in overlap: Point cloud={len(pc_filtered)}, STL={len(stl_filtered)}")

    if len(pc_filtered) == 0 or len(stl_filtered) == 0:
        print("No overlapping points found!")
        return None

    # Build KDTree for STL points
    stl_tree = cKDTree(stl_filtered)

    # Find distances from point cloud to nearest STL points
    distances_pc_to_stl, indices_pc_to_stl = stl_tree.query(pc_filtered)

    # Build KDTree for point cloud
    pc_tree = cKDTree(pc_filtered)

    # Find distances from STL to nearest point cloud points
    distances_stl_to_pc, indices_stl_to_pc = pc_tree.query(stl_filtered)

    # Calculate statistics
    mean_distance_pc_to_stl = np.mean(distances_pc_to_stl)
    mean_distance_stl_to_pc = np.mean(distances_stl_to_pc)
    rms_error = np.sqrt((mean_distance_pc_to_stl ** 2 + mean_distance_stl_to_pc ** 2) / 2)

    # Create results dictionary
    results = {
        'point_cloud': pc_filtered,
        'stl_points': stl_filtered,
        'distances_pc_to_stl': distances_pc_to_stl,
        'distances_stl_to_pc': distances_stl_to_pc,
        'mean_distance_pc_to_stl': mean_distance_pc_to_stl,
        'mean_distance_stl_to_pc': mean_distance_stl_to_pc,
        'rms_error': rms_error,
        'max_distance_pc_to_stl': np.max(distances_pc_to_stl),
        'max_distance_stl_to_pc': np.max(distances_stl_to_pc),
        'distance_threshold': distance_threshold,
        'pc_bounds': pc_bounds,
        'stl_bounds': stl_bounds
    }

    # Visualize results
    visualize_comparison(results)

    return results


def visualize_comparison(results):
    """Visualize point cloud to STL comparison results."""

    point_cloud = results['point_cloud']
    stl_points = results['stl_points']
    distances_pc_to_stl = results['distances_pc_to_stl']
    distances_stl_to_pc = results['distances_stl_to_pc']
    distance_threshold = results['distance_threshold']

    fig = plt.figure(figsize=(20, 15))

    # Plot 1: Both point clouds overlaid
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                c='blue', s=1, alpha=0.6, label='Point Cloud')
    ax1.scatter(stl_points[:, 0], stl_points[:, 1], stl_points[:, 2],
                c='red', s=1, alpha=0.6, label='STL')
    ax1.set_title('Point Cloud vs STL (Overlap Region)')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.legend()

    # Plot 2: Point cloud colored by distance to STL
    ax2 = fig.add_subplot(232, projection='3d')
    scatter = ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                          c=distances_pc_to_stl, cmap='viridis', s=2, alpha=0.8)
    ax2.set_title('Point Cloud (colored by distance to STL)')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    plt.colorbar(scatter, ax=ax2, shrink=0.5, label='Distance (mm)')

    # Plot 3: STL points colored by distance to point cloud
    ax3 = fig.add_subplot(233, projection='3d')
    scatter = ax3.scatter(stl_points[:, 0], stl_points[:, 1], stl_points[:, 2],
                          c=distances_stl_to_pc, cmap='plasma', s=2, alpha=0.8)
    ax3.set_title('STL Points (colored by distance to Point Cloud)')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    plt.colorbar(scatter, ax=ax3, shrink=0.5, label='Distance (mm)')

    # Plot 4: Distance histogram PC→STL
    ax4 = fig.add_subplot(234)
    ax4.hist(distances_pc_to_stl, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(np.mean(distances_pc_to_stl), color='red', linestyle='--',
                label=f'Mean: {np.mean(distances_pc_to_stl):.3f}mm')
    ax4.axvline(distance_threshold, color='orange', linestyle='--',
                label=f'Threshold: {distance_threshold}mm')
    ax4.set_xlabel('Distance (mm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Point Cloud→STL Distance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distance histogram STL→PC
    ax5 = fig.add_subplot(235)
    ax5.hist(distances_stl_to_pc, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax5.axvline(np.mean(distances_stl_to_pc), color='blue', linestyle='--',
                label=f'Mean: {np.mean(distances_stl_to_pc):.3f}mm')
    ax5.axvline(distance_threshold, color='orange', linestyle='--',
                label=f'Threshold: {distance_threshold}mm')
    ax5.set_xlabel('Distance (mm)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('STL→Point Cloud Distance Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Statistics summary
    ax6 = fig.add_subplot(236)
    ax6.axis('off')

    # Count points within threshold
    pc_within_threshold = np.sum(distances_pc_to_stl <= distance_threshold)
    stl_within_threshold = np.sum(distances_stl_to_pc <= distance_threshold)

    stats_text = f"""
    Point Cloud to STL Comparison:

    Points compared:
    Point Cloud: {len(point_cloud):,}
    STL: {len(stl_points):,}

    Distance Analysis:
    PC→STL mean: {results['mean_distance_pc_to_stl']:.3f} mm
    STL→PC mean: {results['mean_distance_stl_to_pc']:.3f} mm

    Max distances:
    PC→STL max: {results['max_distance_pc_to_stl']:.3f} mm
    STL→PC max: {results['max_distance_stl_to_pc']:.3f} mm

    Standard deviations:
    PC→STL std: {np.std(distances_pc_to_stl):.3f} mm
    STL→PC std: {np.std(distances_stl_to_pc):.3f} mm

    Overall RMS error: {results['rms_error']:.3f} mm

    Within threshold ({distance_threshold}mm):
    PC points: {pc_within_threshold:,} ({100 * pc_within_threshold / len(point_cloud):.1f}%)
    STL points: {stl_within_threshold:,} ({100 * stl_within_threshold / len(stl_points):.1f}%)
    """

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig('compare_to_stl_point_cloud.png')
    plt.show()


def apply_circular_mask(point_cloud, center_x=None, center_y=None, radius=29.0):
    """
    Apply a circular mask to keep only points within a specified radius in XY plane.

    Args:
        point_cloud: Point cloud array (N, 3)
        center_x: X center of circle (None for point cloud center)
        center_y: Y center of circle (None for point cloud center)
        radius: Radius in mm to keep points within

    Returns:
        Filtered point cloud array
    """
    if center_x is None:
        center_x = (point_cloud[:, 0].min() + point_cloud[:, 0].max()) / 2
    if center_y is None:
        center_y = (point_cloud[:, 1].min() + point_cloud[:, 1].max()) / 2

    print(f"Applying circular mask: center=({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}mm")

    # Calculate distances from center in XY plane
    distances_xy = np.sqrt((point_cloud[:, 0] - center_x) ** 2 + (point_cloud[:, 1] - center_y) ** 2)

    # Create mask for points within radius
    mask = distances_xy <= radius

    # Apply mask
    filtered_points = point_cloud[mask]

    print(f"Filtered point cloud: {len(point_cloud)} -> {len(filtered_points)} points "
          f"({100 * len(filtered_points) / len(point_cloud):.1f}% kept)")

    return filtered_points


if __name__ == "__main__":
    # Define your files
    point_cloud_file = "extracted_surface_pointcloud.ply"
    stl_file = "3d_files/COC_spheres_10mm_half.stl"

    # Compare point cloud to STL with auto-alignment
    results = compare_point_cloud_to_stl(
        point_cloud_file=point_cloud_file,
        stl_file=stl_file,
        stl_rotation=(-90, 0, 0),  # Rotate STL if needed
        pc_translation=(4, 9, -0.5),  # Manual translation if needed
        auto_align=False,  # Automatically find best alignment
        max_points=50000,
        distance_threshold=1.0,
        point_cloud_delimiter=None
    )

    if results:
        print(f"\nComparison completed successfully!")
        print(f"Mean distance PC→STL: {results['mean_distance_pc_to_stl']:.3f} mm")
        print(f"Mean distance STL→PC: {results['mean_distance_stl_to_pc']:.3f} mm")
        print(f"Overall RMS error: {results['rms_error']:.3f} mm")
