import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.spatial import cKDTree


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


def get_stl_bounds(stl_mesh):
    """Get bounding box of STL mesh."""
    if stl_mesh is None:
        return None

    vertices = stl_mesh.vectors.reshape(-1, 3)
    bounds = {
        'x_min': vertices[:, 0].min(),
        'x_max': vertices[:, 0].max(),
        'y_min': vertices[:, 1].min(),
        'y_max': vertices[:, 1].max(),
        'z_min': vertices[:, 2].min(),
        'z_max': vertices[:, 2].max()
    }
    return bounds


def compare_two_stl_files(stl_file1, stl_file2, rotate1=(0, 0, 0), rotate2=(0, 0, 0),
                          comparison_method='nearest_neighbor'):
    """
    Compare two STL files using different methods.

    Args:
        stl_file1: Path to first STL file (reference)
        stl_file2: Path to second STL file (comparison)
        rotate1: Rotation angles (x,y,z) for first STL in degrees
        rotate2: Rotation angles (x,y,z) for second STL in degrees
        comparison_method: 'nearest_neighbor' or 'hausdorff'

    Returns:
        Dictionary with comparison results
    """

    # Load STL files
    mesh1 = load_stl(stl_file1)
    mesh2 = load_stl(stl_file2)

    if mesh1 is None or mesh2 is None:
        print("Failed to load STL files")
        return None

    # Apply rotations
    if rotate1 != (0, 0, 0):
        mesh1 = rotate_stl(mesh1, *rotate1)
    if rotate2 != (0, 0, 0):
        mesh2 = rotate_stl(mesh2, *rotate2)

    # Get vertices and centroids for better point coverage
    vertices1 = mesh1.vectors.reshape(-1, 3)
    vertices2 = mesh2.vectors.reshape(-1, 3)

    centroids1 = np.mean(mesh1.vectors, axis=1)
    centroids2 = np.mean(mesh2.vectors, axis=1)

    # Combine vertices and centroids
    points1 = np.vstack([vertices1, centroids1])
    points2 = np.vstack([vertices2, centroids2])

    # Remove duplicate points
    points1 = np.unique(points1, axis=0)
    points2 = np.unique(points2, axis=0)

    print(f"STL 1: {len(mesh1.vectors)} triangles, {len(points1)} unique points")
    print(f"STL 2: {len(mesh2.vectors)} triangles, {len(points2)} unique points")

    # Get bounds
    bounds1 = get_stl_bounds(mesh1)
    bounds2 = get_stl_bounds(mesh2)

    print(f"STL 1 bounds: X[{bounds1['x_min']:.2f}, {bounds1['x_max']:.2f}], "
          f"Y[{bounds1['y_min']:.2f}, {bounds1['y_max']:.2f}], "
          f"Z[{bounds1['z_min']:.2f}, {bounds1['z_max']:.2f}]")

    print(f"STL 2 bounds: X[{bounds2['x_min']:.2f}, {bounds2['x_max']:.2f}], "
          f"Y[{bounds2['y_min']:.2f}, {bounds2['y_max']:.2f}], "
          f"Z[{bounds2['z_min']:.2f}, {bounds2['z_max']:.2f}]")

    # Find overlapping region
    x_min_common = max(bounds1['x_min'], bounds2['x_min'])
    x_max_common = min(bounds1['x_max'], bounds2['x_max'])
    y_min_common = max(bounds1['y_min'], bounds2['y_min'])
    y_max_common = min(bounds1['y_max'], bounds2['y_max'])
    z_min_common = max(bounds1['z_min'], bounds2['z_min'])
    z_max_common = min(bounds1['z_max'], bounds2['z_max'])

    print(f"Overlap region: X[{x_min_common:.2f}, {x_max_common:.2f}], "
          f"Y[{y_min_common:.2f}, {y_max_common:.2f}], "
          f"Z[{z_min_common:.2f}, {z_max_common:.2f}]")

    # Filter points to overlapping region
    mask1 = ((points1[:, 0] >= x_min_common) & (points1[:, 0] <= x_max_common) &
             (points1[:, 1] >= y_min_common) & (points1[:, 1] <= y_max_common) &
             (points1[:, 2] >= z_min_common) & (points1[:, 2] <= z_max_common))

    mask2 = ((points2[:, 0] >= x_min_common) & (points2[:, 0] <= x_max_common) &
             (points2[:, 1] >= y_min_common) & (points2[:, 1] <= y_max_common) &
             (points2[:, 2] >= z_min_common) & (points2[:, 2] <= z_max_common))

    points1_filtered = points1[mask1]
    points2_filtered = points2[mask2]

    print(f"Points in overlap: STL1={len(points1_filtered)}, STL2={len(points2_filtered)}")

    if len(points1_filtered) == 0 or len(points2_filtered) == 0:
        print("No overlapping points found!")
        return None

    # Perform comparison based on method
    if comparison_method == 'nearest_neighbor':
        # Build KDTree for STL2
        tree2 = cKDTree(points2_filtered)

        # Find distances from STL1 points to nearest STL2 points
        distances_1to2, indices_1to2 = tree2.query(points1_filtered)

        # Build KDTree for STL1
        tree1 = cKDTree(points1_filtered)

        # Find distances from STL2 points to nearest STL1 points
        distances_2to1, indices_2to1 = tree1.query(points2_filtered)

        # Symmetric distance (both directions)
        mean_distance_1to2 = np.mean(distances_1to2)
        mean_distance_2to1 = np.mean(distances_2to1)

        results = {
            'method': 'nearest_neighbor',
            'distances_1to2': distances_1to2,
            'distances_2to1': distances_2to1,
            'mean_distance_1to2': mean_distance_1to2,
            'mean_distance_2to1': mean_distance_2to1,
            'points1': points1_filtered,
            'points2': points2_filtered,
            'rms_error': np.sqrt((mean_distance_1to2 ** 2 + mean_distance_2to1 ** 2) / 2),
            'max_distance_1to2': np.max(distances_1to2),
            'max_distance_2to1': np.max(distances_2to1),
        }

    elif comparison_method == 'hausdorff':
        # Hausdorff distance (maximum of minimum distances)
        tree2 = cKDTree(points2_filtered)
        tree1 = cKDTree(points1_filtered)

        distances_1to2, _ = tree2.query(points1_filtered)
        distances_2to1, _ = tree1.query(points2_filtered)

        hausdorff_1to2 = np.max(distances_1to2)
        hausdorff_2to1 = np.max(distances_2to1)
        hausdorff_distance = max(hausdorff_1to2, hausdorff_2to1)

        results = {
            'method': 'hausdorff',
            'distances_1to2': distances_1to2,
            'distances_2to1': distances_2to1,
            'hausdorff_1to2': hausdorff_1to2,
            'hausdorff_2to1': hausdorff_2to1,
            'hausdorff_distance': hausdorff_distance,
            'points1': points1_filtered,
            'points2': points2_filtered,
            'mean_distance_1to2': np.mean(distances_1to2),
            'mean_distance_2to1': np.mean(distances_2to1),
        }

    # Visualize results
    visualize_stl_comparison(results, bounds1, bounds2)

    return results


def visualize_stl_comparison(results, bounds1, bounds2):
    """Visualize STL comparison results."""

    fig = plt.figure(figsize=(20, 15))

    points1 = results['points1']
    points2 = results['points2']
    distances_1to2 = results['distances_1to2']
    distances_2to1 = results['distances_2to1']

    # Plot 1: Both STL point clouds
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2],
                c='blue', s=1, alpha=0.6, label='STL 1')
    ax1.scatter(points2[:, 0], points2[:, 1], points2[:, 2],
                c='red', s=1, alpha=0.6, label='STL 2')
    ax1.set_title('Both STL Files (Overlap Region)')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.legend()

    # Plot 2: Distance-colored STL1 points
    ax2 = fig.add_subplot(232, projection='3d')
    scatter = ax2.scatter(points1[:, 0], points1[:, 1], points1[:, 2],
                          c=distances_1to2, cmap='viridis', s=2, alpha=0.8)
    ax2.set_title('STL1 Points (colored by distance to STL2)')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    plt.colorbar(scatter, ax=ax2, shrink=0.5, label='Distance (mm)')

    # Plot 3: Distance-colored STL2 points
    ax3 = fig.add_subplot(233, projection='3d')
    scatter = ax3.scatter(points2[:, 0], points2[:, 1], points2[:, 2],
                          c=distances_2to1, cmap='plasma', s=2, alpha=0.8)
    ax3.set_title('STL2 Points (colored by distance to STL1)')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    plt.colorbar(scatter, ax=ax3, shrink=0.5, label='Distance (mm)')

    # Plot 4: Distance histogram STL1->STL2
    ax4 = fig.add_subplot(234)
    ax4.hist(distances_1to2, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(np.mean(distances_1to2), color='red', linestyle='--',
                label=f'Mean: {np.mean(distances_1to2):.3f}mm')
    ax4.set_xlabel('Distance (mm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('STL1→STL2 Distance Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distance histogram STL2->STL1
    ax5 = fig.add_subplot(235)
    ax5.hist(distances_2to1, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax5.axvline(np.mean(distances_2to1), color='blue', linestyle='--',
                label=f'Mean: {np.mean(distances_2to1):.3f}mm')
    ax5.set_xlabel('Distance (mm)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('STL2→STL1 Distance Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Statistics summary
    ax6 = fig.add_subplot(236)
    ax6.axis('off')

    stats_text = f"""
    STL Comparison Statistics:

    Method: {results['method']}

    Points compared:
    STL1: {len(points1):,}
    STL2: {len(points2):,}

    Distance Analysis:
    STL1→STL2 mean: {results['mean_distance_1to2']:.3f} mm
    STL2→STL1 mean: {results['mean_distance_2to1']:.3f} mm

    Max distances:
    STL1→STL2 max: {np.max(distances_1to2):.3f} mm
    STL2→STL1 max: {np.max(distances_2to1):.3f} mm

    Standard deviations:
    STL1→STL2 std: {np.std(distances_1to2):.3f} mm
    STL2→STL1 std: {np.std(distances_2to1):.3f} mm

    Overall RMS error: {results.get('rms_error', 0):.3f} mm
    """

    if results['method'] == 'hausdorff':
        stats_text += f"""
    Hausdorff distances:
    STL1→STL2: {results['hausdorff_1to2']:.3f} mm
    STL2→STL1: {results['hausdorff_2to1']:.3f} mm
    Overall: {results['hausdorff_distance']:.3f} mm
        """

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define your STL files
    stl_file1 = "3d_files/COC_spheres_10mm_half.stl"  # Reference STL
    stl_file2 = "extracted_surface.stl"  # STL to compare

    # Compare STL files
    results = compare_two_stl_files(
        stl_file1, stl_file2,
        rotate1=(90, 0, 0),  # Rotate first STL if needed
        rotate2=(0, 0, 0),  # Rotate second STL if needed
        comparison_method='nearest_neighbor'  # or 'hausdorff'
    )

    if results:
        print(f"\nComparison completed successfully!")
        print(f"Mean distance STL1→STL2: {results['mean_distance_1to2']:.3f} mm")
        print(f"Mean distance STL2→STL1: {results['mean_distance_2to1']:.3f} mm")

        if 'rms_error' in results:
            print(f"Overall RMS error: {results['rms_error']:.3f} mm")

        if 'hausdorff_distance' in results:
            print(f"Hausdorff distance: {results['hausdorff_distance']:.3f} mm")
