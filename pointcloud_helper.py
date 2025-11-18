from typing import Optional

import numpy as np


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


import numpy as np
from typing import Optional

def save_surface_as_point_cloud(
    surface,
    x_max,
    x_min,
    y_max,
    y_min,
    filename,
    mask_radius: Optional[float] = None,
    format='txt',
    rotation_angles: Optional[np.ndarray] = None,  # [rx, ry, rz] in radians
    translation_vector: Optional[np.ndarray] = None
):
    """
    Save 2D surface data as a point cloud file, with optional rotation (angles) and translation.

    Args:
        surface: 2D numpy array with height values
        x_max, x_min, y_max, y_min: Physical dimensions in mm
        filename: Output filename
        format: 'txt', 'csv', 'xyz', or 'ply'
        rotation_angles: Optional 3-element array for rotation (radians) around X, Y, Z
        translation_vector: Optional 3-element numpy array for translation
    """
    nx, ny = surface.shape
    surface -= np.nanmin(surface)

    x_mm = np.linspace(y_min, y_max, ny)
    y_mm = np.linspace(-x_min, -x_max, nx)
    X, Y = np.meshgrid(x_mm, y_mm, indexing='xy')

    x_points = X.flatten()
    y_points = Y.flatten()
    z_points = surface.flatten()

    valid_mask = np.isfinite(z_points)
    x_points, y_points, z_points = x_points[valid_mask], y_points[valid_mask], z_points[valid_mask]

    if mask_radius is not None:
        width = x_max - x_min
        height = y_max - y_min
        radius = mask_radius * np.min([width, height]) / 2.0
        center_x = (y_max + y_min) / 2.0
        center_y = -(x_max + x_min) / 2.0
        mask = (x_points - center_x) ** 2 + (y_points - center_y) ** 2 <= radius ** 2
        x_points, y_points, z_points = x_points[mask], y_points[mask], z_points[mask]

    point_cloud = np.column_stack([x_points, y_points, z_points])

    # Apply rotation if provided
    if rotation_angles is not None:
        rx, ry, rz = rotation_angles
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        point_cloud = point_cloud @ R.T

    # Apply translation if provided
    if translation_vector is not None:
        point_cloud = point_cloud + translation_vector

    print(f"Saving {len(point_cloud)} masked points to {filename}")
    print(f"Point cloud bounds: X[{point_cloud[:,0].min():.2f}, {point_cloud[:,0].max():.2f}], "
          f"Y[{point_cloud[:,1].min():.2f}, {point_cloud[:,1].max():.2f}], "
          f"Z[{point_cloud[:,2].min():.2f}, {point_cloud[:,2].max():.2f}]")

    fmt = format.lower()
    if fmt in ('txt', 'csv', 'xyz'):
        delim = {'txt': ' ', 'csv': ',', 'xyz': ' '}[fmt]
        header = {'txt': 'X Y Z', 'csv': 'X,Y,Z', 'xyz': ''}[fmt]
        comments = {'txt': '# ', 'csv': '', 'xyz': ''}[fmt]
        np.savetxt(filename, point_cloud, fmt='%.6f', delimiter=delim, header=header, comments=comments)
    elif fmt == 'ply':
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(point_cloud)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            np.savetxt(f, point_cloud, fmt="%.6f")
    else:
        raise ValueError("Format must be 'txt', 'csv', 'xyz', or 'ply'")

    print("✅ Point cloud saved successfully!")
    return point_cloud[:,0], point_cloud[:,1], point_cloud[:,2]
