# surface_geometry/surface_area.py

import numpy as np
from scipy.spatial import Delaunay

def calculate_surface_area(dem, resolution_x, resolution_y):
    """
    Calculate surface area of a DEM using triangulation
    
    Parameters:
    -----------
    dem : 2D numpy array
        Digital Elevation Model data
    resolution_x, resolution_y : float
        Resolution of the DEM in x and y dimensions
        
    Returns:
    --------
    float
        Surface area
    """
    rows, cols = dem.shape
    
    # Create coordinate grid
    y, x = np.mgrid[0:rows, 0:cols]
    x = x * resolution_x
    y = y * resolution_y
    
    # Flatten coordinates and values
    points = np.column_stack((x.flatten(), y.flatten()))
    z = dem.flatten()
    
    # Remove NaN points
    valid = ~np.isnan(z)
    points = points[valid]
    z = z[valid]
    
    # Add z coordinate
    points_3d = np.column_stack((points, z))
    
    # Calculate surface area using triangulation
    try:
        # Create triangulation
        tri = Delaunay(points)
        
        # Get triangles
        triangles = points_3d[tri.simplices]
        
        # Calculate area of each triangle
        surface_area = 0
        for triangle in triangles:
            # Calculate vectors for two sides of the triangle
            v1 = triangle[1] - triangle[0]
            v2 = triangle[2] - triangle[0]
            
            # Calculate cross product
            cross = np.cross(v1, v2)
            
            # Calculate area
            area = 0.5 * np.linalg.norm(cross)
            surface_area += area
            
        return surface_area
    except:
        # Return None if triangulation fails
        return None