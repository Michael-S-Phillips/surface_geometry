# surface_geometry/functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from scipy.spatial import ConvexHull
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import Polygon, Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def R_func(H0, L0):
    """Calculate rugosity from theory"""
    return np.sqrt((H0**2) / (2 * L0**2) + 1)

def D_func(H, R, L, L0):
    """Calculate fractal dimension from theory"""
    return 3 - np.log10(H / (np.sqrt(2) * L0 * np.sqrt(R**2 - 1))) / np.log10(L / L0)

def HL0_func(D, R, L, L0):
    """Calculate height range (normalized) based on fractal dimension and rugosity"""
    return (3 - D) * np.log10(L/L0) + 0.5 * np.log10(R**2 - 1)

def fd_func(x, y, s, data, x0, y0):
    """Calculate height range within a window"""
    # Create a window for the specified area
    window = Window(
        col_off=int((x0 + x - data.bounds.left) / data.res[0]),
        row_off=int((data.bounds.top - (y0 + y + s)) / data.res[1]),
        width=int(s / data.res[0]),
        height=int(s / data.res[1])
    )
    
    # Read the data in the window
    window_data = data.read(1, window=window)
    
    # Calculate the height range (max - min)
    if window_data.size > 0:
        return np.nanmax(window_data) - np.nanmin(window_data)
    else:
        return np.nan

def height_variation(data, x0, y0, L, scl, L0, output="output", rep=1, write=True, return_data=False):
    """
    Calculate height variation at different scales within a patch
    
    Parameters:
    -----------
    data : rasterio.DatasetReader
        Raster dataset (DEM)
    x0, y0 : float
        Bottom-left coordinates of the patch
    L : float
        Scope/extent of the patch
    scl : list or array
        Scales at which to calculate height variation
    L0 : float
        Grain/resolution of processing
    output : str
        Output directory name
    rep : int
        Repetition number for file naming
    write : bool
        Whether to write results to file
    return_data : bool
        Whether to return results as DataFrame
        
    Returns:
    --------
    DataFrame with height variation at different scales
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(f"output/{output}", exist_ok=True)
    
    # Initialize empty DataFrame
    temp = pd.DataFrame()
    
    # Calculate height variation at each scale
    for s in scl:
        inc = np.arange(0, L-s+0.001, s)  # Add small value to handle floating point
        x_vals = np.repeat(inc, int(L/s))  # Use L/s as in original R code
        y_vals = np.tile(inc, int(L/s))    # Use L/s as in original R code
        
        # Calculate height range for each window
        H0_vals = []
        for x, y in zip(x_vals, y_vals):
            H0_vals.append(fd_func(x, y, s, data, x0, y0))
        
        # Add to DataFrame
        scale_df = pd.DataFrame({
            'L0': s,
            'x': x_vals,
            'y': y_vals,
            'H0': H0_vals
        })
        temp = pd.concat([temp, scale_df], ignore_index=True)
    
    # Write to file if requested
    if write:
        filename = f"output/{output}/var_{os.path.basename(data.name)}_{rep:04d}.csv"
        temp.to_csv(filename, index=False)
        print(f"Complete: {os.path.basename(data.name)}_{rep:04d}")
    
    # Return data if requested
    if return_data:
        return temp
    
def rdh(hvar, L, L0, data=None, x0=None, y0=None):
    """
    Calculate rugosity, fractal dimension, and height range
    
    Parameters:
    -----------
    hvar : DataFrame
        Output from height_variation function
    L : float
        Scope/extent of the patch
    L0 : float
        Grain/resolution of processing
    data : rasterio.DatasetReader, optional
        Raster dataset for surface area calculation
    x0, y0 : float, optional
        Bottom-left coordinates for surface area calculation
        
    Returns:
    --------
    Dictionary with calculated metrics
    """
    # Log10 transform
    hvar = hvar.copy()
    hvar['H0'] = np.log10(hvar['H0'])
    hvar['L0'] = np.log10(hvar['L0'])
    
    # Mean of scales to avoid biased sampling at smaller scales
    hvar_m = hvar.groupby('L0')['H0'].mean().reset_index()
    
    # Find height ranges at both ends of the scale
    H = 10**hvar_m.loc[hvar_m['L0'] == np.log10(L), 'H0'].values[0]
    H0 = 10**hvar_m.loc[hvar_m['L0'] == np.log10(L0), 'H0'].values[0]
    
    # Re-centering
    hvar_m['H0'] = hvar_m['H0'] - hvar_m.loc[hvar_m['L0'] == np.log10(L), 'H0'].values[0]
    hvar_m['L0'] = hvar_m['L0'] - np.log10(L)
    
    # Calculate slopes and minus from 3 (to get D from S)
    # Fit linear model
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(hvar_m['L0'], hvar_m['H0'])
    D = 3 - slope
    
    # Calculate D from endpoints only - using first and last rows like in R
    ends = hvar_m.iloc[[0, hvar_m.shape[0]-1]]
    slope_ends, intercept_ends, r_value_ends, p_value_ends, std_err_ends = stats.linregress(ends['L0'], ends['H0'])
    D_ends = 3 - slope_ends
    
    # Calculate rugosity from theory
    R_theory = R_func(H0, L0)
    
    # Calculate rugosity from theory (integral)
    HL0_values = 10**hvar.loc[hvar['L0'] == min(hvar['L0']), 'H0'].values
    R_theory2 = sum(R_func(HL0_values, L0)) / (2/L0)**2
    
    # Calculate R using surface area (if available)
    R = None
    if data is not None and hasattr(data, 'read') and x0 is not None and y0 is not None:
        try:
            from .surface_area import calculate_surface_area
            # Get the resolution
            resolution_x, resolution_y = data.res
            
            # Create window
            from rasterio.windows import from_bounds
            window = from_bounds(x0, y0, x0 + L, y0 + L, data.transform)
            
            # Read the data
            patch_data = data.read(1, window=window)
            
            # Calculate surface area
            surface_area = calculate_surface_area(patch_data, resolution_x, resolution_y)
            
            # Calculate rugosity (R)
            if surface_area is not None:
                R = surface_area / (L**2)
        except Exception as e:
            print(f"Warning: Could not calculate surface area: {e}")
            R = None
    
    # Calculate D from theory
    D_theory = D_func(H, R_theory, L, L0)
    
    return {
        'D': D,
        'D_ends': D_ends,
        'D_theory': D_theory,
        'R': R,
        'R_theory': R_theory,
        'R_theory2': R_theory2,
        'H': H
    }

def rescale(x, x0, xm, n):
    """
    Rescale values from one range to another
    
    Parameters:
    -----------
    x : array_like
        Values to rescale
    x0 : float
        Lower bound of original range
    xm : float
        Upper bound of original range
    n : float
        Upper bound of new range (with lower bound 0)
        
    Returns:
    --------
    array_like
        Rescaled values
    """
    return (x - x0) / (xm - x0) * n

def count_filled_cells(dat, xmin, xmax, ymin, ymax, zmin, zmax, ngrid):
    """
    Count number of filled cells in a 3D grid
    
    Parameters:
    -----------
    dat : array_like
        3D point coordinates (N x 3)
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Bounds of the 3D space
    ngrid : int
        Number of grid cells along each dimension
        
    Returns:
    --------
    int
        Number of filled cells
    """
    import numpy as np
    
    # Rescale coordinates to grid indices
    x_idx = np.ceil(rescale(dat[:, 0], xmin, xmax, ngrid)).astype(int)
    y_idx = np.ceil(rescale(dat[:, 1], ymin, ymax, ngrid)).astype(int)
    z_idx = np.ceil(rescale(dat[:, 2], zmin, zmax, ngrid)).astype(int)
    
    # Combine indices to create a unique cell identifier
    # Ensure indices are within bounds (1 to ngrid)
    x_idx = np.clip(x_idx, 1, ngrid)
    y_idx = np.clip(y_idx, 1, ngrid)
    z_idx = np.clip(z_idx, 1, ngrid)
    
    # Create 3D grid
    grid = np.zeros((ngrid+1, ngrid+1, ngrid+1), dtype=bool)
    
    # Mark occupied cells
    for i in range(len(x_idx)):
        grid[x_idx[i], y_idx[i], z_idx[i]] = True
    
    # Count filled cells
    return np.sum(grid)