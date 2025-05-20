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
    return 10**(0.5 * np.log10(R**2 - 1) + (3 - D) * np.log10(L/L0))

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
        x_vals = np.repeat(inc, len(inc))
        y_vals = np.tile(inc, len(inc))
        
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
    
def rdh(hvar, L, L0):
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
    
    # Calculate D from endpoints only
    ends = hvar_m.iloc[[0, -1]]
    slope_ends, intercept_ends, r_value_ends, p_value_ends, std_err_ends = stats.linregress(ends['L0'], ends['H0'])
    D_ends = 3 - slope_ends
    
    # Calculate rugosity from theory
    R_theory = R_func(H0, L0)
    
    # Calculate rugosity from theory (integral)
    HL0_values = 10**hvar.loc[hvar['L0'] == min(hvar['L0']), 'H0'].values
    R_theory2 = sum(R_func(HL0_values, L0)) / (2/L0)**2
    
    # Calculate D from theory
    D_theory = D_func(H, R_theory, L, L0)
    
    return {
        'D': D,
        'D_ends': D_ends,
        'D_theory': D_theory,
        'R_theory': R_theory,
        'R_theory2': R_theory2,
        'H': H
    }