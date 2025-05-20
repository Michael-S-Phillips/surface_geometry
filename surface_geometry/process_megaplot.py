# surface_geometry/process_megaplot.py

import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import sys

# Add parent directory to path
sys.path.append('..')

from surface_geometry import functions

# Set up variables
functions.output = "megaplot"

# Load geotif
data_path = "data/megaplot/trimodal3set_photoscan_2chunk_DEM_7mm_proj_clip.tif"
data = rasterio.open(data_path)
functions.data = data

# These are midpoints for 2x2m squares, which are our "patch"-level samples
mids = gpd.read_file("data/megaplot/trimodal_patch_grid_midpnts.shp")

# These are the annotated coral colonies, ID'ed to species level
anno = gpd.read_file("data/megaplot/trimodal_ann_aligned_cleaned.shp")

# Scope (extent), scales of variation, and resolution (grain)
functions.L = 2  # Scope
functions.scl = functions.L / np.array([1, 2, 4, 8, 16, 32])  # Scales
functions.L0 = min(functions.scl)  # Grain

# Ensure output directory exists
os.makedirs(os.path.join("output", "megaplot"), exist_ok=True)

store = []

for mid in range(len(mids)):
    rep = mid + 1  # 1-indexed like in R code
    functions.rep = rep
    
    # Get lower corner of 2x2m bounding box
    x0 = mids.iloc[mid].geometry.x - functions.L/2
    y0 = mids.iloc[mid].geometry.y - functions.L/2
    functions.x0 = x0
    functions.y0 = y0
    
    fname = f"output/megaplot/var_{data.name}_{rep:04d}.csv"
    if os.path.exists(fname):
        temp = pd.read_csv(fname)
    else:
        temp = functions.height_variation(write=True, return_results=True)
    
    # Crop annotations to the current patch
    patch_bounds = (x0, y0, x0 + 2, y0 + 2)
    mask = (
        (anno.geometry.x >= patch_bounds[0]) & 
        (anno.geometry.x <= patch_bounds[2]) & 
        (anno.geometry.y >= patch_bounds[1]) & 
        (anno.geometry.y <= patch_bounds[3])
    )
    tax = anno[mask]
    
    # Calculate biodiversity metrics
    spp = len(tax['Species'].unique())
    abd = len(tax)
    pie = 0
    if abd > 0:
        # Calculate probability of interspecific encounter
        species_counts = tax['Species'].value_counts()
        pie = 1 - np.sum((species_counts / abd) ** 2)
    
    # Calculate surface metrics
    results = functions.rdh(temp)
    results['rec'] = data.name
    results['rep'] = mid + 1
    results['spp'] = spp
    results['abd'] = abd
    results['pie'] = pie
    results['site'] = "megaplot"
    
    store.append(results)

# Convert to DataFrame and save
df = pd.DataFrame(store)
df.to_csv("output/megaplot.csv", index=False)