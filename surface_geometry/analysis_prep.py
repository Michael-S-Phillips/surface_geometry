# surface_geometry/analysis_prep.py

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
functions.L0 = 2/32  # Grain, resolution of processing ~ 6 cm

# Load geotifs
data = rasterio.open("data/megaplot/trimodal3set_photoscan_2chunk_DEM_7mm_proj_clip.tif")
mids = gpd.read_file("data/megaplot/trimodal_patch_grid_midpnts.shp")
anno = gpd.read_file("data/megaplot/trimodal_ann_aligned_cleaned.shp")

# Load reef records and megaplot datasets
records = pd.read_csv("output/records.csv")
megaplot = pd.read_csv("output/megaplot.csv")

# Add empty columns for records if they don't exist
if 'spp' not in records.columns:
    records['spp'] = np.nan
if 'abd' not in records.columns:
    records['abd'] = np.nan
if 'pie' not in records.columns:
    records['pie'] = np.nan

# Merge datasets
dat = pd.concat([records, megaplot], ignore_index=True)

# Surface descriptors and transformations
dat['R2_log10'] = np.log10(dat['R_theory']**2 - 1)
dat['R2_log10_sq'] = dat['R2_log10']**2

dat['HL0_log10'] = np.log10(dat['H'] / (functions.L0 * np.sqrt(2)))
dat['HL0_log10_sq'] = dat['HL0_log10']**2

dat['D_theory_sq'] = dat['D_theory']**2

# Save to CSV
dat.to_csv("output/master_20200709.csv", index=False)