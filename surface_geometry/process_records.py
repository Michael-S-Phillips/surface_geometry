# surface_geometry/process_records.py

import os
import glob
import numpy as np
import rasterio
import pandas as pd
import sys

# Add parent directory to path
sys.path.append('..')

from surface_geometry import functions

# Set up variables
functions.L = 2  # Scope
functions.scl = functions.L / np.array([1, 2, 4, 8, 16, 32])  # Scales
functions.L0 = min(functions.scl)  # Grain
functions.output = "record"

# Ensure output directory exists
os.makedirs(os.path.join("output", "records"), exist_ok=True)

# Get reef records
files = glob.glob("data/records/*.tif")
files = [os.path.splitext(os.path.basename(f))[0] for f in files]
files = list(set([f.replace('.tfw', '') for f in files]))

store = []

for rec in files:
    # Load geotif for reef record
    data_path = f"data/records/{rec}.tif"
    data = rasterio.open(data_path)
    functions.data = data
    
    # Get lower corner of 8x8m bounding box
    xb = (data.bounds.left + data.bounds.right) / 2 - 4
    yb = (data.bounds.bottom + data.bounds.top) / 2 - 4
    
    # Iterate through 2x2m quadrats (reps = 16) in reef record
    rep = 1
    for i in [0, 2, 4, 6]:
        for j in [0, 2, 4, 6]:
            x0 = xb + i
            y0 = yb + j
            functions.x0 = x0
            functions.y0 = y0
            functions.rep = rep
            
            fname = f"output/records/var_{data.name}_{rep:04d}.csv"
            if os.path.exists(fname):
                temp = pd.read_csv(fname)
            else:
                temp = functions.height_variation(write=True, return_results=True)
            
            # Calculate surface metrics
            results = functions.rdh(temp)
            results['rec'] = rec
            results['rep'] = rep
            store.append(results)
            
            rep += 1

# Convert to DataFrame
df = pd.DataFrame(store)

# Add site information
site_mapping = {
    "rr201611_004_Osprey_dem_low": "Osprey",
    "rr201611_007_CooksPath_dem_low": "Cooks Path",
    "rr201611_018_Resort_dem_low": "Resort",
    "rr201611_023_NorthReef03_dem_low": "Mermaid Cove",
    "rr201611_033_ConerBeach_dem_low": "Corner Beach",
    "rr201611_037_southeast_dem_low": "Southeast",
    "rr201611_039_EasterPoint_dem_low": "Easter Point",
    "rr201611_040_NoMansLand_dem_low": "No Mans Land",
    "rr201611_041_NorthOfParadise_dem_low": "North of Paradise",
    "rr201611_042_GnarlyTree_dem_low": "Gnarly Tree",
    "rr201611_046_NorthReef02_dem_low": "North Reef 2",
    "rr201611_050_horsehoe_DEM_low": "Horseshoe",
    "rr201611_051_Vickis_dem_low": "Vickis",
    "rr201611_052_SouthIsland_dem_low": "South Island",
    "rr201611_053_Trimodal_dem_low": "Trimodal",
    "rr201611_054_lagoon02_dem_low": "Lagoon 2",
    "rr201611_055_Lagoon01_dem_low": "Lagoon 1",
    "rr201611_055_LizardHead_dem_low": "Lizard Head",
    "rr201611_056_TurtleBeach_dem_low": "Turtle Beach",
    "rr201611_045_WashingMachine_dem_low": "Washing Machine",
    "rr201611_049_NorthReef01_dem_low": "North Reef 1"
}

df['site'] = df['rec'].map(site_mapping)

# Save to CSV
df.to_csv("output/records.csv", index=False)