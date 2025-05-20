# example.py

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import pandas as pd
from surface_geometry.functions import height_variation, rdh

# Create output directories
os.makedirs("output/example", exist_ok=True)

# Scope (extent), scales of variation, and resolution (grain)
L = 2  # Scope, 2 by 2 m reef patches
scl = L / np.array([1, 2, 4, 8, 16, 32, 64, 128])  # Scales, aim for 2 orders of magnitude
L0 = min(scl)  # Grain, resolution of processing ~ 6 cm

# Example surface (an 8x8m section of Horseshoe from Lizard Island)
output = "example"  # For housekeeping

# Load example geotif
with rasterio.open("data/example/horseshoe.tif") as data:
    # Plot the DEM
    fig, ax = plt.subplots(figsize=(10, 8))
    show(data, ax=ax)
    
    rep = 1
    # Choose patch in which to calculate RDH (rugosity, fractal D and height range).
    x0 = data.bounds.left
    y0 = data.bounds.bottom
    
    # Draw rectangle on the plot
    ax.add_patch(plt.Rectangle((x0, y0), L, L, edgecolor='white', 
                              facecolor='none', linestyle='--'))
    plt.savefig(f"output/{output}/horseshoe_dem.png", dpi=300)
    
    # Calculate height variation at different scales within patch
    example_data = height_variation(data, x0, y0, L, scl, L0, output=output, 
                                    rep=rep, write=True, return_data=True)

# Load the saved file (as would be done when starting from a checkpoint)
example_file = f"output/{output}/var_horseshoe.tif_{rep:04d}.csv"
if os.path.exists(example_file):
    example_data = pd.read_csv(example_file)
else:
    print(f"Warning: {example_file} not found. Using calculated data.")

# Calculate rugosity, fractal dimension and height range
results = rdh(example_data, L, L0)

# Print results in a formatted way
print("\nSurface geometry metrics:")
print("-" * 30)
for key, value in results.items():
    print(f"{key:10s}: {value:.4f}")

# Create a visualization of the height variation
plt.figure(figsize=(10, 8))
grouped = example_data.groupby('L0')
for name, group in grouped:
    plt.scatter(group['x'], group['y'], s=group['H0']*50, alpha=0.5, 
                label=f"Scale: {10**name:.4f}")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Height variation at different scales")
plt.legend()
plt.savefig(f"output/{output}/height_variation.png", dpi=300)

# Plot the relationship between scale and height range
plt.figure(figsize=(8, 6))
mean_H0 = example_data.groupby('L0')['H0'].mean().reset_index()
plt.scatter(10**mean_H0['L0'], 10**mean_H0['H0'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Scale (m)")
plt.ylabel("Height range (m)")
plt.title("Scale-dependent height variation")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(f"output/{output}/scale_height_relationship.png", dpi=300)

print(f"\nAll figures saved to output/{output}/")