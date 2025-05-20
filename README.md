# A geometric basis for surface habitat complexity and biodiversity

This is a Python implementation of the tools described in the research posted at [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.02.03.929521v1). The paper has been published at *Nature Ecology & Evolution*.

## Installation

```bash
git clone https://github.com/Michael-S-Phillips/surface_geometry.git
cd surface_geometry
pip install -e .

```

Usage
The examples/example.py script contains a generic example of how the code might be used to calculate the surface descriptors from any Digital Elevation Model (DEM). There are essentially three steps:

Select the size of the patch for your calculations (L)
Select the scales across which fractal dimension will be calculated (scl), the smallest scale automatically is the resolution (L0)
Load a DEM file as data (geotif)
Pick the bottom left point of the patch (x0 and y0)
Run the height_variation() function. This function requires the variables mentioned so far (L, scl, L0, data, x0 and y0). The output is the DEM height range at the prescribed scales.
Calculate the surface descriptor metrics using the rdh() function, which requires the output from height_variation().

Surface Descriptors
The rdh() function returns several metrics:
VariableDescriptionDFractal dimension from model fitD_endsFractal dimension only considering the largest (L) and smallest (L0) scaleD_theoryFractal dimension calculated from theory (i.e., from R and H)RSurface rugosity calculated using surface areaR_theorySurface rugosity calculated from theoryHThe height range (or height range at L)

```python
Example
pythonimport numpy as np
import rasterio
import matplotlib.pyplot as plt
from surface_geometry import functions

# Setup parameters
L = 2  # Scope, 2 by 2 m reef patches
scl = L / np.array([1, 2, 4, 8, 16, 32])  # Scales
L0 = min(scl)  # Grain

# Load DEM
data = rasterio.open("data/example/horseshoe.tif")

# Set global variables
functions.data = data
functions.L = L
functions.scl = scl
functions.L0 = L0
functions.output = "example"
functions.rep = 1

# Set patch coordinates
x0 = data.bounds.left
y0 = data.bounds.bottom
functions.x0 = x0
functions.y0 = y0

# Calculate height variation
height_var = functions.height_variation(write=True, return_results=True)

# Calculate surface descriptors
results = functions.rdh(height_var)
print(results)
```

Future Work

Apply to 3D meshes
Address issues with D when large drop-offs in DEM (can cause D to go below 2)


This Python implementation preserves the core functionality of the original R code while following Python best practices. The key functions for calculating surface geometry metrics have been translated, and the analysis workflow has been maintained. 

The implementation uses libraries such as:
- `rasterio` for handling geospatial raster data (replacing R's `raster`)
- `geopandas` for vector data (replacing R's `rgdal`)
- `numpy` and `pandas` for numerical operations and data manipulation
- `matplotlib` for visualization
- `scipy` for statistical operations
- `scikit-learn` for machine learning components (used in place of R's GAM models)

Note that some of the more advanced visualization and statistical analysis components from the original R code have been simplified, but the core functionality for calculating and analyzing surface geometry is fully preserved.