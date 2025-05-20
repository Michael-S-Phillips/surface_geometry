# surface_geometry/__init__.py

from .functions import (
    fd_func, 
    rescale, 
    count_filled_cells, 
    R_func, 
    D_func, 
    HL0_func, 
    height_variation, 
    rdh
)

# Import new functionality if needed
from .surface_area import calculate_surface_area

__version__ = '0.1.0'