# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:18:38 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from rasterstats import zonal_stats
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch



############################################################################


# SET UP DIRECTORY


############################################################################
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())



############################################################################


# IMPORT DATA


############################################################################
gfc_lossyear_paths = [f"data/intermediate/gfc_lossyear_{year}.tif" for 
                      year in range(2013, 2024)]
tmf_deforyear_paths = [f"data/intermediate/tmf_DeforestationYear_{year}.tif" 
                       for year in range(2013, 2024)]
tmf_degrayear_paths = [f"data/intermediate/tmf_DegradationYear_{year}.tif" 
                       for year in range(2013, 2024)]
tmf_annual_paths = [f"data/jrc_preprocessed/tmf_AnnualChange_{year}_fm.tif" for 
                    year in range(2013,2024)]

# Set nodata value
nodata_val = 255


### POLYGON DATA
villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']])


