# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:56:10 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import pandas as pd
import numpy as np



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


# IMPORT RELEVANT DATASETS (LOCAL DRIVE)


############################################################################

### READ POLYGON DATA
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")
villages = gpd.read_file("data/village polygons/VillagePolygons.geojson")

### READ RASTER DATA
# Rasters of interest
gfc_lossyear = "data/hansen_preprocessed/gfc_lossyear_reclassified.tif"
gfc_treecover = "data/hansen_preprocessed/gfc_treecover2000_clipped.tif"
tmf_deforyear = "data/jrc_preprocessed/tmf_DeforestationYear_clipped.tif"
tmf_degrayear = "data/jrc_preprocessed/tmf_DegradationYear_clipped.tif"

gfc_lossyear = rasterio.open(gfc_lossyear)
