# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 10:49:20 2024

@author: hanna
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import os
import geopandas as gpd



############################################################################


# SET UP DIRECTORY AND NODATA


############################################################################
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())

# Set nodata value
nodata_val = 255

# Set validation directory
val_dir = os.path.join('data', 'validation')

# Set output directory
out_dir = os.path.join('data', 'intermediate')

# Set year range
years = range(2013, 2025)



############################################################################


# REPROJECTING VECTORS


############################################################################
# Read validation points
valpoints = gpd.read_file("data/validation/validation_points_geometry.shp")

# Reproject to EPSG  4326 
valpoints_4326 = valpoints.to_crs('EPSG:4326')
valpoints_4326.to_file("data/validation/validation_points_geometry_4326.shp")

# Read villages
