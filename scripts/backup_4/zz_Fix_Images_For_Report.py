# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:36:07 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import rasterio
import numpy as np



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

# Set output directory
out_dir = os.path.join(os.getcwd(), 'data', 'cc_composites')



############################################################################


# READ DATA


############################################################################
# Define relevant filepaths
l8 = "data/cc_composites/L8_Annual/L8_2023_CC.tif"
s2 = "data/cc_composites/S2_Annual/S2_2023_CC.tif"

# Read l8 raster
with rasterio.open(l8) as rast:
    l8_data = rast.read()
    l8_profile = rast.profile
    
# Read s2 raster
with rasterio.open(s2) as rast:
    s2_data = rast.read()
    s2_profile = rast.profile



############################################################################


# CHANGE NAN VALUES FOR VISUALIZATION


############################################################################
# Update profiles
l8_profile['nodata'] = np.nan
s2_profile['nodata'] = np.nan

# Create new filenames
l8_path = "data/plots/Misc/l8_annual_vis.tif"
s2_path = "data/plots/Misc/s2_annual_vis.tif"

# Write l8 raster with nodata as nan
with rasterio.open(l8_path, "w", **l8_profile) as dst:
    for i in range(l8_data.shape[0]):  
        dst.write(l8_data[i], i + 1)  

print(f"GeoTIFF written to {l8_path}")
    
# Write s2 raster with nodata as nan
with rasterio.open(s2_path, "w", **s2_profile) as dst:
    for i in range(s2_data.shape[0]):  
        dst.write(s2_data[i], i + 1)  

print(f"GeoTIFF written to {s2_path}")
    
    
    
    
    
    
    
    
    
    