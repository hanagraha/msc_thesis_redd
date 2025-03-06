# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:56:52 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import pandas as pd
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask



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
val_dir = os.path.join('data', 'validation')

# Set year range
years = range(2012, 2025)

# Define landsat data folder (with 2012)
l8_folder = os.path.join("data", "cc_composites", "L8_Annual_w_2012")

# Define landsat jan data folder
l8_jan_folder = os.path.join("data", "cc_composites", "L8_Jan")

# Define sentinel data folder
s2_folder = os.path.join("data", "cc_composites", "S2_Annual")

# Define sentinel jan data folder
s2_jan_folder = os.path.join("data", "cc_composites", "S2_Jan")



############################################################################


# READ DATA


############################################################################
# Read landsat data
l8_ann = [os.path.join(l8_folder, file) for file in os.listdir(l8_folder) if \
          file.endswith('.tif')]
    
# Read landsat jan data
l8_jan = [os.path.join(l8_jan_folder, file) for file in os.listdir(l8_jan_folder) \
          if file.endswith('.tif')]
    
# Read sentinel data
s2_ann = [os.path.join(s2_folder, file) for file in os.listdir(s2_folder) if \
      file.endswith('.tif')]
    
# Read sentinel jan data
s2_jan = [os.path.join(s2_jan_folder, file) for file in os.listdir(s2_jan_folder) \
          if file.endswith('.tif')]
    
# Read village shapefile
villages = gpd.read_file("data/village polygons/VillagePolygons.geojson")

# Get village geometry
villages_geom = villages.dissolve().geometry

    
# %%
############################################################################


# CLIP IMAGES TO AOI


############################################################################
# Define function to re-define nodata value
def redef_nd(pathlist, nodata_val, out_dir):
    
    # Iterate over each path
    for path in pathlist:
        
        # Read file
        with rasterio.open(path) as rast:
            
            # Extract rasters
            data = rast.read()
            
            # Extract metadata
            meta = rast.meta
            
            # Update metadata
            meta.update(nodata=nodata_val)
            
        # Extract filename
        basename = os.path.basename(path)
        
        # Define output path
        output_path = os.path.join(out_dir, basename)
            
        # Write file with new metadata
        with rasterio.open(output_path, 'w', **meta) as dest:
            dest.write(data)
            
        # Print statement
        print(f"Updated nodata value for {basename}")

# Define output folder for l8 annual
l8_ann_dir = os.path.join("data", "cc_composites", "L8_Annual_PP")

# Define output folder for l8 jan
l8_jan_dir = os.path.join("data", "cc_composites", "L8_Jan_PP")

# Define output folder for l8 annual
s2_ann_dir = os.path.join("data", "cc_composites", "L8_Annual_PP")

# Define output folder for l8 jan
s2_jan_dir = os.path.join("data", "cc_composites", "L8_Jan_PP")

# Adapt nodata values in l8 annual composites
redef_nd(l8_ann, nodata_val, l8_ann_dir)

# Adapt nodata values in l8 jan composites
redef_nd(l8_jan, nodata_val, l8_jan_dir)

# Adapt nodata values in s2 annual composites
redef_nd(s2_ann, nodata_val, s2_ann_dir)

# Adapt nodata values in s2 jan composites
redef_nd(s2_jan, nodata_val, s2_jan_dir)


