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
import glob
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt


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
# Raster file paths
gfc_lossyear = "data/hansen_preprocessed/gfc_lossyear_AOI.tif"
gfc_treecover = "data/hansen_preprocessed/gfc_treecover2000_AOI.tif"
tmf_deforyear = "data/jrc_preprocessed/tmf_DeforestationYear_AOI.tif"
tmf_degrayear = "data/jrc_preprocessed/tmf_DegradationYear_AOI.tif"
tmf_trans_main = "data/jrc_preprocessed/tmf_TransitionMap_MainClasses_AOI.tif"
tmf_trans_sub = "data/jrc_preprocessed/tmf_TransitionMap_Subtypes_AOI.tif"

# Read rasters
with rasterio.open(gfc_lossyear) as gfc:
    gfc_lossyear = gfc.read(1)
    
with rasterio.open(gfc_treecover) as gfc:
    gfc_treecover = gfc.read(1)
    gfc_treecover_profile = gfc.profile
    
with rasterio.open(tmf_deforyear) as tmf:
    tmf_deforyear = tmf.read(1)
    
with rasterio.open(tmf_degrayear) as tmf:
    tmf_degrayear = tmf.read(1)
    
with rasterio.open(tmf_trans_main) as tmf:
    tmf_trans_main = tmf.read(1)
    
with rasterio.open(tmf_trans_sub) as tmf:
    tmf_trans_sub = tmf.read(1)
    tmf_trans_sub_profile = tmf.profile

# Annual Change data
tmf_folder = "data/jrc_preprocessed"
tmf_annual_files = glob.glob(os.path.join(tmf_folder, "*AnnualChange*"))

tmf_annual_dict = {}

for file in tmf_annual_files:
    filename = os.path.basename(file)
    year = filename.split("AnnualChange_")[1].split('_')[0]
    var = f"tmf_{year}"
    
    with rasterio.open(file) as tmf:
        tmf_annual = tmf.read(1)
        tmf_annual_dict[var] = tmf_annual

    print(f"Stored {var}, data shape: {tmf_annual.shape}")
    

    
############################################################################


# INVESTIGATE FOREST MASK


############################################################################

### CREATE TMF BASELINE FOREST 
"""
"The initial tropical moist forest domain can be derived from the Transition 
Map - Sub types map by selecting all pixels belonging to classes 10 to 89 
excluding classes 71 and 72."
- taken from https://forobs.jrc.ec.europa.eu/TMF/resources#how_to
"""

# Create baseline forest mask
tmf_baseforest_mask = ((tmf_trans_sub >= 10) & (tmf_trans_sub <= 89) & 
                       (tmf_trans_sub != 71) & (tmf_trans_sub != 72))

# Mask TMF forest
tmf_baseforest = np.where(tmf_baseforest_mask, tmf_trans_sub, 255)

# Update data type and nodata value for saving
tmf_trans_sub_profile.update(dtype=rasterio.float32, nodata=255)  

# Save to drive
tmf_baseforest_file = 'data/jrc_preprocessed/tmf_baselineforest.tif'

with rasterio.open(tmf_baseforest_file, 'w', **tmf_trans_sub_profile) as dst:
    dst.write(tmf_baseforest.astype(rasterio.float32), 1)

print(f"TMF baseline forest saved as {tmf_baseforest_file}")


### CREATE GFC BASELINE FOREST 
"""
Forest mask will use 50% forest cover threshold from Malan et al. (2024)
"""

# Create baseline forest mask
gfc_baseforest_mask = (gfc_treecover >= 50)

# Mask GFC forest
gfc_baseforest = np.where(gfc_baseforest_mask, gfc_treecover, 255)

# Update data type and nodata value for saving
gfc_treecover_profile.update(dtype=rasterio.float32, nodata=255)  

# Save to drive
gfc_baseforest_file = 'data/hansen_preprocessed/gfc_baselineforest.tif'

with rasterio.open(gfc_baseforest_file, 'w', **gfc_treecover_profile) as dst:
    dst.write(gfc_baseforest.astype(rasterio.float32), 1)

print(f"GFC baseline forest saved as {gfc_baseforest_file}")


### FOREST MASK SPATIAL AGREEMENT
nodata_val = 255

# Create binary masks indicating where each raster has valid values (not 255)
gfc_mask = gfc_baseforest != nodata_val 
tmf_mask = tmf_baseforest != nodata_val  

# Create empty raster
spatial_agreement = np.zeros_like(gfc_baseforest, dtype=np.uint8)

# Assign values based on agreement
spatial_agreement[gfc_mask & tmf_mask] = 1  # Both rasters have values
spatial_agreement[gfc_mask & ~tmf_mask] = 2  # Only raster1 has values
spatial_agreement[~gfc_mask & tmf_mask] = 3  # Only raster2 has values



############################################################################


# TEST PLOTTING


############################################################################
plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
show(tmf_baseforest, title='TMF Baseline Forest', cmap='gist_ncar')













