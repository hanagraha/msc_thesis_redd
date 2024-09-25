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
gfc_lossyear = "data/hansen_preprocessed/gfc_lossyear_fm.tif"
gfc_treecover = "data/hansen_preprocessed/gfc_treecover2000_fm.tif"
tmf_deforyear = "data/jrc_preprocessed/tmf_DeforestationYear_fm.tif"
tmf_degrayear = "data/jrc_preprocessed/tmf_DegradationYear_fm.tif"
tmf_trans_main = "data/jrc_preprocessed/tmf_TransitionMap_MainClasses_fm.tif"
tmf_trans_sub = "data/jrc_preprocessed/tmf_TransitionMap_Subtypes_fm.tif"

# Read rasters
with rasterio.open(gfc_lossyear) as gfc:
    gfc_lossyear = gfc.read(1)
    
with rasterio.open(gfc_treecover) as gfc:
    gfc_treecover = gfc.read(1)
    
with rasterio.open(tmf_deforyear) as tmf:
    tmf_deforyear = tmf.read(1)
    
with rasterio.open(tmf_degrayear) as tmf:
    tmf_degrayear = tmf.read(1)
    
with rasterio.open(tmf_trans_main) as tmf:
    tmf_trans_main = tmf.read(1)
    
with rasterio.open(tmf_trans_sub) as tmf:
    tmf_trans_sub = tmf.read(1)

# Annual Change data
tmf_folder = "data/jrc_preprocessed"
tmf_annual_files = glob.glob(os.path.join(tmf_folder, "*AnnualChange*"))
tmf_annual_files = ([file for file in tmf_annual_files if file.endswith('_fm.tif') and not 
              file.endswith(('.xml', '.ovr'))])

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


# EXTRACT DEFORESTATION PER YEAR


############################################################################





############################################################################


# TEST PLOTTING


############################################################################
plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
show(tmf_baseforest, title='TMF Baseline Forest', cmap='gist_ncar')













