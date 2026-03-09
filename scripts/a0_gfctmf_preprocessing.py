"""
Created on Mar 6 2026

@author: hanna

This file preprocesses Global Forest Change (GFC) and Tropical Moist Forests (TMF) data for 2013-2023.

Expected runtime XX min
"""

# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterstats import zonal_stats

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"Z:\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# GFC DATA
# -------------------------------------------------------------------------
# Read gfc tree cover 2000
with rasterio.open('temp/gfc_treecover2000_reprojected_clipped.tif') as rast:     
    gfc_tc2000 = rast.read(1)

# Read gfc lossyear
with rasterio.open('temp/gfc_lossyear_reprojected_clipped.tif') as rast:     
    gfc_lossyear = rast.read(1)
    gfc_profile = rast.profile
    gfc_res = rast.res

# Calculate pixel area (ha)
gfc_pixarea = gfc_res[0] * gfc_res[1] / 10000

# Define gfc forest mask (50% canopy cover)
gfc_mask = (gfc_tc2000 >= 50)

# Mask disturbances
gfc_lossyear50 = np.where(gfc_mask, gfc_lossyear, 255)


# -------------------------------------------------------------------------
# TMF DATA
# -------------------------------------------------------------------------
# Read tmf deforestation year
with rasterio.open('temp/tmf_DeforestationYear_reprojected_clipped.tif') as rast:     
    tmf_deforyear = rast.read(1)
    tmf_profile = rast.profile
    tmf_res = rast.res
    tmf_crs = rast.crs

# Calculate pixel area (ha)
tmf_pixarea = tmf_res[0] * tmf_res[1] / 10000

# Read tmf degradation year
with rasterio.open('temp/tmf_DegradationYear_reprojected_clipped.tif') as rast:     
    tmf_degrayear = rast.read(1)

# Read tmf annual change 2012
with rasterio.open('temp/tmf_AnnualChange_2012_reprojected_clipped.tif') as rast:     
    tmf_ac2012 = rast.read(1)

# Read tmf transition map
with rasterio.open('temp/tmf_TransitionMap_MainClasses_reprojected_clipped.tif') as rast:     
    tmf_transition = rast.read(1)

# Define annual change paths
tmf_acpaths = sorted(f"temp/{file}" for file in os.listdir('temp')
    if file.startswith("tmf_AnnualChange") and file.endswith("_clipped.tif"))

# Initialize empty list to hold annual change data
tmf_annualchange = []

# Iterate over each path and read raster data
for path in tmf_acpaths:
    with rasterio.open(path) as rast:
        tmf_annualchange.append(rast.read(1))


# -------------------------------------------------------------------------
# AREA OF INTEREST
# -------------------------------------------------------------------------
# Read grnp shapefile
grnp = gpd.read_file("gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Read village shapefile
villages = gpd.read_file("village polygons/VillagePolygons.geojson")

# Combine grnp and villages to create aoi
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()

with rasterio.open("reference_raster.tif") as src:
    ref_transform = src.transform
    ref_crs = src.crs
    ref_shape = src.shape

    # Create AOI mask (True = inside AOI)
    from rasterio.features import geometry_mask
    aoi_mask = ~geometry_mask(
        aoi_geoms,
        transform=ref_transform,
        invert=False,
        out_shape=ref_shape
    )


# -------------------------------------------------------------------------
# CREATE COMBINED FOREST MASK
# -------------------------------------------------------------------------
# Define gfc tree cover in 2012
gfc_treecover2012 = (gfc_tc2000 >= 50) & ((gfc_lossyear == 0) | (gfc_lossyear > 2012)).astype(np.uint8)

# Define tmf tree cover in 2012
tmf_undisturbed2012 = (tmf_ac2012 == 1).astype(np.uint8)
tmf_treecover2012 = (tmf_ac2012 ==1) | (tmf_ac2012 == 2) | (tmf_ac2012 == 4).astype(np.uint8)

# Create combined forest mask
agreement = np.zeros(gfc_treecover2012.shape, dtype=np.uint8)

agreement = np.where(gfc_treecover2012,           agreement | 1, agreement)
agreement = np.where(tmf_undisturbed2012, agreement | 2, agreement)
agreement = np.where(tmf_treecover2012,   agreement | 4, agreement)

# --- Apply nodata outside AOI ---
agreement = np.where(aoi_mask, agreement, 255).astype(np.uint8)