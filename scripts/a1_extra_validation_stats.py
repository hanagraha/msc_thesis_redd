# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 15:32:49 2026

@author: hanna

This file plots runs extra analysis to compare GFC and TMF datasets before resampling. Attempting analysis on 
annual deforestation vs degradation year area, validation of transition map and annual change maps.

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

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"\\mefe38.gfz.de\mefe_glm1\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# GEOMETRY DATA
# -------------------------------------------------------------------------
# Read geometries
grnp = gpd.read_file("gola gazetted polygon/Gola_Gazetted_Polygon.shp")
villages = gpd.read_file("village polygons/VillagePolygons.geojson")
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()

# Define years
years = list(range(2013, 2024))


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

# -------------------------------------------------------------------------
# TMF DATA
# -------------------------------------------------------------------------
# Read tmf deforestation year
with rasterio.open('temp/tmf_DeforestationYear_reprojected_clipped.tif') as rast:     
    tmf_deforyear = rast.read(1)
    tmfdefor_profile = rast.profile
    tmf_res = rast.res

# Calculate pixel area (ha)
tmf_pixarea = tmf_res[0] * tmf_res[1] / 10000

# Read tmf degradation year
with rasterio.open('temp/tmf_DegradationYear_reprojected_clipped.tif') as rast:     
    tmf_degrayear = rast.read(1)
    tmfdegra_profile = rast.profile

# Read tmf transition map
with rasterio.open('temp/tmf_TransitionMap_MainClasses_reprojected_clipped.tif') as rast:     
    tmf_transition = rast.read(1)
    tmftrans_profile = rast.profile

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
# REFERENCE DATA
# -------------------------------------------------------------------------
# Define function to read csv validation data
def csv_read(datapath):
    
    # Read validation data
    data = pd.read_csv(datapath, delimiter = ",", index_col = 0)
    
    # Convert csv geometry to WKT
    data['geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
    
    # Convert dataframe to geodataframe
    data = gpd.GeoDataFrame(data, geometry = 'geometry', crs="EPSG:32629")
    
    return data

# Read validation data
valdata = csv_read("validation/validation_datasets/validation_points_780.csv")


# -------------------------------------------------------------------------
# STRATIFICATION MAPS
# -------------------------------------------------------------------------
# Define function to extract area from map
def map_area(path):

    # Read raster data
    with rasterio.open(path) as rast:
        data = rast.read()

    # Calculate area (hectares)
    pixels = np.sum(data != 255)
    ha = pixels * 0.09
        
    return data, pixels, ha

# Calculate aoi map area
stratmap, total_pix, total_ha = map_area("validation/stratification_maps/stratification_layer_nogrnp.tif")

# Calculate redd+ map area
strat_redd, redd_pix, redd_ha = map_area("validation/stratification_maps/stratification_layer_redd.tif")

# Calculate nonredd+ map area
strat_nonredd, nonredd_pix, nonredd_ha = map_area("validation/stratification_maps/stratification_layer_nonredd.tif")


# -------------------------------------------------------------------------
# EXTRACT SUMMARY COUNTS (ANNUAL)
# -------------------------------------------------------------------------
# Define function to get count summaries
def count_summary(raster, valuename = 'year'):

    # Filter out nodata values
    raster = raster[raster != 255]

    # Get unique values and counts
    vals, counts = np.unique(raster, return_counts=True)

    # Create dataframe 
    df = pd.DataFrame({valuename: vals, 'counts': counts})

    return df

# Extract disturbance counts
gfc_lossyear_counts = count_summary(gfc_lossyear)
tmf_deforyear_counts = count_summary(tmf_deforyear)
tmf_degrayear_counts = count_summary(tmf_degrayear)

# Extract annual change counts
tmf_annualchange_counts = []
for changeraster in tmf_annualchange:
    annual_summary = count_summary(changeraster, valuename = 'class')
    tmf_annualchange_counts.append(annual_summary)

# Extract transition map counts
tmf_transition_counts = count_summary(tmf_transition, valuename = 'class')


# -------------------------------------------------------------------------
# COMPARE TOTAL AREAS
# -------------------------------------------------------------------------
# Create dataframe of total areas
area_comparison = pd.DataFrame({
    'dataset': [
        'GFC lossyear',
        'TMF deforestation year',
        'TMF degradation year',
        'TMF transition',
        'TMF annual change'
    ],
    'pixels': [
        gfc_lossyear_counts['counts'].sum(),
        tmf_deforyear_counts['counts'].sum(),
        tmf_degrayear_counts['counts'].sum(),
        tmf_transition_counts['counts'].sum(),
        tmf_transition_counts['counts'].sum()
    ],
    'area_ha': [
        gfc_lossyear_counts['counts'].sum() * gfc_pixarea,
        tmf_deforyear_counts['counts'].sum() * tmf_pixarea,
        tmf_degrayear_counts['counts'].sum() * tmf_pixarea,
        tmf_transition_counts['counts'].sum() * tmf_pixarea,
        tmf_transition_counts['counts'].sum() * tmf_pixarea
    ]   
})


# -------------------------------------------------------------------------
# CALCULATE PROPORTIONAL DISTURBANCE AREAS (AOI)
# -------------------------------------------------------------------------
# Define color palatte 
bluecols = ['brown', "#1E2A5E", "#83B4FF"]

# Calculate proportaional deforestation area
gfc_loss_aoi = gfc_lossyear_counts.copy()
gfc_loss_aoi['area_ha'] = gfc_loss_aoi['counts'] * gfc_pixarea
gfc_loss_aoi['prop_dist'] = gfc_loss_aoi['area_ha'] / gfc_loss_aoi['area_ha'].sum()

tmf_defor_aoi = tmf_deforyear_counts.copy()
tmf_defor_aoi['area_ha'] = tmf_defor_aoi['counts'] * tmf_pixarea
tmf_defor_aoi['prop_dist'] = tmf_defor_aoi['area_ha'] / tmf_defor_aoi['area_ha'].sum()

tmf_degra_aoi = tmf_degrayear_counts.copy()
tmf_degra_aoi['area_ha'] = tmf_degra_aoi['counts'] * tmf_pixarea
tmf_degra_aoi['prop_dist'] = tmf_degra_aoi['area_ha'] / tmf_degra_aoi['area_ha'].sum()

# Filter for years 2013-2023
gfc_loss_aoi = gfc_loss_aoi[gfc_loss_aoi['year'].between(2013, 2023)].reset_index(drop=True)
tmf_defor_aoi = tmf_defor_aoi[tmf_defor_aoi['year'].between(2013, 2023)].reset_index(drop=True)
tmf_degra_aoi = tmf_degra_aoi[tmf_degra_aoi['year'].between(2013, 2023)].reset_index(drop=True)

# Add tmf disturbances
tmf_defordegra_aoi = pd.DataFrame({
    'year': years,
    'prop_dist': tmf_defor_aoi['prop_dist'] + tmf_degra_aoi['prop_dist']
})


# -------------------------------------------------------------------------
# PLOT DISTURBANCE (AOI)
# -------------------------------------------------------------------------
# Initialize figure
plt.figure(figsize=(10, 6))

# Add gfc deforestation line
plt.plot(years, gfc_loss_aoi['prop_dist']*100, color=bluecols[0], linewidth = 2,
         label='GFC Lossyear')

# Add tmf deforestation line
plt.plot(years, tmf_defor_aoi['prop_dist']*100, color=bluecols[1], linewidth = 2,
         label='TMF Deforestation Year')

# Add tmf deforestation + degradation line
plt.plot(years, tmf_defordegra_aoi['prop_dist']*100, color=bluecols[2], linewidth = 2,
         label='TMF Deforestation + Degradation Year')

# Add labels and title
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Proportional Disturbance Area (%)', fontsize = 17)

# Add x tickmarks
plt.xticks(years, fontsize = 16)

# Edit y tickmark fontsize
plt.yticks(fontsize = 16)

# Add legend
plt.legend(fontsize = 16, loc="upper right")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Show plot
plt.tight_layout()

# Save plot
plt.savefig("figs/gfc_tmf_native_comparison", dpi=300, bbox_inches='tight', transparent=True)

plt.show()