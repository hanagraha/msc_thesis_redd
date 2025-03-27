# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:02:49 2024

@author: hanna

This file plots GFC and TMF data to compare deforestation and degradation 
over 2013-2023. Deforestation rates are calculated for the AOI, GRNP, REDD+, 
and non-REDD+ village areas. 

Expected runtime <1min
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
import matplotlib.pyplot as plt



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

# Define year range
years = list(range(2013, 2024))

# Define pixel area
pixel_area = 0.09

# Color Palatte (3 colors)
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]

# Color palatte (GFC and TMF)
gfc_col = "#820300"  # Darker Red
tmf_col = "#4682B4"  # Darker Blue - lighter



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read multiple rasters
def read_files(pathlist):
    
    # Create empty list to hold arrays
    arrlist = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read file
        with rasterio.open(path) as rast:
            data = rast.read(1)
            
            # Add array to list
            arrlist.append(data)
            
    return arrlist

# Define multiyear gfc lossyear path
gfc_lossyear_path = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Define multiyear tmf defordegra path
tmf_defordegra_path = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# Define singleyear gfc paths
gfc_lossyear_paths = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif" for 
                      year in years]

# Define file paths for annual tmf rasters
tmf_defordegra_paths = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif" for 
                      year in years]

# Read multiyear gfc rasters
with rasterio.open(gfc_lossyear_path) as rast:
    gfc = rast.read(1)
    gfc_profile = rast.profile

# Read multiyear tmf rasters
with rasterio.open(tmf_defordegra_path) as rast:
    tmf = rast.read(1)
    tmf_profile = rast.profile
    
# Read singleyear gfc rasters
gfc_arrlist = read_files(gfc_lossyear_paths)

# Read singleyear tmf rasters
tmf_arrlist = read_files(tmf_defordegra_paths)

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp data
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create aoi
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']])

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k').reset_index()

# Extract redd+ polygon area (ha)
redd_area = villages.loc[1].geometry.area / 10000

# Extract non-redd+ polygon area (ha)
nonredd_area = villages.loc[0].geometry.area / 10000

# Extract grnp area
grnp_area = grnp.dissolve().geometry.area / 10000

# Extract village area
village_area = redd_area + nonredd_area


# %%
############################################################################


# EXTRACT DEFORESTATION PER YEAR FOR AOI, REDD+, AND NON_REDD+ AREA


############################################################################
# Define function to extract multiyear statistics
def multiyear_atts(arr, yearrange):
    
    # Extract unique values and pixel counts
    unique_values, pixel_counts = np.unique(arr, return_counts=True)
    
    # Create dataframe with unique values and pixel counts
    attributes = pd.DataFrame({"Year": unique_values, 
                               "AOI": pixel_counts})
    
    # Filter only for attributes within yearrange
    filt_atts = attributes[(attributes['Year'] >= min(yearrange)) & \
                           (attributes['Year'] <= max(yearrange))]
    
    return filt_atts

# Define function to extract singleyear zonal statistics
def singleyear_zonal(arrlist, yearrange, geom_df, nodata_val, affine):
    
    # Create empty list to hold statistics
    stats = []
    
    # Iterate over each array
    for arr, year in zip(arrlist, yearrange):
    
        # Calculate zonal statistics
        deforestation_stats = zonal_stats(geom_df, arr, nodata = nodata_val, 
                                          affine = affine, stats="count")
        
        # Extract redd deforestation (proportion of area)
        redd_stats = (deforestation_stats[1]['count'] * pixel_area) / redd_area
        
        # Extract nonredd (proportion of area)
        nonredd_stats = (deforestation_stats[0]['count'] * pixel_area) / nonredd_area
        
        # Add deforestation stats to list
        stats.append({'Year': year, 'REDD+': redd_stats, 'Non-REDD+': nonredd_stats})
        
    # Convert stats list to dataframe
    stats_df = pd.DataFrame(stats)
    
    return stats_df

# Extract annual deforestation data from gfc
gfc_atts = multiyear_atts(gfc, years)

# Extract annual deforestation data from tmf
tmf_atts = multiyear_atts(tmf, years)

# Extract gfc data for redd+ and non-redd+ area
gfc_zonal = singleyear_zonal(gfc_arrlist, years, villages, nodata_val, 
                             gfc_profile['transform'])

# Extract tmf data for redd+ and non-redd+ area
tmf_zonal = singleyear_zonal(tmf_arrlist, years, villages, nodata_val, 
                             tmf_profile['transform'])


# %%
############################################################################


# EXTRACT DEFORESTATION PER YEAR FOR GRNP AREA


############################################################################
def grnp_zonal(arrlist, yearrange, affine):
    
    # Create empty list to hold statistics
    stats = []
    
    # Iterate over each array
    for arr, year in zip(arrlist, yearrange):
    
        # Calculate zonal statistics
        deforestation_stats = zonal_stats(grnp.dissolve(), arr, nodata = nodata_val, 
                                          affine = affine, stats="count")
        
        # Extract redd deforestation (proportion of area)
        grnp_stats = (deforestation_stats[0]['count'] * pixel_area) / grnp_area
        
        # Add deforestation stats to list
        stats.append({'Year': year, 'GRNP': grnp_stats[0]})
        
    # Convert stats list to dataframe
    stats_df = pd.DataFrame(stats)
    
    return stats_df

def villages_zonal(arrlist, yearrange, affine):
    
    # Create empty list to hold statistics
    stats = []
    
    # Iterate over each array
    for arr, year in zip(arrlist, yearrange):
    
        # Calculate zonal statistics
        deforestation_stats = zonal_stats(villages.dissolve(), arr, nodata = nodata_val, 
                                          affine = affine, stats="count")
        
        # Extract redd deforestation (proportion of area)
        village_stats = (deforestation_stats[0]['count'] * pixel_area) / village_area
        
        # Add deforestation stats to list
        stats.append({'Year': year, 'Villages': village_stats})
        
    # Convert stats list to dataframe
    stats_df = pd.DataFrame(stats)
    
    return stats_df

# Extract gfc data for grnp area
gfc_grnp = grnp_zonal(gfc_arrlist, years, gfc_profile['transform'])

# Extract tmf data for grnp area
tmf_grnp = grnp_zonal(tmf_arrlist, years, tmf_profile['transform'])

# Extract gfc data for village area
gfc_villages = villages_zonal(gfc_arrlist, years, gfc_profile['transform'])

# Extract tmf data for village area
tmf_villages = villages_zonal(tmf_arrlist, years, tmf_profile['transform'])


# %%
############################################################################


# PLOT DEFORESTATION PER AREA (LINE)


############################################################################
# Initialize figure
plt.figure(figsize=(10, 6.5))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_zonal['REDD+']*100, color=gfc_col, linewidth = 2,
         label='GFC Deforestation in REDD+ Villages')

# Plot the pixel values for REDD+ villages
plt.plot(years, tmf_zonal['REDD+']*100, color=tmf_col, linewidth = 2,
         label='TMF Deforestation in REDD+ Villages')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, gfc_zonal['Non-REDD+']*100, color=gfc_col, linewidth = 2, 
         label='GFC Deforestation in Non-REDD+ Villages', linestyle = '--')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, tmf_zonal['Non-REDD+']*100, color=tmf_col, linewidth = 2,
         label='TMF Deforestation in Non-REDD+ Villages',
         linestyle = '--')

# Add labels and title
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Proportional Deforestation Area (%)', fontsize = 17)

# Add x tickmarks
plt.xticks(years, fontsize = 16)

# Edit y tickmark fontsize
plt.yticks(fontsize = 16)

# Add legend
plt.legend(fontsize = 14, loc="lower left")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# PLOT DEFORESTATION PER AREA (AOI)


############################################################################

# Initialize figure
plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_villages['Villages']*100, color=bluecols[0], linewidth = 2,
         label='GFC Deforestation')

# Plot the pixel values for REDD+ villages
plt.plot(years, tmf_villages['Villages']*100, color=bluecols[1], linewidth = 2,
         label='TMF Deforestation')

# Add labels and title
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Proportional Deforestation Area (%)', fontsize = 17)

# Add x tickmarks
plt.xticks(years, fontsize = 16)

# Edit y tickmark fontsize
plt.yticks(fontsize = 16)

# Add legend
plt.legend(fontsize = 16, loc="lower left")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Show plot
plt.tight_layout()

plt.show()



# %%
############################################################################


# PLOT DEFORESTATION PER AREA (GRNP)


############################################################################
# Initialize figure
plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_grnp['GRNP']*100, color=bluecols[0], linewidth = 2,
         label='GFC Deforestation')

# Plot the pixel values for REDD+ villages
plt.plot(years, tmf_grnp['GRNP']*100, color=bluecols[1], linewidth = 2,
         label='TMF Deforestation')

# Add labels and title
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Proportional Deforestation Area (%)', fontsize = 17)

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
plt.show()



