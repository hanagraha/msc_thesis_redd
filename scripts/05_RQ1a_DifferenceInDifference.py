# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:02:49 2024

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
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator



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


# PLOT DEFORESTATION PER AREA (LINE)


############################################################################
# Color palatte
redd_col1 = "#820300"  # Darker Red

nonredd_col1 = "#4682B4"  # Darker Blue - lighter

gfc_col = "#820300"  # Darker Red

tmf_col = "#4682B4"  # Darker Blue - lighter

# Initialize figure
plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_zonal['REDD+']*100, color=gfc_col, linewidth = 2,
         label='GFC Deforestation in REDD+ Villages')

# Plot the pixel values for REDD+ villages
plt.plot(years, tmf_zonal['REDD+']*100, color=tmf_col, linewidth = 2,
         label='TMF Deforestation and Degradation in REDD+ Villages')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, gfc_zonal['Non-REDD+']*100, color=gfc_col, linewidth = 2, 
         label='GFC Deforestation in Non-REDD+ Villages', linestyle = '--')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, tmf_zonal['Non-REDD+']*100, color=tmf_col, linewidth = 2,
         label='TMF Deforestation and Degradation in Non-REDD+ Villages',
         linestyle = '--')

# Add labels and title
plt.xlabel('Year', fontsize = 14)
plt.ylabel('Proportional Deforestation Area (%)', fontsize = 14)
# plt.title('Deforestation in REDD+ vs Non-REDD+ Villages (2013-2023)')

# Add x tickmarks
plt.xticks(years, fontsize = 14)

# Edit y tickmark fontsize
plt.yticks(fontsize = 14)

# Add legend
plt.legend(fontsize = 14, loc="lower left")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# DIFFERENCE IN DIFFERENCES ANALYSIS


############################################################################
# Calculate gfc redd+/non-redd+ difference
gfc_diffs = gfc_zonal['Non-REDD+'] - gfc_zonal['REDD+']

# Calculate tmf redd+/non-redd+ difference
tmf_diffs = tmf_zonal['Non-REDD+'] - tmf_zonal['REDD+']

# Calculate gfc counterfactual treatment
gfc_cf = gfc_zonal['Non-REDD+'] - gfc_diffs[0]

# Calculate tmf counterfactual treatment
tmf_cf = tmf_zonal['Non-REDD+'] - tmf_diffs[0]

# Calculate gfc difference in differences
gfc_dd = gfc_zonal['REDD+'] - gfc_cf

# Calculate tmf difference in differences
tmf_dd = tmf_zonal['REDD+'] - tmf_cf


# %%
############################################################################


# PLOT DEFORESTATION RATES WITH COUNTERFACTUAL


############################################################################
plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_zonal['REDD+'], color=redd_col1,
         label='GFC REDD+ Deforestation')

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_cf, color=redd_col1,
         label='GFC Counterfactual Non-REDD+ Deforestation', 
         linestyle = '--')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, tmf_zonal['REDD+'], color=nonredd_col1, 
          label='TMF REDD+ Deforestation')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, tmf_cf, color=nonredd_col1,
          label='TMF Counterfactual Non-REDD+ Deforestation',
          linestyle = '--')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('% of Deforestation Pixels Per REDD+/Non-REDD+ Area')
plt.title('Deforestation in REDD+ vs Non-REDD+ Villages (2013-2023)')

# Add x tickmarks
plt.xticks(years, rotation=45)

# Add legend
plt.legend()

# Show the grid and plot
plt.grid(True)
plt.tight_layout()  # Adjust layout for better spacing
plt.show()


# %%
############################################################################


# PLOT DIFFERENCE IN DIFFERENCES (BAR)


############################################################################
# Set bar width
bar_width = 0.3

# Define x values
x = np.arange(len(years))

# Initiate figure
plt.figure(figsize=(10, 6))

# Add gfc data to figure
plt.bar(x - bar_width/2, gfc_dd, width=bar_width, label='GFC Deforestation', 
        color=redd_col1)

# Add tmf data to figure
plt.bar(x + bar_width/2, tmf_dd, width=bar_width, label='TMF Deforestation', 
        color=nonredd_col1)

# Add axes lables
plt.xlabel('Year')
plt.ylabel('Difference in Difference for Deforestation %')

# Add title
plt.title('Difference in Difference Analysis of GFC and TMF Deforestation')

# Add legend
plt.legend()

# Set major tickmark spacing
plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))

# Set minor tickmark spacing
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.005))

# Add gridlines
plt.grid(axis='y', which='major', linestyle = "-")
plt.grid(axis='y', which='minor', linestyle = "--")

# Add x tickmarks
plt.xticks(x, years)

# Adjust plot layout
plt.tight_layout()

# Show plot
plt.show()


# %%
############################################################################


# CALCULATE AND PLOT DIFFERENCES BETWEEN REDD / NONREDD


############################################################################
# Calculate differences for gfc data
gfc_rdif = gfc_zonal['Non-REDD+'] - gfc_zonal['REDD+']

# Calculate differences for tmf data
tmf_rdif = tmf_zonal['Non-REDD+'] - tmf_zonal['REDD+']

# Set bar width
bar_width = 0.3

# Define x values
x = np.arange(len(years))

# Initiate figure
plt.figure(figsize=(6.9, 4.5))

# Add gfc data to figure
plt.bar(x - bar_width/2, gfc_rdif, width=bar_width, label='GFC Deforestation', 
        color=redd_col1)

# Add tmf data to figure
plt.bar(x + bar_width/2, tmf_rdif, width=bar_width, label='TMF Deforestation', 
        color=nonredd_col1)

# Add axes lables
plt.xlabel('Year')
plt.ylabel('Difference b/w Non-REDD+ and REDD+ Deforestation (%)')

# Add title
plt.title('Estimated deforestation (2013-2023)')

# Add legend
plt.legend()

# Set major tickmark spacing
plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))

# Set minor tickmark spacing
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.005))

# Add gridlines
plt.grid(axis='y', which='major', linestyle = "-")
plt.grid(axis='y', which='minor', linestyle = "--")

# Add x tickmarks
plt.xticks(x, years)

# Adjust plot layout
plt.tight_layout()

# Show plot
plt.show()


# %%
############################################################################


# SIDE BY SIDE PLOT: RATES AND DIFFERENCES BETWEEN REDD / NONREDD


############################################################################

# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot 1: line of redd/nonredd, gfc/tmf defor
axes[0].plot(years, gfc_zonal['REDD+']*100, color=gfc_col, linewidth=2,
             label='GFC Deforestation in REDD+ Villages')
axes[0].plot(years, tmf_zonal['REDD+']*100, color=tmf_col, linewidth=2,
             label='TMF Deforestation and Degradation in REDD+ Villages')
axes[0].plot(years, gfc_zonal['Non-REDD+']*100, color=gfc_col, linewidth=2, 
             label='GFC Deforestation in Non-REDD+ Villages', linestyle='--')
axes[0].plot(years, tmf_zonal['Non-REDD+']*100, color=tmf_col, linewidth=2,
             label='TMF Deforestation and Degradation in Non-REDD+ Villages', 
             linestyle='--')

# Add x axis label
axes[0].set_xlabel('Year', fontsize=17)

# Add y axis label
axes[0].set_ylabel('Relative Deforestation Area (%)', fontsize=17)

# Add tickmarks
axes[0].set_xticks(years)
axes[0].tick_params(axis='both', labelsize=16)

# Edit ticklabel fontsize
axes[0].tick_params(axis='both', labelsize = 16)

# Add legend
axes[0].legend(fontsize = 14, loc = 'lower left')

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# Define bar width
bar_width = 0.4

# Define x values for bar chart
x = range(len(years))  # Assuming `x` represents years as indices

# Plot 2: bar of nonredd-redd, gfc/tmf defor
axes[1].bar(x, gfc_rdif*100, width=bar_width, label='GFC Deforestation', 
            color=redd_col1, align='center')
axes[1].bar([i + bar_width for i in x], tmf_rdif*100, width=bar_width, 
            label='TMF Deforestation', color=nonredd_col1, align='center')

# Add x axis label
axes[1].set_xlabel('Year', fontsize = 17)

# Add y axis label
axes[1].set_ylabel('Deforestation Area Difference (%)', 
                   fontsize=17)

# Add x tickmarks
axes[1].set_xticks([i + bar_width / 2 for i in x])  
axes[1].set_xticklabels(years)

# Add y tickmarks
axes[1].yaxis.set_major_locator(MultipleLocator(0.5))

# Edit ticklabel fontsize
axes[1].tick_params(axis='both', labelsize = 16)

# Add gridlines
axes[1].grid(axis='y', which='major', linestyle='--')
axes[1].grid(axis='x', linestyle = "--")

# Add legend
axes[1].legend(fontsize=16)

# Show plot
plt.tight_layout()
plt.show()






