# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:34:35 2024

@author: hanna

This file calculates spatial agreement statistics: overall agreement, 
agreement proportional to undisturbed area, and agreement proportional to 
deforested/disturbed area. 

Estimated runtime: ~1min
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Define study range years
years = range(2013, 2024)

# Set default plotting colors
defaultblue = "#4682B4"
reddcol = "brown"
nonreddcol = "dodgerblue"
grnpcol = "darkgreen"

reddcol = "#820300"  # Darker Red
grnpcol = "#4682B4"  # Darker Blue - lighter

# Define Color Palatte (3 colors)
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read list of files
def read_files(pathlist):
    
    # Create empty list to hold arrays
    arrlist = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read file
        with rasterio.open(path) as rast:
            data = rast.read(1)
            profile = rast.profile
            
            # Add array to list
            arrlist.append(data)
            
    return arrlist, profile

# Define spatial agreement paths
spatagree_paths = [f"data/intermediate/agreement_gfc_combtmf_{year}.tif" for 
                   year in years]

# Define gfc lossyear paths
gfc_paths = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif" for year 
             in years]

# Define tmf defordegra paths
tmf_paths = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif" for 
             year in years]

# Read spatial agreement rasters
spatagree_arrs, spatagree_profile = read_files(spatagree_paths)

# Read gfc paths
gfc_arrs, gfc_profile = read_files(gfc_paths)

# Read tmf paths
tmf_arrs, tmf_profile = read_files(tmf_paths)

# Read vector data
villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create REDD+ and non-REDD+ polygons
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k')

# Create REDD+ and non-REDD+ geometries
redd_geom = [villages.loc[1, 'geometry']]
nonredd_geom = [villages.loc[0, 'geometry']]

# Create GRNP geometry
grnp_geom = grnp.geometry


# %%
############################################################################


# SPLIT DATA INTO REDD+, NON-REDD+, GRNP


############################################################################
# Define function for clipping stack of agreement rasters
def regions_clip(pathlist):
    
    # Create empty list to hold arrays
    redd_arrs = []
    nonredd_arrs = []
    grnp_arrs = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read raster data
        with rasterio.open(path) as rast:
            
            # Mask + crop data to redd area
            redd_arr, transform = mask(rast, redd_geom, crop = True, 
                                       nodata = nodata_val)
            
            # Mask + crop data to nonredd area
            nonredd_arr, transform = mask(rast, nonredd_geom, crop = True, 
                                          nodata = nodata_val)
            
            # Mask + crop data to redd area
            grnp_arr, transform = mask(rast, grnp_geom, crop = True, 
                                       nodata = nodata_val)
            
        # Add array to list
        redd_arrs.append(redd_arr)
        nonredd_arrs.append(nonredd_arr)
        grnp_arrs.append(grnp_arr)

    return redd_arrs, nonredd_arrs, grnp_arrs

# Clip agreement to redd+, nonredd+, and grnp area
ag_redd, ag_nonredd, ag_grnp = regions_clip(spatagree_paths)

# Clip gfc arrays to redd+, nonredd+, and grnp area
gfc_redd, gfc_nonredd, gfc_grnp = regions_clip(gfc_paths)

# Clip tmf arrays to redd+, nonredd+, and grnp area
tmf_redd, tmf_nonredd, tmf_grnp = regions_clip(tmf_paths)



# %%
############################################################################


# PLOTTING OPTIONS


############################################################################
# Define function to plot line graph
def lineplot(ydata, title, ylab, yaxis, lim_low, lim_up):
    
    # Initialize figure
    plt.figure(figsize = (10,6))
    
    # Plot line data
    plt.plot(years, ydata, label = ylab, color = defaultblue)
    
    # Add title
    plt.title(title, fontsize = 16)
    
    # Add axes labels
    plt.xlabel("Year", fontsize = 16)
    plt.ylabel(yaxis, fontsize = 16)
    
    # Add gridlines
    plt.grid(True, linestyle = "--", alpha = 0.6)
    
    # Add legend
    plt.legend(loc='best', fontsize = 16)
    
    # Add x tickmarks
    plt.xticks(years, rotation=0, fontsize = 16)
    plt.yticks(fontsize = 16)
    
    # Adjust yaxis limits
    plt.ylim(lim_low, lim_up)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Define function to plot line graph with redd, nonredd, and grnp data
def tripleplot(redd, nonredd, grnp, title, yaxis, lim_low, lim_up):
    
    # Initialize figure
    plt.figure(figsize = (10, 6))
    
    # Plot redd+ data
    plt.plot(years, redd, label = "REDD+", color = reddcol)
    
    # Plot nonredd data
    plt.plot(years, nonredd, label = "Non-REDD+", color = reddcol, 
             linestyle = "--")
    
    # Plot grnp data
    plt.plot(years, grnp, label = "GRNP", color = grnpcol)
    
    # Add title
    plt.title(title, fontsize = 16)
    
    # Add axes labels
    plt.xlabel("Year", fontsize = 16)
    plt.ylabel(yaxis, fontsize = 16)
    
    # Add gridlines
    plt.grid(True, linestyle = "--")
    
    # Add legend
    plt.legend(loc='best', fontsize = 16)
    
    # Add x tickmarks
    plt.xticks(years, fontsize = 16)
    plt.yticks(fontsize = 16)
    
    # Adjust yaxis limits
    plt.ylim(lim_low, lim_up)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# %%
############################################################################


# OVERALL AGREEMENT


############################################################################
# Define function to calculate overall agreement
def ov_agree(spatagree_arrs):
    
    # Create empty list to hold statistics
    prop_agree = []
    
    # Iterate over each array
    for arr in spatagree_arrs:
        
        # Calculate total agreement
        ag = (np.sum(arr == 5)) + (np.sum(arr == 8))
        
        # Calculate total disagreement
        disag = (np.sum(arr == 6)) + (np.sum(arr == 7))
        
        # Calculate overall agreement
        ov_ag = (ag / (ag + disag))*100
        
        # Add overall agreement to list
        prop_agree.append(ov_ag)
        
    return prop_agree
    
# Calculate overall agreement
ov_ag_aoi = ov_agree(spatagree_arrs)

# Calculate overall agreement for redd
ov_ag_redd = ov_agree(ag_redd)

# Calculate overall agreement for nonredd area
ov_ag_nonredd = ov_agree(ag_nonredd)

# Calculate overall agreement for grnp area
ov_ag_grnp = ov_agree(ag_grnp)

# Plot overall agreement for aoi
lineplot(ov_ag_aoi, "Overall Spatial Agreement between GFC and TMF Datasets", 
         "AOI", "Overall Agreement (%)", 95, 98)

# Plot overall agreement for redd, nonredd, and grnp area
tripleplot(ov_ag_redd, ov_ag_nonredd, ov_ag_grnp, 
           "Overall Spatial Agreement between GFC and TMF Datasets", 
           "Overall Agreement (%)", 92, 100)


# %%
############################################################################


# UNDISTURBED AGREEMENT


############################################################################
# Define function to calculate overall agreement
def for_agree(spatagree_arrs):
    
    # Create empty list to hold statistics
    prop_agree = []
    
    # Iterate over each array
    for arr in spatagree_arrs:
        
        # Calculate total agreement
        ag = (np.sum(arr == 5))
        
        # Calculate total disagreement
        disag = (np.sum(arr == 6)) + (np.sum(arr == 7))
        
        # Calculate overall agreement
        ov_ag = (ag / (ag + disag))*100
        
        # Add overall agreement to list
        prop_agree.append(ov_ag)
        
    return prop_agree
    
# Calculate overall agreement
for_ag_aoi = for_agree(spatagree_arrs)

# Calculate overall agreement for redd
for_ag_redd = for_agree(ag_redd)

# Calculate overall agreement for nonredd area
for_ag_nonredd = for_agree(ag_nonredd)

# Calculate overall agreement for grnp area
for_ag_grnp = for_agree(ag_grnp)

# Plot overall agreement for aoi
lineplot(for_ag_aoi, "Undisturbed Spatial Agreement between GFC and TMF Datasets", 
         "AOI", "Undisturbed Agreement (%)", 95, 98)

# Plot overall agreement for redd, nonredd, and grnp area
tripleplot(for_ag_redd, for_ag_nonredd, for_ag_grnp, 
           "Spatial Agreement Relative to Undisturbed Area", 
           "Undisturbed Agreement (%)", 92, 100)


# %%
############################################################################


# DEFORESTATION AGREEMENT


############################################################################
# Define function to calculate deforestation agreement
def defor_agree(spatagree_arrs):
    
    # Create empty list to hold statistics
    prop_agree = []
    
    # Iterate over each array
    for arr in spatagree_arrs:
        
        # Calculate total agreement
        ag = np.sum(arr == 8)
        
        # Calculate total disagreement
        disag = (np.sum(arr == 6)) + (np.sum(arr == 7))
        
        # Calculate overall agreement
        ov_ag = (ag / (ag + disag))*100
        
        # Add overall agreement to list
        prop_agree.append(ov_ag)
        
    return prop_agree

# Calculate deforestation agreement for aoi
defor_ag_aoi = defor_agree(spatagree_arrs)

# Calculate deforestation agreement for redd
defor_ag_redd = defor_agree(ag_redd)

# Calculate deforestation agreement for nonredd area
defor_ag_nonredd = defor_agree(ag_nonredd)

# Calculate deforestation agreement for grnp area
defor_ag_grnp = defor_agree(ag_grnp)

# Plot deforestation agreement for aoi
lineplot(defor_ag_aoi, "Deforestation Spatial Agreement between GFC and TMF Datasets", 
         "AOI", "Deforestation Agreement (%)", 10, 30)

# Plot deforestation agreement for redd, nonredd, and grnp area
tripleplot(defor_ag_redd, defor_ag_nonredd, defor_ag_grnp, 
           "Spatial Agreement Relative to Deforestation Area", 
           "Deforestation Agreement (%)", 0, 35)


# %%
############################################################################


# ADJUST PLOTTING: DEFORESTATION AGREEMENT


############################################################################
# Initialize figure
plt.figure(figsize = (10, 6))

# Plot redd+ data
plt.plot(years, defor_ag_redd, label = "REDD+", color = reddcol, 
         linewidth = 2)

# Plot nonredd data
plt.plot(years, defor_ag_nonredd, label = "Non-REDD+", color = reddcol, 
         linewidth = 2, linestyle = "--")

# Plot grnp data
plt.plot(years, defor_ag_grnp, label = "GRNP+", color = grnpcol, 
         linewidth = 2, linestyle = "-")

# Add axes labels
plt.xlabel("Year", fontsize = 16)
plt.ylabel("Deforestation Agreement (%)", fontsize = 16)

# Add gridlines
plt.grid(True, linestyle = "--", alpha = 0.6)

# Add legend
plt.legend(loc='best', fontsize = 16)

# Add x tickmarks
plt.xticks(years, fontsize = 16)
plt.yticks(fontsize = 16)

# Adjust yaxis limits
plt.ylim(0, 35)

# Show the plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# PLOT SIDE BY SIDE: OVERALL AND DEFORESTATION AGREEMENT


############################################################################
# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: overall spatial agreement
axes[0].plot(years, ov_ag_redd, color=reddcol, linewidth=2,
             label='REDD+')
axes[0].plot(years, ov_ag_nonredd, color=reddcol, linewidth=2,
             label='non-REDD', linestyle = "--")
axes[0].plot(years, ov_ag_grnp, color=grnpcol, linewidth=2, 
             label='GRNP')

# Add x axis label
axes[0].set_xlabel('Year', fontsize=16)

# Add y axis label
axes[0].set_ylabel('Overall Agreement (%)', fontsize=16)

# Add tickmarks
axes[0].set_xticks(years)
axes[0].tick_params(axis='both', labelsize=14)

# Add legend
axes[0].legend(fontsize=16, loc = 'lower right')

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# Adjust yaxis limits
axes[0].set_ylim(92, 100)

# Plot 2: deforestation spatial agreement
axes[1].plot(years, defor_ag_redd, label = "REDD+", color = reddcol)
axes[1].plot(years, defor_ag_nonredd, label = "Non-REDD+", color = reddcol, 
             linestyle = "--")
axes[1].plot(years, defor_ag_grnp, label = "GRNP", color = grnpcol)

# Add tickmarks
axes[1].set_xticks(years)
axes[1].tick_params(axis='both', labelsize=14)

# Add x axis label
axes[1].set_xlabel('Year', fontsize=16)

# Add y axis label
axes[1].set_ylabel('Deforestation Agreement (%)', fontsize=16)

# Add gridlines
axes[1].grid(True, linestyle = "--")

# Adjust yaxis limits
axes[1].set_ylim(0, 35)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# CALCULATE POTENTIAL DEFORESTATION RANGES


############################################################################
# Define function to calculate potential deforestation range
def defor_range(spatagree_arrs):
    
    # Create empty list to hold statistics
    min_defors = []
    max_defors = []
    ranges = []
    
    # Iterate over each array
    for arr in spatagree_arrs:
        
        # Calculate min potential defor (defor agree)
        min_defor = np.sum(arr == 8)
        
        # Calculate max potential defor (all defor)
        max_defor = (np.sum(arr == 6)) + (np.sum(arr == 7))
        
        # Add min and max defor to lists
        min_defors.append(min_defor)
        max_defors.append(max_defor)
        ranges.append(max_defor - min_defor)
        
    # Combine lists into dataframe
    defor_range = pd.DataFrame({"Min": min_defors,
                                "Max": max_defors,
                                "Range": ranges})
        
    return defor_range
    
# Calculate potential deforestation range
potdefor = defor_range(spatagree_arrs)

# Calculate sum potential deforestation
print(f"Potential Minimum Deforestation: {potdefor['Min'].sum()} ha")
print(f"Potential Maximum Deforestation: {potdefor['Max'].sum()} ha")
print(f"Potential Variation in Deforestation: {potdefor['Range'].sum()} ha")





















