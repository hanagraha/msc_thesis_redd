# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:37:07 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
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

# Set output directory
val_dir = os.path.join('data', 'validation')

# Set year range
years = range(2013, 2024)

# Define Color Palatte (3 colors)
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]

# Define color palatte (2 colors)
gfc_col = "#820300"  # Darker Red
tmf_col = "#4682B4"  # Darker Blue - lighter

# Define dataset names
datanames = ["GFC", "TMF", "Sensitive Early"]

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k').reset_index()

# Extract redd+ polygon area (ha)
redd_ha = villages.loc[1].geometry.area / 10000

# Extract non-redd+ polygon area (ha)
nonredd_ha = villages.loc[0].geometry.area / 10000


# %%
############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read files in subfolder
def folder_files(folder, suffix):
    
    # Define folder path
    folderpath = os.path.join(val_dir, folder)
    
    # Create empty list to store files
    paths = []

    # Iterate over every item in folder
    for file in os.listdir(folderpath):
        
        # Check if file ends in suffix
        if file.endswith(suffix):
            
            # Create path for file
            filepath = os.path.join(folderpath, file)
            
            # Add file to list
            paths.append(filepath)
    
    return paths

# Define function to read files from list
def list_read(pathlist, suffix, filt = False):
    
    # Create empty dictionary to store outputs
    files = {}
    
    # Iterate over each file in list
    for path in pathlist:
        
        # Read file
        data = pd.read_csv(path)
        
        # Extract file name
        filename = os.path.basename(path)
        
        # Remove suffix from filename
        var = filename.replace(suffix, "")
        
        # Add data to dictionary
        files[var] = data
    
    # If filter is true
    if filt == True:
        
        # Iterate over each read file
        for key in files:
            
            # Subset to only keep years 2013-2023
            files[key] = files[key][(files[key]['year'] >= 2013) & \
                                    (files[key]['year'] <= 2023)]
            
            # Reset index
            files[key] = files[key].reset_index(drop = True)
    
    return files

# Define function to extract area from map
def map_area(path):
    
    # Read raster
    with rasterio.open(path) as rast:
        
        # Read data
        data = rast.read()
        
        # Calculate number of non-na pixels
        pixels = np.sum(data != nodata_val)
        
        # Convert pixels to ha
        ha = pixels * 0.09
        
    return ha

# Read protocol b statistics (no buffer)
protb_statpaths = folder_files("val_protb", "stehmanstats.csv")
protb_stats = list_read(protb_statpaths, "_stehmanstats.csv", filt = True)

# Read protocol b statistics (with buffer)
protb_statpaths_buff = folder_files("val_protb_buff", "stehmanstats.csv")
protb_stats_buff = list_read(protb_statpaths_buff, "_stehmanstats.csv", filt = True)

# Read protocol c statistics (no buffer)
protc_statpaths = folder_files("val_protc", "stehmanstats.csv")
protc_stats = list_read(protc_statpaths, "_stehmanstats.csv", filt = True)

# Read protocol c statistics (with buffer)
protc_statpaths_buff = folder_files("val_protc_buff", "stehmanstats.csv")
protc_stats_buff = list_read(protc_statpaths_buff, "_stehmanstats.csv", filt = True)

# Calculate aoi map area
total_ha = map_area("data/validation/stratification_maps/stratification_layer_nogrnp.tif")

# Calculate redd+ map area
redd_ha = map_area("data/validation/stratification_maps/stratification_layer_redd.tif")

# Calculate nonredd+ map area
nonredd_ha = map_area("data/validation/stratification_maps/stratification_layer_nonredd.tif")

# Read gfc deforestation data
with rasterio.open("data/hansen_preprocessed/gfc_lossyear_fm.tif") as gfc:
    gfc_defor = gfc.read(1)
    
# Read tmf deforestation data
with rasterio.open("data/jrc_preprocessed/tmf_defordegrayear_fm.tif") as tmf:
    tmf_defor = tmf.read(1)

# Read se deforestation data
with rasterio.open("data/intermediate/gfc_tmf_sensitive_early.tif") as se:
    se_defor = se.read(1)


# %%
############################################################################


# CALCULATE ERROR ADJUSTED AREA AND CONFIDENCE INTERVALS


############################################################################
# Define function to calculate error adjusted area and ci
def calc_eea(data_dict):
    
    # Create a copy of the input dictionary
    eea_dict = data_dict.copy()
    
    # Iterate over each dictionary iem
    for key, value in eea_dict.items():
        
        # If the key is for nonredd areas
        if "nonredd" in key:
            
            # Calculate error adjsuted area
            area = value['area'] * nonredd_ha
            
            # Calculate area standard error
            error = value['se_a'] * nonredd_ha
            
        # If the key is for redd areas
        elif "redd" in key:
            
            # Calculate error adjusted area
            area = value['area'] * redd_ha
            
            # Calculate area standard error
            error = value['se_a'] * redd_ha 
            
        # If the key is for the whole area
        else: 
    
            # Calculate error adjusted area
            area = value['area'] * total_ha
            
            # Extract area standard error
            error = value['se_a'] * total_ha
        
        # Calculate 95% confidence interval
        ci95 = 1.96 * error
        
        # Calculate 50% confidence interval
        ci50 = 0.67 * error
        
        # Add error adjusted area to df
        value['eea'] = area
        
        # Add 95ci to df
        value['ci95'] = ci95
        
        # Add 50ci to df
        value['ci50'] = ci50
    
    return eea_dict

# Calculate eea and ci for prot b (no buffer)
protb_eea = calc_eea(protb_stats) 

# Calculate eea and ci for prot b (with buffer)
protb_eea_buff = calc_eea(protb_stats_buff)    

# Calculate eea and ci for prot c (no buffer)
protc_eea = calc_eea(protc_stats)   

# Calculate eea and ci for prot c (with buffer)
protc_eea_buff = calc_eea(protc_stats_buff)  


# %%
############################################################################


# EXTRACT TOTAL DEFORESTATION FROM TMF, GFC, AND SE


############################################################################
# Define function to extract multiyear statistics
def multiyear_defor(arr, yearrange):
    
    # Extract unique values and pixel counts
    unique_values, pixel_counts = np.unique(arr, return_counts=True)
    
    # Create dataframe with unique values and pixel counts
    attributes = pd.DataFrame({"year": unique_values, 
                               "defor_ha": pixel_counts * 0.09})
    
    # Filter only for attributes within yearrange
    filt_atts = attributes[(attributes['year'] >= min(yearrange)) & \
                           (attributes['year'] <= max(yearrange))]
    
    return filt_atts

# Calculate gfc deforestation
gfc_att = multiyear_defor(gfc_defor, years)

# Calculate tmf deforestation
tmf_att = multiyear_defor(tmf_defor, years)

# Calculate se deforestation
se_att = multiyear_defor(se_defor, years)


# %%
############################################################################


# PLOTTING FUNCTIONS


############################################################################
# Define function to plot redd and non-redd changes
def redd_comp(defor_dict, lab):
    
    # Initialize figure with subplots
    fig, axes = plt.subplots(1, 2, figsize = (18,6))
    
    # Calculate the common y-axis limits
    min_y = min((defor_dict[f"{lab}_redd"]['eea'] - 
                 defor_dict[f"{lab}_redd"]['ci50']).min(), 
                (defor_dict[f"{lab}_nonredd"]['eea'] - 
                 defor_dict[f"{lab}_nonredd"]['ci50']).min())
    max_y = max((defor_dict[f"{lab}_redd"]['eea'] + 
                 defor_dict[f"{lab}_redd"]['ci50']).max(), 
                (defor_dict[f"{lab}_nonredd"]['eea'] + 
                 defor_dict[f"{lab}_nonredd"]['ci50']).max())
    
    # Add padding for better visualization
    padding = 0.05 * (max_y - min_y)
    min_y -= padding
    max_y += padding
    
    # Set y-axis limits for both axes
    axes[0].set_ylim(min_y, max_y)
    axes[1].set_ylim(min_y, max_y)
    
    # PLOT 1: REDD DEFOR AREA
    
    # Create 95% ci rectangle
    axes[0].fill_between(
        years, 
        defor_dict[f"{lab}_redd"]['eea'][0] - defor_dict[f"{lab}_redd"]['ci95'][0],
        defor_dict[f"{lab}_redd"]['eea'][0] + defor_dict[f"{lab}_redd"]['ci95'][0],
        color = bluecols[1],
        alpha = 0.2,
        label = "95% confidence interval"
        )
    
    # Create 50% ci rectangle
    axes[0].fill_between(
        years, 
        defor_dict[f"{lab}_redd"]['eea'][0] - defor_dict[f"{lab}_redd"]['ci50'][0],
        defor_dict[f"{lab}_redd"]['eea'][0] + defor_dict[f"{lab}_redd"]['ci50'][0],
        color = bluecols[1],
        alpha = 0.3,
        label = "50% confidence interval"
        )
    
    # Plot redd defor
    axes[0].errorbar(
        years,
        defor_dict[f"{lab}_redd"]['eea'],
        yerr = defor_dict[f"{lab}_redd"]['ci50'],
        fmt="-o",
        capsize = 5,
        color = bluecols[0],
        label = "REDD+ Deforestation"
    )
    
    # Add x-axis tick marks
    axes[0].set_xticks(years)
    
    # Adjust fontsize of tick labels
    axes[0].tick_params(axis='both', which='major', labelsize=14)

    # Add axes labels
    axes[0].set_xlabel("Year", fontsize=14)
    axes[0].set_ylabel("Error-Adjusted Deforestation Area (ha)", fontsize=14)

    # Add a title and legend
    # axes[0].set_title("REDD+ Error-Adjusted Deforestation Area")
    axes[0].legend(fontsize=16, loc = "upper right")

    # Add gridlines
    axes[0].grid(linestyle="--", alpha=0.6)
    
    # PLOT 2: NONREDD DEFOR AREA
    
    # Create 95% ci rectangle
    axes[1].fill_between(
        years, 
        defor_dict[f"{lab}_nonredd"]['eea'][0] - defor_dict[f"{lab}_nonredd"]['ci95'][0],
        defor_dict[f"{lab}_nonredd"]['eea'][0] + defor_dict[f"{lab}_nonredd"]['ci95'][0],
        color = bluecols[1],
        alpha = 0.2,
        label = "95% confidence interval"
        )
    
    # Create 50% ci rectangle
    axes[1].fill_between(
        years, 
        defor_dict[f"{lab}_nonredd"]['eea'][0] - defor_dict[f"{lab}_nonredd"]['ci50'][0],
        defor_dict[f"{lab}_nonredd"]['eea'][0] + defor_dict[f"{lab}_nonredd"]['ci50'][0],
        color = bluecols[1],
        alpha = 0.3,
        label = "50% confidence interval"
        )
    
    # Plot nonredd defor
    axes[1].errorbar(
        years,
        defor_dict[f"{lab}_nonredd"]['eea'],
        yerr = defor_dict[f"{lab}_nonredd"]['ci50'],
        fmt="-o",
        capsize = 5,
        color = bluecols[0],
        label = "Non-REDD+ Deforestation"
    )
    
    # Add x-axis tick marks
    axes[1].set_xticks(years)
    
    # Adjust font size of tick labels
    axes[1].tick_params(axis='both', which='major', labelsize=14)

    # Add axes labels
    axes[1].set_xlabel("Year", fontsize=14)
    axes[1].set_ylabel("Error-Adjusted Deforestation Area (ha)", fontsize=14)

    # Add a title and legend
    # axes[1].set_title("Non-REDD+ Error-Adjusted Deforestation Area")
    axes[1].legend(fontsize=16, loc = "upper right")

    # Add gridlines
    axes[1].grid(linestyle="--", alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.show()
    
# Define function to plot eea and pred defor
def defor_comp(defor_dict, lab):
    
    # Initialize figure
    plt.figure(figsize = (10, 6))
    
    # Add gfc data to figure
    plt.plot(years, gfc_att['defor_ha'], color = bluecols[0], linewidth = 2, 
             label = "GFC Deforestation")
    
    # Add tmf data to figure
    plt.plot(years, tmf_att['defor_ha'], color = bluecols[1], linewidth = 2, 
             label = "TMF Deforestation")
    
    # Add se data to figure
    plt.plot(years, se_att['defor_ha'], color = bluecols[1], linewidth = 2, 
             label = "Sensitive Early Deforestation", linestyle = "--")
    
    # Add eea data to figure
    plt.errorbar(years, defor_dict[lab]['eea'], yerr = defor_dict[f"{lab}_nonredd"]['ci95'], 
                 fmt="-o", capsize = 5, color = bluecols[2], linewidth = 2, 
                 label = "Error Adjusted Deforestation")
    
    # Add axes abels
    plt.xlabel("Year", fontsize = 14)
    plt.ylabel("Deforestation Area (ha)", fontsize = 14)
    
    # Add x tickmarks
    plt.xticks(years)
    
    # Adjust font size of tick labels
    plt.tick_params(axis='both', which='major', labelsize = 14)
    
    # Add legend
    plt.legend(fontsize = 16)
    
    # Add gridlines 
    plt.grid(linestyle = "--", alpha = 0.6)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    

# %%
############################################################################


# PLOT ERROR ADJUSTED AREA WITH CONFIDENCE INTERVALS


############################################################################
# Plot prot b redd and nonredd (gfc)
redd_comp(protb_eea, "protb_gfc")

# Plot prot b redd and nonredd (tmf)
redd_comp(protb_eea, "protb_tmf")

# Plot prot b redd and nonredd (se)
redd_comp(protb_eea, "protb_se")

# Plot prot c redd and nonredd (gfc)
redd_comp(protc_eea, "protc_gfc")

# Plot prot c redd and nonredd (tmf)
redd_comp(protc_eea, "protc_tmf")

# Plot prot c redd and nonredd (se)
redd_comp(protc_eea, "protc_se")

# Plot prot b redd and nonredd (gfc, buffered)
redd_comp(protb_eea_buff, "protb_gfc")

# Plot prot b redd and nonredd (tmf, buffered)
redd_comp(protb_eea_buff, "protb_tmf")

# Plot prot b redd and nonredd (se, buffered)
redd_comp(protb_eea_buff, "protb_se")

# Plot prot c redd and nonredd (gfc, buffered)
redd_comp(protc_eea_buff, "protc_gfc")

# Plot prot c redd and nonredd (tmf, buffered)
redd_comp(protc_eea_buff, "protc_tmf")

# Plot prot c redd and nonredd (se, buffered)
redd_comp(protc_eea_buff, "protc_se")


# %%
############################################################################


# PLOT MAP AND ERROR ADJUSTED DEFORESTATION AREA


############################################################################
# Plot prot c map and eea
defor_comp(protc_eea, "protc_gfc")

# Plot prot c map and eea (buffered)
defor_comp(protc_eea_buff, "protc_gfc")












