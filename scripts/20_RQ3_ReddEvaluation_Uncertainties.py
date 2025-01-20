# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:56:35 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
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
def list_read(pathlist, suffix):
    
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
        
    return files
        
# Read validation data
val_data = pd.read_csv("data/validation/validation_points.csv")

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

# Read protocol a data
prota_filepaths = folder_files("val_prota", ".csv")
prota_files = list_read(prota_filepaths, ".csv")

# Read protocol b statistics
protb_statpaths = folder_files("val_protb", "stehmanstats.csv")
protb_stats = list_read(protb_statpaths, "_stehmanstats.csv")

# Read protocol c statistics
protc_statpaths = folder_files("val_protc", "stehmanstats.csv")
protc_stats = list_read(protc_statpaths, "_stehmanstats.csv")

# Read protocol d statistics
protd_statpaths = folder_files("val_protd", "stehmanstats.csv")
protd_stats = list_read(protd_statpaths, "_stehmanstats.csv")

# # Read gfc stehman statistic data (calculated in R, pre-processed 2)
# gfc_stats = pd.read_csv("data/validation/proc2_gfc_stehmanstats.csv", delimiter=",")

# # Read tmf stehman statistic data (calculated in R, pre-processed 2)
# tmf_stats = pd.read_csv("data/validation/proc2_tmf_stehmanstats.csv", delimiter=",")

# # Read se stehman statistic data (calculated in R, pre-processed 2)
# se_stats = pd.read_csv("data/validation/proc2_se_stehmanstats.csv", delimiter=",")

# Define gfc lossyear filepath
gfc_lossyear_file = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Define tmf defordegra filepath
tmf_defordegra_file = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# Define sensitive early filepath
se_file = "data/intermediate/gfc_tmf_sensitive_early.tif"

# Read gfc deforestation data
with rasterio.open(gfc_lossyear_file) as gfc:
    gfc_defor = gfc.read(1)
    gfc_profile = gfc.profile

# Read tmf deforestation data
with rasterio.open(tmf_defordegra_file) as tmf:
    tmf_defor = tmf.read(1)
    profile = tmf.profile

# Read se deforestation data
with rasterio.open(se_file) as se:
    se_defor = se.read(1)
    profile = se.profile


# %%
############################################################################


# PLOTTING FUNCTIONS


############################################################################
# Define function to plot gfc, tmf, and se data
def defor_ci(gfc_area, tmf_area, se_area, gfc_ci, tmf_ci, se_ci):
    
    # Initiate figure
    fig, ax = plt.subplots(figsize = (8,6))
    
    # Plot gfc data
    ax.errorbar(years, gfc_area, yerr = gfc_ci, capsize = 5, color = 
                bluecols[0], label = "GFC")
    
    # Plot tmf data
    ax.errorbar(years, tmf_area, yerr = tmf_ci, capsize = 5, color = 
                bluecols[1], label = "TMF")
    
    # Plot se data
    ax.errorbar(years, se_area, yerr = se_ci, capsize = 5, color = 
                bluecols[2], label = "Sensitive Early")
    
    # Add x tickmarks
    ax.set_xticks(list(years))
    
    # Add axes labels
    ax.set_xlabel("Year")
    ax.set_ylabel("Error-Adjusted Deforestation Area (ha)")
    
    # Add legend
    ax.legend()
    
    # Add gridlnies
    ax.grid(linestyle = '--', alpha = 0.6)
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Define function to plot defor with ci shaded regions
def defor_uncert(defordata, ci95, ci50, lab):
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot 95% ci rectangle
    ax.fill_between(
        years,
        defordata[1] - ci95[1],
        defordata[1] + ci95[1],
        color=bluecols[1],
        alpha=0.2,
        label="95% Confidence Interval"
    )
    
    # Plot 50% ci rectangle
    ax.fill_between(
        years,
        defordata[1] - ci50[1],
        defordata[1] + ci50[1],
        color=bluecols[1],
        alpha=0.3,
        label="50% Confidence Interval"
    )

    # Plot GFC data with error bars
    ax.errorbar(
        years,
        defordata[1:],
        yerr=ci95[1:],
        fmt="-o",
        capsize=5,
        color=bluecols[0],
        label=f"{lab} Deforestation"
    )

    # Add x-axis tick marks
    ax.set_xticks(years)

    # Add axes labels
    ax.set_xlabel("Year")
    ax.set_ylabel("Error-Adjusted Deforestation Area (ha)")

    # Add a title and legend
    ax.set_title(f"{lab} Deforestation Area with Confidence Interval (Shaded)")
    ax.legend()

    # Add gridlines
    ax.grid(linestyle="--", alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.show()
    
# Define function to plot multiple lines
def multline(maparea, errorarea, dataname):
    
    # Initialize figure
    plt.figure(figsize=(10, 6))

    # Plot map deforestation
    plt.plot(years, maparea, label=f'{dataname} Map Deforestation', 
             color=bluecols[0], linewidth=2)
    
    # Plot error adjusted deforestation
    plt.plot(years, errorarea, label=f'{dataname} Error-Adjusted Deforestation', 
             color=bluecols[1], linewidth=2)
    
    # Add labels and legend
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Deforestation Area (ha)', fontsize=12)
    plt.legend(fontsize=11)
    
    # Add gridlines
    plt.grid(linestyle='--', alpha=0.6)
    
    # Customize ticks
    plt.xticks(years, fontsize=11)
    plt.yticks(fontsize=11)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
# Define function to plot deforestation prediction and error adjusted area
def pred_eaa(gfc_pred, tmf_pred, error_defor, error_ci):
    
    # Initiate figure
    fig, ax = plt.subplots(figsize = (10,6))
    
    # Plot gfc area
    ax.plot(years, gfc_pred, color=bluecols[0], linewidth = 2,
             label='GFC Deforestation')

    # Plot tmf area
    ax.plot(years, tmf_pred, color=bluecols[1], linewidth = 2,
             label='TMF Deforestation and Degradation')
    
    # Plot error adjusted area
    ax.errorbar(years, error_defor, yerr = error_ci, capsize = 5, color = 
                bluecols[2], linewidth = 2, linestyle = "--",
                label = "Error-Adjusted Deforestation (95% CI)")
    
    # Add x tickmarks
    ax.set_xticks(list(years))
    
    # Add axes labels
    ax.set_xlabel("Year", fontsize = 12)
    ax.set_ylabel("Deforestation Area (ha)", fontsize = 12)
    
    # Add legend
    ax.legend(fontsize = 11)
    
    # Add gridlnies
    ax.grid(linestyle = '--', alpha = 0.6)
    
    # Show plot
    plt.tight_layout()
    plt.show()


# %%
############################################################################


# PROTOCOL B: CALCULATE ERROR-ADJUSTED AREA OF DEFORESTATION + CI


############################################################################
# Calculate total map pixels
total_pix = np.sum(gfc_defor != np.nan)

# Convert map pixels to map area (ha)
total_ha = total_pix * 0.09

# Calculate gfc error-adjusted defor
gfc_ha = protb_stats['protb_gfc']['area'] * total_ha

# Calculate tmf error-adjusted defor
tmf_ha = protb_stats['protb_tmf']['area'] * total_ha

# Calculate se error-adjusted defor
se_ha = protb_stats['protb_se']['area'] * total_ha


# %%
############################################################################


# PROTOCOL D: CALCULATE ERROR-ADJUSTED AREA OF DEFORESTATION + CI


############################################################################
"""
Because protocol D always takes the first deforestation event from the 
validation dataset, the area estimation between gfc, tmf, and se are the same
"""

# Calculate error-adjusted defor area
gfc_ha = protd_stats['protd_gfc']['area'] * total_ha

# Calculate redd error-adjusted defor area
gfc_ha_redd = protd_stats['protd_gfc_redd']['area'] * total_ha

# Calculate nonredd error-adjusted defor area
gfc_ha_nonredd = protd_stats['protd_gfc_nonredd']['area'] * total_ha

# Calculate standard error of gfc area estimate
gfc_se = total_ha * protd_stats['protd_gfc']['se_a']

# Calculate standard error of redd area estimate
gfc_se_redd = total_ha * protd_stats['protd_gfc_redd']['se_a']

# Calculate standard error of nonredd area estimate
gfc_se_nonredd = total_ha * protd_stats['protd_gfc_nonredd']['se_a']

# Calculate gfc 95% confidence interval
gfc_95ci = 1.96 * gfc_se

# Calculate tmf 95% confidence interval
redd_95ci = 1.96 * gfc_se_redd

# Calculate se 95% confidence interval
nonredd_95ci = 1.96 * gfc_se_nonredd

# Calculate gfc 50% confidence interval
gfc_50ci = 0.67 * gfc_se

# Calculate tmf 50% confidence interval
redd_50ci = 0.67 * gfc_se_redd

# Calculate se 50% confidence interval
nonredd_50ci = 0.67 * gfc_se_nonredd


# %%
############################################################################


# PROTOCOL D: PLOT ERROR ADJUSTED AREA WITH CONFIDENCE INTERVALS


############################################################################
# Plot area estimates with error bars
defor_ci(gfc_ha[1:], gfc_ha_redd[1:], gfc_ha_nonredd[1:], gfc_95ci[1:], 
         redd_95ci[1:], nonredd_95ci[1:])

# Plot gfc deforestation with uncertainty
defor_uncert(gfc_ha, gfc_95ci, gfc_50ci, "GFC")

# Plot tmf deforestation with uncertainty
defor_uncert(tmf_ha, tmf_95ci, tmf_50ci, "TMF")

# Plot se deforestation with uncertainty
defor_uncert(se_ha, se_95ci, se_50ci, "Sensitive Early")


# %%
############################################################################


# CALCULATE CONFIDENCE INTERVALS


############################################################################
"""
Conversion of standard error to confidence interval uses constants from 
z-tables: https://www.ztable.net/
"""

# Calculate standard error of gfc area estimate
gfc_se = total_ha * gfc_stats['se_a']

# Calculate standard error of tmf area estimate
tmf_se = total_ha * tmf_stats['se_a']

# Calculate standard error of se area estimate
se_se = total_ha * se_stats['se_a']

# Calculate gfc 95% confidence interval
gfc_95ci = 1.96 * gfc_se

# Calculate tmf 95% confidence interval
tmf_95ci = 1.96 * tmf_se

# Calculate se 95% confidence interval
se_95ci = 1.96 * se_se

# Calculate gfc 50% confidence interval
gfc_50ci = 0.67 * gfc_se

# Calculate tmf 50% confidence interval
tmf_50ci = 0.67 * tmf_se

# Calculate se 50% confidence interval
se_50ci = 0.67 * se_se



############################################################################


# PLOT ERROR ADJUSTED AREA WITH CONFIDENCE INTERVALS


############################################################################
# Plot area estimates with error bars
defor_ci(gfc_ha[1:], tmf_ha[1:], se_ha[1:], gfc_95ci[1:], tmf_95ci[1:], se_95ci[1:])

# Plot gfc deforestation with uncertainty
defor_uncert(gfc_ha, gfc_95ci, gfc_50ci, "GFC")

# Plot tmf deforestation with uncertainty
defor_uncert(tmf_ha, tmf_95ci, tmf_50ci, "TMF")

# Plot se deforestation with uncertainty
defor_uncert(se_ha, se_95ci, se_50ci, "Sensitive Early")



############################################################################


# PLOT MAP AND ERROR ADJUSTED DEFORESTATION AREA


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

# Extract annual deforestation data from gfc
gfc_atts = multiyear_atts(gfc_defor, years)

# Extract annual deforestation data from tmf
tmf_atts = multiyear_atts(tmf_defor, years)

# Convert gfc pixel number to ha
gfc_atts['AOI'] = gfc_atts['AOI'] * 0.09

# Convert tmf pixel number to ha
tmf_atts["AOI"] = tmf_atts["AOI"] * 0.09

# Plot gfc map and error adjusted area
multline(gfc_atts["AOI"], gfc_ha[1:], "GFC")

# Plot tmf map and error adjusted area
multline(tmf_atts["AOI"], tmf_ha[1:], "TMF")

# Plot gfc and tmf deforestation with area adjusted deforestation
pred_eaa(gfc_atts["AOI"], tmf_atts["AOI"], gfc_ha[1:], gfc_50ci[1:])












