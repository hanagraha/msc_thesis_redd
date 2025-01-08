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
import seaborn as sns



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

# Set year range
years = range(2013, 2024)

# other colors test
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]

# Define dataset names
datanames = ["GFC", "TMF", "Sensitive Early"]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Read gfc stehman statistic data (calculated in R, pre-processed 2)
gfc_stats = pd.read_csv("data/validation/proc2_gfc_stehmanstats.csv", delimiter=",")

# Read tmf stehman statistic data (calculated in R, pre-processed 2)
tmf_stats = pd.read_csv("data/validation/proc2_tmf_stehmanstats.csv", delimiter=",")

# Read se stehman statistic data (calculated in R, pre-processed 2)
se_stats = pd.read_csv("data/validation/proc2_se_stehmanstats.csv", delimiter=",")

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



############################################################################


# CALCULATE ERROR-ADJUSTED AREA OF DEFORESTATION


############################################################################
# Calculate total map pixels
total_pix = np.sum(gfc_defor != np.nan)

# Convert map pixels to map area (ha)
total_ha = total_pix * 0.09

# Calculate gfc error-adjusted defor
gfc_ha = gfc_stats['area'] * total_ha

# Calculate tmf error-adjusted defor
tmf_ha = tmf_stats['area'] * total_ha

# Calculate se error-adjusted defor
se_ha = se_stats['area'] * total_ha



############################################################################


# CALCULATE 95% CONFIDENCE INTERVAL


############################################################################
"""
Conversion of standard error to confidence interval uses the constant 1.96
taken from https://pmc.ncbi.nlm.nih.gov/articles/PMC1255808/
"""

# Calculate standard error of gfc area estimate
gfc_se = total_ha * gfc_stats['se_a']

# Calculate standard error of tmf area estimate
tmf_se = total_ha * tmf_stats['se_a']

# Calculate standard error of se area estimate
se_se = total_ha * se_stats['se_a']

# Calculate gfc 95% confidence interval
gfc_ci = 1.96 * gfc_se

# Calculate tmf 95% confidence interval
tmf_ci = 1.96 * tmf_se

# Calculate se 95% confidence interval
se_ci = 1.96 * se_se



############################################################################


# PLOT ERROR ADJUSTED AREA WITH CONFIDENCE INTERVALS


############################################################################
# Plot area estimates with error bars
defor_ci(gfc_ha[1:], tmf_ha[1:], se_ha[1:], gfc_ci[1:], tmf_ci[1:], se_ci[1:])

# Plot uncertainty rectangles
fig, ax = plt.subplots(figsize=(8, 6))

# Add the shaded rectangle based on the first GFC data point
ax.fill_between(
    years,
    gfc_ha[1] - gfc_ci[1],
    gfc_ha[1] + gfc_ci[1],
    color=bluecols[1],
    alpha=0.2,
    label="95% Confidence Interval"
)

# Plot GFC data with error bars
ax.errorbar(
    years,
    gfc_ha[1:],
    yerr=gfc_ci[1:],
    fmt="-o",
    capsize=5,
    color=bluecols[0],
    label="GFC Deforestation"
)

# Add x-axis tick marks
ax.set_xticks(years)

# Add axes labels
ax.set_xlabel("Year")
ax.set_ylabel("Error-Adjusted Deforestation Area (ha)")

# Add a title and legend
ax.set_title("GFC Deforestation Area with Confidence Interval (Shaded)")
ax.legend()

# Add gridlines
ax.grid(linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()




