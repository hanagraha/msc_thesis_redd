# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:42:18 2024

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
import matplotlib.pyplot as plt
from scipy.ndimage import label
import time
import seaborn as sns
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

# Set year range
years = range(2013, 2024)

# Set output directory
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Set default plotting colors
defaultblue = "#4682B4"
reddcol = "brown"
nonreddcol = "dodgerblue"
grnpcol = "darkgreen"



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
            profile = rast.profile
            
            # Add array to list
            arrlist.append(data)
            
    return arrlist, profile

# Define file paths for clustered gfc rasters
gfc_forclust_paths = [f"data/intermediate/gfc_forclust_{year}.tif" for 
                      year in years]

# Define file paths for clustered tmf rasters
tmf_forclust_paths = [f"data/intermediate/tmf_forclust_{year}.tif" for 
                      year in years]

# Define file paths for clustered agreement rasters
agree_forclust_paths = [f"data/intermediate/agree_forclust_{year}.tif" for 
                        year in years]

# Define file paths for clustered disagreement rasters
disagree_forclust_paths = [f"data/intermediate/disagree_forclust_{year}.tif" 
                            for year in years]

# Read gfc rasters 
gfc_forclust_arrs, profile = read_files(gfc_forclust_paths)

# Read tmf rasters
tmf_forclust_arrs, profile = read_files(tmf_forclust_paths)

# Read agreement rasters
agree_forclust_arrs, agprofile = read_files(agree_forclust_paths)

# Read disagreement rasters
disagree_forclust_arrs, agprofile = read_files(disagree_forclust_paths)



############################################################################


# CREATE BOX PLOTS OF DEFORESTATION PATCH SIZES


############################################################################
# Define function to create attribute tables
def att_table(arr_list, expected_classes = None, nodata = None):
    
    # Create empty list to hold dataframes
    att_tables = []
    
    # Iterate over each array
    for arr in arr_list:
        
        if nodata is not None:
            unique_values, pixel_counts = np.unique(arr[arr != nodata], 
                                                    return_counts=True)
        else:
            unique_values, pixel_counts = np.unique(arr, return_counts=True)
        
        # Create a DataFrame with unique values and pixel counts
        attributes = pd.DataFrame({"Class": unique_values, 
                                    "Frequency": pixel_counts})
        
        # If expected_classes is provided, run the following:
        if expected_classes is not None:
            
            # Reindex DataFrame to include all expected_classes
            attributes = attributes.set_index("Class").reindex(expected_classes, 
                                                                fill_value=0)
            
            # Reset index to have Class as a column again
            attributes.reset_index(inplace=True)
            
        # Switch rows and columns of dataframe
        attributes = attributes.transpose()
        
        # Add attributes to list
        att_tables.append(attributes)
    
    return att_tables

# Define function to restructure attribute tables for boxplots
def boxplot_dat(df_list, yearrange):
    
    # Create empty list to hold data
    boxplot_data = []

    # Iterate over each DataFrame (yearly data)
    for df, year in zip(df_list, years):
        
        # Extract class and frequency
        classes = df.iloc[0].values
        frequencies = df.iloc[1].values  
        
        # Create a list of patch sizes based on their frequency
        for class_size, freq in zip(classes, frequencies):
            
            # Append each class size 'freq' times to the boxplot_data
            boxplot_data.extend([(class_size, year)] * freq)  

    # Convert to list to dataframe
    boxplot_df = pd.DataFrame(boxplot_data, columns=['Patch Size', 'Year'])
    
    return boxplot_df

# Define function to plot data in boxplot
def forclust_bxplt(boxplot_dflist, titles):
    
    # Initialize a new DataFrame to hold all data
    combined_df = pd.DataFrame()

    # Iterate over each dataframe
    for df, title in zip(boxplot_dflist, titles):
        
        # Add dataset column
        df['Dataset'] = title
        
        # Combine with overall dataframe
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Initialize figure
    plt.figure(figsize=(14, 10))

    # Plot boxplot data
    sns.boxplot(x='Year', y='Patch Size', hue='Dataset', data=combined_df, 
                width = 0.7)

    # Edit y tickmarks
    plt.yticks(range(0, 801, 100))
    
    # Add y major gridlines
    plt.grid(axis='y', which='major', color='gray', linestyle='--', 
              linewidth=0.5)
    
    # Set locations for minor gridlines
    plt.gca().yaxis.set_minor_locator(MultipleLocator(50))
    
    # Add y minor gridlines
    plt.grid(axis='y', which='minor', color='lightgray', linestyle=':', 
              linewidth=0.5)

    # Add title
    plt.title('Distribution of Deforestation Patch Sizes by Year')

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('Deforestation Patch Sizes (# Pixels)')

    # Rotate x tickmarks for readability
    plt.xticks(rotation=45)

    # Show plot
    plt.show()

# Create attribute table for gfc clustered forest
gfc_forclust_atts = att_table(gfc_forclust_arrs, nodata = nodata_val)

# Create attribute table for tmf clustered forest
tmf_forclust_atts = att_table(tmf_forclust_arrs, nodata = nodata_val)

# Create attribute table for agreement clusters
agree_forclust_atts = att_table(agree_forclust_arrs, nodata = nodata_val)

# Create attribute table for disagreement clusters
disagree_forclust_atts = att_table(disagree_forclust_arrs, nodata = nodata_val)

# Create boxplot data for gfc clustered forest
gfc_boxplot_df = boxplot_dat(gfc_forclust_atts, years)

# Create boxplot data for tmf clustered forest
tmf_boxplot_df = boxplot_dat(tmf_forclust_atts, years)

# Create boxplot with gfc and tmf data
forclust_bxplt([gfc_boxplot_df, tmf_boxplot_df], ["GFC Deforestation", 
                                                  "TMF Deforestation"])



############################################################################


# BIN AGREEMENT AND DISAGREEMENT CLUSTER SIZES


############################################################################
# Define function to bin cluster sizes from dataframe
def bin_clusts(df_list, bins_list, bin_labels):

    # Create empty list to hold the binned frequencies
    binned_frequencies = []
    
    # Iterate through each DataFrame
    for df in df_list:
        
        # Create empty list for frequency sum 
        df_binned_freq = []
        
        # Iterate through each bin (min_patch_size, max_patch_size)
        for min_patch, max_patch in bins_list:
            
            # Create a mask to select patch sizes within the current bin range
            mask = (df.iloc[0] >= min_patch) & (df.iloc[0] < max_patch)
            
            # Sum the frequencies corresponding to patch sizes in this bin range
            bin_freq_sum = df.iloc[1][mask].sum()
            
            # Append the summed frequency for this bin to the list
            df_binned_freq.append(bin_freq_sum)
        
        # Append the binned frequencies for this DataFrame to the overall list
        binned_frequencies.append(df_binned_freq)
    
    # Convert the binned frequencies to a new DataFrame with bin labels as columns
    result_df = pd.DataFrame(binned_frequencies, columns=bin_labels)
    
    return result_df

# Define bin ranges (min_patch_size, max_patch_size)
bins = [(1, 10), (10,50), (50,100), (100,200), (200,600)]

# Define bin labels
bin_labels = ['1-10', '10-50', '50-100', '100-200', '200-600']

# Bin agreement clusters
agree_binned = bin_clusts(agree_forclust_atts, bins, bin_labels)

# Bin disagreement clusters
disagree_binned = bin_clusts(disagree_forclust_atts, bins, bin_labels)

# Calculate agreement/disagreement ratios
agratio_binned = agree_binned / (disagree_binned + agree_binned)

# Copy agreement ratio dataframe
agratio_cleaned = agratio_binned.copy()

# Convert 0 values to na
agratio_cleaned[agratio_cleaned == 0] = np.nan

# Calculate patch frequencies
patch_freq = disagree_binned + agree_binned



############################################################################


# PLOT 


############################################################################
# Define function to create line plot
def agratio_plot(df, yearrange):

    # Initiate figure
    plt.figure(figsize=(6.9, 4.5))
    
    # Iterate over each column (series)
    for lab in df.columns:
        
        # Plot series as line graph
        plt.plot(years, df[lab], label=f"{lab} Pixels")

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('Proportion of Deforestation Agreement')
    
    # Add title
    plt.title('Spatial Agreement between GFC and TMF by Deforestation Patch Size')
    
    # Add tickmarks
    plt.xticks(years)
    
    # Add legend
    plt.legend(title='Deforestation Patches', loc='lower left')
    
    # Add gridlines
    plt.grid(True, linestyle = "--")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
# Define function to plot patch population sizes
def patchfreq_plot(df):
    # Define figure size
    plt.figure(figsize=(13.8, 9))
    
    # Initialize the bottom values for stacking
    bottom = np.zeros(len(df.index))
    
    # Iterate over each column (series) to create stacked bars
    for lab in df.columns:
        # Plot each series as a bar with the bottom set to the current cumulative total
        plt.bar(years, df[lab], bottom=bottom, label=f"{lab} Pixels")
        # Update the bottom for the next series to stack on top
        bottom += df[lab]
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Pixel Count')
    plt.title('Frequency of Deforestation Agreement + Disagreement Patches by Size')
    
    # Add tick marks
    plt.xticks(years)
    
    # Add legend
    plt.legend(title='Deforestation Patches', loc='upper right')
    
    # Add gridlines
    plt.grid(True, linestyle="--")
    
    # Show plot with tight layout
    plt.tight_layout()
    plt.show()
    
# Plot agreement ratios for different deforestation patches
agratio_plot(agratio_cleaned, years)

# Plot frequencies for deforestation patch sizes
patchfreq_plot(patch_freq)



############################################################################


# SIDE BY SIDE: SPAT AGREE AND POPULATION SIZE 


############################################################################
# %%
# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
# Plot 1: gfc and tmf spatagree by defor patch size
for lab in agratio_cleaned.columns:
    
    # Plot series as line graph
    axes[0].plot(years, agratio_cleaned[lab], label=f"{lab} Pixels")

# Add x axis label
axes[0].set_xlabel('Year', fontsize=12)

# Add y axis label
axes[0].set_ylabel('Proportion of Deforestation Agreement', fontsize=12)

# Add tickmarks
axes[0].set_xticks(years)
axes[0].tick_params(axis='both', labelsize=11)

# Add legend
# axes[0].legend(fontsize=11)

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# Initialize bottom value for stacking
bottom = np.zeros(len(patch_freq.index))

# Define bar width
bar_width = 0.7

# Plot 2: population of each defor patch size
for lab in patch_freq.columns:
    
    # Plot each series as a bar with the bottom set to the current cumulative total
    
    axes[1].bar(years, patch_freq[lab], width=bar_width, bottom=bottom, 
                label=f"{lab} Pixels")
    
    # Update the bottom for the next series to stack on top
    bottom += patch_freq[lab]

# Add x axis label
axes[1].set_xlabel('Year', fontsize=12)

# Add y axis label
axes[1].set_ylabel('Pixel Count', fontsize=12)

# Add tickmarks
axes[1].set_xticks(years)
axes[1].tick_params(axis='both', labelsize=11)

# Add x tickmarks
# axes[1].set_xticks([i + bar_width / 2 for i in x])  
# axes[1].set_xticklabels(years)

# Add y tickmarks
# axes[1].yaxis.set_major_locator(MultipleLocator(0.01))
# axes[1].yaxis.set_minor_locator(MultipleLocator(0.005))

# # Add gridlines
# axes[1].grid(axis='y', which='major', linestyle='-')
# axes[1].grid(axis='y', which='minor', linestyle='--')
# axes[1].grid(axis='x', linestyle = "--")

# Add gridlines
axes[1].grid(linestyle="--", alpha=0.6)

# Add legend
axes[1].legend(fontsize=11)

# Show plot
plt.tight_layout()
plt.show()










