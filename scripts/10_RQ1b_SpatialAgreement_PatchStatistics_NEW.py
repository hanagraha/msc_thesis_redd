# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:10:35 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
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

# Set year range
years = range(2013, 2024)

# Set output directory
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Set default plotting colors
defaultblue = "#4682B4"
reddcol = "brown"
nonreddcol = "dodgerblue"
grnpcol = "darkgreen"

# Define color palatte (2 colors)
gfc_col = "#820300"  # Darker Red
tmf_col = "#4682B4"  # Darker Blue - lighter

# Define Color Palatte (3 colors)
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



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

# Define file paths for ratio rasters
agratio_forclust_paths = [f"data/intermediate/disag_ratio_{year}.tif" for year 
                        in years]

# Define file paths for size rasters
agsize_forclust_paths = [f"data/intermediate/disag_size_{year}.tif" for year 
                       in years]

# Read gfc rasters 
gfc_forclust_arrs, profile = read_files(gfc_forclust_paths)

# Read tmf rasters
tmf_forclust_arrs, profile = read_files(tmf_forclust_paths)

# Read ratio rasters
agratio_forclust_arrs, ratprofile = read_files(agratio_forclust_paths)

# Read size rasters
agsize_forclust_arrs, sizeprofile = read_files(agsize_forclust_paths)


# %%
############################################################################


# CALCULATE PATCH SIZE FREQUENCY BY PATCH COUNT


############################################################################
# Define function to create patch size attribute table
def patch_att(arrlist):
    
    # Create empty dataframe
    atts = pd.DataFrame()
    
    # Iterate over each array
    for arr, year in zip(arrlist, years):
        
        # Extract pixel counts in unique patch sizes
        vals, pixels = np.unique(arr[arr != nodata_val], return_counts = True)
        
        # Convert pixel counts to patch counts
        patches = pixels / vals
        
        # Create attribute table
        att = pd.DataFrame({"Year": year,
                            "Patch Size": vals,
                            "Patch Count": patches})
        
        # Add attribute table to parent dataframe
        atts = pd.concat([atts, att], ignore_index = True)
        
    return atts

# Define function to convert patch attributes to boxplot data
def patch_boxdata(att_df, dataname):
    
    # Create empty list to hold data
    data = []
    
    # Iterate over each year
    for year in years:
        
        # Extract patch sizes present in that year
        sizes = att_df[att_df["Year"] == year]["Patch Size"]
        
        # Extract patch counts present in that year
        counts = att_df[att_df["Year"] == year]["Patch Count"]
        
        # Iterate over each size-count pair
        for size, count in zip(sizes, counts):
            
            # Multiply size row by number of counts
            data.extend([(size, year, dataname)] * int(count))
            
    # Convert list to dataframe
    boxplot_data = pd.DataFrame(data, columns = ["Patch Size", "Year", "Dataset"])
    
    return boxplot_data

# Define function to calculate mean patch size
def patch_statsum(boxdata):
    
    # Create empty list to hold means and maximums
    means = []
    maxs = []
    
    # Iterate over each year
    for year in years:
        
        # Calculate mean patch size for that year
        yearmean = boxdata[boxdata["Year"] == year]["Patch Size"].mean() * 0.09
        
        # Add mean to list
        means.append(yearmean)
        
        # Calculate max patch size for that year
        yearmax = boxdata[boxdata["Year"] == year]["Patch Size"].max() * 0.09
        
        # Add max to list
        maxs.append(yearmax)
        
    # Create dataframe from list
    stats_df = pd.DataFrame({"Year": years, 
                             "Mean Patch Size": means, 
                             "Max Patch Size": maxs})
        
    return stats_df

# Create gfc patch attribute table
gfc_patch_atts = patch_att(gfc_forclust_arrs)

# Create tmf patch attribute table
tmf_patch_atts = patch_att(tmf_forclust_arrs)

# Create spatial agreement size attribute table
agsize_patch_atts = patch_att(agsize_forclust_arrs)

# Convert gfc attribute table to boxplot data
gfc_boxdata = patch_boxdata(gfc_patch_atts, "GFC Deforestation")

# Convert tmf attribute table to boxplot data
tmf_boxdata = patch_boxdata(tmf_patch_atts, "TMF Deforestation")

# Convert size attribute table to boxplot data
agsize_boxdata = patch_boxdata(agsize_patch_atts, "Potential Deforestation") 

# Combine gfc and tmf boxplot data
gfctmf_boxdata = pd.concat([gfc_boxdata, tmf_boxdata], ignore_index=True)

# Calculate mean and max gfc patch size
gfc_patchstats = patch_statsum(gfc_boxdata)

# Calculate mean and max tmf patch size
tmf_patchstats = patch_statsum(tmf_boxdata)

# Create copy of boxplot data
gfctmf_boxdata_ha = gfctmf_boxdata.copy()

# Convert patch size from pixels to ha
gfctmf_boxdata_ha['Patch Size']*=0.09


# %%
############################################################################


# PLOT PATCH BOX PLOT


############################################################################
# Initialize figure
plt.figure(figsize=(10, 6))

# Define dataset colors for seaborne
data_cols = {"GFC Deforestation": bluecols[0], "TMF Deforestation": bluecols[1]}

# Plot boxplot data
sns.boxplot(x='Year', y='Patch Size', hue='Dataset', data=gfctmf_boxdata_ha, 
            width = 0.7, showfliers=False, palette = data_cols)

# Edit tickmark fontsizes
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)

# Add y major gridlines
plt.grid(axis='y', which='major', color='gray', linestyle='--', alpha = 0.6)

# Add axes labels
plt.xlabel('Year', fontsize = 16)
plt.ylabel('Deforestation Patch Size (ha)', fontsize = 16)

# Adapt legend fontsize
plt.legend(fontsize = 16)

# Show plot
plt.show()


# %%
############################################################################


# PLOT MAXIMUM PATCH SIZES


############################################################################
# Initialize figure
plt.figure(figsize = (10, 6))

# Add gfc patch statistics
plt.plot(years, gfc_patchstats["Max Patch Size"], color = bluecols[0], 
         linewidth = 2, label = "GFC Deforestation Patches")

# Add tmf patch statsitics
plt.plot(years, tmf_patchstats["Max Patch Size"], color = bluecols[1],
         linewidth = 2, label = "TMF Deforestation Patches")

# Add axes labels
plt.xlabel("Year", fontsize = 16)
plt.ylabel("Maximum Patch Area (ha)", fontsize = 16)

# Plot years as x axis
plt.xticks(years)

# Adjust font size of tick labels
plt.tick_params(axis='both', which='major', labelsize = 14)

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Add legend
plt.legend(fontsize = 16)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# PLOT MEAN PATCH SIZES


############################################################################
# Initialize figure
plt.figure(figsize = (10, 6))

# Add gfc patch statistics
plt.plot(years, gfc_patchstats["Mean Patch Size"], color = bluecols[0], 
         linewidth = 2, label = "GFC Deforestation Patches")

# Add tmf patch statsitics
plt.plot(years, tmf_patchstats["Mean Patch Size"], color = bluecols[1],
         linewidth = 2, label = "TMF Deforestation Patches")

# Add axes labels
plt.xlabel("Year", fontsize = 16)
plt.ylabel("Mean Patch Area (ha)", fontsize = 16)

# Plot years as x axis
plt.xticks(years)

# Adjust font size of tick labels
plt.tick_params(axis='both', which='major', labelsize = 14)

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Add legend
plt.legend(fontsize = 16)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# COMBINE AGREEMENT RATIO AND CLUSTER SIZE


############################################################################
# Flatten agreement ratio rasters
agratio_forclust_flat = [arr.flatten() for arr in agratio_forclust_arrs]

# Flatten size rasters
agsize_forclust_flat = [arr.flatten() for arr in agsize_forclust_arrs]

# Remove nodata values
datamasks = [(arr != 255) for arr in agratio_forclust_flat]

# Create empty dictionary to hold data
ratsize = {}

# Iterate over each array and mask
for ratio, size, mask, year in zip(agratio_forclust_flat, agsize_forclust_flat, 
                                   datamasks, years):
    
    # Create dataframe
    df = pd.DataFrame({
        "Agreement Ratio": ratio[mask],
        "Patch Size": size[mask]
        })
    
    # Add dataframe to dictionary
    ratsize[year] = df

# Extract list of patch sizes
allsizes = [np.unique(size) for size in agsize_forclust_arrs]

# Test with 2013
size2013 = allsizes[0]
arr2013 = agratio_forclust_arrs[0]

# Iterate over each patch size
for size in size2013:
    
    # Extract pixels in size arary
    arr = arr2013[size]


# %%
############################################################################


# PLOT RATIO AND SIZE SCATTERPLOT


############################################################################
# Initialize figure
plt.figure(figsize = (10, 6))

# Add scatterplot data
sns.scatterplot(data = ratsize[2023], x='Patch Size', y='Agreement Ratio')

# Add axes labels
plt.xlabel("Deforestation Patch Size", fontsize = 16)
plt.ylabel("Deforestation Agreement Ratio", fontsize = 16)

# Adjust font size of tick labels
# plt.tick_params(axis='both', which='major', labelsize = 14)

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Add legend
plt.legend(fontsize = 16)

# Show plot
plt.tight_layout()
plt.show()




# %%
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


# %%
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


# %%
############################################################################


# SIDE BY SIDE: SPAT AGREE AND POPULATION SIZE 


############################################################################
# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
# Plot 1: gfc and tmf spatagree by defor patch size
for lab in agratio_cleaned.columns:
    
    # Plot series as line graph
    axes[0].plot(years, agratio_cleaned[lab], label=f"{lab} Pixels")

# Add x axis label
axes[0].set_xlabel('Year', fontsize=16)

# Add y axis label
axes[0].set_ylabel('Proportion of Deforestation Agreement', fontsize=16)

# Add tickmarks
axes[0].set_xticks(years)
axes[0].tick_params(axis='both', labelsize=14)

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
axes[1].set_xlabel('Year', fontsize=16)

# Add y axis label
axes[1].set_ylabel('Pixel Count', fontsize=16)

# Add tickmarks
axes[1].set_xticks(years)
axes[1].tick_params(axis='both', labelsize=14)

# Add gridlines
axes[1].grid(linestyle="--", alpha=0.6)

# Add legend
axes[1].legend(fontsize=16)

# Show plot
plt.tight_layout()
plt.show()



patch_freq.sum()






