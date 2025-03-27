# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:59:00 2025

@author: hanna

This file clusters predicted GFC and TMF deforestation into patches. Next, the
distribution, minimum, and maximum deforestation patch sizes is calculated and 
plotted. 

Expected runtime ~1 hour and 45min
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import pandas as pd
import os
import numpy as np
from scipy.ndimage import label
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
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

# Define pixel area
pixel_area = 0.09

# Set output directory
out_dir = os.path.join('data', 'intermediate')

# Define color palatte
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

# Define file paths for annual gfc rasters
gfc_lossyear_paths = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif" for 
                      year in years]

# Define file paths for annual tmf rasters
tmf_defordegra_paths = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif" for 
                      year in years]

# Read GFC rasters 
gfc_lossyear_arrs, profile = read_files(gfc_lossyear_paths)

# Read TMF rasters
tmf_defordegra_arrs, profile = read_files(tmf_defordegra_paths)


# %%
############################################################################


# CLUSTER GFC AND TMF DEFORESTATION PATCHES


############################################################################
"""
GFC: 2312.139471 seconds, TMF: 3726.648931 seconds, 
"""
# Define function that clusters and labels deforestation (takes 5min for 1 array)
def patch_label(arr_list, yearrange, nodata_val):
    
    # Note start time to track operation duration
    start_time = time.time()
    
    # Create empty list to hold labeled arrays
    labeled_arrs = []
    
    # Iterate through each array and year
    for arr, year in zip(arr_list, yearrange):
    
        # Create mask where pixels have non-na data in the given year
        mask = (arr != nodata_val) & (arr != 0)
        
        # Label clusters in the valid mask
        labeled_array, num_features = label(mask)
        
        # Initialize output array with NoData values
        cluster_size_array = np.full_like(arr, nodata_val, dtype=np.int32)
        
        # Iterate over each cluster
        for clust in range(1, num_features + 1):
            
            # Calculate number of pixels in cluster
            clust_size = np.sum(labeled_array == clust)
            
            # Assign the cluster size to all pixels in the cluster
            cluster_size_array[labeled_array == clust] = clust_size
        
        # Add the labeled array (with cluster sizes) to the list
        labeled_arrs.append(cluster_size_array)
        
        # Print statement for status
        print(f"Labeled array for {year}")
    
    # Note end time to track operation duration
    end_time = time.time()
    
    # Calculate operation duration
    elapsed_time = end_time - start_time
    
    print(f"Operation took {elapsed_time:.6f} seconds")
    
    return labeled_arrs

# Define function to check values of array
def valcheck(array, dataname):
    uniquevals = np.unique(array)
    print(f"Unique values in the {dataname} are {uniquevals}")

# Define function to write a list of arrays to file
def filestack_write(arraylist, yearrange, dtype, fileprefix):
    
    # Create empty list to store output filepaths
    filelist = []
    
    # Save each array to drive
    for var, year in zip(arraylist, yearrange):
        
        # Adapt file datatype
        data = var.astype(dtype)
        
        # Define file name and path
        output_filename = f"{fileprefix}_{year}.tif"
        output_filepath = os.path.join(out_dir, output_filename)
        
        # Update profile with dtype string
        profile['dtype'] = data.dtype.name
        
        # Write array to file
        with rasterio.open(output_filepath, "w", **profile) as dst:
            dst.write(data, 1)
            
        # Append filepath to list
        filelist.append(output_filepath)
        
        print(f"{output_filename} saved to file")
    
    return filelist

# Cluster gfc forests
gfc_forclust = patch_label(gfc_lossyear_arrs, years, nodata_val)

# Write gfc clusters to file
gfc_forclust_files = filestack_write(gfc_forclust, years, rasterio.uint32, 
                                      "gfc_forclust")

# Cluster tmf forests
tmf_forclust = patch_label(tmf_defordegra_arrs, years, nodata_val)

# Write tmf clusters to file
tmf_forclust_files = filestack_write(tmf_forclust, years, rasterio.uint32, 
                                      "tmf_forclust")


# %%
############################################################################


# READ DATA TO AVOID RE-RUNNING CODE


############################################################################
# Define file paths for clustered gfc rasters
gfc_forclust_paths = [f"data/intermediate/gfc_forclust_{year}.tif" for 
                      year in years]

# Define file paths for clustered tmf rasters
tmf_forclust_paths = [f"data/intermediate/tmf_forclust_{year}.tif" for 
                      year in years]

# Read gfc rasters 
gfc_forclust_arrs, profile = read_files(gfc_forclust_paths)

# Read tmf rasters
tmf_forclust_arrs, profile = read_files(tmf_forclust_paths)


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

# Convert gfc attribute table to boxplot data
gfc_boxdata = patch_boxdata(gfc_patch_atts, "GFC Deforestation")

# Convert tmf attribute table to boxplot data
tmf_boxdata = patch_boxdata(tmf_patch_atts, "TMF Deforestation")

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

# Save figure
savepath = "data/plots/Presentation Plots/rq1_deforpatches.png"
savefig(savepath, transparent=True)

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
