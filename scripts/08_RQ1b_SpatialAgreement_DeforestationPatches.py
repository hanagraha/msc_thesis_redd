# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:57:46 2024

@author: hanna

This file clusters deforestation into patches, assigning spatial agreement
and patch size to the pixel value. 

Estimated runtime: ~2 hours
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import os
import numpy as np
from scipy.ndimage import label
import time



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

# Define file paths for agreement rasters
agreement_files = [f"data/intermediate/agreement_gfc_combtmf_{year}.tif" for 
                   year in years]

# Read agreement rasters
agreement_arrs, profile = read_files(agreement_files)


# %%
############################################################################


# CLUSTER SPATIAL AGREEMENT PATCHES (NEW)


############################################################################
# Define function to cluster by any deforestation
def clustratio_patch(arr_list, yearrange, nodata_val):
    
    # Note start time to track operation duration
    start_time = time.time()
    
    # Create empty list to hold arrays with agreement ratios
    agratio_arrs = []
    
    # Create empty list to hold arrays with cluster size labels
    clustsize_arrs = []
    
    # Iterate through each array and year
    for arr, year in zip(arr_list, yearrange):
        
        # Initialize output array with NoData values (for clustsize)
        clustsize_array = np.full_like(arr, nodata_val, dtype=np.int32)
        
        # Initialize output array with NoData values (for ratio)
        agratio_array = np.full_like(arr, nodata_val, dtype=np.float32)
        
        # Extract agreement pixels
        agree_pixels = (arr == 8)
        
        # Extract disagreement pixels
        disagree_pixels = np.isin(arr, [6, 7])
    
        # Mask for deforestation agreement and disagreement pixels
        mask = (arr == 6) | (arr == 7) | (arr == 8)
        
        # Label clusters in the valid mask
        labeled_array, num_features = label(mask)
        
        # Initialize output array with NoData values (for clustsize)
        clustsize_array = np.full_like(arr, nodata_val, dtype=np.int32)
        
        # Initialize output array with NoData values (for ratio)
        agratio_array = np.full_like(arr, nodata_val, dtype=np.float32)
        
        # Iterate over each cluster
        for clust in range(1, num_features + 1):
            
            # Create cluster mask
            cluster_mask = labeled_array == clust
            
            # Calculate number of pixels in cluster
            clust_size = np.sum(cluster_mask)
            
            # Assign cluster size to all pixels in cluster
            clustsize_array[cluster_mask] = clust_size
            
            # Calculate number of agreement pixels in cluster
            agree = np.sum(agree_pixels[cluster_mask])
            
            # Calculate number of disagreement pixels in cluster
            disagree = np.sum(disagree_pixels[cluster_mask])
        
            # Calculate agreement ratio
            agratio = agree / (disagree + agree) if (disagree + agree) > 0 else 0.0
            
            # Assign agreement ratio to all pixels in cluster
            agratio_array[cluster_mask] = agratio
        
        # Add labeled array to clustsize list
        clustsize_arrs.append(clustsize_array)
        
        # Add original array to id list
        agratio_arrs.append(agratio_array)
        
        # Print statement for status
        print(f"Labeled array for {year}")
    
    # Note end time to track operation duration
    end_time = time.time()
    
    # Calculate operation duration
    elapsed_time = end_time - start_time
    
    print(f"Operation took {elapsed_time:.6f} seconds")
    
    return agratio_arrs, clustsize_arrs

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

# Cluster spatial agreement deforestation
defor_ratio, defor_size = clustratio_patch(agreement_arrs, years, nodata_val)

# Write spatial agreement clusters to file
ratio_files = filestack_write(defor_ratio, years, rasterio.float32, "disag_ratio")
size_files = filestack_write(defor_size, years, rasterio.uint32, "disag_size")




