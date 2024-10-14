# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:50:37 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import glob
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import Point
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

# Set year range
years = range(2013, 2024)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Agreement raster filepaths
data_folder = "data/intermediate"
agreement_files = glob.glob(os.path.join(data_folder, "*agreement_gfc_combtmf*.tif"))

# Create empty list to store agreement rasters
agreements = []

# Read raster data
for file, year in zip(agreement_files, years):
    with rasterio.open(file) as rast:
        data = rast.read(1)
        agreements.append(data)
        profile = rast.profile
        transform = rast.transform

# Read GRNP vector data
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create GRNP geometry
grnp_geom = grnp.geometry

# Read vector data
villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create REDD+ and non-REDD+ polygons
villages = villages[['grnp_4k', 'geometry']]
villages = villages.dissolve(by='grnp_4k')
villages = villages.reset_index()

# Create REDD+ and non-REDD+ geometries
redd_geom = villages.loc[1, 'geometry']
nonredd_geom = villages.loc[0, 'geometry']



############################################################################


# RECLASSIFY AGREEMENT ARRAYS 


############################################################################
"""
Sampling strata are based on forest disagreement and agreement area for each 
year between 2013-2023. Therefore, the classes "only GFC deforestation" (pixel 
value 6) and "only TMF deforestation" (pixel value 7) are combined to create
a general "disagreement" class
"""
# Define function to check unique values in a given array
def valcheck(array, dataname):
    uniquevals = np.unique(array)
    print(f"Unique values in the {dataname} are {uniquevals}")

# Check values of agreement array
valcheck(agreements[0], "agreement 2013")

# Create empty list to hold combined disagreement class arrays
combdis_arrays = []

# Combine the two disagreement classes (classes 6 and 7) into one class
for array in agreements:
    # Copy agreement array
    modified_array = array.copy()
    
    # Replace class 7 with class 6
    modified_array[modified_array == 7] = 6
    
    # Append modified array to combined disagreement list
    combdis_arrays.append(modified_array)

# Check values of combined disagreement arrays
valcheck(combdis_arrays[0], "combined disagreement array 2013")

# Initialize counter to check for temporal class overlaps
count_conflicts = 0

# Get number of arrays
num_arrays = len(combdis_arrays)

# Iterate over all pairs of arrays
for i in range(num_arrays):
    for j in range(i + 1, num_arrays):
        # Create boolean mask for pixels where value in one array is 6 and 
        # value in another array is 8
        conflict_mask = (combdis_arrays[i] == 6) & (combdis_arrays[j] == 8) | \
        (combdis_arrays[i] == 8) & (combdis_arrays[j] == 6)
        
        # Count number of True values in conflict mask
        count_conflicts += np.sum(conflict_mask)

print(f'Total number of pixel locations with conflicts (6 in one array and 8 \
      in another, or vice versa): {count_conflicts}')


############################################################################


# CREATE STRATIFICATION ARRAY


############################################################################
"""
Agreement arrays per year are merged into a single stratification array using 
the reclassification key defined below. A total of 23, non-overlapping strata 
should cover the new array. 

Note: execution time for this segment is ~15min
"""
# Initialize output array filled with 23 (strata for no deforestation)
strat_arr = np.full(combdis_arrays[0].shape, 23)

# Reclassification key for pixel values per each year (2013 to 2023)
reclass_key = [
    {6: 1, 8: 2},  # 2013
    {6: 3, 8: 4},  # 2014
    {6: 5, 8: 6},  # 2015
    {6: 7, 8: 8},  # 2016
    {6: 9, 8: 10},  # 2017
    {6: 11, 8: 12},  # 2018
    {6: 13, 8: 14},  # 2019
    {6: 15, 8: 16},  # 2020
    {6: 17, 8: 18},  # 2021
    {6: 19, 8: 20},  # 2022
    {6: 21, 8: 22}   # 2023
]

# Iterate through each array (corresponding to each year)
for i in range(len(combdis_arrays)):
    # Apply reclassification for each pixel in the array
    for j in range(combdis_arrays[i].shape[0]):  # Rows
        for k in range(combdis_arrays[i].shape[1]):  # Columns
            pixel_value = combdis_arrays[i][j, k]
            
            # Reclassify based on the reclass_key
            if pixel_value in (6, 8):  # Only interested in values 6 or 8
                strat_arr[j, k] = reclass_key[i][pixel_value]
            elif pixel_value == 255:  # NoData case
                strat_arr[j, k] = 255

# Check values
valcheck(strat_arr, "stratification array")
    
# Create mask to remove GRNP area
grnp_mask = geometry_mask(grnp.geometry, transform=transform, invert=False, 
                          out_shape=strat_arr.shape)

# Remove GRNP area from stratification array
strat_arr_nogrnp = np.where(grnp_mask, strat_arr, nodata_val)

# Define output path
output_filename = "stratification_layer_nogrnp.tif"
output_filepath = os.path.join(out_dir, output_filename)

# Save stratification layer to file
with rasterio.open(output_filepath, "w", **profile) as dst:
    dst.write(strat_arr_nogrnp, 1)


############################################################################


# RANDOM SAMPLING PER STRATA


############################################################################
"""
This thesis validation methodology will sample 22 points per 23 strata, 
creating 506 total validation points
"""
# (Re-)read stratification layer to extract transform information
# with rasterio.open(output_filepath) as src:
#     strat_arr = src.read(1)
#     transform = src.transform

# Set random seed for reproducibility
np.random.seed(42)

# Create empty list to hold sample point coordinates
sample_points = []

# Loop through classes 1 to 23
for strata in range(1, 24):
    # Find indices of all pixels belonging to current class
    pixel_indices = np.argwhere(strat_arr_nogrnp == strata)

    # If there are fewer than 22 pixels in this class, sample all of them
    if len(pixel_indices) <= 22:
        selected_indices = pixel_indices
    else:
        # Randomly sample 22 pixels
        selected_indices = pixel_indices[np.random.choice(
            pixel_indices.shape[0], 22, replace=False)]

    # Convert pixel indices (row, col) to coordinates
    for idx in selected_indices:
        # Format pixel location as x, y
        row, col = idx
        
        # Use transform information from strat_arr to convert
        x, y = rasterio.transform.xy(transform, row, col)
        
        # Append strata and Point geometry (x, y) to list
        sample_points.append({"strata": strata, "geometry": Point(x, y)})

# Convert sample point list to geodataframe
sample_point_gdf = gpd.GeoDataFrame(sample_points, crs=profile['crs'])

# Define output path
output_filename = "validation_points_geometry.shp"
output_filepath = os.path.join(out_dir, output_filename)

# Save sample point coordinates to file
sample_point_gdf.to_file(output_filepath)



















