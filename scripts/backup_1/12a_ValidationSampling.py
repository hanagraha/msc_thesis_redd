# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:53:21 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import Point
import pandas as pd


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

# Define output directory
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Define output directory
val_dir = os.path.join("data", "validation")

# Set year range
years = range(2013, 2024)

# Set random seed for reproducibility
np.random.seed(42)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read list of paths
def read_files(pathlist):
    
    # Create empty list to hold arrays
    arrlist = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read file
        with rasterio.open(path) as rast:
            data = rast.read(1)
            profile = rast.profile
            transform = rast.transform
            
            # Add array to list
            arrlist.append(data)
            
    return arrlist, profile, transform

# Define agreement filepaths
agreement_filepaths = [f"data/intermediate/agreement_gfc_combtmf_{year}.tif"
                       for year in years]

# Define gfc lossyear filepath
gfc_lossyear_path = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Define tmf defordegra filepath
tmf_defordegra_path = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# Define sensitive early filepath 
sensitive_early_path = "data/intermediate/gfc_tmf_sensitive_early.tif"

# Read agreement files
agreement_arrs, profile, transform = read_files(agreement_filepaths)

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

# Define function to reclassify list of arrays
def arr_reclass(arrlist, oldclass, newclass):
    
    # Create empty list to hold reclassified arrays
    reclassed = []
    
    # Iterate over each array
    for arr in arrlist:
        
        # Copy array
        mod_arr = arr.copy()
        
        # Replace old class with new class
        mod_arr[mod_arr == oldclass] = newclass
        
        # Append modified array to list
        reclassed.append(mod_arr)
        
    return reclassed

# Define function to check for overlaps in disagreement/agreement over time
def class_overlaps(arrlist):
    
    # Initialize counter for class overlaps
    conflicts = 0
    
    # Iterate over number of arrays
    for i in range(len(arrlist)):
        
        # Iterate over pairs of arrays
        for j in range(i+1, len(arrlist)):
            
            # Create conflict mask where pixel values differ between arrays
            conflict_mask = (arrlist[i] == 6) & (arrlist[j] == 8) | \
                (arrlist[i] == 8) & (arrlist[j] == 6)
                
            # Count number of conflicts in mask (True values)
            conflicts += np.sum(conflict_mask)
            
    # Print results
    print("Total pixels with deforestation disagreement and agreement at "
          f"different years but the same location: {conflicts}")

# Check values of agreement array
valcheck(agreement_arrs[0], "agreement 2013")

# Merge disagreement classes (7 and 6)
combdis_arrays = arr_reclass(agreement_arrs, 7, 6)

# Check unique values in combined disagreement array
valcheck(combdis_arrays[0], "combined disagreement array 2013")

# Check for temporal overlaps between deforestation classes
class_overlaps(combdis_arrays)


    
############################################################################


# CREATE STRATIFICATION ARRAY


############################################################################
"""
Agreement arrays per year are merged into a single stratification array using 
the reclassification key defined below. A total of 23, non-overlapping strata 
should cover the new array. 

Note: execution time for this segment is ~15min
"""
def create_reclass_key(class1, class2, num_dicts):
    
    # Create empty list to hold reclassification dictionaries
    reclass_key = []
    
    # Initiate starting reclass value
    reclass_value = 1
    
    # Iterate over the number of dictionaries needed
    for _ in range(num_dicts):
        
        # Input reclass values to dictionary
        reclass_key.append({class1: reclass_value, class2: reclass_value + 1})
        
        # Increment 2 for the next class pair dictionary
        reclass_value += 2
    
    return reclass_key

# Define function to create stratification layer
def strat_layer(arrlist, reclass_key):
    
    # Create array filled with no deforestation class (value = 23)
    strat_arr = np.full(combdis_arrays[0].shape, 23)
    
    # Iterate over each array
    for i in range(len(arrlist)):
        
        # Iterate over each row
        for j in range(arrlist[i].shape[0]):
            
            # Iterate over each column
            for k in range(arrlist[i].shape[1]):
                
                # Extract pixel value at array, row, column
                pixel_value = arrlist[i][j, k]
                
                # Reclassify disagreement areas (value 6 or 8)
                if pixel_value in (6, 8):
                    strat_arr[j, k] = reclass_key[i][pixel_value]
                
                # Keep nodata pixels as nodata
                elif pixel_value == nodata_val: 
                    strat_arr[j, k] = nodata_val
        
        # Print statement for status
        print(f"Array {i} incorporated into stratification layer")
                    
    return strat_arr

# Define function to remove polygon space from array
def rmv_polyspace(array, geometry, transform, nodataval):
    
    # Create mask to remove geometry area
    mask = geometry_mask(geometry, transform = transform, invert=False, 
                         out_shape = array.shape)
    
    # Remove geometry area from array
    arr_masked = np.where(mask, array, nodataval)
    
    return arr_masked

# Define function to write file to drive
def rast_write(array, filename, out_dir, profile):
    
    # Define output filepath
    output_filepath = os.path.join(out_dir, filename)
    
    # Save to file
    with rasterio.open(output_filepath, "w", **profile) as dst:
        dst.write(array, 1)

# Create reclassification key for pixel values per each year (2013 to 2023)
reclass_key = create_reclass_key(6, 8, len(years))

# Create stratification layer for sampling
strat_arr = strat_layer(combdis_arrays, reclass_key)

# Check values in stratified array
valcheck(strat_arr, "stratification array")

# Remove GRNP from sample space
nogrnp_stratarr = rmv_polyspace(strat_arr, grnp.geometry, transform, nodata_val)

# Write stratified layer to file
rast_write(nogrnp_stratarr, "stratification_layer_nogrnp.tif", out_dir, profile)



############################################################################


# RANDOM SAMPLING PER STRATA


############################################################################
"""
This thesis validation methodology will sample 22 points per 23 strata, 
creating 506 total validation points
"""
# Define function to sample points from array
def strat_sample(array, sample_size, transform, profile, min_strata = None, 
                 max_strata = None):
    
    # Identify number of clasess
    classnum = len(np.unique(array))
    
    # Select strata of interest (if defined)
    if min_strata is not None or max_strata is not None:
        
        # Replace values outside range with nodata
        array = np.where(
            (array >= (min_strata if min_strata is not None else array.min())) &
            (array <= (max_strata if max_strata is not None else array.max())),
            array, nodata_val
            )
    
    # Create empty list to hold point coordinates
    sample_points = []
    
    # Iterate over each class
    for strata in range(np.min(array), classnum):
        
        # Find indices of pixels in class
        pixel_indices = np.argwhere(array == strata)
        
        # If pixels in class < sample size
        if len(pixel_indices) <= sample_size:
            
            # Sample all pixels
            selected_indices = pixel_indices
            
        # If pixels in class > sample size
        else:
            
            # Take random sample
            selected_indices = pixel_indices[np.random.choice(
                pixel_indices.shape[0], sample_size, replace=False)]
            
        # Iterate over each pixel index
        for idx in selected_indices:
            
            # Extract row and column for each pixel
            row, col = idx
            
            # Use transform information convert indices to coordinates
            x, y = rasterio.transform.xy(transform, row, col)
            
            # Append strata and Point geometry (x, y) to list
            sample_points.append({"strata": strata, "geometry": Point(x, y)})
            
    # Convert sample point list to geodataframe
    sample_point_gdf = gpd.GeoDataFrame(sample_points, crs=profile['crs'])
    
    return sample_point_gdf

# Define function to write points to file
def shp_write(gdf, filename, out_dir):
    
    # Define output filepath
    output_filepath = os.path.join(out_dir, filename)
    
    # Write to file
    gdf.to_file(output_filepath)
    
# Create sample points from stratified array
sample_points = strat_sample(nogrnp_stratarr, 22, transform, profile)

# Write sample points to file
shp_write(sample_points, "validation_points_geometry.shp", val_dir)


"""
Also try a strategy only sampling from 2016-2023, strata7-23, 17 total)
"""
# Create sample points from subsection of stratified array
points_strata7up = strat_sample(nogrnp_stratarr, 30, transform, profile,
                                min_strata = 7)

# Define function to replace already sampled points
def pnt_replace(gdf1, gdf2):
    
    # Define minimum strata
    strata_min = min(gdf2['strata'])
    
    # Define maximum strata
    strata_max = max(gdf2['strata'])
    
    # Create empty list to hold updated gdf2 strata
    gdf2_updated = []
    
    # Iterate over each strata
    for strata in range(strata_min, strata_max + 1):
        
        # Subset gdf1 for strata
        gdf1_sub = gdf1[gdf1['strata'] == strata].reset_index(drop=True)
        
        # Subset gdf2 for strata
        gdf2_sub = gdf2[gdf2['strata'] == strata].reset_index(drop=True)
        
        # Define sample size of gdf1 strata
        gdf1_ss = len(gdf1_sub) - 1
        
        # Check for common geometries
        common_points = gdf1_sub[gdf1_sub.geometry.isin(gdf2_sub.geometry)]
        
        # If there are common geometries
        if not common_points.empty:
            print(f"Identical points found in strata {strata}:")
            print(common_points)
        
        # If there are no common geometries
        else:
            
            # Print statement
            print(f"No identical points found in strata {strata}.")
            
            # Replace geometries of gdf2 with gdf1 (only 22)
            gdf2_sub.loc[:gdf1_ss, 'geometry'] = gdf1_sub.loc[:,'geometry'].values
            
            # Add updated strata geometry to list
            gdf2_updated.append(gdf2_sub)
            
    # Add updated strata to gdf2
    updated_gdf2 = pd.concat(gdf2_updated, ignore_index = True)
            
    return updated_gdf2

# Merge original sample points to new dataset
sample_points_strata7up = pnt_replace(sample_points, points_strata7up)

# Write sample points to file
shp_write(sample_points_strata7up, "validation_points_geometry_minstrata7.shp", val_dir)


############################################################################


# EXTRACT DATASET VALUES PER SAMPLE POINT


############################################################################
# Define function to extract raster values per point
def extract_val(points_gdf, tiflist, tifnames):
    
    # Copy points gdf
    gdf = points_gdf.copy()
    
    # Iterate over each tif file
    for tif, name in zip(tiflist, tifnames):
        
        # Create empty list to store pixel values
        pix_vals = []
        
        # Read tif file
        with rasterio.open(tif) as src:
        
            # Iterate over each point
            for pnt in gdf.geometry:
                
                # Get row and column indices
                row, col = src.index(pnt.x, pnt.y)
                
                # Extract pixel value at point location
                pix_val = src.read(1)[row, col]
                
                # Append pixel value to list
                pix_vals.append(pix_val)
            
        # Add new column to geodataframe
        gdf[name] = pix_vals
        
    return gdf

# Define function to save gdf as csv
def write_csv(gdf, out_dir, outfilename):
    
    # Convert the geometry to WKT format
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.wkt)
    
    # Define output path
    outfilepath = os.path.join(out_dir, f"{outfilename}.csv")

    # Save the GeoDataFrame as a CSV file
    gdf.to_csv(outfilepath, index=True)
    
    # Print statement
    print(f"File saved to {outfilepath}")

# Define list of rasters
tiflist = [gfc_lossyear_path, tmf_defordegra_path, sensitive_early_path]

# Define names of rasters
tifnames = ['gfc', 'tmf', 'se']

# Extract raster values
valpoints = extract_val(sample_points, tiflist, tifnames)

# Write points to file
write_csv(valpoints, val_dir, "validation_points_labelling")

# Extract raster values for strata 7+ points
valpoints_strata7up = extract_val(sample_points_strata7up, tiflist, tifnames)

# Write points to file
write_csv(valpoints_strata7up, val_dir, "validation_points_labelling_minstrata7")
