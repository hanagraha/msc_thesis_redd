# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:03:29 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point



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

# Define year range
years = range(2013, 2024)

# Set output directory
val_dir = os.path.join(os.getcwd(), 'data', 'validation')



############################################################################


# READ DATA


############################################################################
# Read validation data (2013-2023)
valdata690 = pd.read_csv("data/validation/validation_datasets/validation_points_2013_2023_690.csv", 
                         sep = ",", index_col = 0)

# Read validation data (2016-2023)
valdata510 = pd.read_csv("data/validation/validation_datasets/validation_points_2016_2023_510.csv",
                         sep = ",", index_col = 0)

# Read validation data (2013-2015)
valdata180 = pd.read_csv("data/validation/validation_datasets/validation_points_2013_2015_180.csv",
                         sep = ",", index_col = 0)

# Read validation data (2013-2023)
valdata1440 = pd.read_csv("data/validation/validation_datasets/validation_points_2013_2023_1440.csv",
                          sep = ",", index_col = 0)

# Read validation data (buffer)
valdata_buff = pd.read_csv("data/validation/validation_datasets/validation_points_buffer_60.csv",
                          sep = ",", index_col = 0)

# Define stratification map filepath
stratmap_path = "data/intermediate/stratification_layer_nogrnp.tif"

# Define gfc lossyear filepath
gfc_lossyear_path = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Define tmf defordegra filepath
tmf_defordegra_path = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# Define sensitive early filepath 
sensitive_early_path = "data/intermediate/gfc_tmf_sensitive_early.tif"

# Read stratification map
with rasterio.open(stratmap_path) as rast:
    
    # Extract data
    stratmap = rast.read(1)
    
    # Extract profile
    stratmap_profile = rast.profile


# %%
############################################################################


# COMBINE DATASETS BY STRATA


############################################################################
# Combine the datasets
valdata_comb = pd.concat([valdata690, valdata510], ignore_index=True)
    
# Sort by strata column
valdata_comb = valdata_comb.sort_values(by = "strata", ascending = True)

# Reset index
valdata_comb = valdata_comb.reset_index(drop = True)
    
# Check for any duplicated geometries
duplicate_points = valdata_comb[valdata_comb['geometry'].duplicated()]

# Print the duplicate points
print(f"Number of duplicate geometries: {len(duplicate_points)}")
print(duplicate_points)

# Count points in each strata
strata_counts = valdata_comb['strata'].value_counts().sort_index()

# Print results
print(strata_counts)



############################################################################


# GET MORE SAMPLES 


############################################################################
# Define function to sample points from array
def strat_sample(array, sample_size, transform, profile, min_strata = None, 
                 max_strata = None, random_state=None):
    
    # Set local random generator with given random state (for reproducibility)
    rng = np.random.default_rng(random_state)
    
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
            selected_indices = pixel_indices[rng.choice(
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

# Create sample points with 30 points per strata
points_30 = strat_sample(stratmap, 30, stratmap_profile['transform'], stratmap_profile, 
                         max_strata = 6, random_state = 35)

# Define list of rasters
tiflist = [gfc_lossyear_path, tmf_defordegra_path, sensitive_early_path]

# Define names of rasters
tifnames = ['gfc', 'tmf', 'se']

# Extract raster values
valpoints_30 = extract_val(points_30, tiflist, tifnames)



############################################################################


# COMBINE DATASETS BY STRATA


############################################################################
# Combine new sample points with larger dataset
valdata_exp = pd.concat([valdata_comb, valpoints_30], ignore_index=True)
    
# Sort by strata column
valdata_exp = valdata_exp.sort_values(by = "strata", ascending = True)

# Reset index
valdata_exp = valdata_exp.reset_index(drop = True)
    
# Check for any duplicated geometries
duplicate_points = valdata_exp[valdata_exp['geometry'].duplicated()]

# Print the duplicate points
print(f"Number of duplicate geometries: {len(duplicate_points)}")
print(duplicate_points)

# Count points in each strata
strata_counts = valdata_exp['strata'].value_counts().sort_index()

# Print results
print(strata_counts)



############################################################################


# WRITE TO FILE


############################################################################
# Define function to write points to file
def shp_write(gdf, filename, out_dir):
    
    # Define output filepath
    output_filepath = os.path.join(out_dir, filename)
    
    # Write to file
    gdf.to_file(output_filepath)
    
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
    
# Write  points to file (vector)
shp_write(valpoints_30, "validation_points_30extra.shp", val_dir)

# Write points to file (csv)
write_csv(valpoints_30, val_dir, "validation_points_30extra")

# Convert csv geometry to WKT
valdata_comb['geometry'] = gpd.GeoSeries.from_wkt(valdata_comb['geometry'])

# Write points to file (csv)
write_csv(valdata_comb, val_dir, "validation_points_1200")



############################################################################


# AFTER VALIDATING, COMBINE AGAIN


############################################################################
# Combine new sample points with larger dataset
valdata_all = pd.concat([valdata690, valdata510, valdata180], ignore_index=True)
    
# Sort by strata column
valdata_all = valdata_exp.sort_values(by = "strata", ascending = True)

# Reset index
valdata_all = valdata_exp.reset_index(drop = True)
    
# Check for any duplicated geometries
duplicate_points = valdata_all[valdata_all['geometry'].duplicated()]

# Print the duplicate points
print(f"Number of duplicate geometries: {len(duplicate_points)}")
print(duplicate_points)

# Count points in each strata
strata_counts = valdata_all['strata'].value_counts().sort_index()

# Print results
print(strata_counts)



############################################################################


# WRITE TO FILE


############################################################################
# Convert csv geometry to WKT
valdata_all['geometry'] = gpd.GeoSeries.from_wkt(valdata_all['geometry'])

# Convert valdata to gdf
valdata_all = gpd.GeoDataFrame(valdata_all, geometry='geometry')

# Write  points to file (vector)
shp_write(valdata_all, "validation_points_1380.shp", val_dir)

# Write points to file (csv)
write_csv(valdata_all, val_dir, "validation_points_1380")


# %%
############################################################################


# COMBINE BUFFER AND 1380 DATASETS


############################################################################
# Read 1380 dataset
valdata1380 = pd.read_csv("data/validation/validation_points_labelled.csv", 
                          index_col = 0)

# Convert csv geometry to WKT
valdata1380['geometry'] = gpd.GeoSeries.from_wkt(valdata1380['geometry'])

# Convert dataframe to geodataframe
valdata1380 = gpd.GeoDataFrame(valdata1380, geometry='geometry', crs="EPSG:32629") 

# Read buffer dataset
valdata60 = pd.read_csv("data/validation/validation_points_missing_labelled.csv", 
                        sep = ";", index_col = 0)

# Convert csv geometry to WKT
valdata60['geometry'] = gpd.GeoSeries.from_wkt(valdata60['geometry'])

# Convert dataframe to geodataframe
valdata60 = gpd.GeoDataFrame(valdata60, geometry='geometry', crs="EPSG:32629") 

# Define strata file path
strat_path = "data/intermediate/stratification_layer_buffered.tif"

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

# Combine new sample points with larger dataset
valdata_comb = pd.concat([valdata1380, valdata60], ignore_index=True)
    
# Sort by strata column
valdata_comb = valdata_comb.sort_values(by = "strata", ascending = True)

# Reset index
valdata_comb = valdata_comb.reset_index(drop = True)

# Check strata counts
strat_counts = valdata_comb['strata'].value_counts()    

# Extract buffered strata
valdata_strat = extract_val(valdata_comb, [strat_path], ["buff_strat"])

# Check strata counts again
valdata_strat_counts = valdata_strat['buff_strat'].value_counts()

# Copy validation data gdf
valdata_copy = valdata_strat.copy()

# Remove strata column
valdata_copy = valdata_copy.drop(columns=['strata'])

# # Convert csv geometry to WKT
# valdata_copy['geometry'] = gpd.GeoSeries.from_wkt(valdata_copy['geometry'])

# # Convert dataframe to geodataframe
# valdata_copy = gpd.GeoDataFrame(valdata_copy, geometry='geometry', crs="EPSG:32629") 

# Rename 'buff_strat' to 'strata'
valdata_copy = valdata_copy.rename(columns={'buff_strat': 'strata'})

# Define new column order
new_column_order = ['strata'] + [col for col in valdata_copy.columns if col != 'strata']

# Reorder columns
valdata_copy = valdata_copy[new_column_order]

# Write to file
write_csv(valdata_copy, val_dir, "validation_points_with_buffer_labelled")


# %%

# Extract strata 1-22 from valdata 660
valdata660 = valdata690[valdata690['strata'].between(1,22)]

# Take strata 23 and 24 from valdata 1440
valdata120 = valdata1440[valdata1440['strata'].between(23,24)]

# Combine all strata
valdata_comb = pd.concat([valdata660, valdata120], ignore_index = True)

# # Convert csv geometry to WKT
valdata_comb['geometry'] = gpd.GeoSeries.from_wkt(valdata_comb['geometry'])

# Convert dataframe to geodataframe
valdata_comb = gpd.GeoDataFrame(valdata_comb, geometry='geometry', crs="EPSG:32629") 

# Define folder path
valdat_dir = os.path.join("data", "validation", "validation_datasets")

# Write to file
write_csv(valdata_comb, valdat_dir, "validation_points_2013_2023_780")















