# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:25:35 2024

@author: hanna
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from rasterio.mask import mask
from statsmodels.stats.contingency_tables import mcnemar 
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score



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

# Define study range years
years = range(2013, 2024)

# Define temporary folder path
temp_folder = os.path.join('data', "temp")

# Create temporary folder, if it doesn't already exist
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    print(f"{temp_folder} created.")
else:
    print(f"{temp_folder}' already exists.")



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read list of files
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

# Define spatial agreement paths
spatagree_paths = [f"data/intermediate/agreement_gfc_combtmf_{year}.tif" for 
                   year in years]

# Define gfc lossyear paths
gfc_files = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif" for year 
             in years]

# Define tmf defordegra paths
tmf_files = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif" for 
             year in years]

# Read spatial agreement rasters
spatagree_arrs, spatagree_profile = read_files(spatagree_paths)

# Read gfc paths
gfc_arrs, gfc_profile = read_files(gfc_files)

# Read tmf paths
tmf_arrs, tmf_profile = read_files(tmf_files)



############################################################################


# SPLIT AGREEMENT MAP INTO TILES


############################################################################
# Define function to split array into tiles
def tilesplit(pathlist, yearrange, tilesize_m=1000):
    
    # Create empty dictionary to store tile sets per year
    yearly_tiles= {}
    
    # Iterate over each path
    for path, year in zip(pathlist, yearrange):
    
        # Create an empty list to store tiles
        tiles = []
        
        # Open the raster file
        with rasterio.open(path) as src:
            
            # Calculate the tile size in pixels based on the raster resolution
            tilesize_pix = int(tilesize_m / src.res[0])
    
            # Get the dimensions of the raster
            raster_width, raster_height = src.width, src.height
    
            # Loop through the raster and extract tiles
            for i in range(0, raster_width, tilesize_pix):
                for j in range(0, raster_height, tilesize_pix):
                    
                    # Calculate the window for the tile
                    window = Window(i, j, tilesize_pix, tilesize_pix)
    
                    # Read the window (tile) as an array
                    tile = src.read(window=window)
                    
                    # Add tile to list
                    tiles.append(tile)
        
        # Add tiles to yearly list
        yearly_tiles[year] = tiles
                
    return yearly_tiles


# Define function to create contingency tables from agreement layers
def contingency_table(image):
    # Mask out NoData (255) values
    valid_pixels = image[image != nodata_val]
    
    # Count pixels with values 0, 1, and 2
    count_5 = np.sum(valid_pixels == 5) # agreement not deforestation
    count_6 = np.sum(valid_pixels == 6) # only GFC says deforestation
    count_7 = np.sum(valid_pixels == 7) # only TMF says deforestation
    count_8 = np.sum(valid_pixels == 8) # agreement on deforestation
    
    # Create contingency table
    matrix = [[count_8, count_6], 
              [count_7, count_5]]

    return matrix

def mcnemar_per_tile(tiledict):
    
    # Create empty dictionary to store results
    annual_mcnemar = {}

    # Iterate over each dictionary item (years)
    for year, tiles in tiledict.items():
        
        # Create empty list to store per-tile mcnemar statistics
        mcnemars = []
        
        # Iterate over each tile in the year
        for tile in tiles:
            
            # Calculate contingency table
            matrix = contingency_table(tile)
            
            # Extract discoordinant pairs from table
            b, c = matrix[0][1], matrix[1][0]
            
            # If the pairs have data:
            if b + c > 0:
                
                # Calculate mcNemar statistic
                result = mcnemar(matrix)
                
                # Extract mcNemar statistic
                stat = result.statistic
                
                # Extract p-value
                pvalue = result.pvalue
                
            # If the pairs do not have data
            else:
                stat = np.nan
                pvalue = np.nan

            # Store tile results in list
            mcnemars.append({
                'statistic': stat,
                'pvalue': pvalue
            })
        
        # Store list results in dictionary
        annual_mcnemar[year] = mcnemars
    
    return annual_mcnemar

# Define function to write McNemar statistics back into tiles
def mcnemar_rast(annual_mcnemar, pathlist, yearrange, out_dir, fileprefix, 
                 tilesize_m=1000):
    
    # Iterate over each dictionary item (years)
    for year, stats in annual_mcnemar.items():
        
        # Extract corresponding raster path (for that year)
        path = pathlist[yearrange.index(year)]
        
        # Read raster
        with rasterio.open(path) as src:
            
            # Extract raster metadata
            meta = src.meta.copy()
            
            # Define raster dimensions
            raster_width, raster_height = src.width, src.height
            
            # Calculate tile size in pixels
            tilesize_pix = int(tilesize_m / src.res[0])
            
            # Create an empty array to hold mcnemar statistics
            mcnemar_raster = np.full((raster_height, raster_width), nodata_val, 
                                     dtype=np.float32)

            # Initiate tile index counter
            tile_idx = 0
            
            # Iterate over width (by tile)
            for i in range(0, raster_width, tilesize_pix):
                
                # Iteraate over height (by tile)
                for j in range(0, raster_height, tilesize_pix):
                    
                    # Extract McNemar statistic from tile
                    stat = stats[tile_idx]['statistic']
                    
                    # If statistic is NA
                    if np.isnan(stat):
                        
                        # Set pixel value to nodata_val (255)
                        mcnemar_raster[j:j+tilesize_pix, i:i+tilesize_pix] = nodata_val
                        
                    else:
                        # Set pixel value to McNemar statistic
                        mcnemar_raster[j:j+tilesize_pix, i:i+tilesize_pix] = stat
                    
                    # Add 1 to index counter
                    tile_idx += 1

            # Update the metadata to reflect the NoData value
            meta.update({
                'dtype': 'float32',
                'nodata': nodata_val
            })
            
            # Define the output path
            output_path = os.path.join(out_dir, f"{fileprefix}_{year}.tif")
            
            # Write raster to drive
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(mcnemar_raster, 1)
            
        # Print statement
        print(f"Saved mcnemar raster to {output_path}")

# Split agreement rasters into tiles
agtiles = tilesplit(spatagree_paths, years, 1000)

# Calculate mcnemar statistic
ag_mcnemar = mcnemar_per_tile(agtiles)

# Create mcnemar raster
mcnemar_rast(ag_mcnemar, spatagree_paths, years, temp_folder, "ag_mcnemar")



















