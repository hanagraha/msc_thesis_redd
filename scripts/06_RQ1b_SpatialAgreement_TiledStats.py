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
import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt


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

# Set default plotting colors
defaultblue = "#4682B4"
reddcol = "brown"
nonreddcol = "dodgerblue"
grnpcol = "darkgreen"



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

# Split agreement rasters into tiles
agtiles = tilesplit(spatagree_paths, years, 1000)



############################################################################


# PLOTTING OPTIONS


############################################################################
# Define function to plot with standard deviation
def stdevplt(means, std_devs, ylab, title):

    # Initialize figure
    plt.figure(figsize=(10, 6))
    
    # Plot mean line
    plt.plot(years, means, color=defaultblue)
    
    # Assign 0 as lower boundary
    lower_bound = np.maximum(0, np.array(means) - np.array(std_devs))
    
    # Assign upper standard deviation as upper boundary
    upper_bound = np.array(means) + np.array(std_devs)
    
    # Create shading for standard deviation area
    plt.fill_between(years, lower_bound, upper_bound, color=defaultblue, alpha=0.2)
    
    # Add legend
    plt.legend(['Mean Statistic', 'Standard Deviation'], loc='best')
    
    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel(ylab)
    
    # Add title
    plt.title(title)
    
    # Add x tickmarks
    plt.xticks(years, rotation=45)
    
    # Add gridlines
    plt.grid(True, linestyle = "--")
    
    # Show plot
    plt.show()
    
    

############################################################################


# MCNEMAR STATISTIC


############################################################################
# Define function to create contingency tables from agreement layers
def contingency_table(image):
    
    # Count pixels with values 0, 1, and 2
    count_5 = np.sum(image == 5) # agreement not deforestation
    count_6 = np.sum(image == 6) # only GFC says deforestation
    count_7 = np.sum(image == 7) # only TMF says deforestation
    count_8 = np.sum(image == 8) # agreement on deforestation
    
    # Create contingency table
    matrix = [[count_8, count_6], 
              [count_7, count_5]]

    return matrix

# Define function to calculate mcnemar's statistic per tile
def tile_mcnemar(tiledict):
    
    # Create empty dictionary to store results
    annual_mcnemars = []
    annual_pvals= []

    # Iterate over each dictionary item (years)
    for year, tiles in tiledict.items():
        
        # Create empty list to store per-tile mcnemar statistics
        mcnemars = []
        pvals = []
        
        # Iterate over each tile in the year
        for tile in tiles:
            
            # Calculate contingency table
            matrix = contingency_table(tile)
            
            # Extract disagreement pairs from table
            b, c = matrix[0][1], matrix[1][0]
            
            # Extract agreement pairs from table
            a, d = matrix[0][0], matrix[1][1]
            
            # If there is disagreement in the tile
            if b + c > 0:
                
                # Calculate mcnemar statistic
                result = mcnemar(matrix)
                
                # Extract mcnemar statistic
                stat = result.statistic
                
                # Extract p-value
                pvalue = result.pvalue
                
            # If there is no disagreement in the tile
            else:
                
                # If there is agreement in the tile
                if a + d > 0:
                    stat = 0
                    pvalue = np.nan
                    
                # If there is no data in the tile
                else: 
                    stat = np.nan
                    pvalue = np.nan

            # Store tile results in list
            mcnemars.append(stat)
            pvals.append(pvalue)
        
        # Store list results in dictionary
        annual_mcnemars.append(mcnemars)
        annual_pvals.append(pvals)
    
    return annual_mcnemars, annual_pvals

# Define function to write McNemar statistics back into tiles
def tilestat_rast(statlist, pathlist, out_dir, fileprefix, tilesize_m=1000):
    
    # Iterate over each dictionary item (years)
    for path, stats, year in zip(pathlist, statlist, years):
        
        # Read raster
        with rasterio.open(path) as src:
            
            # Extract raster metadata
            meta = src.meta.copy()
            
            # Define raster dimensions
            raster_width, raster_height = src.width, src.height
            
            # Calculate tile size in pixels
            tilesize_pix = int(tilesize_m / src.res[0])
            
            # Create an empty array to hold mcnemar statistics
            stat_raster = np.full((raster_height, raster_width), nodata_val, 
                                  dtype=np.float32)

            # Initiate tile index counter
            tile_idx = 0
            
            # Iterate over width (by tile)
            for i in range(0, raster_width, tilesize_pix):
                
                # Iteraate over height (by tile)
                for j in range(0, raster_height, tilesize_pix):
                    
                    # Extract McNemar statistic from tile
                    stat = stats[tile_idx]
                    
                    # If statistic is NA
                    if np.isnan(stat):
                        
                        # Set pixel value to nodata_val (255)
                        stat_raster[j:j+tilesize_pix, i:i+tilesize_pix] = nodata_val
                        
                    else:
                        # Set pixel value to McNemar statistic
                        stat_raster[j:j+tilesize_pix, i:i+tilesize_pix] = stat
                    
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
                dst.write(stat_raster, 1)
            
        # Print statement
        print(f"Saved tiled stat raster to {output_path}")

# Calculate mcnemar statistic
ag_mcnemar, ag_mcnemar_pval = tile_mcnemar(agtiles)

# Create mcnemar statistic raster
tilestat_rast(ag_mcnemar, spatagree_paths, out_dir, "tiled_mcnemar")


############################################################################


# PLOTTING OPTIONS


############################################################################
# Define function to calculate mean statistic
def stat_mean(statlist):
    
    # Create empty list to hold statistics
    means = []
    stdevs = []
    
    # Iterate over each year
    for stats in statlist:
        
        # Convert stats to np array
        stats = np.array(stats)
        
        # Exclude nodata statistics
        stats = stats[~np.isnan(stats)]
        
        # Calculate mean
        mean = np.mean(stats)
        
        # Calculate standard deviation
        std_dev = np.std(stats)
        
        # Append results to the lists
        means.append(mean)
        stdevs.append(std_dev)
    
    return means, stdevs

# Calculate mean and standard deviation of mcnemar statistic
mcnemar_mean, mcnemar_std = stat_mean(ag_mcnemar)

# Plot mcnemar data
stdevplt(mcnemar_mean, mcnemar_std, 'Tiled McNemar Statistics (Excluding NaN)', 
         'McNemar Statistics per 1000m x 1000m Tile')



############################################################################


# CALCULATE DEFORESTATION AGREEMENT


############################################################################
# Define function to calculate disagreeemnt proportions
def defor_props(tiles):
    
    # Create empty list to hold proportions for each year
    years_props =[]
    
    # Iterate over each year
    for year in years:
        
        # Extract arrays for year of interest
        arrs = agtiles[year]
        
        # Create empty list to hold proportions for each array
        arrs_props = []
        
        # Iterate over each array
        for arr in arrs:
            
            # Calculate total agreement
            ag = np.sum(arr == 8)
            
            # Calculate total disagreement
            disag = (np.sum(arr == 6)) + (np.sum(arr == 7))
            
            # Calculate overall agreement
            ov_ag = (ag / (ag + disag))*100
            
            # Add proportional agreement to list
            arrs_props.append(ov_ag)
            
        # Add all array proportions to list
        years_props.append(arrs_props)
        
    return years_props

# Calculate deforestation proportions
deforprops = defor_props(agtiles)

# Calculate means of deforestation proportions
deforprops_mean, deforprops_stdv = stat_mean(deforprops)

# Plot deforestation proportion data
stdevplt(deforprops_mean, deforprops_stdv, "Tiled Deforestation Agreement", 
         "Deforestation Agreement per 1000m x 1000m Tile")

# Save proportion tiles as raster
tilestat_rast(deforprops, spatagree_paths, out_dir, "tiled_deforprops")



############################################################################


# MCNEMAR STATISTIC


############################################################################


# Define function to write McNemar statistics back into tiles
def mcnemar_rast(annual_mcnemar, pathlist, val, yearrange, out_dir, fileprefix, 
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
                    stat = stats[tile_idx][val]
                    
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
