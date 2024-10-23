# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:57:46 2024

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
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from scipy.ndimage import label
import time
import matplotlib.ticker as ticker



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
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')



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
gfc_lossyear_paths = [f"data/intermediate/gfc_lossyear_{year}.tif" for 
                      year in years]

# Define file paths for annual tmf rasters
tmf_defordegra_paths = [f"data/intermediate/tmf_defordegra_{year}.tif" for 
                      year in years]

# Read GFC rasters 
gfc_lossyear_arrs, profile = read_files(gfc_lossyear_paths)

# Read TMF rasters
tmf_defordegra_arrs, profile = read_files(tmf_defordegra_paths)

# Read village polygons
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp polygons
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']])

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k')

# Create village polygon with no subgroups
villages_merged = villages.dissolve()



############################################################################


# REMOVE SMALL DEFORESTATION PATCHES


############################################################################
"""
Note: this segment takes ~ 1 hour 33 minutes to run!!
"""
# Define function to check values of array
def valcheck(array, dataname):
    uniquevals = np.unique(array)
    print(f"Unique values in the {dataname} are {uniquevals}")

# Define function to remove small patches of deforestation
def patch_filter(arr_list, yearrange, min_patch_size):
    
    # Note start time to track operation duration
    start_time = time.time()
    
    # Create empty list to hold filtered arrays
    filtered_arrs = []
    
    # Iterate through each array and year
    for arr, year in zip(arr_list, yearrange):
    
        # Create true/false mask
        mask = arr == year
        
        # Label clusters in array
        labeled_array, num_features = label(mask)
        
        # Iterate through each labeled cluster
        for clust in range(1, num_features + 1):
            
            # Calculate number of pixels in cluster
            clust_size = np.sum(labeled_array == clust)
            
            # If clusters don't meet threshold, convert to NoData
            if clust_size < min_patch_size:
                arr[labeled_array == clust] = nodata_val
        
        # Add filtered array to list
        filtered_arrs.append(arr)    
        
        # Print statement for status
        print(f"Filtered array for {year}")
    
    # Note end time to track operation duration
    end_time = time.time()
    
    # Calculate operation duration
    elapsed_time = end_time - start_time
    
    print(f"Operation took {elapsed_time:.6f} seconds")
    
    return filtered_arrs

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

# Remove small forest patches from GFC arrays (takes ~33min)
gfc_filtered_arrs = patch_filter(gfc_lossyear_arrs, years, 9)

# Write filtered GFC arrays to file
gfc_filtered_files = filestack_write(gfc_filtered_arrs, years, 
                                     rasterio.uint16, "gfc_patch9")

# Remove small forest patches from TMF arrays (start 1:47, end 2:45)
tmf_filtered_arrs = patch_filter(tmf_defordegra_arrs, years, 9)

# Write filtered TMF arrays to file
tmf_filtered_files = filestack_write(tmf_filtered_arrs, years, 
                                     rasterio.uint16, "tmf_patch9")



############################################################################


# EXTRACT YEARLY DEFORESTATION FOR REDD+/NON-REDD+ VILLAGE GROUPS


############################################################################
# Define function to calculate areas of polygons (in ha)
def poly_area(polylist):
    area_list = []
    for poly in polylist:
        poly_area = poly.geometry.area / 10000
        area_list.append(poly_area)
    
    return area_list

# Define function to calculate zonal statistic per image and polygon
def multi_zonal_stats(polylist, polynames, tif_list, yearrange, arealist):
    
    # Create empty list to hold statistics
    stats_df = pd.DataFrame(index=yearrange).transpose()
    
    # Iterate over each polygon file
    for poly, name in zip(polylist, polynames):
        
        # Create empty dictionary to store statistics
        data = {f"{name}": []}
                
        # Iterate over each file
        for file in tif_list:
            
            # Calculate zonal statistics for that polygon
            stats = zonal_stats(poly, file, stats="count", nodata=nodata_val)
            
            # Add statistics to dictionary
            data[f"{name}"].append(stats[0]['count'])
        
        # Convert dictionary to dataframe
        data_df = pd.DataFrame(data, index=yearrange).transpose()
        
        # Add statistics dataframes to list
        stats_df = pd.concat([stats_df, data_df])
        
    # Multiply pixel counts by pixel size to get area
    area_df = stats_df * pixel_area
    
    # Divide each row by respective area
    area_df = area_df.div(arealist, axis=0)
        
    return stats_df, area_df

# To avoid re-running the filtering code, get the files here:
gfc_filtered_files = [f"data/intermediate/gfc_patch9_{year}.tif" for 
                      year in years]
tmf_filtered_files = [f"data/intermediate/tmf_patch9_{year}.tif" for 
                      year in years]

# Polygons of interest for zonal statistics
zones = [villages.loc[1], villages.loc[0], villages_merged.loc[0], aoi.loc[0]]

# List of polygon names
names = ["REDD+", "Non-REDD+", "All Villages", "AOI"]

# Calculate area of each polygon
areas = poly_area(zones)

# Calculate zonal statistics with GFC data
gfc_defor_stats, gfc_defor_area = multi_zonal_stats(zones, names, 
                                    gfc_lossyear_paths, years, areas)

# Calculate zonal statistics with TMF data
tmf_defor_stats, tmf_defor_area = multi_zonal_stats(zones, names, 
                                    tmf_defordegra_paths, years, areas)

# Calculate zonal statistics with filtered GFC data
gfc_filtered_stats, gfc_filtered_area = multi_zonal_stats(zones, names, 
                                    gfc_filtered_files, years, areas)

# Calcualte zonal statistics with filtered TMF data
tmf_filtered_stats, tmf_filtered_area = multi_zonal_stats(zones, names, 
                                    tmf_filtered_files, years, areas)

# Calculate difference between GFC and TMF deforestation 
gfc_tmf_diff = gfc_defor_area - tmf_defor_area

# Calculate difference between GFC and TMF deforestation after filtering
filt_gfc_tmf_diff = gfc_filtered_area - tmf_filtered_area



############################################################################


# PLOT DEFORESTATION RATES


############################################################################
# Define function to plot deforestation rates
def defor_rate_lines(df_list, series, colors, labels, title):
    
    # Initialize plot figure
    plt.figure(figsize=(10, 6))
        
    # Initialize empty list to hold data
    df_data = []
    
    # Iterate over dataframes
    for df in df_list:
        
        # Iterate over zones (rows) of interest
        for zone in series:
            
            # Get row data from that zone
            row_data = df.loc[zone]
        
            # Add row data to dataframe data
            df_data.append(row_data)
    
    # Check minimum value in dataset
    min_value = min([df.min() for df in df_data])
    
    # Conditional statement if any values are less than 0
    if min_value < 0:
        
        # Add black horizontal line at y=0
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Iterate over data in list
    for df, col, lab in zip(df_data, colors, labels):
        plt.plot(years, df, color = col, label = lab)

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('% of Deforestation Pixels Per Area')
    
    # Add title
    plt.title(title)
    
    # Add legend
    plt.legend()

    # Add gridlines
    plt.grid(linestyle='--')
    
    # Rotate x ticks for readability
    plt.xticks(years, rotation=45)
    
    # Adjust layout for spacing
    plt.tight_layout()
    
    plt.show()

def defor_rate_bars(df_list, series, colors, labels, title):
    
    # Initialize plot figure
    plt.figure(figsize=(12, 6))
        
    # Initialize empty list to hold data
    df_data = []
    
    # Iterate over dataframes
    for df in df_list:
        
        # Iterate over zones (rows) of interest
        for zone in series:
            
            # Get row data from that zone
            row_data = df.loc[zone]
        
            # Add row data to dataframe data
            df_data.append(row_data)
    
    # Check minimum value in dataset
    min_value = min([df.min() for df in df_data])
    
    # Conditional statement if any values are less than 0
    if min_value < 0:
        
        # Add black horizontal line at y=0
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        
    # Set width of bars
    bar_width = 0.3 
    
    # Set position of bars
    x_positions = np.arange(len(years)) 

    # Iterate over data in list and plot bars
    for i, (df, col, lab) in enumerate(zip(df_data, colors, labels)):
        
        # Offset the bars for each dataset to avoid overlap
        plt.bar(x_positions + i * bar_width, df, color=col, width=bar_width, label=lab)

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('GFC - TMF Deforestation %')
    
    # Add title
    plt.title(title)
    
    # Add legend
    plt.legend()

    # Add gridlines
    ax = plt.gca()
    
    # Add major ticks every 0.005 units
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    
    # Add gridlines for major ticks
    ax.grid(True, which='major', axis='y', linestyle='--', color='gray')
    
    # Rotate x ticks for readability
    plt.xticks(x_positions + bar_width * (len(df_data) - 1) / 2, years, rotation=45)
    
    # Adjust layout for spacing
    plt.tight_layout()
    
    # Show plot
    plt.show()

# Define reusable parameters for plotting
data = [gfc_defor_area, tmf_defor_area, gfc_filtered_area, tmf_filtered_area, 
        gfc_tmf_diff, filt_gfc_tmf_diff]

colors = ["#8B0000", "#D2691E", "#228B22", "#4682B4"]

series = ['REDD+', 'Non-REDD+', 'All Villages', 'AOI']

# Plot GFC data for all zones
title = "GFC Deforestation Area"
labels = ["REDD+", "Non-REDD+", "All Villages", "AOI"]
defor_rate_lines([data[0]], series, colors, labels, title)

# Plot GFC and TMF data for village areas
title = "GFC and TMF Deforestation in Village Area"
labels = ["Unfiltered GFC Deforestation", "Unfiltered TMF Deforestation",
          "Filtered GFC Deforestation (min patch size = 9)", 
          "Filtered TMF Deforestation (min patch size = 9)"]
defor_rate_lines(data[:4], [series[2]], colors, labels, title)

# Plot dataset differences for REDD+, Non-REDD+, unfiltered, filtered (line)
title = "Difference between GFC and TMF Deforestation Area"
labels = ["Unfiltered REDD+ Deforestation", 
          "Unfiltered Non-REDD+ Deforestation", 
          "Filtered REDD+ Deforestation", 
          "Filtered Non-REDD+ Deforestation"]
defor_rate_lines(data[4:], series[0:2], colors, labels, title)

# Plot dataset differences for REDD+, Non-REDD+, unfiltered, filtered (bar)
defor_rate_bars(data[4:], series[0:2], colors, labels, title)

# Plot dataset differences for AOI with unfiltered and filtered data (bar)
labels = ["Unfiltered Deforestation for AOI", 
          "Filtered Deforestation for AOI (min patch size = 9)"]
defor_rate_bars(data[4:], [series[3]], colors, labels, title)

# Plot dataset differences for AOI with unfiltered and filtered data (bar)
labels = ["Unfiltered Deforestation for All Villages", 
          "Filtered Deforestation for All Villages (min patch size = 9)"]
defor_rate_bars(data[4:], [series[2]], colors, labels, title)



############################################################################


# CLUSTER FOREST PATCHES


############################################################################
"""
GFC: 3577.453431 seconds
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

# Cluster GFC forests
gfc_forclust = patch_label(gfc_lossyear_arrs, years, nodata_val)

# Write GFC clusters to file
gfc_forclust_files = filestack_write(gfc_forclust, years, rasterio.uint32, "gfc_forclust")

# Cluster TMF forests
tmf_forclust = patch_label(tmf_defordegra_arrs, years, nodata_val)

# Write TMF clusters to file
tmf_forclust_files = filestack_write(tmf_forclust, years, rasterio.uint32, "tmf_forclust")



# started 5:02 end 5:14, one array done in 10min next at 5:24

############################################################################


# STATISTICS ON FOREST PATCHES


############################################################################





