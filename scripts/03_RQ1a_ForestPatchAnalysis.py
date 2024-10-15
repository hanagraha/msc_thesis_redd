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

# Remove small forest patches from GFC arrays
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



############################################################################


# PLOT DEFORESTATION RATES


############################################################################
# Define function to plot deforestation rates
def defor_rate_plot(df_list, colors, labels):
    
    # Initialize plot figure
    plt.figure(figsize=(10, 6))
    
    # Iterate over each dataframe
    for df, col, lab in zip(df_list, colors, labels):
        plt.plot(years, df, color = col, label = lab)

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('% of Deforestation Pixels Per REDD+/Non-REDD+ Area')
    
    # Add title
    plt.title('Deforestation in REDD+ vs Non-REDD+ Villages (2013-2023)')
    
    # Add legend
    plt.legend()

    # Add gridlines
    plt.grid(linestyle='--')
    
    # Rotate x ticks for readability
    plt.xticks(years, rotation=45)
    
    # Adjust layout for spacing
    plt.tight_layout()
    
    plt.show()

# Define reusable parameters for plotting
df_list = [gfc_defor_areas.loc['REDD+'], tmf_defor_areas.loc['REDD+'], 
           gfc_filtered_areas.loc["REDD+"], tmf_filtered_areas.loc["REDD+"], 
           gfc_defor_areas.loc['Non-REDD+'], tmf_defor_areas.loc['Non-REDD+'], 
           gfc_filtered_areas.loc["Non-REDD+"], tmf_filtered_areas.loc["Non-REDD+"]]

colors = ["#8B0000", "#D2691E", "#228B22", "#4682B4"]

labels = ["GFC Deforestation in REDD+ Villages", 
          "TMF Deforestation and Degradation in REDD+ Villages", 
          "GFC Deforestation > 0.81ha in REDD+ Villages", 
          "TMF Deforestation and Degradation > 0.81ha in REDD+ Villages", 
          "GFC Deforestation in Non-REDD+ Villages", 
          "TMF Deforestation and Degradation in Non-REDD+ Villages", 
          "GFC Deforestation > 0.81ha in Non-REDD+ Villages", 
          "TMF Deforestation and Degradation > 0.81ha in Non-REDD+ Villages"]

# Plot GFC and TMF datasets for REDD+ Villages
defor_rate_plot(df_list[:4], colors, labels[:4])

# Plot GFC and TMF datasets for Non-REDD+ Villages
defor_rate_plot(df_list[4:], colors, labels[4:])



############################################################################


# ESTIMATE PATCH SIZE FOR OPTIMAL COMPARABILITY


############################################################################
# Calculate difference between GFC and TMF deforestation 
gfc_tmf_diff = gfc_defor_areas - tmf_defor_areas

# Calculate difference between GFC and TMF deforestation after filtering
filt_gfc_tmf_diff = gfc_filtered_areas - tmf_filtered_areas

# Data list
df_list = [gfc_tmf_diff.loc["REDD+"], gfc_tmf_diff.loc["Non-REDD+"], 
           filt_gfc_tmf_diff.loc["REDD+"], filt_gfc_tmf_diff.loc["Non-REDD+"]]
labels = ["Difference in Unfiltered REDD+ % Deforestation Area", 
          "Difference in Unfiltered Non-REDD+ % Deforestation Area", 
          "Difference in Filtered REDD+ % Deforestation Area", 
          "Difference in Filtered Non-REDD+ % Deforestation Area"]

# Plot differences
defor_rate_plot(df_list, colors, labels)

# Iterate over different patch sizes
for size in range(2,56):
    
    # Filter GFC array 
    gfc_filtered_arrs = patch_filter(gfc_lossyear_arrs, years, size)
    
    # Filter TMF array
    tmf_filtered_arrs = patch_filter(tmf_defordegra_arrs, years, size)
    
    # 

















