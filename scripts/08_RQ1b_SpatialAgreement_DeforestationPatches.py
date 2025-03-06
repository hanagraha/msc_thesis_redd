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
import matplotlib.pyplot as plt
from scipy.ndimage import label
import time
import seaborn as sns
from matplotlib.ticker import MultipleLocator



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
gfc_lossyear_paths = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif" for 
                      year in years]

# Define file paths for annual tmf rasters
tmf_defordegra_paths = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif" for 
                      year in years]

# Define file paths for agreement rasters
agreement_files = [f"data/intermediate/agreement_gfc_combtmf_{year}.tif" for 
                   year in years]

# Read GFC rasters 
gfc_lossyear_arrs, profile = read_files(gfc_lossyear_paths)

# Read TMF rasters
tmf_defordegra_arrs, profile = read_files(tmf_defordegra_paths)

# Read agreement rasters
agreement_arrs, profile_agr = read_files(agreement_files)

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


# %%
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

# Cluster spatial agreement deforestation
defor_ratio, defor_size = clustratio_patch(agreement_arrs, years, nodata_val)

# Write spatial agreement clusters to file
ratio_files = filestack_write(defor_ratio, years, rasterio.float32, "disag_ratio")
size_files = filestack_write(defor_size, years, rasterio.uint32, "disag_size")


# %%
# DEBUGGING 11:22

# # Create copy of agreement arrays
# agarrs_merg = agreement_arrs.copy()    
    
# # Combine deforestation agreement classes
# agarrs_merg = [np.where(arr == 5, 0, np.where(np.isin(arr, [6, 7, 8]), 1, arr)) 
#                for arr in agarrs_merg]

# arr = agreement_arrs[0]
# year = 2013

# # Create empty list to hold arrays with agreement ratios
# agratio_arrs = []

# # Create empty list to hold arrays with cluster size labels
# clustsize_arrs = []

# # Mask for deforestation agreement and disagreement pixels
# mask = (arr == 6) | (arr == 7) | (arr == 8)

# # Label clusters in the valid mask
# labeled_array, num_features = label(mask)

# # Initialize output array with NoData values (for clustsize)
# clustsize_array = np.full_like(arr, nodata_val, dtype=np.int32)

# # Initialize output array with NoData values (for ratio)
# agratio_array = np.full_like(arr, nodata_val, dtype=np.float32)

# # Extract agreement pixels
# agree_pixels = (arr == 8)

# # Extract disagreement pixels
# disagree_pixels = np.isin(arr, [6, 7])

# # Iterate over each cluster
# for clust in range(1, num_features + 1):
    
#     # Create cluster mask
#     cluster_mask = labeled_array == clust
    
#     # Calculate number of pixels in cluster
#     clust_size = np.sum(cluster_mask)
    
#     # Assign cluster size to all pixels in cluster
#     clustsize_array[cluster_mask] = clust_size
    
#     # Calculate number of agreement pixels in cluster
#     agree = np.sum(agree_pixels[cluster_mask])
    
#     # Calculate number of disagreement pixels in cluster
#     disagree = np.sum(disagree_pixels[cluster_mask])

#     # Calculate agreement ratio
#     agratio = agree / (disagree + agree) if (disagree + agree) > 0 else 0.0
    
#     # Assign agreement ratio to all pixels in cluster
#     agratio_array[cluster_mask] = agratio
    


# %%
############################################################################


# CLUSTER SPATIAL AGREEMENT PATCHES (OLD)


############################################################################
"""
Agreement: 10min per year, start 3:42 estimate 5:30 end 4 done at 4:27
5601.917588 seconds (93min, approx 1.5 hours)
"""
# Define function to cluster agreement and disagreement deforestation areas
def agpatch_label(arr_list, yearrange, nodata_val):
    
    # Note start time to track operation duration
    start_time = time.time()
    
    # Create empty lists to hold labeled arrays for each mask
    labeled_arrs_8 = [] 
    labeled_arrs_67 = [] 
    
    # Iterate through each array and year
    for arr, year in zip(arr_list, yearrange):
    
        # Mask 1: deforestation agreement pixels
        mask_8 = (arr == 8)
        
        # Mask 2: deforestation disagreement pixels
        mask_67 = (arr == 6) | (arr == 7)
        
        # Label agreement pixel clusters
        labeled_array_8, num_features_8 = label(mask_8)
        
        # Label disagreement pixel clusters
        labeled_array_67, num_features_67 = label(mask_67)
        
        # Create NoData array for agreement clusters
        cluster_size_array_8 = np.full_like(arr, nodata_val, dtype=np.int32)
        
        # Create NoData array for disagreement clusters
        cluster_size_array_67 = np.full_like(arr, nodata_val, dtype=np.int32)
        
        # Iterate over each agreement cluster
        for clust in range(1, num_features_8 + 1):
            
            # Calculate number of pixels in cluster
            clust_size_8 = np.sum(labeled_array_8 == clust)
            
            # Assign cluster size to all pixels in cluster
            cluster_size_array_8[labeled_array_8 == clust] = clust_size_8
        
        # Iterate over each disagreement cluster
        for clust in range(1, num_features_67 + 1):
            
            # Calculate number of pixels in cluster
            clust_size_67 = np.sum(labeled_array_67 == clust)
            
            # Assign cluster size to all pixels in cluster
            cluster_size_array_67[labeled_array_67 == clust] = clust_size_67
        
        # Add labeled agreement clusters to list
        labeled_arrs_8.append(cluster_size_array_8)
        
        # Add labeled disagreement clusters to list
        labeled_arrs_67.append(cluster_size_array_67)
        
        # Print statement for status
        print(f"Labeled arrays for {year}: Mask 8 and Mask 6 or 7")
    
    # Note end time to track operation duration
    end_time = time.time()
    
    # Calculate operation duration
    elapsed_time = end_time - start_time
    
    print(f"Operation took {elapsed_time:.6f} seconds")
    
    # Return both sets of labeled arrays
    return labeled_arrs_8, labeled_arrs_67

# Cluster spatial agreement arrays
agree_clust, disagree_clust = agpatch_label(agreement_arrs, years, nodata_val)

# Write agreement clusters to file
agree_clust_files = filestack_write(agree_clust, years, rasterio.uint32, 
                                    "agree_forclust")

# Write disagreement clusters to file
disagree_clust_files = filestack_write(disagree_clust, years, rasterio.uint32, 
                                       "disagree_forclust")


# %%
############################################################################


# (RE-) READ CLUSTERED FILES


############################################################################
"""
This is so the clustering functions do not have to be re-run to execute
subsequent code
"""
# Define file paths for clustered gfc rasters
gfc_forclust_paths = [f"data/intermediate/gfc_forclust_{year}.tif" for 
                      year in years]

# Define file paths for clustered tmf rasters
tmf_forclust_paths = [f"data/intermediate/tmf_forclust_{year}.tif" for 
                      year in years]

# Define file paths for clustered agreement rasters
agree_forclust_paths = [f"data/intermediate/agree_forclust_{year}.tif" for 
                        year in years]

# Define file paths for clustered disagreement rasters
disagree_forclust_paths = [f"data/intermediate/disagree_forclust_{year}.tif" 
                            for year in years]

# Read gfc rasters 
gfc_forclust_arrs, profile = read_files(gfc_forclust_paths)

# Read tmf rasters
tmf_forclust_arrs, profile = read_files(tmf_forclust_paths)

# Read agreement rasters
agree_forclust_arrs, agprofile = read_files(agree_forclust_paths)

# Read disagreement rasters
disagree_forclust_arrs, agprofile = read_files(disagree_forclust_paths)



############################################################################


# CREATE BOX PLOTS OF DEFORESTATION PATCH SIZES


############################################################################
# Define function to create attribute tables
def att_table(arr_list, expected_classes = None, nodata = None):
    
    # Create empty list to hold dataframes
    att_tables = []
    
    # Iterate over each array
    for arr in arr_list:
        
        if nodata is not None:
            unique_values, pixel_counts = np.unique(arr[arr != nodata], 
                                                    return_counts=True)
        else:
            unique_values, pixel_counts = np.unique(arr, return_counts=True)
        
        # Create a DataFrame with unique values and pixel counts
        attributes = pd.DataFrame({"Class": unique_values, 
                                    "Frequency": pixel_counts})
        
        # If expected_classes is provided, run the following:
        if expected_classes is not None:
            
            # Reindex DataFrame to include all expected_classes
            attributes = attributes.set_index("Class").reindex(expected_classes, 
                                                                fill_value=0)
            
            # Reset index to have Class as a column again
            attributes.reset_index(inplace=True)
            
        # Switch rows and columns of dataframe
        attributes = attributes.transpose()
        
        # Add attributes to list
        att_tables.append(attributes)
    
    return att_tables

# Define function to restructure attribute tables for boxplots
def boxplot_dat(df_list, yearrange):
    
    # Create empty list to hold data
    boxplot_data = []

    # Iterate over each DataFrame (yearly data)
    for df, year in zip(df_list, years):
        
        # Extract class and frequency
        classes = df.iloc[0].values
        frequencies = df.iloc[1].values  
        
        # Create a list of patch sizes based on their frequency
        for class_size, freq in zip(classes, frequencies):
            
            # Append each class size 'freq' times to the boxplot_data
            boxplot_data.extend([(class_size, year)] * freq)  

    # Convert to list to dataframe
    boxplot_df = pd.DataFrame(boxplot_data, columns=['Patch Size', 'Year'])
    
    return boxplot_df

# Define function to plot data in boxplot
def forclust_bxplt(boxplot_dflist, titles):
    
    # Initialize a new DataFrame to hold all data
    combined_df = pd.DataFrame()

    # Iterate over each dataframe
    for df, title in zip(boxplot_dflist, titles):
        
        # Add dataset column
        df['Dataset'] = title
        
        # Combine with overall dataframe
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Initialize figure
    plt.figure(figsize=(14, 10))

    # Plot boxplot data
    sns.boxplot(x='Year', y='Patch Size', hue='Dataset', data=combined_df, 
                width = 0.7)

    # Edit y tickmarks
    plt.yticks(range(0, 801, 100))
    
    # Add y major gridlines
    plt.grid(axis='y', which='major', color='gray', linestyle='--', 
              linewidth=0.5)
    
    # Set locations for minor gridlines
    plt.gca().yaxis.set_minor_locator(MultipleLocator(50))
    
    # Add y minor gridlines
    plt.grid(axis='y', which='minor', color='lightgray', linestyle=':', 
              linewidth=0.5)

    # Add title
    plt.title('Distribution of Deforestation Patch Sizes by Year')

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('Deforestation Patch Sizes (# Pixels)')

    # Rotate x tickmarks for readability
    plt.xticks(rotation=45)

    # Show plot
    plt.show()

# Create attribute table for gfc clustered forest
gfc_forclust_atts = att_table(gfc_forclust_arrs, nodata = nodata_val)

# Create attribute table for tmf clustered forest
tmf_forclust_atts = att_table(tmf_forclust_arrs, nodata = nodata_val)

# Create attribute table for agreement clusters
agree_forclust_atts = att_table(agree_forclust_arrs, nodata = nodata_val)

# Create attribute table for disagreement clusters
disagree_forclust_atts = att_table(disagree_forclust_arrs, nodata = nodata_val)

# Create boxplot data for gfc clustered forest
gfc_boxplot_df = boxplot_dat(gfc_forclust_atts, years)

# Create boxplot data for tmf clustered forest
tmf_boxplot_df = boxplot_dat(tmf_forclust_atts, years)

# Create boxplot with gfc and tmf data
forclust_bxplt([gfc_boxplot_df, tmf_boxplot_df], ["GFC Deforestation", 
                                                  "TMF Deforestation"])


# %%
############################################################################


# CALCULATE AGREEMENT STATISTICS BY FOREST PATCH SIZE


############################################################################
# Define function to calculate zonal statistics with spatial agreement
def patch_agreement(forclust_arrlist, agreement_arrlist, nodataval):
    
    # Create list to hold statistics per year
    annual_cluststats = []
    
    # Iterate over each array
    for clustarr, agree in zip(forclust_arrlist, agreement_arrlist):
        
        # Calculate number of unique clusters labels
        clustnum = np.unique(clustarr[clustarr != nodataval])
        
        # Create empty list to store zonal stats
        zone_stats = []
        
        # Iterate over each unique cluster
        for clust in clustnum:
            
            # Create mask for pixels in cluster
            clustmask = (clustarr == clust)
            
            # Extract agreement values
            clust_agree = agree[clustmask]
            
            # Calculate count of each agreement type
            count_6 = np.sum(clust_agree == 6) # only gfc deforestation
            count_7 = np.sum(clust_agree == 7) # only tmf deforestation
            count_8 = np.sum(clust_agree == 8) # agreement on deforestation
            
            # Calculate agreement/ disagreement ratio
            if count_8 != 0:
                ratio = (count_6 + count_7) / count_8
            else:
                ratio = 0
            
            # Append results
            zone_stats.append({
                'Patch Size': clust, 
                'Disagreement': count_6 + count_7, 
                'Agreement': count_8, 
                'Disagreement Ratio': ratio})
            
            # Convert to dataframe
            cluster_agstats = pd.DataFrame(zone_stats).transpose()
        
        # Append statistics to list
        annual_cluststats.append(cluster_agstats)
            
    return annual_cluststats

# Define function to plot cluster statistics
def clust_scatplot(clust_statlist, yearrange, dataname):
    
    # Initialize figure
    plt.figure(figsize=(10, 6))
    
    # Iterate over each dataframe
    for df, year in zip(clust_statlist, yearrange):
        
        # Plot ratio and patch size
        plt.scatter(df.loc['Patch Size'], df.loc['Disagreement Ratio'], 
                    label = year, s=10)
    
    # Add axes labels
    plt.xlabel("Forest Patch Size (# Pixels)")
    plt.ylabel("Disagreement/Agreement Ratio")
    
    # Add title
    plt.title(f"Disagreement to Agreement Ratio by {dataname} Deforestation Patches (2013-2023)")
              
    # Add legend
    plt.legend(title="Year")
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Calculate cluster statistics for gfc
gfc_cluststats = patch_agreement(gfc_forclust_arrs, agreement_arrs, nodata_val)

# Calculate cluster statistics for tmf
tmf_cluststats = patch_agreement(tmf_forclust_arrs, agreement_arrs, nodata_val)

# Plot gfc cluster statistics
clust_scatplot(gfc_cluststats, years, "GFC")

# Plot tmf cluster statistics
clust_scatplot(tmf_cluststats, years, "TMF")



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
bins = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7,8), (8,9), (9,10),
        (10,20), (20,40), (40,80), (80,160), (160,320), (320,640)]

# Define bin labels
bin_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10-20', '20-40', 
              '40-80', '80-160', '160-320', '320-640']
    

bins = [(1, 10), (10,50), (50,100), (100,200), (200,600)]

# Define bin labels
bin_labels = ['1-10', '10-50', '50-100', '100-200', '200-600']

# Bin agreement clusters
agree_binned = bin_clusts(agree_forclust_atts, bins, bin_labels)

# Bin disagreement clusters
disagree_binned = bin_clusts(disagree_forclust_atts, bins, bin_labels)

# Calculate agreement/disagreement ratios
agratio_binned = agree_binned / (disagree_binned + agree_binned)

# Replace na values with 0
agratio_binned_cleaned = agratio_binned.fillna(0)



############################################################################


# PLOT 


############################################################################
# Define function to create line plot
def agratio_plot(df, yearrange):

    # Initiate figure
    plt.figure(figsize=(10,6))
    
    # Iterate over each column (series)
    for lab in df.columns:
        
        # Plot series as line graph
        plt.plot(years, df[lab], label=f"{lab} Pixels")

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('Proportion of Agreement')
    
    # Add title
    plt.title('Spatial Agreement between GFC and TMF by Deforestation Patch Size')
    
    # Add tickmarks
    plt.xticks(years)
    
    # Add legend
    plt.legend(title='Deforestation Patches', loc='lower left')
    
    # Add gridlines
    plt.grid(True)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
# Plot agreement ratios for different deforestation patches
agratio_plot(agratio_binned_cleaned, years)



############################################################################


# IDENTIFY PATCH SIZE FOR MAXIMUM AGREEMENT 


############################################################################
# # Define function to find maximum frequency in multiple dataframes
# def max_freq(df_list, yearrange):
    
#     # Create empty list to hold results
#     yearly_maxes = []
    
#     # Iterate over each DataFrame
#     for df, year in zip(df_list, yearrange):
        
#         # Extract patch sizes
#         patch_sizes = df.iloc[0, :]
        
#         # Extract frequencies
#         frequencies = df.iloc[1, :]
        
#         # Find index of maximum frequency
#         max_freqindex = frequencies.idxmax()
        
#         # Find patch size of maximum frequency
#         max_patchsize = patch_sizes[max_freqindex]
        
#         # Append maximum patch size to list
#         yearly_maxes.append(max_patchsize)
        
#     # Create dataframe
#     max_patchsize = pd.DataFrame({
#         'Year': list(yearrange), 
#         'Patch Size': yearly_maxes})

#     return max_patchsize

# # Calculate maximum agreement patchsize
# agree_max = max_freq(agratio_binned_cleaned, years)

# # Calculate maximum disagreement patchsize
# disagree_max = max_freq(disagree_forclust_atts, years)





