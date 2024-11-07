# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:18:38 2024

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



############################################################################


# IMPORT AND READ DATA


############################################################################
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

# Raster data filepaths
gfc_lossyear_file = "data/hansen_preprocessed/gfc_lossyear_fm.tif"
tmf_defordegra_file = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# gfc lossyear filepaths
gfc_annual_files = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif"
                    for year in years]

tmf_annual_files = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif"
                    for year in years]

gfc_arrs, gfc_profile = read_files(gfc_annual_files)

tmf_arrs, tmf_profile = read_files(tmf_annual_files)


# Read raster data
with rasterio.open(gfc_lossyear_file) as gfc:
    gfc_lossyear = gfc.read(1)
    
with rasterio.open(tmf_defordegra_file) as tmf:
    tmf_defordegra = tmf.read(1)
    profile = tmf.profile

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

# Create GRNP geometry
grnp_geom = grnp.geometry



############################################################################


# CREATE BINARY GFC AND TMF LAYERS


############################################################################
# Define function to reclassify multi-year array to binary single-year arrays
def binary_reclass(yeardata, yearrange, class1, class2, nodata):
    binary_list = []
    for year in yearrange:
        binary_data = np.where(yeardata == year, class1, np.where(
            yeardata == nodata, nodata, class2))
        binary_list.append(binary_data)
    print("Binary reclassification complete")    
    return binary_list

# Define function to check values of new array
def valcheck(array, dataname):
    uniquevals = np.unique(array)
    print(f"Unique values in the {dataname} are {uniquevals}")

# Reclassify GFC lossyear array
gfc_binary_arrs = binary_reclass(gfc_lossyear, years, 2, 1, nodata_val)
valcheck(gfc_binary_arrs[1], "gfc binary array")

# Reclassify TMF deforestation and degradation array
tmf_binary_arrs = binary_reclass(tmf_defordegra, years, 6, 4, nodata_val)
valcheck(tmf_binary_arrs[1], "tmf binary array")



############################################################################


# CREATE SPATIAL AGREEMENT RASTERS


############################################################################
# Define function to create attribute table
def att_table(arr):
    
    # Extract unique values and pixel counts
    unique_values, pixel_counts = np.unique(arr, return_counts=True)
    
    # Create a DataFrame with unique values and pixel counts
    attributes = pd.DataFrame({"Class": unique_values, 
                               "Frequency": pixel_counts})
    
    # Switch rows and columns of dataframe
    attributes = attributes.transpose()
    
    return attributes

# Define function to create spatial agreement maps
def spatagree(arrlist1, arrlist2, nd_overlap=False):
    
    # Create empty list to store agreement layers
    aoi_agreement = []
    
    # Ignoring pixels where tmf has nodata but gfc has data
    if nd_overlap == False:
        
        # Iterate over arrays
        for gfc, tmf in zip(gfc_binary_arrs, tmf_binary_arrs):
            
            # Add binary arrays together 
            agreement = np.where((gfc == nodata_val) | (tmf == nodata_val), nodata_val, 
                                 gfc + tmf)
            
            # Add array to list
            aoi_agreement.append(agreement)
    
    else:
        
        # Iterate over arrays
        for gfc, tmf in zip(gfc_binary_arrs, tmf_binary_arrs):
            
            # Create agreement array with conditions
            agreement = np.where(
                
                # Condition 1: Both gfc and tmf are NoData
                (gfc == nodata_val) & (tmf == nodata_val), nodata_val,
                
                # Condition 2: gfc is NoData, tmf is not NoData
                np.where((gfc == nodata_val) & (tmf != nodata_val), 10,
                
                         # Condition 3: tmf is NoData, gfc is not NoData
                         np.where((tmf == nodata_val) & (gfc != nodata_val), 20,
                                  
                                  # Condition 4: Both gfc and tmf have valid data
                                  gfc + tmf)))
            
            aoi_agreement.append(agreement)
    
    return aoi_agreement

# Define function to save a list of files by year
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

# Define function for clipping stack of agreement rasters
def filestack_clip(array_files, yearrange, geometry, nodataval):
    clipped_list = []
    for file, year, in zip(array_files, yearrange):
        with rasterio.open(file) as rast:
            agree_clip, agree_trans = mask(rast, geometry, crop=True,
                                            nodata=nodataval)
        clipped_list.append(agree_clip)
    filenum = len(clipped_list)
    print(f"Clipping complete for {filenum} files")
    return clipped_list

# Define function to clip agreement rasters to multiple geometries
def filestack_clip_multi(array_files, yearrange, geom1, geom2, geom3, nodataval):
    redd_clip = filestack_clip(array_files, yearrange, geom1, nodataval)
    nonredd_clip = filestack_clip(array_files, yearrange, geom2, nodataval)
    grnp_clip = filestack_clip(array_files, yearrange, geom3, nodataval)
    
    return redd_clip, nonredd_clip, grnp_clip

# Create spatial agreement layer for gfc and tmf
aoi_agreement = spatagree(gfc_binary_arrs, tmf_binary_arrs)

# Check values for spatial agreement (should be 5, 6, 7, 8, 255)
valcheck(aoi_agreement[1], "aoi spatial agreement")

# Save each agreement raster to drive
agreement_files = filestack_write(aoi_agreement, years, rasterio.uint8, 
                                  "agreement_gfc_combtmf")

# Clip agreement rasters to REDD+, non-REDD+, and GRNP area
redd_agreement, nonredd_agreement, grnp_agreement = filestack_clip_multi(
    agreement_files, years, [redd_geom], [nonredd_geom], grnp_geom, nodata_val)

# Double check values
valcheck(redd_agreement[1], "redd+ agreement")
valcheck(nonredd_agreement[1], "non-redd+ agreement")
valcheck(grnp_agreement[1], "grnp agreement")



############################################################################


# CALCULATE SPATIAL AGREEMENT STATISTICS (RELATIVE TO AOI)


############################################################################
# Define function to calculate agreement statistics for one image
def agreestats(image, class1=5, class2=6, class3=7, class4=8):
    # Mask out NoData (255) values
    valid_pixels = image[image != 255]
    
    # Count pixels with values 5, 6, 7, and 8
    total_pixels = valid_pixels.size
    count_5 = np.sum(valid_pixels == class1) # agreement undisturbed
    count_6 = np.sum(valid_pixels == class2) # only gfc detects deforested
    count_7 = np.sum(valid_pixels == class3) # only tmf detects deforested
    count_8 = np.sum(valid_pixels == class4) # agreement deforested
    
    # Reclassify counts to agreement and disagreement
    agreement_undisturbed = count_5
    disagreement = count_6 + count_7
    agreement_deforested = count_8
    
    # Calculate ratios
    perc_5 = (agreement_undisturbed / total_pixels)*100
    perc_67 = (disagreement / total_pixels)*100
    perc_8 = (agreement_deforested / total_pixels)*100
    
    return perc_5, perc_67, perc_8

# Define function to calculate agreement statistics for multiple images
def agreestat_summary(imagelist, yearrange):
    # Create an empty list
    agree_stats = []
    
    # Calculate statistics for each image
    for var, year, in zip(imagelist, yearrange):
        perc_5, perc_67, perc_8 = agreestats(var)
        
        # Append results to list as a dictionary
        agree_stats.append({
            'Year': year,
            'Agree_Undisturbed': perc_5,
            'Disagree': perc_67,
            'Agree_Deforested': perc_8
        })   
    
    # Convert list to dataframe
    agree_stats = pd.DataFrame(agree_stats)
    
    return agree_stats

# Calculate summary statistics for AOI
aoi_agree_stats = agreestat_summary(aoi_agreement, years)

# Calculate summary statistics for REDD+ area
redd_agree_stats = agreestat_summary(redd_agreement, years)

# Calculate summary statistics for non-REDD+ area
nonredd_agree_stats = agreestat_summary(nonredd_agreement, years)

# Calculate summary statistics for GRNP area
grnp_agree_stats= agreestat_summary(grnp_agreement, years)



############################################################################


# CALCULATE SPATIAL AGREEMENT STATISTICS (RELATIVE TO DEFORESTATION AREA)


############################################################################
# Define function to calculate agreement statistics for one image
def rel_agreestats(image, class1=5, class2=6, class3=7, class4=8):
    # Mask out NoData (255) values
    valid_pixels = image[image != 255]
    
    # Count pixels with values 5, 6, 7, and 8
    count_5 = np.sum(valid_pixels == class1) # agreement undisturbed
    count_6 = np.sum(valid_pixels == class2) # only gfc detects deforested
    count_7 = np.sum(valid_pixels == class3) # only tmf detects deforested
    count_8 = np.sum(valid_pixels == class4) # agreement deforested
    
    # Reclassify counts to agreement and disagreement
    agreement_undisturbed = count_5
    disagreement = count_6 + count_7
    agreement_deforested = count_8
    
    # Calculate ratios
    perc_5 = (agreement_undisturbed / (agreement_undisturbed + disagreement))*100
    perc_67 = (disagreement / (disagreement + agreement_deforested))*100
    perc_8 = (agreement_deforested / (disagreement + agreement_deforested))*100
    
    return perc_5, perc_67, perc_8

# Define function to create agreement relative statistic summary
def rel_agreestat_summary(imagelist, yearrange):
    # Create an empty list
    agree_stats = []
    
    # Calculate statistics for each image
    for var, year, in zip(imagelist, yearrange):
        perc_5, perc_67, perc_8 = rel_agreestats(var)
        
        # Append results to list as a dictionary
        agree_stats.append({
            'Year': year,
            'Agree_Undisturbed': perc_5,
            'Disagree': perc_67,
            'Agree_Deforested': perc_8
        })   
    
    # Convert list to dataframe
    agree_stats = pd.DataFrame(agree_stats)
    
    return agree_stats

# Calculate summary statistics for AOI
aoi_agree_rel_stats = rel_agreestat_summary(aoi_agreement, years)

# Calculate summary statistics for REDD+ area
redd_agree_rel_stats = rel_agreestat_summary(redd_agreement, years)

# Calculate summary statistics for non-REDD+ area
nonredd_agree_rel_stats = rel_agreestat_summary(nonredd_agreement, years)

# Calculate summary statistics for GRNP area
grnp_agree_rel_stats= rel_agreestat_summary(grnp_agreement, years)



############################################################################


# PLOT STATISTICS FOR SPATIAL AGREEMENT


############################################################################
# Define function to plot spatial agreement statistics
def spatagree_plot(datasetlist, linestyles, upperylim, lowerylim, colors, labels):
    """
    The following is created with help from ChatGPT, especially the diagonal lines
    """
    
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                   gridspec_kw={'height_ratios': [0.8, 1]})
    
    # Plot data on the first subplot
    for data, line in zip(datasetlist, linestyles):
        data.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                               'Agree_Deforested'], kind='line', ax=ax1, 
                                legend=False, color=colors, linestyle=line)
    
    # Set boundaries of first subplot
    ax1.set_ylim(upperylim, 100) 
    ax1.grid(True, linestyle='--')
    ax1.set_ylabel('Proportion of Pixels (%)')
    ax1.spines['bottom'].set_visible(False) 
    
    # Plot data on the second subplot
    for data, line in zip(datasetlist, linestyles):
        data.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                               'Agree_Deforested'], kind='line', ax=ax2, 
                                legend=False, color=colors, linestyle=line)
        
    # Set boundaries of second subplot
    ax2.set_ylim(0, lowerylim)
    ax2.grid(True, linestyle='--')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Proportion of Pixels (%)')
    ax2.spines['top'].set_visible(False)
    
    # Add diagonal lines to indicate axes breaks
    d = .015  
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs) 
    
    kwargs.update(transform=ax2.transAxes) 
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  
    
    # Create legend for both subplots
    lines, _ = ax1.get_legend_handles_labels()
    
    # Add legend
    fig.legend(lines, labels, loc='upper center', 
               bbox_to_anchor=(0.5, +0.01), ncol=3)
    
    # Set X axis ticks
    ax2.set_xticks(redd_agree_stats['Year']) 
    ax2.set_xticklabels(redd_agree_stats['Year'], rotation=45)
    
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    plt.show()


# Define reusable parameters for plotting
datasetlist = [aoi_agree_stats, redd_agree_stats, nonredd_agree_stats, 
               grnp_agree_stats, redd_agree_rel_stats, nonredd_agree_rel_stats]
colors = ['green', 'orange', 'red']
linestyles = ["-", "--", "dashdot"]
labels = ['AOI Agreement on Not Deforested', 
          'AOI Disagreement on Deforested', 
          'AOI Agreement on Deforested',
          'REDD+ Agreement on Not Deforested', 
          'REDD+ Disagreement on Deforested', 
          'REDD+ Agreement on Deforested',
          'Non-REDD+ Agreement on Not Deforested', 
          'Non-REDD+ Disagreement on Deforested', 
          'Non-REDD+ Agreement on Deforested',
          'GRNP Agreement on Not Deforested', 
          'GRNP Disagreement on Deforested', 
          'GRNP Agreement on Deforested']

# Plot statistics for whole AOI
spatagree_plot([datasetlist[0]], linestyles[0], 90, 5, colors, labels[0:3])

# Plot statistics for REDD+ and non-REDD+
spatagree_plot(datasetlist[1:3], linestyles[0:2], 90, 8, colors, labels[3:9])

# Plot relativestatistics for REDD+ and non-REDD+
spatagree_plot(datasetlist[4:6], linestyles[0:2], 65, 35, colors, labels[3:9])

# Plot statistics for REDD+, non-REDD+, and GRNP
spatagree_plot(datasetlist[1:], linestyles, 90, 5, colors, labels[3:]) 



############################################################################


# MCNEMAR'S TEST


############################################################################
"""
McNemar's test requires a square contingency table. This format should be:
    
                               [TMF Deforestation]   [TMF Not Deforestation]      
    [GFC Deforestation]                a                        b
    [GFC Not Deforesetation]           c                        d
    
Examples on conducting McNemar's test with Python found in:
https://www.geeksforgeeks.org/how-to-perform-mcnemars-test-in-python/

"""
# Define function to create contingency tables from agreement layers
def contingency_table(image):
    # Mask out NoData (255) values
    valid_pixels = image[image != 255]
    
    # Count pixels with values 0, 1, and 2
    count_5 = np.sum(valid_pixels == 5) # agreement not deforestation
    count_6 = np.sum(valid_pixels == 6) # only GFC says deforestation
    count_7 = np.sum(valid_pixels == 7) # only TMF says deforestation
    count_8 = np.sum(valid_pixels == 8) # agreement on deforestation
    
    # Create contingency table
    matrix = [[count_8, count_6], 
              [count_7, count_5]]

    return matrix

# Define function to calculate McNemar's statistic
def mcnemar_df(agreement_arrs, yearrange):
    # Create empty list to store contingency tables
    contingency_tables = []
    
    # Create contingency tables for each agreement raster
    for var in agreement_arrs:
        matrix = contingency_table(var)
        contingency_tables.append(matrix)
    
    # Create empty list to store McNemar statistics
    mcnemar_results = []
    
    # Calculate McNemar statistic on each contingency table
    for matrix in contingency_tables:
        result = mcnemar(matrix)
        mcnemar_results.append({'statistic': result.statistic, 
                                'pvalue': result.pvalue})
    
    # Convert list to dataframe
    mcnemar_df = pd.DataFrame(mcnemar_results, index=yearrange)
    
    # Print results
    print(mcnemar_df)
    
    return mcnemar_df
    
# Calculate McNemar's statistic for AOI agreements
aoi_mcnemar = mcnemar_df(aoi_agreement, years)

# Calculate McNemar's statistic for REDD+ agreements
redd_mcnemar = mcnemar_df(redd_agreement, years)

# Calculate McNemar's statistic for non-REDD+ agreements
nonredd_mcnemar = mcnemar_df(nonredd_agreement, years)

# Calculate McNemar's statistic for GRNP agreements
grnp_mcnemar = mcnemar_df(grnp_agreement, years)



############################################################################


# PLOT MCNEMAR'S TEST RESULTS


############################################################################
# Define function to plot McNemar results
def mcnemar_plot(datasetlist, colors, labels):
    plt.figure(figsize=(10, 6))
    
    for data, color, label in zip(datasetlist, colors, labels):
        plt.plot(data.index, data['statistic'], linestyle="-", color=color, 
                 label=label)
    
    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('McNemar Statistic')
    plt.xticks(datasetlist[0].index)  # Show all years as ticks
    
    # Add gridlines
    plt.grid(linestyle='--')
    
    # Add legend
    plt.legend()

    # Display the plot
    plt.show()

# Define reusable plotting parameters
datasetlist = [aoi_mcnemar, redd_mcnemar, nonredd_mcnemar, grnp_mcnemar]
colors = ["brown", "dodgerblue", "darkgreen"]
labels = ["AOI", "REDD+", "Non-REDD+", "GRNP"]

# Plot McNemar results for the whole AOI
mcnemar_plot([datasetlist[0]], [colors[1]], [labels[0]])

# Plot McNemar results for REDD+ and non-REDD+ villages
mcnemar_plot(datasetlist[1:3], colors[0:2], labels[1:3])

# Plot McNemar results of REDD+, nonREDD+, and GRNP
mcnemar_plot(datasetlist[1:], colors, labels[1:])



############################################################################


# RECLASSIFY ALL BINARY LAYERS TO 1, 2, NODATA


############################################################################
"""
To calculate Cohen's Kappa (next processing step), each dataset must have the 
same values. This preliminary step ensures each deforestation array has 
comparable values of 1 (not deforested), 2 (deforested), and nodata
"""
# Reclassify GFC lossyear array with 1, 2, and numpy nodata
gfc_simpbin = binary_reclass(gfc_lossyear, years, 2, 1, nodata_val)
valcheck(gfc_simpbin[1], "gfc simple binary")

# Save each gfc simple binary raster to drive
gfc_simpbin_files = filestack_write(gfc_simpbin, years, rasterio.uint8, 
                                    "gfc_simple_binary")

# Reclassify TMF deforestation and degradation array with 1, 2, and numpy nodata
tmf_simpbin = binary_reclass(tmf_defordegra, years, 2, 1, nodata_val)
valcheck(tmf_simpbin[1], "tmf simple binary")

# Save each tmf simple binary raster to drive
tmf_simpbin_files = filestack_write(tmf_simpbin, years, rasterio.uint8, 
                                    "tmf_simple_binary")

# Clip reclassified GFC binary arrays to REDD+, non-REDD+, and GRNP area
gfc_redd_simpbin, gfc_nonredd_simpbin, gfc_grnp_simpbin = filestack_clip_multi(
    gfc_simpbin_files, years, [redd_geom], [nonredd_geom], grnp_geom, nodata_val)

# Clip reclassified TMF binary arrays to REDD+, non-REDD+, and GRNP area
tmf_redd_simpbin, tmf_nonredd_simpbin, tmf_grnp_simpbin = filestack_clip_multi(
    tmf_simpbin_files, years, [redd_geom], [nonredd_geom], grnp_geom, nodata_val)



############################################################################


# CALCULATE COHEN'S KAPPA


############################################################################
"""
"The kappa statistic, which is a number between -1 and 1. The maximum value 
means complete agreement; zero or lower means chance agreement."
From: https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.cohen_kappa_score.html

"""
# Define function for calculating Cohen's Kappa
def cohen_kappa(arrlist1, arrlist2, nodataval):    
    # Create empty list to store Kappa values
    kappa_results = []
    
    for arr1, arr2 in zip(arrlist1, arrlist2):
        # Convert both arrays to float
        arr1 = arr1.astype(float)
        arr2 = arr2.astype(float)
        
        # Replace nodata value with np.nodata
        arr1[arr1 == nodataval] = np.nan
        arr2[arr2 == nodataval] = np.nan
        
        # Create nodata mask
        mask = ~np.isnan(arr1) & ~np.isnan(arr2)
        
        # Filter arrays to remove NaN pixels
        filtered_arr1 = arr1[mask]
        filtered_arr2 = arr2[mask]
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(filtered_arr1, filtered_arr2)
        
        # Store results in list
        kappa_results.append(kappa)

    return kappa_results

# Calculate Cohen's Kappa per year in the AOI
aoi_kappa = cohen_kappa(gfc_simpbin, tmf_simpbin, nodata_val)

# Calculate Cohen's Kappa for REDD+ villages
redd_kappa = cohen_kappa(gfc_redd_simpbin, tmf_redd_simpbin, nodata_val)

# Calculate Cohen's Kappa for non-REDD+ villages
nonredd_kappa = cohen_kappa(gfc_nonredd_simpbin, tmf_nonredd_simpbin, nodata_val)

# Calculate Cohen's Kappa for GRNP
grnp_kappa = cohen_kappa(gfc_grnp_simpbin, tmf_grnp_simpbin, nodata_val)

# Calculate Cohen's Kappa for whole 2013-2023 range
multiyear_kappa = cohen_kappa([gfc_lossyear], [tmf_defordegra], nodata_val)



############################################################################


# PLOT COHEN'S KAPPA


############################################################################
# Define function to plot Cohen's Kappa
def cohen_plot(datasetlist, yearrange, colors, labels):
    # Initialize figure
    plt.figure(figsize=(12, 6))
    
    for yvals, color, label in zip(datasetlist, colors, labels):
        # Plot results for yearly Kappa values
        plt.plot(yearrange, yvals, linestyle='-', color=color, label=label)

    # Add axes ticks and labels
    plt.xticks(years)
    plt.xlabel('Year')
    plt.ylabel("Cohen's Kappa")
        
    # Add gridlines
    plt.grid(linestyle='--')

    # Add title
    plt.title("Cohen's Kappa for GFC and TMF data")
    
    # Add legend
    plt.legend()

    # Show plot
    plt.show()

# Define reusable plot parameters
datasetlist = [aoi_kappa, redd_kappa, nonredd_kappa, grnp_kappa]
colors = ["brown", "dodgerblue", "darkgreen"]
labels = ["AOI", "REDD+", "Non-REDD+", "GRNP"]

# Plot Cohen's Kappa for AOI
cohen_plot([datasetlist[0]], years, [colors[1]], [labels[0]])

# Plot Cohen's Kappa for REDD+, non-REDD+, and GRNP areas
cohen_plot(datasetlist[1:], years, colors, labels[1:])



############################################################################


# CHI SQUARED TEST


############################################################################
from scipy.stats import chi2_contingency

aoi_tab = contingency_table(aoi_agreement[0])

# defining the table
data = [[207, 282, 241], [234, 242, 232]]
stat, p, dof, expected = chi2_contingency(aoi_tab)

# interpret p-value
alpha = 0.05
print("p value is " + str(p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')


############################################################################


# CREATE SENSITIVE EARLY AGREEMENT MAPS


############################################################################
"""
The sensitive early combination is defined by Bos et al. (2019) as recording 
the earliest deforestation year between two datasets, regardless of the other
product's detection
"""
# Combine tmf and gfc maps
with rasterio.open(gfc_lossyear_file) as src1, \
    rasterio.open(tmf_defordegra_file) as src2:
    
    # Extract raster data
    gfc = src1.read(1)  
    tmf = src2.read(1) 
    
    # Extract metadata
    profile = src1.profile

    # Combine datasets with conditions
    combined_data = np.where(
        
        # Where both datasets have nodata
        (gfc == nodata_val) & (tmf == nodata_val), nodata_val,
        np.where(
            
            # Where only tmf has data, take tmf
            gfc == nodata_val, tmf,
            np.where(
                
                # Where only gfc has data, take gfc
                tmf == nodata_val, gfc,
                np.where(
                    
                    # Where only tmf detects deforestation, take tmf
                    (gfc == 0) & ((tmf >= min(years)) & (tmf <= max(years))), tmf,  
                    np.where(
                        
                        # Where only gfc detects deforestation, take gfc
                        (tmf == 0) & ((gfc >= min(years)) & (gfc <= max(years))), gfc,  
                        np.where(
                            
                            # Where both datasets detect deforestation
                            ((gfc >= min(years)) & (gfc <= max(years))) & \
                                ((tmf >= min(years)) & (tmf <= max(years))), \
                                    np.minimum(gfc, tmf),  
                            np.where(
                                
                                # Where both datasets do NOT detect deforestation
                                (gfc == 0) & (tmf == 0), 0, 
                                
                                # Leftover values
                                nodata_val
                            )
                        )
                    )
                )
            )
        )
    )

    # Define output filename
    gfc_tmf_outfile = "data/intermediate/gfc_tmf_sensitive_early.tif"

    # Write the output raster
    with rasterio.open(gfc_tmf_outfile, 'w', **profile) as dst:
        dst.write(combined_data, 1)

print(f"Combined raster saved to {gfc_tmf_outfile}")

# View unique values to check
comb_vals = np.unique(combined_data)
print(f"Values in agreement map are {comb_vals}")

























