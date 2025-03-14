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


# %%
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


# %%
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


# %%
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


# %%
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


# %%
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






















