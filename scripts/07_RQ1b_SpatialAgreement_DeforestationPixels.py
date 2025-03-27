# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:18:38 2024

@author: hanna

This file uses annual rasters for gfc lossyear and tmf defordegra created
in 02_DeforestationData_PreProcessing.py to calculate spatial agreement. 

Expected runtime: <1min
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

# Define gfc lossyear file
gfc_lossyear_file = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Define tmf defordegra file
tmf_defordegra_file = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# Define annual gfc filepaths
gfc_annual_files = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif"
                    for year in years]

# Define annual tmf filepaths
tmf_annual_files = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif"
                    for year in years]

# Read annual gfc rasters
gfc_arrs, gfc_profile = read_files(gfc_annual_files)

# Read annual tmf rasters
tmf_arrs, tmf_profile = read_files(tmf_annual_files)

# Read gfc lossyear
with rasterio.open(gfc_lossyear_file) as gfc:
    gfc_lossyear = gfc.read(1)
    
# Read tmf defordegra
with rasterio.open(tmf_defordegra_file) as tmf:
    tmf_defordegra = tmf.read(1)
    profile = tmf.profile

# Read villages shapefile
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp shapefile
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Simplify villages to only redd and nonredd villages
villages = villages[['grnp_4k', 'geometry']]
villages = villages.dissolve(by='grnp_4k')
villages = villages.reset_index()

# Create redd and nonredd geometries
redd_geom = villages.loc[1, 'geometry']
nonredd_geom = villages.loc[0, 'geometry']

# Create grnp geometry
grnp_geom = grnp.geometry


# %%
############################################################################


# RECLASSIFY ANNUAL GFC AND TMF LAYERS FOR SPATIAL AGREEMENT


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

























