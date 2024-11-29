# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:17:44 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import shutil
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import time




############################################################################


# SET UP DIRECTORY AND NODATA


############################################################################
# Define function to create temporary folders of choice
def folder_create(folderpath):
    
    # If the folder doesn't already exist
    if not os.path.exists(folderpath):
        
        # Create the folder
        os.makedirs(folderpath)
        
        # Print statement
        print(f"{folderpath} created.")
    
    # If the folder already exists
    else:
        
        # Print statement
        print(f"{folderpath}' already exists.")
        
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())

# Set nodata value
nodata_val = 255

# Set year range 
years = range(2013, 2025)

# Set input directory
in_dir = os.path.join('data', 'landsat_raw')

# Set output directory
out_dir = os.path.join('data', 'validation')

# Define temporary folder path for intermediate landsat folders
temp_folder = os.path.join('data', "landsat_intermediate")

# Define temporary subfolder for january images
jan_temp = os.path.join(temp_folder, "jan")

# Define temporary subfolder for february images
feb_temp = os.path.join(temp_folder, "feb")

# Define temporary subfolder for february images
dec_temp = os.path.join(temp_folder, "dec")

# Create temporary landsat intermediate folder
folder_create(temp_folder)

# Create temporary january subfolder
folder_create(jan_temp)

# Create temporary february subfolder
folder_create(feb_temp)

# Create temporary december subfolder
folder_create(dec_temp)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to return filepaths in a subfolder
def subfolder_paths(month):
    
    # Define filepaths for subfolder
    folder = os.path.join(in_dir, f"GEE_Landsat_{month}")
    
    # Define images in subfolder
    files = [f for f in os.listdir(folder) if f.endswith('.tif')]
    
    # Define image filepaths
    filepaths = [os.path.join(folder, f) for f in files]
    
    return filepaths
    
# Read village polygons
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp polygons
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']]).geometry

# Define january filepaths
jan_filepaths = subfolder_paths("Jan")

# Define february filepaths
feb_filepaths = subfolder_paths("Feb")

# Define december filepaths
dec_filepaths = subfolder_paths("Dec")



############################################################################


# HANDLE FILE NODATA VALUES


############################################################################ 
"""
Expected runtime: ~6mins
"""
# Define function to handle nodata values
def landsat_clean(filepath_list, outfolder):
    
    # Define total number of files
    total = len(filepath_list)
    
    # Create empty list to hold new filepaths
    files = []

    for p, path in enumerate(filepath_list):
        
        with rasterio.open(path) as rast:
            
            # Read data with all bands
            data = rast.read()
            
            # Read profile
            profile = rast.profile
            
        # Extract number of bands
        bandno = profile['count']
        
        # Create empty list to hold manipulated rasters
        data_new = []
        
        # Iterate over each band
        for i in range(bandno):
            
            # Convert data to floating point for nodata processing
            data[i] = data[i].astype('float16')
        
            # Identify data minimum
            # min_val = data[i].min()
            
            # Identify data maximum
            # max_val = data[i].max()
        
            # Rescale data to int8 (0-254, saving 255 for nodata)
            # data_int8 = 254 * (data[i] - min_val) / (max_val - min_val)
            
            # Set all values with pixel value 0 to np.nodata
            data_int8 = np.where(data[i] == 0, np.nan, data[i])
        
            # Convert nodata pixels to value 255
            # data_int8 = np.where(np.isnan(data_int8), nodata_val, data_int8)
            
            # Add new data to list
            data_new.append(data_int8)
            
        # Create a raster stack out of new rasters
        data_stack = np.stack(data_new, axis = 0)
        
        # Update profile
        profile['nodata'] = np.nan
        profile['dtype'] = 'float32'
        
        # Define filename
        filename = os.path.basename(path)
        
        # Define output filepath
        output_filepath = os.path.join(outfolder, filename)
        
        # Write updated data to file
        with rasterio.open(output_filepath, "w", **profile) as dest:
            dest.write(data_stack)
            
        # Add filename to list
        files.append(output_filepath)
            
        # Print statement
        print(f"Exported image {p+1} of {total} to {outfolder}")
    
    # Create space to separate re-runs
    print()
    
    return files

# Handle nodata and scaling for january data (2mins)
jan_filepaths_c = landsat_clean(jan_filepaths, jan_temp)

# Handle nodata and scaling for february data (2mins)
feb_filepaths_c = landsat_clean(feb_filepaths, feb_temp)

# Handle nodata and scaling for december data (2mins)
dec_filepaths_c = landsat_clean(dec_filepaths, dec_temp)



############################################################################


# CHECK REPROJECTIONS


############################################################################ 
"""
Expected execution time: ~410 seconds
"""   
# Define function to reproject raster
def reproject_raster(path, epsg):
        
    # Open raster
    with rasterio.open(path) as rast:
        
        # Calculate transform data
        transform, width, height = calculate_default_transform(
            rast.crs, epsg, rast.width, rast.height, *rast.bounds)
        
        # Copy raster metadata
        kwargs = rast.meta.copy()
        
        # Update raster metadata with transform data
        kwargs.update({'crs': epsg,
                       'transform': transform,
                       'width': width,
                       'height': height,
                       'nodata': np.nan})
        
        # Define new filepath
        filepath = path.replace('LC08', 'pp')
        
        # Write reprojected file to drive
        with rasterio.open(filepath, 'w', **kwargs) as dst:
            for i in range(1, rast.count + 1):
                reproject(
                    source=rasterio.band(rast, i),
                    destination=rasterio.band(dst, i),
                    rast_transform=rast.transform,
                    rast_crs=rast.crs,
                    dst_transform=transform,
                    dst_crs=epsg,
                    resampling=Resampling.nearest)
        
        # Print statement
        print(f"Reprojection Complete for {filepath}")
        
        return filepath

# Define function to check epsg with reference epsg
def epsgcheck(ref_gdf, pathlist):
    
    # Record start time    
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    # Extract first epsg (reference)
    epsg1 = ref_gdf.crs.to_epsg()
    
    # Create string format for reference epsg
    epsg_string = f"EPSG:{epsg1}"
    
    # Create empty list to hold new filepaths
    filepaths = []
    
    # Iterate over each year
    for i, path in enumerate(pathlist):
            
        # Read path
        with rasterio.open(path) as rast:
            
            # Extract raster data
            data = rast.read()
            
            # Extract raster profile
            profile = rast.profile
            
            # Extract epsg (input)
            epsg2 = rast.meta['crs'].to_epsg()
        
        # If epsgs match
        if epsg1 == epsg2:
            
            # Print statement
            print(f"Landsat image {i} maches reference EPSG: {epsg1}")
            
            # Define new filename
            filepath = path.replace('LC08', 'pp')
            
            # (Re)-write with new filename for consistency
            with rasterio.open(filepath, "w", **profile) as dest:
                dest.write(data)
                
            # Add filepath to list
            filepaths.append(filepath)
        
        # If epsgs don't match
        else:
            
            # Print statements
            print(f"Different EPSG codes: Reference has {epsg1}, Landsat image {i} has {epsg2}")
            
            # Reproject raster
            reproj_path = reproject_raster(path, epsg_string)
            
            # Add filepath to list
            filepaths.append(reproj_path)
            
    # Record end time
    end_time = time.time()
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    # Print the total execution time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    return filepaths

# Reproject january images (if necessary)
jan_reproj_paths = epsgcheck(villages, jan_filepaths_c)

# Reproject february images (if necessary)
feb_reproj_paths = epsgcheck(villages, feb_filepaths_c)

# Reproject december images (if necessary)
dec_reproj_paths = epsgcheck(villages, dec_filepaths_c)



############################################################################


# CREATE MONTHLY COMPOSITES


############################################################################
# Define function to group files into years for compositing
def comp_groups(filepath_list):
    
    # Create empty dictionary with keys for each year
    file_dict = {year: [] for year in years}

    # Iterate over each path
    for path in filepath_list:
        
        # Extract year for image
        year = int(path.split('_')[3][:4])
        
        # Add file to year groupp
        file_dict[year].append(path)
        
    return file_dict

def composite(path_dict, out_dir):
    
    # Record start time    
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # Create empty list to store output files
    outfilepaths = []
    
    # Iterate over each year
    for year, group in path_dict.items():
        
        # Ensure pairs is a list of lists
        if isinstance(group[0], str):
            
            # Put items in parent list
            group = [group]
        
        # Iterate over each pair
        for img in group:
            
            # Open both files
            with rasterio.open(img[0]) as src1, rasterio.open(pair[1]) as src2:
                
                # Union of the two images with priority to image 1
                comp_arr, comp_trans = merge([src1, src2], method="first")

                # Copy profile
                out_profile = src1.profile.copy()
                
                # Update profile to new raster shape
                out_profile.update({
                    "height": comp_arr.shape[1],
                    "width": comp_arr.shape[2],
                    "transform": comp_trans
                })
            
            # Extract month from filepath
            mondat1 = pair[0].split('\\')[-1].split('_')[1][4:8]
            mondat2 = pair[1].split('\\')[-1].split('_')[1][4:8]
            
            # If both images are the same date
            if mondat1 == mondat2:
                
                # Define filename
                filename = f"S2_{year}_{mondat1}.tif"
                
            # If images have different acquisition dates    
            else:
                # Define filename
                filename = f"S2_{year}_{mondat1}_{mondat2}_comp.tif"
        
            # Define filepath
            filepath = os.path.join(out_dir, filename)
            
            # Save the composite as a new .tif file
            with rasterio.open(filepath, 'w', **out_profile) as dst:
                dst.write(comp_arr)
            
            # Add filepath to list
            outfilepaths.append(filepath)
                
            # Print statement
            print(f"Created Sentinel-2 Composite: {filepath}")
    
    # Record end time
    end_time = time.time()
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    # Print the total execution time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
                
    return outfilepaths

# Group files for january composites
jan_groups = comp_groups(jan_reproj_paths)

# Group files for february composites
feb_groups = comp_groups(feb_reproj_paths)

# Group files for december composites
dec_groups = comp_groups(dec_reproj_paths)


tes tes test

file_dict = {year: [] for year in years}

# Iterate over each path
for path in pathlist_jan:
    
    # Extract year for image
    year = int(path.split('_')[3][:4])
    
    # Add file to year groupp
    file_dict[year].append(path)





























