# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:33:01 2024

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
from collections import Counter



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
jan_temp = os.path.join(temp_folder, "jan_qa")

# Create temporary landsat intermediate folder
folder_create(temp_folder)

# Create temporary january subfolder
folder_create(jan_temp)



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

# Define january filepaths (with qa band)
jan_filepaths = subfolder_paths("Jan_QA")



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
            
            # Set all values with pixel value 0 to np.nodata
            data_int8 = np.where(data[i] == 0, np.nan, data[i])
            
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
jan_nd_paths = landsat_clean(jan_filepaths, jan_temp)



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
        filepath = path.replace('LC08', 'reproj')
        
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
            filepath = path.replace('LC08', 'reproj')
            
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
jan_reproj_paths = epsgcheck(villages, jan_nd_paths)



############################################################################


# CLIP IMAGES TO AOI


############################################################################
# Define function to clip rasters
def clip_raster(raster_pathlist, aoi_geom, nodata_value, out_dir):
    
    # Create empty list to hold clipped raster paths
    clipped_paths = []
    
    # Iterate over rasters
    for file in raster_pathlist:
    
        # Read raster
        with rasterio.open(file) as rast:
            
            # Extract the number of bands
            bands = rast.count
            
            # Only process the first three bands (RGB)
            indices = list(range(1, bands + 1))
            
            # Mask pixels outside aoi with NoData values
            raster_clip, out_transform = mask(rast, aoi_geom, crop = True, 
                                              nodata = nodata_value, 
                                              indexes = indices)
            
            # Copy metadata
            out_meta = rast.meta.copy()
            
            # Update metadata
            out_meta.update({
                'driver': 'GTiff',
                'dtype': 'uint16',
                'count': len(indices),
                'height': raster_clip.shape[1],
                'width': raster_clip.shape[2],
                'transform': out_transform,
                'nodata': nodata_value})
        
        # Define new file name
        outfilepath = file.replace("reproj", "clipped")
        
        # Replace old file with new file (same file name)
        with rasterio.open(outfilepath, 'w', **out_meta) as dest:
            
            # Iterate over each band of interest
            for band_index, band in enumerate(indices, start=1):
                
                # Write each band to file
                dest.write(raster_clip[band_index - 1], band_index)
        
        # Print statement
        print(f"Clipping Complete for {os.path.basename(outfilepath)}")
        
        # Add path to list
        clipped_paths.append(outfilepath)
        
    return clipped_paths

# Clip january composites
jan_clipped_paths = clip_raster(jan_reproj_paths, aoi, nodata_val, jan_temp)



############################################################################


# CREATE PAIRS FOR COMPOSITING


############################################################################
# Define function to group files into unique shapes
def shape_groups(pathlist):
    
    # Create empty dictionary to store filepaths
    shape_dict = {}
    
    # Iterate over each path
    for path in pathlist:
        
        # Read raster
        with rasterio.open(path) as rast:
            
            # Extract shape
            shape = rast.shape
            
        # If the shape doesn't already exist in the dictionary
        if shape not in shape_dict:
            
            # Add shape to dictionary
            shape_dict[shape] = []
            
        # Add path to shape key
        shape_dict[shape].append(path)
        
    # Create list of unique shapes
    unique_shapes = list(shape_dict.keys())
    
    return shape_dict, unique_shapes

# Define function to group files into years for compositing
def year_groups(filepath_list):
    
    # Create empty dictionary with keys for each year
    file_dict = {year: [] for year in years}

    # Iterate over each path
    for path in filepath_list:
        
        # Extract year for image
        year = int(path.split('_')[4][:4])
        
        # Add file to year groupp
        file_dict[year].append(path)
        
    return file_dict
    
# Group january images by shape
jan_shp_groups, jan_shps = shape_groups(jan_clipped_paths)

# Group january images by year
jan_shp1yr_groups = year_groups(jan_shp_groups[jan_shps[0]])

# Group january images by year
jan_shp2yr_groups = year_groups(jan_shp_groups[jan_shps[1]])



############################################################################


# COMPOSITE IMAGES


############################################################################
img1_path = jan_shp2yr_groups[2023][0]
img2_path = jan_shp2yr_groups[2023][1]
out = os.path.join(jan_temp, "composite_shp2_2023.tif")

with rasterio.open(img1_path) as src_a, rasterio.open(img2_path) as src_b:
    
    # Read all bands
    img1 = src_a.read()
    img2 = src_b.read()
    
    # Extract profile data
    profile = src_a.profile
    
    # Extract pixel quality data
    qa1 = img1[7]
    qa2 = img2[7]
    
    # Prepare output array
    combined = np.zeros_like(img1, dtype=img1.dtype)
    
    # Define mask where both images have data
    valid_mask = (qa1 != nodata_val) & (qa2 != nodata_val)
    
    # Apply combination logic across all bands
    for band in range(img1.shape[0]):
        
        # Extract band data
        band1 = img1[band]
        band2 = img2[band]
        
        # If both are NoData, keep NoData
        combined[band][(qa1 == nodata_val) & (qa2 == nodata_val)] = nodata_val
    
        # If only image 1 has data keep image 1's pixel values
        combined[band][(qa1 != nodata_val) & (qa2 == nodata_val)] = band1[
            (qa1 != nodata_val) & (qa2 == nodata_val)]
    
        # If only image 2 has data keep image 2's pixel values
        combined[band][(qa1 == nodata_val) & (qa2 != nodata_val)] = band2[
            (qa1 == nodata_val) & (qa2 != nodata_val)]
    
        # If both have data, take the minimum value for this band
        combined[band][valid_mask] = np.where(qa1[valid_mask] < qa2[valid_mask],
                                              band1[valid_mask], band2[valid_mask])

with rasterio.open(out, 'w', **profile) as dst:
    dst.write(combined)






















