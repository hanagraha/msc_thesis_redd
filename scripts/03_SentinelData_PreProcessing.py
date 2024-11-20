# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:35:29 2024

@author: hanna

Estimated runtime: ~42min
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

# Set year range (specific for sentinel2!)
years = range(2015, 2025)

# Set output directory
out_dir = os.path.join('data', 'validation')

# Define temporary folder path
temp_folder = os.path.join('data', "sentinel_intermediate")

# Create temporary folder, if it doesn't already exist
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    print(f"{temp_folder} created.")
else:
    print(f"{temp_folder}' already exists.")
    
    
    
############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to extract tif filepaths in subfolder
def folder_tifs(years):
    
    # Create empty dictionary to hold files for each year
    r_paths = {}
    g_paths = {}
    b_paths = {}
    
    # Iterate over each year
    for year in years:
        
        # Create empty listst to hold files for each year
        r_paths_year = []
        g_paths_year = []
        b_paths_year = []
        
        # Define folder path for that year
        folder_path = os.path.join('data', 'sentinel_raw', str(year))
        
        # Define subfolder paths
        subfolders = os.listdir(folder_path)
        
        # Iterate over each subfolder
        for subfolder in subfolders:
            
            # Define subfolder path
            subfolder_path = os.path.join(folder_path, subfolder)
            
            # Define granule folder
            granule_path = os.path.join(subfolder_path, "GRANULE")
            
            # Define subfolder
            sub_granule = os.listdir(granule_path)[0]
            
            # Define 10m resolution image folder
            img_folder = os.path.join(granule_path, sub_granule, "IMG_DATA", "R10m")
            
            # Extract rgb files
            r = [f for f in os.listdir(img_folder) if f.endswith('B04_10m.jp2')][0]
            g = [f for f in os.listdir(img_folder) if f.endswith('B03_10m.jp2')][0]
            b = [f for f in os.listdir(img_folder) if f.endswith('B02_10m.jp2')][0]
            
            # Create paths for rgb files
            r_path = os.path.join(img_folder, r)
            g_path = os.path.join(img_folder, g)
            b_path = os.path.join(img_folder, b)
            
            # Add paths to year list
            r_paths_year.append(r_path)
            g_paths_year.append(g_path)
            b_paths_year.append(b_path)
        
        # Add year paths to dictionary
        r_paths[year] = r_paths_year
        g_paths[year] = g_paths_year
        b_paths[year] = b_paths_year
    
    return r_paths, g_paths, b_paths

# Define sentinel2 filepaths
s2_r, s2_g, s2_b = folder_tifs(years)

# Read village polygons
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp polygons
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']]).geometry



############################################################################


# COMBINE BANDS INTO ONE IMAGE


############################################################################
"""
Total execution time: 1289.19 seconds, 303.69 seconds
"""

# Create function to read list of paths into 3d array
def read_rgb(r_dict, g_dict, b_dict, out_dir):
    
    # Record start time    
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    # Create empty dictionary to store all output filepaths
    rgb_paths = {}
    
    # Iterate over each year
    for year in r_dict.keys():
        
        # Get file paths
        r_paths = r_dict[year]
        g_paths = g_dict[year]
        b_paths = b_dict[year]
        
        # Create empty list to hold output files for that year
        rgb_paths_year = []
        
        # Iterate over each set of files
        for r_path, g_path, b_path in zip(r_paths, g_paths, b_paths):
            
            # Read red file
            with rasterio.open(r_path) as rast:
                
                # Extract data
                r = rast.read(1)
                
                # Extract profile
                profile = rast.profile
                
            # Read green file
            with rasterio.open(g_path) as rast:
                
                # Extract data
                g = rast.read(1)
                
            # Read blue file
            with rasterio.open(b_path) as rast:
                
                # Extract data
                b = rast.read(1)
            
            # Stack into 3d array
            rgb = np.stack((r, g, b), axis = 0)
            
            # Update profile
            profile['count'] = 3
            profile['driver'] = 'GTiff'
            
            # Extract filename segment
            seg = r_path.split('\\')[8].split('_B')[0]
            
            # Define output filename
            filename = f"{seg}.tif"
            
            # Define output filepath
            outfilepath = os.path.join(out_dir, filename)
            
            # Write rgb stack to file
            with rasterio.open(outfilepath, 'w', **profile) as dst:
                dst.write(rgb)
            
            # Print statement
            print(f"Created rgb image {outfilepath}")
            
            # Add outfilepath to list
            rgb_paths_year.append(outfilepath)
        
        # Add that year's file list to dictionary
        rgb_paths[year] = rgb_paths_year
    
    # Record end time
    end_time = time.time()
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    # Print the total execution time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
        
    return rgb_paths

# Create rgb images from sentinel data
s2_rgb = read_rgb(s2_r, s2_g, s2_b, temp_folder)



############################################################################


# CHECK REPROJECTIONS


############################################################################ 
"""
Total execution time: 0.23 seconds, 0.39 seconds
"""  
# Define function to check epsg with reference epsg
def epsgcheck(ref_gdf, path_dict):
    
    # Record start time    
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    # Extract first epsg
    epsg1 = ref_gdf.crs.to_epsg()
    
    # Iterate over each year
    for year in path_dict.keys():
        
        # Define paths in year
        paths = path_dict[year]
        
        # Iterate over paths
        for path in paths:
            
            # Extract epsg
            with rasterio.open(path) as rast:
                epsg2 = rast.meta['crs'].to_epsg()
            
            # If epsgs match
            if epsg1 == epsg2:
                print(f"Image from {year} maches reference EPSG: {epsg1}")
            
            # If epsgs don't match
            else:
                print(f"Different EPSG codes: Reference has {epsg1}, Image from {year} has {epsg2}")
    
    # Record end time
    end_time = time.time()
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    # Print the total execution time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Check projection sentinel images
epsgcheck(villages, s2_rgb)



############################################################################


# CREATE IMAGE PAIRS


############################################################################
"""
Total execution time: 0.00 seconds, 0.0 seconds
"""
# Define function to create lists of pairs
def img_pair(path_dict):

    # Record start time    
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    # Create empty list to hold pairs
    img_pairs = {}
    
    # Iterate over each year
    for year in path_dict.keys():
        
        # Define paths in year
        paths = path_dict[year]
        
        # If there are only two paths
        if len(paths) == 2:
            
            # Add paths to dictionary
            img_pairs[year] = paths
        
        # If there are more than two paths
        else:
            
            # Initialize lists for Jan/Feb/March and Nov/Dec
            jan_mar = []
            nov_dec = []
            
            # Iterate over each path
            for path in paths:
                
                # Extract month from filepath
                month = int(path.split('\\')[-1].split('_')[1][4:6])
            
                # If month is either jan, feb, or mar
                if month in [1, 2, 3]:
                    jan_mar.append(path)
                    
                # If month is either nov, dec
                elif month in [11, 12]:
                    nov_dec.append(path)
                    
                # Add pairs to dictionary
                img_pairs[year] = [jan_mar, nov_dec]
    
    # Record end time
    end_time = time.time()
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    # Print the total execution time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    return img_pairs
            
# Pair sentinel images by acquisition month
s2_pairs = img_pair(s2_rgb)



############################################################################


# COMPOSITE IMAGE PAIRS


############################################################################
"""
Total execution time: 1347.24 seconds, 162.17 seconds
"""
# Define function to combine two arrays (only partially overlapping)
def img_union(path_dict, out_dir):
    
    # Record start time    
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # Create empty list to store output files
    outfilepaths = []
    
    # Iterate over each year
    for year, pairs in path_dict.items():
        
        # Ensure pairs is a list of lists
        if isinstance(pairs[0], str):
            
            # Put items in parent list
            pairs = [pairs]
        
        # Iterate over each pair
        for pair in pairs:
            
            # Open both files
            with rasterio.open(pair[0]) as src1, rasterio.open(pair[1]) as src2:
                
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

# Create sentinel composites
s2_comps = img_union(s2_pairs, out_dir)

    

############################################################################


# CLIP RASTERS 


############################################################################
"""
Total execution time: 140.76 seconds, 62.19 seconds
"""
# Define function to clip rasters to aoi
def clip_raster(raster_pathlist, aoi_geom, nodata_value):
    
    # Record start time    
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    
    # Iterate over rasters
    for file in raster_pathlist:
    
        # Read raster
        with rasterio.open(file) as rast:
            
            # Only process the first three bands (RGB)
            indices = [1,2,3]
            
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
        
        # Replace old file with new file (same file name)
        with rasterio.open(file, 'w', **out_meta) as dest:
            
            # Iterate over each band of interest
            for band_index, band in enumerate(indices, start=1):
                
                # Write each band to file
                dest.write(raster_clip[band_index - 1], band_index)
        
        # Print statement
        print(f"Clipping Complete for {file}")
    
    # Record end time
    end_time = time.time()
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    # Print the total execution time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Clip rapideye imagery
clip_raster(s2_comps, aoi, nodata_val)



############################################################################


# DELETE TEMPORARY FOLDER


############################################################################
# If the temporary folder still exists, delete it
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
    print(f"{temp_folder} has been deleted.")
else:
    print(f"{temp_folder} does not exist.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    