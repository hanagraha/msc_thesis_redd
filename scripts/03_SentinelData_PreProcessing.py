# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:35:29 2024

@author: hanna

Estimated runtime: ~3min
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

# Set output directory
out_dir = os.path.join('data', 'validation')

# Define temporary folder path
temp_folder = os.path.join('data', "planet_intermediate")

# Create temporary folder, if it doesn't already exist
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    print(f"{temp_folder} created.")
else:
    print(f"{temp_folder}' already exists.")
    
    
    
############################################################################


# IMPORT AND READ DATA


############################################################################
# Define paths for 2016 sentinel imagery (image 1)
s2_2016_r = "data/planet_raw/2016/Sentinel2/03_15/GRANULE/L2A_T29NKJ_A003809_20160315T111236/IMG_DATA/R10m/T29NKJ_20160315T110042_B04_10m.jp2"
s2_2016_g = "data/planet_raw/2016/Sentinel2/03_15/GRANULE/L2A_T29NKJ_A003809_20160315T111236/IMG_DATA/R10m/T29NKJ_20160315T110042_B03_10m.jp2"
s2_2016_b = "data/planet_raw/2016/Sentinel2/03_15/GRANULE/L2A_T29NKJ_A003809_20160315T111236/IMG_DATA/R10m/T29NKJ_20160315T110042_B02_10m.jp2"

# Define paths for 2016 sentinel imagery (image 2)
s2_2016_r1 = "data/planet_raw/2016/Sentinel2/04_20/GRANULE/L2A_T29NLJ_A003237_20160204T111022/IMG_DATA/R10m/T29NLJ_20160204T110242_B04_10m.jp2"
s2_2016_g1 = "data/planet_raw/2016/Sentinel2/04_20/GRANULE/L2A_T29NLJ_A003237_20160204T111022/IMG_DATA/R10m/T29NLJ_20160204T110242_B03_10m.jp2"
s2_2016_b1 = "data/planet_raw/2016/Sentinel2/04_20/GRANULE/L2A_T29NLJ_A003237_20160204T111022/IMG_DATA/R10m/T29NLJ_20160204T110242_B02_10m.jp2"

# Read village polygons
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp polygons
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']])
aoi_geom = aoi.geometry

# Create list for first sentinel images
s2_img1 = [s2_2016_r, s2_2016_g, s2_2016_b]

# Create list for second sentinel image
s2_img2 = [s2_2016_r1, s2_2016_g1, s2_2016_b1]



############################################################################


# COMBINE BANDS INTO ONE IMAGE


############################################################################
# Create function to read list of paths into 3d array
def read_rgb(pathlist, out_dir, filename):
    
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
    
    # Stack single layers to 3d array shape (rgb)
    rgb = np.stack((arrlist[0], arrlist[1], arrlist[2]), axis=0)

    # Define output filepath
    outfilepath = os.path.join(out_dir, filename)
    
    # Update profile
    profile['count'] = 3
    
    # Write rgb array to drive
    with rasterio.open(outfilepath, 'w', **profile) as dst:
        dst.write(rgb)
        
    # Print statement
    print(f"Created rgb image {outfilepath}")
        
    return outfilepath

# Read sentinel image 1
img1_rgb = read_rgb(s2_img1, temp_folder, "HighRes_2016_Sentinel_1.tif")

# Read sentinel image 2
img2_rgb = read_rgb(s2_img2, temp_folder, "HighRes_2016_Sentinel_2.tif")



############################################################################


# CHECK REPROJECTIONS


############################################################################   
# Define function to check epsg with reference epsg
def epsgcheck(ref_gdf, path):
    
    # Extract first epsg
    epsg1 = ref_gdf.crs.to_epsg()
    
    # Extract epsg for input paths
    with rasterio.open(path) as rast:
        epsg2 = rast.meta['crs'].to_epsg()
        
    # If epsgs match
    if epsg1 == epsg2:
        print(f"Input maches the same EPSG: {epsg1}")
    
    # If epsgs don't match
    else:
        print(f"Different EPSG codes: Reference has {epsg1}, Input has {epsg2}")

# Check projection of first sentinel image
epsgcheck(villages, img1_rgb)

# Check projection of second sentinel image
epsgcheck(villages, img2_rgb)



############################################################################


# COMPOSITE IMAGES TO FILL WHOLE AOI


############################################################################
# Define function to combine two arrays (only partially overlapping)
def img_union(path1, path2, out_dir, filename):
    
    # Open both files together
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        
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

    # Define output filepath
    outfilepath = os.path.join(out_dir, filename)

    # Save the composite as a new .tif file
    with rasterio.open(outfilepath, 'w', **out_profile) as dst:
        dst.write(comp_arr)
        
    # Print statement
    print(f"Combined images to {outfilepath}")
        
    return outfilepath

# Combine sentinel images
s2_comb = img_union(img1_rgb, img2_rgb, out_dir, "HighRes_2016_S2.tif")

    

############################################################################


# CLIP RASTERS 


############################################################################
# Define function to clip rasters to aoi
def clip_raster(raster_pathlist, aoi_geom, nodata_value):
    
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

# Clip rapideye imagery
clip_raster([s2_comb], aoi_geom, nodata_val)



############################################################################


# DELETE TEMPORARY FOLDER


############################################################################
# If the temporary folder still exists, delete it
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
    print(f"{temp_folder} has been deleted.")
else:
    print(f"{temp_folder} does not exist.")
    
    
    
    
    