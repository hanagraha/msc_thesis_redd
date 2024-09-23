# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:58:43 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import pandas as pd
import numpy as np



############################################################################


# SET UP DIRECTORY


############################################################################
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())

# Get file paths
gfc_folder = "data/hansen_raw"
tmf_folder = "data/jrc_raw"

# Get list of all tif files in the folder
gfc_files = [f"{gfc_folder}/{file}" for file in os.listdir(gfc_folder) if file.endswith('.tif')]
tmf_files = [f"{tmf_folder}/{file}" for file in os.listdir(tmf_folder) if file.endswith('.tif')]

# Print the file paths to check
print(gfc_files)
print(tmf_files)



############################################################################


# IMPORT DEFORESTATION DATASETS (LOCAL DRIVE)


############################################################################

### READ DATA
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")
villages = gpd.read_file("data/village polygons/VillagePolygons.geojson")

### CHECK PROJECTIONS 
# Get epsg codes
villages_epsg = villages.crs.to_epsg()
grnp_epsg = grnp.crs.to_epsg()

# Check if epsgs match:
if villages_epsg == grnp_epsg:
    print(f"Both GeoDataFrames have the same EPSG: {villages_epsg}")
else:
    print(f"Different EPSG codes: Villages has {villages_epsg}, GRNP has {grnp_epsg}")
    # Reproject if necessary
    grnp = grnp.to_crs(epsg=villages_epsg)
    print(f"GRNP has been reprojected to EPSG: {villages_epsg}")

# Create string of EPSG code for raster reprojection
epsg_string = f"EPSG:{villages_epsg}"

### CREATE AOI
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()
aoi_geom = aoi.geometry



############################################################################


# REPROJECT RASTERS


############################################################################

### DEFINE FUNCTION TO REPROJECT RASTERS
def reproject_raster(raster_path, epsg, output_folder):
    # open raster
    with rasterio.open(raster_path) as rast:
        transform, width, height = calculate_default_transform(
            rast.crs, epsg, rast.width, rast.height, *rast.bounds)
        
        kwargs = rast.meta.copy()
        kwargs.update({'crs': epsg,
                       'transform': transform,
                       'width': width,
                       'height': height})
        
        output_path = raster_path.replace(".tif", "_reprojected.tif")
        folder_name = os.path.basename(os.path.dirname(output_path))
        output_path = output_path.replace(folder_name, output_folder)

        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, rast.count + 1):
                reproject(
                    source=rasterio.band(rast, i),
                    destination=rasterio.band(dst, i),
                    rast_transform=rast.transform,
                    rast_crs=rast.crs,
                    dst_transform=transform,
                    dst_crs=epsg,
                    resampling=Resampling.nearest)
        
        print(f"Reprojection Complete for {output_path}")
        
        return output_path
    

gfc_processed_folder = "hansen_preprocessed"
tmf_processed_folder = "jrc_preprocessed"


# Reproject GFC images
gfc_reproj_files = []

for tif in gfc_files:
    tif_reproj_file = reproject_raster(tif, epsg_string, gfc_processed_folder)
    gfc_reproj_files.append(tif_reproj_file)
    
    
# Reproject TMF images
tmf_reproj_files = []

for tif in tmf_files:
    tif_reproj_file = reproject_raster(tif, epsg_string, tmf_processed_folder)
    tmf_reproj_files.append(tif_reproj_file)
    
    
    
############################################################################


# CLIP RASTERS


############################################################################

### DEFINE FUNCTION TO CLIP RASTERS
def clip_raster(raster_path, aoi_geom, nodata_val):
    
    # open raster
    with rasterio.open(raster_path) as rast:
        
        # Clip raster
        raster_clip, out_transform = mask(rast, aoi_geom, crop=True, nodata=nodata_val)
        
        out_meta = rast.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'count': 1,
            'height': raster_clip.shape[1],
            'width': raster_clip.shape[2],
            'transform': out_transform,
            'nodata': nodata_val})
        
    raster_clip = raster_clip.astype('float32')
    # raster_clip[raster_clip == nodata_val] = np.nan
    
    outpath = raster_path.replace("_reprojected.tif", "_clipped.tif")
    
    # if os.path.exists(outpath): 
    #     os.remove(outpath)
    
    with rasterio.open(outpath, 'w', **out_meta) as dest:
        dest.write(raster_clip)
    
    print(f"Clipping Complete for {outpath}")
    
    return raster_clip, out_meta  
    

# Clip GFC rasters
for file in gfc_reproj_files:
    clip_raster(file, aoi_geom, 255)
    
# Clip TMF rasters
for file in tmf_reproj_files:
    clip_raster(file, aoi_geom, 255)
    
    
    
############################################################################


# RECLASSIFICATION


############################################################################
"""
GFC lossyear will be reclassified to match the year format of TMF degradation year
"""

# Re-do the clip function to get metadata
gfc_lossyear, gfc_lossyear_meta = clip_raster(gfc_reproj_files[0], aoi_geom, 255)

# Get path to save the reclassified file
gfc_lossyear_path = gfc_reproj_files[0].replace("_reprojected.tif", "_reclassified.tif")

# Add 2000 to non-NA years
gfc_lossyear_new = np.where(gfc_lossyear != 255, gfc_lossyear + 2000, gfc_lossyear)

# Make sure data type can hold 2000 values
gfc_lossyear_meta.update(dtype='int16')

# Write file to drive
with rasterio.open(gfc_lossyear_path, 'w', **gfc_lossyear_meta) as dest:
    dest.write(gfc_lossyear_new)
print(f"Reclassification Complete for {gfc_lossyear_path}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    