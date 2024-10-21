# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:51:03 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.merge import merge
import shutil



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
# Define function to extract tif filepaths in subfolder
def folder_tifs(years):
    
    # Create empty dictionary to hold files for each year
    yearly_lists = {}
    
    # Iterate over each year
    for year in years:
        
        # Define folder path for that year
        folder_path = os.path.join('data', 'planet_raw', str(year))
    
        # Extract filepaths ending in .tif
        tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        
        # Put filepaths in list
        file_paths = [os.path.join(folder_path, f) for f in tif_files]
        
        # Add files to dictionary 
        yearly_lists[year] = file_paths
    
    return yearly_lists

# Read village polygons
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp polygons
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']])
aoi_geom = aoi.geometry

# Create string of EPSG code for raster reprojection
villages_epsg = villages.crs.to_epsg()
epsg_string = f"EPSG:{villages_epsg}"

# Define RapidEye filepaths (2013-2015)
re_years = range(2013, 2016)
re_2013_2015_pathdict = folder_tifs(re_years)
re_pathlist = [file_path for paths in re_2013_2015_pathdict.values() for \
               file_path in paths]

# Define year range for planetscope
ps_years = range(2017, 2024)

# Define PlanetScope filepaths (2018-2023)
ps_2017_2023_pathdict = folder_tifs(ps_years)



############################################################################


# CONNECT PLANETSCOPE TILES


############################################################################
"""
start time: 5:19
end time: 5:35?

approx 15min for planetscope
"""

# Define function to create image from tiles
def connect_tiles(yearrange, tile_dict, file_prefix, out_dir):
    
    # Create empty list to hold filepaths of connected images
    connected_paths = []
    
    # Iterate over keys in dictionary
    for year in yearrange:
        
        # Access pathlist for that year
        pathlist = tile_dict[year]
        
        # List to hold opened datasets
        mosaic_files = []
    
        # Open each file and append it to the list
        for tif in pathlist:
            src = rasterio.open(tif)
            mosaic_files.append(src)
    
        # Merge the datasets
        mosaic, out_trans = merge(mosaic_files)
    
        # Update the metadata for the output file
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
    
        # Define file name and path
        outfilename = f"{file_prefix}_{year}.tif"
        output_filepath = os.path.join(out_dir, outfilename)
                
        # Write the merged raster to the output file
        with rasterio.open(output_filepath, "w", **out_meta) as dest:
            dest.write(mosaic)
    
        # Close all source files
        for src in mosaic_files:
            src.close()
            
        # Add merged image path to list
        connected_paths.append(output_filepath)
        
        # Print statement
        print(f"Merging Complete for {outfilename}")
        
    return connected_paths

# Merge planetscope tiles to images
ps_pathlist = connect_tiles(ps_years, ps_2017_2023_pathdict, "Composite", 
                            temp_folder)



############################################################################


# REPROJECT PLANET RASTERS


############################################################################
"""
start time: 5:36 (one image took 3min)
end time: 5:56

approx 20min for planetscope
"""

# Define function to reproject raster
def reproject_raster(raster_pathlist, epsg, out_dir, yearrange, nodata_value):
    
    # Create empty list to store new raster paths
    reprojected_rasters = []
    
    # Iterate over each raster
    for path, year in zip(raster_pathlist, yearrange):
        
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
                           'nodata': nodata_value})
            
            # Define file name and path
            filename = f"HighRes_{year}.tif"
            output_filepath = os.path.join(out_dir, filename)
            
            # Write reprojected file to drive
            with rasterio.open(output_filepath, 'w', **kwargs) as dst:
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
            print(f"Reprojection Complete for {output_filepath}")
            
            # Add reprojected filepath to list
            reprojected_rasters.append(output_filepath)
            
    return reprojected_rasters

# Reproject rapideye imagery
re_proc_files = reproject_raster(re_pathlist, epsg_string, out_dir, re_years, 
                                 nodata_val)

# Reproject planetscope imagery
ps_proc_files = reproject_raster(ps_pathlist, epsg_string, out_dir, ps_years, 
                                 nodata_val)



############################################################################


# CLIP PLANET RASTERS


############################################################################
"""
start time: 5:58
end time: 6:00
for planetscope
"""

# Define function to clip rasters
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
                'dtype': 'uint8',
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
clip_raster(re_proc_files, aoi_geom, nodata_val)

# Clip planetscope imagery
clip_raster(ps_proc_files, aoi_geom, nodata_val)



############################################################################


# DELETE TEMPORARY FOLDER


############################################################################
# If the temporary folder still exists, delete it
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
    print(f"{temp_folder} has been deleted.")
else:
    print(f"{temp_folder} does not exist.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    