# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:23:05 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.mask import mask
from matplotlib.patches import Rectangle
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

# Set output directory
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Set year range
years = range(2013, 2024)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define file paths for validation rasters
planet_paths = [f"data/validation/HighRes_{year}.tif" for year in years]
planet_paths = ['data/validation/HighRes_2013.tif',
 'data/validation/HighRes_2014.tif',
 'data/validation/HighRes_2015.tif',
 'data/validation/HighRes_2016.tif',
 'data/validation/HighRes_2016_S2.tif',
 'data/validation/HighRes_2017.tif',
 'data/validation/HighRes_2018.tif',
 'data/validation/HighRes_2019.tif',
 'data/validation/HighRes_2020.tif',
 'data/validation/HighRes_2021.tif',
 'data/validation/HighRes_2022.tif',
 'data/validation/HighRes_2023.tif']

# Read validation points
valpoints = gpd.read_file("data/validation/validation_points_geometry.shp")



############################################################################


# FORMAT ARRAYS TO RGB STACKS


############################################################################
# Define function to create frames for each point
def point_frame(point_gdf, framesize):
    
    # Create empty list to store frames
    bbox_list = []
    
    # Iterate over each point
    for i in range(len(valpoints)):
        
        # Extract point geometry
        geom = valpoints.geometry[i]
        
        # Extract pixel bounds
        minx, miny, maxx, maxy = geom.bounds
        
        # Create a rectangular bounding box
        bbox = box(minx - framesize / 2, miny - framesize / 2, 
                    maxx + framesize / 2, maxy + framesize / 2)
        
        bbox_list.append(bbox)
    
    # Copy validation points gdf
    bbox_gdf = point_gdf.copy()
    
    # Add bbox infomration to gdf
    bbox_gdf['frame'] = bbox_list
    
    # Make sure crs stays the same
    bbox_gdf.crs = point_gdf.crs
    
    return bbox_gdf

# Define function to clip validation data to each frame
def clip_raster(raster_pathlist, geom, nodata_value):
    
    # Create empty lists to hold clipped arrays and metadata
    clipped_arrs = []
    metadata = []
    
    # Iterate over rasters
    for file in raster_pathlist:

        # Read raster
        with rasterio.open(file) as rast:
            
            # Only process the first three bands (RGB)
            indices = [1,2,3]
            
            # Mask pixels outside aoi with NoData values
            raster_clip, out_transform = mask(rast, geom, crop = True, 
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
        
        # Add clipped array to list
        clipped_arrs.append(raster_clip)
        
        # Add metadata to list
        metadata.append(out_meta)
        
    return clipped_arrs, metadata

# Define function to plot frames based on point
def pnt_plot(raster_pathlist, val_frames, pntindex, pxsize_m):
    
    # Extract relevant point
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    clipped_arrs, metas = clip_raster(raster_pathlist, frame, nodata_val)
    
    # Transpose clipped array to match format for imshow
    clipped_rgb_arrs = [arr.transpose(1, 2, 0) for arr in clipped_arrs]
    
    # Normalize to range 0-255 just in case
    clipped_rgb_arrs = [(arr / arr.max() * 255).astype(np.uint8) for arr 
                        in clipped_rgb_arrs]
    
    # Define labels for subplots
    # labels = ["2013 RapidEye", "2014 RapidEye", "2015 RapidEye", 
    #           "2016 RapidEye + PlanetScope", "2016 Sentinel 2", 
    #           "2017 PlanetScope", "2018 PlanetScope", "2019 PlanetScope", 
    #           "2020 PlanetScope", "2021 PlanetScope", "2022 PlanetScope",
    #           "2023 PlanetScope"]
    labels = [2013, 2014, 2015, "2016p", "2016s", 2017, 2018, 2019, 2020, 
              2021, 2022, 2023]
    
    # Initialize figure with 3x4 subplots
    fig, axs = plt.subplots(3, 4, figsize=(12, 9))
    
    # Flatten axes array
    axs = axs.flatten()
    
    # Iterate over axis, arrays, and metadata
    for i, (rgb_array, meta) in enumerate(zip(clipped_rgb_arrs, metas)):
        
        # Display rgb image
        axs[i].imshow(rgb_array)
        
        # Remove axis labels
        axs[i].axis('off')
        
        # Set subplot titles
        axs[i].set_title(f'{labels[i]}')
        
        # Extract specific transform for each year
        transform = meta['transform']
        
        # Calculate pixel size in pixels for each raster
        pxsize_px = pxsize_m / abs(transform.a)
        
        # Convert xy coordinate to image coordinates
        px, py = ~transform * (point.x, point.y)
        
        # Create pixel rectangle
        rect = Rectangle(
            (px - pxsize_px / 2, py - pxsize_px / 2),
            pxsize_px, pxsize_px, linewidth=1, edgecolor='red', facecolor='none'
        )
        
        # Overlay validation pixel area
        axs[i].add_patch(rect)
        
        # Remove empty subplot axes
        for j in range(len(clipped_rgb_arrs), len(axs)):
            axs[j].axis('off')

    # Show plot
    plt.tight_layout()
    plt.show()

# Create frames for each validation point
val_frames = point_frame(valpoints, 500)

# Plot for defined point
pnt_plot(planet_paths, val_frames, 256, 30)


