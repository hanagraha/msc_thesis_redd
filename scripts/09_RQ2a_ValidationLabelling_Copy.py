# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:41 2024

@author: hanna
"""

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

# Set validation directory
val_dir = os.path.join('data', 'validation')

# Set output directory
out_dir = os.path.join('data', 'intermediate')

# Set year range
years = range(2013, 2025)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to define filepaths in subfolder
def pathlist(folder, prefix):
    
    # Create empty list to hold filepaths
    outfiles = []
    
    # List all files in folder
    files = [f for f in os.listdir(folder) if f.endswith('.tif')]
    
    # Iterate over each file
    for file in files:
        
        # Split filepath into parts
        parts = file.split('\\')[-1].split('_')
        
        # If filepath matches prefix
        if parts[0] == prefix:
            
            # Create filepath
            filepath = os.path.join(folder, file)
            
            # Add filepath to list
            outfiles.append(filepath)
    
    return outfiles

# Define planet filepaths
planet_paths = [f"data/validation/HighRes_{year}.tif" for year in years]
# planet_paths = pathlist(val_dir, "HighRes")

# Define sentinel filepaths
sentinel_paths = pathlist(val_dir, "S2")

# Read validation points
valpoints = gpd.read_file("data/validation/validation_points_geometry.shp")
st7_valpoints = gpd.read_file("data/validation/validation_points_geometry_minstrata7.shp")



############################################################################


# FORMAT ARRAYS TO RGB STACKS


############################################################################
# Define function to create frames for each point
def point_frame(point_gdf, framesize):
    
    # Create empty list to store frames
    bbox_list = []
    
    # Iterate over each point
    for i in range(len(point_gdf)):
        
        # Extract point geometry
        geom = point_gdf.geometry[i]
        
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

# Create validation points for regular dataset
val_frames = point_frame(valpoints, 500)

# Create validation points for strata 7+ dataset
st7_val_frames = point_frame(st7_valpoints, 500)



############################################################################


# PLANET PLOTTING


############################################################################

# Define function to plot frames based on point
def planet_plot(raster_pathlist, val_frames, pntindex, pxsize_m):
    
    # Extract relevant point
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    clipped_arrs, metas = clip_raster(raster_pathlist, frame, nodata_val)
    
    # Transpose clipped array to match format for imshow
    clipped_rgb_arrs = [arr.transpose(1, 2, 0) for arr in clipped_arrs]
    
    # Define labels for subplots
    labels = list(years)
    
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

# Plot planet data for defined point
planet_plot(planet_paths, val_frames, 100, 30)



############################################################################


# SENTINEL PLOTTING


############################################################################
# Define function to plot frames based on point
def sentinel_plot(raster_pathlist, val_frames, pntindex, pxsize_m):
    
    # Extract relevant point
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    clipped_arrs, metas = clip_raster(raster_pathlist, frame, nodata_val)
    
    # Transpose clipped array to match format for imshow
    clipped_rgb_arrs = [arr.transpose(1, 2, 0) for arr in clipped_arrs]
    
    # Create empty list to hold normalized arrays
    norm_arrs = []
    
    # Iterate over each clipped array
    for arr in clipped_rgb_arrs:
        
        # Normalize data (0-1)
        norm_arr = (arr - arr.min()) / (arr.max() - arr.min()) 
        
        # Add normalized data to list
        norm_arrs.append(norm_arr)
    
    # Create empty list to hold labels
    labels = []
    
    # Iterate over each path
    for path in raster_pathlist:
        
        # Extract year
        year = path.split("\\")[-1].split("_")[1]
        
        # Extract month
        mondat = path.split("\\")[-1].split("_")[-1].split(".tif")[0]
        
        # Create label
        label = f"{year}-{mondat[0:2]}-{mondat[2:4]}"
        
        # Add label to list
        labels.append(label)

    # Initialize figure with 3x4 subplots
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    
    # Flatten axes array
    axs = axs.flatten()
    
    # Iterate over axis, arrays, and metadata
    for i, (rgb_array, meta) in enumerate(zip(norm_arrs, metas)):
        
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
        for j in range(len(norm_arrs), len(axs)):
            axs[j].axis('off')

    # Show plot
    plt.tight_layout()
    plt.show()

# Create frames for each validation point
val_frames = point_frame(valpoints, 500)

# Plot sentinel data for defined point
sentinel_plot(sentinel_paths, val_frames, 100, 30)



############################################################################


# SENTINEL PLOTTING FOR STRATA 7+


############################################################################
# Define years for only strata 7+
st7_years = range(2017, 2025)

# Define rasters for only strata 7+
st7_planet_paths = planet_paths[4:]

# Define function to plot frames based on point
def planet_plot(raster_pathlist, val_frames, pntindex, pxsize_m):
    
    # Extract relevant point
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    clipped_arrs, metas = clip_raster(raster_pathlist, frame, nodata_val)
    
    # Transpose clipped array to match format for imshow
    clipped_rgb_arrs = [arr.transpose(1, 2, 0) for arr in clipped_arrs]
    
    # Define labels for subplots
    labels = list(st7_years)
    
    # Initialize figure with 3x4 subplots
    fig, axs = plt.subplots(2, 4, figsize=(14, 8))
    
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
st7_val_frames = point_frame(st7_valpoints, 500)

# Plot planet data for defined point
planet_plot(st7_planet_paths, st7_val_frames, 129, 30)        
















