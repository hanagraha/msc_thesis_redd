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


# RESAMPLE DATA (IF NECESSARY)


############################################################################
"""
2014 images are used as reference shapes
"""
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

# Define function to resample to match image shapes
def resample(arr1path, arr2path):
    
    # Read first array (reference)
    with rasterio.open(arr1path) as src1:
        
        # Get metadata of first array
        profile = src1.profile
        
        # Get the transform and dimensions of first array
        dst_transform = src1.transform
        dst_width = src1.width
        dst_height = src1.height

    # Read second array (array to resample)
    with rasterio.open(arr2path) as src2:
        
        # Read the data from the second TIFF
        data = src2.read(
            
            # Match shape of first array
            out_shape=(src2.count, dst_height, dst_width),
            
            # Use bilinear resampling (good for RGB)
            resampling=Resampling.bilinear
        )

        # Adjust metadata to match first array
        profile.update(
            transform=dst_transform,
            width=dst_width,
            height=dst_height
        )
        
        # Define output filename
        outfilepath = arr2path.replace("pp", "resamp")

        # Save resampled array (override)
        with rasterio.open(outfilepath, 'w', **profile) as dst:
            dst.write(data)

    # Print statement
    print("Resampled mismatched array")
    
    return outfilepath

# Define function to check shape of rasters
def shapecheck(path_dict, ref_img):
    
    # Create an empty list to hold new filenames
    resamp_files = []

    # Read reference image
    with rasterio.open(ref_img) as rast:
        
        # Extract shape of image
        ref_shape = rast.shape
    
    # Iterate over each year and path group
    for year, paths in path_dict.items():
        
        # If there are no paths for that year
        if not paths: 
            
            # Print statement for skipping
            print(f"No rasters for year {year}. Skipping...")
            print()
            
            # Skip processing for this year
            continue
        
        # Create empty list to hold raster shapes
        shapes = []
        
        # Iterate over each path in group
        for path in paths:
            
            # Read raster in path
            with rasterio.open(path) as rast:
                
                # Extract raster metadata
                shape = rast.shape
                
                # Add width to list
                shapes.append(shape)
        
        # Check if shapes are identical to reference shape
        shapecheck = all(shp == ref_shape for shp in shapes)
        
        # Print statement if identical
        if shapecheck == True:
            
            print(f"Images from {year} share identical shapes")
        
        # Print statement if not identical
        else:
            
            print(f"Images from {year} have different shapes")
        
        # Iterate over each path (again)
        for path in paths:
            
            # Read raster in path
            with rasterio.open(path) as rast:
                
                # Extract raster metadata
                shape = rast.shape
                
                # Check if shape matches reference
                if shape == ref_shape:
                    
                    # Define new file name for consistency
                    resamp_filename = path.replace("pp", "resamp")
                    
                    # Copy file under new name
                    shutil.copy2(path, resamp_filename)
                    
                    # Add filename to list
                    resamp_files.append(resamp_filename)
                
                # If the shape does not match reference
                else:
                    
                    # Resample path to match reference
                    resamp_filename = resample(ref_img, path)
                    
                    # Add filename to list
                    resamp_files.append(resamp_filename)
            
        # Print statement to create gap in results
        print()
    
    return resamp_files

# Group files for january composites
jan_groups = comp_groups(jan_reproj_paths)

# Group files for february composites
feb_groups = comp_groups(feb_reproj_paths)

# Group files for december composites
dec_groups = comp_groups(dec_reproj_paths)

# Reference image path
ref_img = jan_groups[2014][0]

# Check group shapes for january
jan_resamp_paths = shapecheck(jan_groups, ref_img)

# Check group shapes for february
feb_resamp_paths = shapecheck(feb_groups, ref_img)

# Check group shapes for december
dec_resamp_paths = shapecheck(dec_groups, ref_img)



############################################################################


# CREATE MONTHLY COMPOSITES


############################################################################
# Define function to composite images 
def composite(path_dict, out_dir, month, nodata_value=np.nan):
    
    # Record start time    
    start_time = time.time()
    
    # Print statement for start time
    print(f"Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # Create empty list to store output files
    outfilepaths = []
    
    # Iterate over each year and path group
    for year, paths in path_dict.items():
        
        # If there are no paths for that year
        if not paths: 
            
            # Print statement for skipping
            print(f"No rasters for year {year}. Skipping...")
            
            continue
        
        # Print statement for compositing
        print(f"Processing year {year} with {len(paths)} files...")
        
        # Placeholder for stacking data and metadata
        band_data_stack = None
        meta = None
        
        # Iterate over each path in group
        for path in paths:
            
            # Open raster image
            with rasterio.open(path) as src:
                
                # If there is no metadata (yet)
                if meta is None:
                    
                    # Add metadata to stack
                    meta = src.meta.copy()
                    
                    # Extract number of bands
                    num_bands = meta['count']
                    
                    # Add empty list in band_data_stack for each band
                    band_data_stack = [[] for _ in range(num_bands)]
                
                # Iterate over each band
                for b in range(1, num_bands + 1):
                    
                    # Extract band data
                    data = src.read(b)
                    
                    # Create masked array to ignore nodata values
                    data_masked = np.ma.masked_equal(data, nodata_value)
                    
                    # Add masked band data to band dictionary
                    band_data_stack[b - 1].append(data_masked)
        
        # Create empty list to hold composite data
        composites = []
        
        # Iterate over each band
        for b, band_stack in enumerate(band_data_stack):
            
            # Convert data list to masked raster stack
            band_stack = np.ma.stack(band_stack, axis=0)
            
            # Extract minimum pixel value, ignoring masked (NoData) values
            composite = np.nanmin(band_stack, axis=0)
            
            # Add minimum pixel values to list
            composites.append(composite)
        
        # Create masked raster stack out of composite values
        composite_array = np.ma.stack(composites, axis=0)
        
        # Define output filepath
        output_path = os.path.join(out_dir, f"composite_{year}_{month}.tif")
        
        # Update metadata with correct dtype of masked arrays
        meta.update(dtype=composite_array.dtype, count=num_bands)
        
        # Write composite to file, 
        with rasterio.open(output_path, 'w', **meta) as dst:
            
            # Replaced masked values with no data value 
            dst.write(composite_array.filled(nodata_value))
        
        # Print statement
        print(f"Composite for year {year} saved to {output_path}")
        
        # Add filepath to list
        outfilepaths.append(output_path)
    
    # Record end time
    end_time = time.time()
    
    # Print statement for end time
    print(f"End time: {time.strftime('%H:%M:%S', time.localtime(end_time))}")
    
    # Print statement for total executino time
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
                
    return outfilepaths

# Group files for january composites
jan_resamp_groups = comp_groups(jan_resamp_paths)

# Group files for february composites
feb_resamp_groups = comp_groups(feb_resamp_paths)

# Group files for december composites
dec_resamp_groups = comp_groups(dec_resamp_paths)

# Composite january images
jan_comp_paths = composite(jan_resamp_groups, jan_temp, "jan")

# Composite february images
feb_comp_paths = composite(feb_resamp_groups, feb_temp, "feb")

# Composite december images
dec_comp_paths = composite(dec_resamp_groups, dec_temp, "dec")



############################################################################


# CLIP IMAGES TO AOI


############################################################################
# Define function to clip rasters
def clip_raster(raster_pathlist, aoi_geom, nodata_value, out_dir):
    
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
        
        # Extract file name
        basename = os.path.basename(file)
        
        # Split file name components
        splits = basename.split("_")
        
        # Define new file name
        outfilename = f"Landsat8_{splits[1]}_{splits[2][:3]}.tif"
        
        # Define new filepath
        outfilepath = os.path.join(out_dir, outfilename)
        
        # Replace old file with new file (same file name)
        with rasterio.open(outfilepath, 'w', **out_meta) as dest:
            
            # Iterate over each band of interest
            for band_index, band in enumerate(indices, start=1):
                
                # Write each band to file
                dest.write(raster_clip[band_index - 1], band_index)
        
        # Print statement
        print(f"Clipping Complete for {outfilename}")

# Clip january composites
clip_raster(jan_comp_paths, aoi, nodata_val, out_dir)

# Clip february composites
clip_raster(feb_comp_paths, aoi, nodata_val, out_dir)

# Clip december composites
clip_raster(dec_comp_paths, aoi, nodata_val, out_dir)




















