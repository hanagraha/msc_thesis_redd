# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:15:56 2024

@author: hanna

start time: 3:30
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import geopandas as gpd
import rasterio
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
years = range(2013, 2025)

# Define landsat data folder
l8_folder = os.path.join("data", "cc_composites", "L8_Annual")

# Define landsat jan data folder
l8_jan_folder = os.path.join("data", "cc_composites", "L8_Jan")

# Create temporary indices folder
temp_folder = os.path.join('data', "indices_intermediate")

# Create temporary folder, if it doesn't already exist
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    print(f"{temp_folder} created.")
else:
    print(f"{temp_folder}' already exists.")




############################################################################


# READ INPUT DATA


############################################################################
# Read landsat data
l8 = [os.path.join(l8_folder, file) for file in os.listdir(l8_folder) if \
      file.endswith('.tif')]
    
# Read landsat jan data
l8_jan = [os.path.join(l8_jan_folder, file) for file in os.listdir(l8_jan_folder) \
          if file.endswith('.tif')]
    
# Because landsat jan data misses 2013, use annual 2013 data
l8_jan.insert(0, l8[0])
    
# Read validation points
valpoints = gpd.read_file("data/validation/validation_points.shp")



############################################################################


# CALCULATE INDICES


############################################################################
# %%
# Define function to calculate ndvi
def calc_ndvi(l8_path, years):
    
    # Create empty list to store ndvi files
    ndvi_files = []
    
    # Iterate over each landsat path
    for path, year in zip(l8_path, years):
        
        # Read raster
        with rasterio.open(path) as rast:
            
            # Extract profile
            profile = rast.profile
            
            # Extract nir data
            nir = rast.read(5)
            
            # Extract red data
            red = rast.read(4)
            
            # Update profile for indices
            profile["count"] = 1
            
        # Calculate ndvi
        ndvi = (nir - red) / (nir + red)
        
        # Define output filename for ndvi
        ndvi_filename = os.path.join(temp_folder, f"L8_ndvi_{year}.tif")
            
        # Write ndvi file to drive
        with rasterio.open(ndvi_filename, "w", **profile) as dst:
            dst.write(ndvi, 1)
            
        # Print statement
        print(f"NDVI calculated for {year}")

        # Add ndvi filename to list
        ndvi_files.append(ndvi_filename)
    
    return ndvi_files

# Define function to calculate ndmi
def calc_ndmi(l8_path, years):
    
    # Create empty list to store ndvi files
    ndmi_files = []
    
    # Iterate over each landsat path
    for path, year in zip(l8_path, years):
        
        # Read raster
        with rasterio.open(path) as rast:
            
            # Extract profile
            profile = rast.profile
            
            # Extract nir data
            nir = rast.read(5)
            
            # Extract swir1 data
            swir1 = rast.read(6)
            
            # Update profile for indices
            profile["count"] = 1
            
        # Calculate ndvi
        ndmi = (nir - swir1) / (nir + swir1)
        
        # Define output filename for ndmi
        ndmi_filename = os.path.join(temp_folder, f"L8_ndmi_{year}.tif")
            
        # Write ndvi file to drive
        with rasterio.open(ndmi_filename, "w", **profile) as dst:
            dst.write(ndmi, 1)
            
        # Print statement
        print(f"NDMI calculated for {year}")

        # Add ndvi filename to list
        ndmi_files.append(ndmi_filename)
    
    return ndmi_files

# Define function to extract raster values per point
def extract_val(points_gdf, tiflist, tifnames):
    
    # Copy points gdf
    gdf = points_gdf.copy()
    
    # Iterate over each tif file
    for tif, name in zip(tiflist, tifnames):
        
        # Create empty list to store pixel values
        pix_vals = []
        
        # Read tif file
        with rasterio.open(tif) as src:
        
            # Iterate over each point
            for pnt in gdf.geometry:
                
                # Get row and column indices
                row, col = src.index(pnt.x, pnt.y)
                
                # Extract pixel value at point location
                pix_val = src.read(1)[row, col]
                
                # Append pixel value to list
                pix_vals.append(pix_val)
            
        # Add new column to geodataframe
        gdf[name] = pix_vals
        
        # Print statement
        print(f"Added data for {name}")
        
    return gdf
# %%
# Calculate annual ndvi
ndvi_files = calc_ndvi(l8, years)

# Create list for yearly ndvi labels
ndvi_labs = [f"ndvi_{year}" for year in years]

# Extract ndvi values at validation points
ann_ndvi = extract_val(valpoints, ndvi_files, ndvi_labs)

# Calculate annual ndmi
ndmi_files = calc_ndmi(l8, years)

# Create list for yearly ndvi labels
ndmi_labs = [f"ndmi_{year}" for year in years]

# Extract ndvi values at validation points
ann_ndmi = extract_val(valpoints, ndmi_files, ndmi_labs)

# Calculate jan ndvi
ndvi_jan_files = calc_ndvi(l8_jan, years)

# Extract ndvi values at validation points
jan_ndvi = extract_val(valpoints, ndvi_files, ndvi_labs)

# Calculate jan ndmi
ndmi_jan_files = calc_ndmi(l8_jan, years)

# Extract ndmi values at validation points
jan_ndmi = extract_val(valpoints, ndmi_files, ndmi_labs)



############################################################################


# SAVE DATA TO CSV


############################################################################
# Save annual ndvi to file
ann_ndvi.to_csv("data/validation/annual_ndvi.csv", index=False)

# Save annual ndmi to file
ann_ndmi.to_csv("data/validation/annual_ndmi.csv", index=False)

# Save january ndvi to file
jan_ndvi.to_csv("data/validation/jan_ndvi.csv", index=False)

# Save january ndmi to file
jan_ndmi.to_csv("data/validation/jan_ndmi.csv", index=False)



############################################################################


# DELETE TEMPORARY FOLDER


############################################################################
# If the temporary folder still exists, delete it
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
    print(f"{temp_folder} has been deleted.")
else:
    print(f"{temp_folder} does not exist.")
    