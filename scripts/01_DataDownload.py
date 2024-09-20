# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:08:53 2024

@author: hanna
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import requests
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


# IMPORT DEFORESTATION DATASETS (WEB DOWNLOAD TO DRIVE)


############################################################################

### DOWNLOAD GFC DATA
gfc_urls = ["https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_treecover2000_10N_020W.tif",
            "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_lossyear_10N_020W.tif"]

# Directory to save the files
gfc_dir = "data/hansen"

# Create the directory if it doesn't exist
os.makedirs(gfc_dir, exist_ok=True)  

# List to store newly created filenames
gfc_files = []

# Loop through each URL
for url in gfc_urls:
    
    # Extract filename from the URL
    filename = url.split("/")[-1]
    filename = filename.replace("Hansen_GFC-2023-v1.11", "gfc").replace("_10N_020W", "")
    local_filename = os.path.join(gfc_dir, filename)

    # Send GET request to the URL
    response = requests.get(url, stream=True)

    # Open the file in write-binary mode and write the content
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            
    # Add the downloaded file to the list
    gfc_files.append(local_filename)

    print(f"Download complete for {filename}")


### DOWNLOAD TMF DATA
tmf_urls = ["https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=DegradationYear&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=DeforestationYear&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2013&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2014&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2015&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2016&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2017&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2018&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2019&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2020&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2021&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2022&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2023&lat=N10&lon=W20"]


# Directory to save the files
tmf_dir = "data/jrc"

# Create the directory if it doesn't exist
os.makedirs(tmf_dir, exist_ok=True)  

# List to store newly created filenames
tmf_files = []

# Loop through each URL
for url in tmf_urls:
    
    # Extract filename from the URL
    filename = url.split("dataset=")[1].split("&")[0]
    filename = f"tmf_{filename}.tif"
    local_filename = os.path.join(tmf_dir, filename)

    # Send GET request to the URL
    response = requests.get(url, stream=True)

    # Open the file in write-binary mode and write the content
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Add the downloaded file to the list
    tmf_files.append(local_filename)

    print(f"Download complete for {filename}")



############################################################################


# REPROJECT AND CLIP RASTERS


############################################################################

def reproject_raster(input_raster_path, epsg_string):
    # Open the source raster
    with rasterio.open(input_raster_path) as src:
        # Calculate the new transform, width, and height for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, epsg_string, src.width, src.height, *src.bounds)
        
        # Copy the metadata and update with new projection details
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': epsg_string,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create an empty array to hold the reprojected data
        reprojected_array = np.empty((src.count, height, width), dtype=src.meta['dtype'])
        
        # Loop through each band and reproject
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=reprojected_array[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=epsg_string,
                resampling=Resampling.nearest)
        
        return reprojected_array
    
    
### REPROJECT AND CLIP GFC
gfc_clipped_dict = {}
tif = gfc_files[1]

for tif in gfc_files:
    clipped_image = reproject_raster(tif, epsg_string)
    
    # Save to drive
    output_path = tif.replace(".tif", "_clipped.tif")
    output_path = tif.replace("data/hansen\\Hansen_GFC-2023-v1.11", "gfc").replace("_10N_020W", "").replace(".tif", "_clipped.tif")

    gfc_clipped_dict[output_path] = clipped_image
    print(f"Preprocessing Complete for {output_path}")


### DEFINE FUNCTION TO REPROJECT AND CLIP
def reproject_and_clip_raster(tif_path, target_crs, clip_boundary_gdf):
    with rasterio.open(tif_path) as src:
        
        # Calculate the transform and dimensions for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        # Create a metadata template for the reprojected raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject the raster in memory
        reprojected_array = rasterio.MemoryFile().open(**kwargs)
        
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(reprojected_array, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
        
        # Clip the reprojected raster with the boundary
        clip_boundary = [feature['geometry'] for feature in clip_boundary_gdf.__geo_interface__['features']]
        clipped_image, clipped_transform = mask(reprojected_array, clip_boundary, crop=True, nodata=255)
        clipped_image = clipped_image.astype('float32')   
        clipped_image[clipped_image == 255] = np.nan
        
        return clipped_image, clipped_transform, kwargs


### REPROJECT AND CLIP GFC
gfc_clipped_dict = {}

for tif in gfc_files:
    clipped_image, clipped_transform, metadata = reproject_and_clip_raster(tif, epsg_string, aoi_geom)
    non_nan_mask = ~np.isnan(clipped_image) 
    clipped_image[non_nan_mask] += 2000
    
    # Save to drive
    output_path = tif.replace(".tif", "_clipped.tif")
    gfc_clipped_dict[output_path] = clipped_image
    print(f"Preprocessing Complete for {output_path}")
    
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(clipped_image)
        
        
### REPROJECT AND CLIP TMF
tmf_clipped_dict = {}

for tif in tmf_files:
    clipped_image, clipped_transform, metadata = reproject_and_clip_raster(tif, epsg_string, aoi_geom)
    
    # Save to drive
    output_path = tif.replace(".tif", "_clipped.tif")
    tmf_clipped_dict[output_path] = clipped_image
    print(f"Preprocessing Complete for {output_path}")
    
    
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(clipped_image)


### READ RELEVANT DATA FILES
with rasterio.open("data/jrc/tmf_AnnualChange_2018_clipped.tif") as tmf:
    tmf2018 = tmf.read(1)



############################################################################


# PLOTTING


############################################################################

# ### FOR RASTERS (ARRAY)
plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
show(tmf2018, title='TMF Deforestation Year', cmap='gist_ncar')

# # ### FOR VECTORS (GDF)
# # aoi.plot()
# # plt.show()


# # Figure with three subplots, unpack directly
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

# # Populate the three subplots with raster data
# show(tmf_defor_clipped, ax=ax1, title='TMF Deforestation Year')
# show(gfc_defor_clipped, ax=ax2, title='GFC Deforestation Year')

# plt.show()
























