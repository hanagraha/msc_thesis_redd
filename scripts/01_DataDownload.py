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

### DEFINE FUNCTION TO REPROJECT RASTERS
def reproject_raster(raster_path, epsg):
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
    
    
# Reproject GFC images
gfc_reproj_files = []

for tif in gfc_files:
    tif_reproj_file = reproject_raster(tif, epsg_string)
    gfc_reproj_files.append(tif_reproj_file)
    
    
# Reproject TMF images
tmf_reproj_files = []

for tif in tmf_files:
    tif_reproj_file = reproject_raster(tif, epsg_string)
    tmf_reproj_files.append(tif_reproj_file)



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
    
    outpath = raster_path.replace(".tif", "_clipped.tif")
    
    # if os.path.exists(outpath): 
    #     os.remove(outpath)
    
    with rasterio.open(outpath, 'w', **out_meta) as dest:
        dest.write(raster_clip)
    
    print(f"Clipping Complete for {outpath}")
    
    return raster_clip, out_meta


# Clip images of interest
gfc_lossyear, gfc_lossyear_meta = clip_raster(gfc_reproj_files[1], aoi_geom, 255)
gfc_treeloss2000, gfc_treeloss2000_meta = clip_raster(gfc_reproj_files[0], aoi_geom, 255)
tmf_deforyear, tmf_deforyear_meta = clip_raster(tmf_reproj_files[1], aoi_geom, 255)
tmf_degrayear, tmf_degrayear_meta = clip_raster(tmf_reproj_files[0], aoi_geom, 255)



############################################################################


# RECLASSIFICATION


############################################################################
# Reclassify GFC 
gfc_lossyear_path = gfc_reproj_files[1].replace(".tif", "_clipped_reclassified.tif")
gfc_lossyear_new = np.where(gfc_lossyear != 255, gfc_lossyear + 2000, gfc_lossyear)
gfc_lossyear_meta.update(dtype='int16')
gfc_lossyear_new = gfc_lossyear + 2000

with rasterio.open(gfc_lossyear_path, 'w', **gfc_lossyear_meta) as dest:
    dest.write(gfc_lossyear_new)

### READ RELEVANT DATA FILES
with rasterio.open(tmf_reproj_files[1]) as tmf:
    deforyear = tmf.read(1)



############################################################################


# PLOTTING


############################################################################

# ### FOR RASTERS (ARRAY)
plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
show(tmf_deforyear, title='TMF Deforestation Year', cmap='gist_ncar')

# # ### FOR VECTORS (GDF)
# # aoi.plot()
# # plt.show()


### FIGURE WITH SUBPLOTS (ARRAY)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

# Populate the three subplots with raster data
show(gfc_lossyear, ax=ax1, title='TMF Deforestation Year')
show(gfc_lossyear_new, ax=ax2, title='GFC Deforestation Year')

plt.show()
























