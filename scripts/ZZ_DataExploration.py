# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:39:18 2024

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



############################################################################


# IMPORT DEFORESTATION DATASETS (WEB DOWNLOAD TO DRIVE)


############################################################################

### REPROJECT TMF
with rasterio.open('data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, epsg_string, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': epsg_string,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('data/jrc/tmf_defor_reproj.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=epsg_string,
                resampling=Resampling.nearest)
            

### REPROJECT GFC
with rasterio.open('data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, epsg_string, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': epsg_string,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('data/hansen/gfc_defor_reproj.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=epsg_string,
                resampling=Resampling.nearest)


### READ REPROJECTED FILES
with rasterio.open("data/jrc/tmf_defor_reproj.tif") as defor:
    tmf_defor = defor.read(1)
    
with rasterio.open("data/hansen/gfc_defor_reproj.tif") as gfc:
    gfc_defor = gfc.read(1)

# Check if epsgs match:
if defor.crs == epsg_string:
    print(f"Both GeoDataFrames have the same {epsg_string}")
else:
    print(f"Different EPSG codes: raster dataset has {defor.crs}, destination crs is {epsg_string}")



############################################################################


# CLIP TO AOI


############################################################################

### CREATE AOI
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()
aoi_geom = aoi.geometry


### MASK TMF
with rasterio.open("data/jrc/tmf_defor_reproj.tif") as src:
    # Read the raster's CRS and bounds
    raster_crs = src.crs
    raster_transform = src.transform
    
    # Clip the raster to the polygon shape
    tmf_defor_clipped, out_transform = mask(src, aoi_geom, crop=True, nodata=9999)
    
    out_meta = src.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count': 1,
        'height': tmf_defor_clipped.shape[1],
        'width': tmf_defor_clipped.shape[2],
        'transform': out_transform })
    
tmf_defor_clipped = tmf_defor_clipped.astype('float32')   
tmf_defor_clipped[tmf_defor_clipped == 9999] = np.nan

# Save the clipped raster to a new file
tmf_defor_clipped_path = 'data/jrc/tmf_defor_reproj_clipped.tif'
with rasterio.open(tmf_defor_clipped_path, 'w', **out_meta) as dest:
    dest.write(tmf_defor_clipped)
    

### MASK GFC
with rasterio.open("data/hansen/gfc_defor_reproj.tif") as src:
    # Read the raster's CRS and bounds
    raster_crs = src.crs
    raster_transform = src.transform
    
    # Clip the raster to the polygon shape
    gfc_defor_clipped, out_transform = mask(src, aoi_geom, crop=True, nodata = 255)
    
    out_meta = src.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count': 1,
        'height': gfc_defor_clipped.shape[1],
        'width': gfc_defor_clipped.shape[2],
        'transform': out_transform })
    
# gfc_defor_clipped = gfc_defor_clipped + 2000
gfc_defor_clipped = gfc_defor_clipped.astype('float32')   
gfc_defor_clipped[gfc_defor_clipped == 255] = np.nan
gfc_defor_clipped = gfc_defor_clipped + 2000


# Save the clipped raster to a new file
gfc_defor_clipped_path = 'data/hansen/gfc_defor_reproj_clipped.tif'
with rasterio.open(gfc_defor_clipped_path, 'w', **out_meta) as dest:
    dest.write(gfc_defor_clipped)


############################################################################


# PLOTTING


############################################################################

### FOR RASTERS (ARRAY)
plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
show(gfc_defor_clipped, title='TMF Deforestation Year', cmap='gist_ncar')

# ### FOR VECTORS (GDF)
# aoi.plot()
# plt.show()


# Figure with three subplots, unpack directly
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

# Populate the three subplots with raster data
show(tmf_defor_clipped, ax=ax1, title='TMF Deforestation Year')
show(gfc_defor_clipped, ax=ax2, title='GFC Deforestation Year')

plt.show()


############################################################################


# ARCHIVE


############################################################################

# vil_shape = villages[['geometry']]
# vil_new = vil_shape.explode(ignore_index=True)
# gola_shape = grnp[['geometry']]
# non_na_pixel_count = np.sum(~np.isnan(out_image))
# ref_pix_count = np.sum(out_image)*0.01
# ref_pix_count

### DOWNLOAD DATA

# # URL of the file to download
# url = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# # Local filename to save the file
# local_filename = "data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# # Send GET request to the URL
# response = requests.get(url, stream=True)

# # Open the file in write-binary mode and write the content
# with open(local_filename, 'wb') as file:
#     for chunk in response.iter_content(chunk_size=8192):
#         file.write(chunk)

# print("Download complete.")


### READ DATA
    
# with rasterio.open("data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as defor:
#     tmf_defor = defor.read(1)

# with rasterio.open("data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as degra:
#     tmf_degra = degra.read(1)
    
# with rasterio.open("data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif") as gfc:
#     gfc_defor = gfc.read(1)
    
# # Add 2000 to GFC so instead of 22 looks like 2022
# gfc_defor = gfc_defor + 2000

############################################################################


# VISUALIZE LANDSAT


############################################################################


from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep


landsat_bands_data_path = "data\\landsat\\LC09_L2SP_200055_20240207_20240209_02_T1_SR_B*[1-7]*.TIF"
stack_band_paths = glob(landsat_bands_data_path)
stack_band_paths.sort()

# Create image stack and apply nodata value for Landsat
arr_st, meta = es.stack(stack_band_paths, nodata=-9999)

# Create figure with one plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot red, green, and blue bands, respectively
ep.plot_rgb(arr_st, rgb=(3, 2, 1), ax=ax, title="Landsat 9 RGB Image")
plt.show()


# Create figure with one plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot bands with stretched applied
ep.plot_rgb(
    arr_st,
    rgb=(3, 2, 1),
    ax=ax,
    stretch=True,
    str_clip=0.5,
    title="Landsat 8 RGB Image with Stretch Applied",
)
plt.show()











