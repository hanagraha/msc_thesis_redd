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

### DOWNLOAD DATA

# URL of the file to download
url = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# Local filename to save the file
local_filename = "data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# Send GET request to the URL
response = requests.get(url, stream=True)

# Open the file in write-binary mode and write the content
with open(local_filename, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print("Download complete.")


### REPROJECT DATA FILES
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

with rasterio.open("data/jrc/tmf_defor_reproj.tif") as defor:
    tmf_defor = defor.read(1)

# Check if epsgs match:
if defor.crs == epsg_string:
    print(f"Both GeoDataFrames have the same {epsg_string}")
else:
    print(f"Different EPSG codes: raster dataset has {defor.crs}, destination crs is {epsg_string}")


### READ DATA
    
with rasterio.open("data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as defor:
    tmf_defor = defor.read(1)

with rasterio.open("data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as degra:
    tmf_degra = degra.read(1)
    
with rasterio.open("data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif") as gfc:
    gfc_defor = gfc.read(1)
    


############################################################################


# CROP TO AOI


############################################################################




############################################################################


# PLOTTING


############################################################################


# plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
# show(dat, title='Chech Area', cmap='gist_ncar')































