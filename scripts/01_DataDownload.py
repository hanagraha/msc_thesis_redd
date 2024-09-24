# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:08:53 2024

@author: hanna
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import requests
import geopandas as gpd
import os
import pandas as pd



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
gfc_dir = "data/hansen_raw"

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
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2023&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=TransitionMap_Subtypes&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=TransitionMap_MainClasses&lat=N10&lon=W20"]


# Directory to save the files
tmf_dir = "data/jrc_raw"

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


























