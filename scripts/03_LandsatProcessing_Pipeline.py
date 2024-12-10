# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:53:21 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import geopandas as gpd
import os
import pandas as pd
import ee
import matplotlib.pyplot as plt
import numpy as np
import calendar



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

# Define years of interest
years = range(2013, 2025)

# Define 



############################################################################


# IMPORT DEFORESTATION DATASETS (LOCAL DRIVE)


############################################################################

# Read data
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")
villages = gpd.read_file("data/village polygons/VillagePolygons.geojson")

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

# Create aoi
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()

# Download aoi as shapefile
aoi_gdf = aoi[['geometry']]

aoi_gdf = aoi_gdf.to_crs(epsg=4326)

# Specify the path where you want to save the shapefile
outfilepath = os.path.join("data", "village polygons", "villages_simple.shp")

# Export as shapefile
aoi_gdf.to_file(outfilepath)


############################################################################


# IDENTIFY AVAILABLE LANDSAT IMAGERY


############################################################################
# Initialize Earth Engine API
ee.Authenticate()
ee.Initialize()

# Adjust aoi crs to match GEE default
aoi_4326 = aoi.to_crs(epsg=4326)

# Create bounding box from aoi
aoi_bbox = aoi_4326['geometry'].bounds

# Create ee geometry with bounding box
ee_aoi = ee.Geometry.Rectangle([
    aoi_bbox["minx"].iloc[0], aoi_bbox["miny"].iloc[0], 
    aoi_bbox["maxx"].iloc[0], aoi_bbox["maxy"].iloc[0]])

# Define function to create date lists for each month
def date_list(yearrange):
    
    # Create empty list to hold all month dates
    alldates = []
    
    for month in range(1,13):
        
        # Create empty list for dates
        dates = []
        
        # Define ending month
        end_month = month + 1
        
        # Iterate over each year
        for year in yearrange:
            
            # Define start date
            start = f"{year}-{month}-01"
            
            # If the starting month is december
            if month == 12: 
                
                # Convert ending month to january (avoid month 13)
                end_month = 1
                
                # Add 1 to the ending year (to represent the following year)
                year = year + 1
            
            # Define end date
            end = f"{year}-{end_month}-01"
            
            # Add dates to list
            dates.append((start, end))
        
        # Add dates to list
        alldates.append(dates)
        
    return alldates

# Define function to create an image collection
def landsat_collect(monthdates, aoi, yearrange, cc=30):
    
    # Create empty list to hold ee objects for each month and year
    ee_collections = []
    
    # Create empty list to hold sum of available 
    ee_availability = []
    
    # Iterate over each month list
    for dates in monthdates:
        
        # Create empty list to hold ee objects for each year
        ee_objlist = []
        
        # Create empty list to hold ee sizes for each year
        ee_sizelist = []
    
        # Iterate over each date pair
        for (start, end) in dates:
            
            # Filter Landsat 8 collection
            landsat = (
                
                # Extract from landsat surface reflectance
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                
                # Filter collection for date range
                .filterDate(start, end)
                
                # Filter collection for aoi bounding box
                .filterBounds(aoi)
                
                # Filter collection for cloud cover < 30% (default)
                .filterMetadata('CLOUD_COVER', 'less_than', cc)
            )
        
            # Convert image collection to list
            landsat_list = landsat.toList(landsat.size())
            
            # Extract size of image collection
            landsat_size = landsat.size().getInfo()
            
            # Add image collection list to list
            ee_objlist.append(landsat_list)
            
            # Add size to list
            ee_sizelist.append(landsat_size)
            
        # Add list of yearly ee objects to list
        ee_collections.append(ee_objlist)
        
        # Add number of available images to list
        ee_availability.append(ee_sizelist)
        
    # Convert availability to dataframe
    ee_availability_df = pd.DataFrame(
        ee_availability,
        index = list(calendar.month_abbr)[1:],
        columns = yearrange)
    
    # Add sum column
    ee_availability_df["sum"] = ee_availability_df.sum(axis = 1)
    
    return ee_collections, ee_availability_df

# Create date ranges for each month between 2013-2024
dates = date_list(years)

# Extract available landsat images within date ranges (cloud cover < 30%)
cc30_collection, cc30_availability = landsat_collect(dates, ee_aoi, years, cc = 30)

# Extract available landsat images within date ranges (cloud cover < 15%)
cc15_collection, cc15availability = landsat_collect(dates, ee_aoi, years, cc = 15)



############################################################################


# EXPORT LANDSAT IMAGERY


############################################################################
# Define function to mask out clouds
def cloudmask(image):
    
    # Get the QA_PIXEL band
    QA_PIXEL = image.select('QA_PIXEL')
    
    # Mask for cloud (0th bit) and cloud shadow (3rd bit)
    cloud_mask = QA_PIXEL.bitwiseAnd(1).eq(0)  # Cloud bit is 0 for clear sky
    shadow_mask = QA_PIXEL.bitwiseAnd(8).eq(0)  # Shadow bit is 0 for no shadow
    
    # Combine both masks (only keep pixels that are neither cloud nor shadow)
    combined_mask = cloud_mask.And(shadow_mask)
    
    # Apply the mask to image
    return image.updateMask(combined_mask)

# Define function to create cloud-free mosaics
def mosaic(ee_list):
    
    # Convert ee.List to ee.ImageCollection
    collection = ee.ImageCollection(ee_list)
    
    # Apply cloud masking to all images in the collection
    cloud_free_collection = collection.map(cloudmask)
    
    # Create a mosaic (most recent pixels are prioritized, or use median)
    cloud_free_mosaic = cloud_free_collection.median()
    
    return cloud_free_mosaic

# Define function to export mosaics to google drive
def mosaic_export(mosaics, month, outfolder):
    
    # Iterate over each mosaic
    for mosaic, year in zip(mosaics, years):
        
        # Define ee export task
        export_task= ee.batch.Export.image.toDrive(
            image = mosaic, 
            description = f"{month}_mosaic_{year}", 
            folder = outfolder,
            fileNamePrefix = f"{month}_mosaic_{year}",
            scale = 30, 
            crs = "EPSG: 32629",
            region = ee_aoi, 
            maxPixels = 1e8)
        
        # Start export task
        export_task.start()
        
        # Print statement
        print(f"Exporting cloud-free mosaic for {month} {year}")
        
# Create empty list to hold mosaics
mosaics = []

# Iterate over each ee list
for eelist in cc15_collection:
    
    # Iterate over each ee list in ee.ee_list.List
    for eecoll in eelist:
    
        # Check if the list is empty
        if eecoll.size().getInfo() == 0:
            
            # Print statement
            print("Empty list encountered.")
            
            # Skip empty lists
            continue
        
    # Apply mosaic function
    ee_mosaic = mosaic(eelist)
    
    # Add mosaic to list
    mosaics.append(ee_mosaic)
    
# Export mosaics to google drive
mosaic_export(mosaics, "Jan", "GEE_Jan_Mosaics")



############################################################################


# EXPORT LANDSAT IMAGERY


############################################################################
# Define function to export images
def gee_export(image, bands, folder):
    
    # Define GEE task
    task = ee.batch.Export.image.toDrive(
        
        # Select bands and convert to int16 data type
        image = image.select(bands).toUint16(),  
        
        # Define filename description
        description = f'Landsat images in Sierra Leone saved to {folder}',
        
        # Define output folder (in Google Drive)
        folder = folder,  
        
        # Define filename
        fileNamePrefix = image.get('system:index').getInfo(),
        
        # Landsat spatial resolution
        scale = 30,
        
        # Bounding box by AOI
        region = ee_aoi,
        
        # Maximum pixels to export
        maxPixels = 1e13,
    )
    
    # Start GEE task
    task.start()
        
# Define function to download gee images per month
def month_imgs(imglist, bands, outfolder):
    
    # Iterate over each image list
    for eelist, year in zip(imglist, years):
        
        # Print statement to create gap between each year
        print()
        print(f"Image processing for {year}...")
        
        # If images exist for that year
        try:
            
            # Extract number of images in that year
            size = eelist.size().getInfo()
    
            # Iterate over each image
            for i in range(size):
                
                # Extract image from list
                image = ee.Image(eelist.get(i))
                
                # Save landsat image to Google Drive
                gee_export(image, bands, outfolder)
                
                # Print statement
                print(f"Export task started for image {i}. Check Google Drive.")
                
        # Handle cases where no image is available
        except:
            print("No images <30% cloud cover exist.")
            
# Define bands of interest
bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', "QA_PIXEL"]

# Download january images
month_imgs(cc15_collection[0], bands, "GEE_Landsat_Jan_cc30")



## Quick reformatting 
with 

