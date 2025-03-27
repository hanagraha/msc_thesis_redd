# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:15:37 2025

@author: hanna
"""
############################################################################


# IMPORT PACKAGES


############################################################################

import os
import ee
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd


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

# Create AOI
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()

# Extract AOI geometry
aoi_geom = aoi.geometry



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

# Define function to create date lists
def date_list(month_num):
    
    # Create empty list for dates
    datelist = []
    
    # Define ending month
    end_month = month_num + 1
    
    # Iterate over each year
    for year in years:
        
        # Define start date
        start = f"{year}-{month_num}-01"
        
        # If the starting month is december
        if month_num == 12: 
            
            # Convert ending month to january (avoid month 13)
            end_month = 1
            
            # Add 1 to the ending year (to represent the following year)
            year = year + 1
        
        # Define end date
        end = f"{year}-{end_month}-01"
        
        # Add dates to list
        datelist.append((start, end))
        
    return datelist

# Define function to create an image collection
def landsat_collect(month, datelist, aoi, return_sizes=False):
    
    # Create a gap to separate results
    print()
    
    # Print statement
    print(f"Images to create {month} composite:")
    
    # Create empty list to hold image collections
    landsat_all = []
    
    # Create empty list to hold image collection sizes
    landsat_sizes = []
    
    # Iterate over each acquisition date
    for (start, end), year in zip(datelist, years):
    
        # Filter Landsat 8 collection
        landsat = (
            
            # Extract from landsat surface reflectance
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            
            # Filter collection for date range
            .filterDate(start, end)
            
            # Filter collection for aoi bounding box
            .filterBounds(aoi)
            
            # Filter collection for cloud cover < 30%
            .filterMetadata('CLOUD_COVER', 'less_than', 30)
        )
    
        # Convert image collection to list
        landsat_list = landsat.toList(landsat.size())
        
        # Extract size of image collection
        landsat_size = landsat.size().getInfo()
        
        # Print number of landsat images
        print(f'Number of images in {year} with <30% cloud cover:', landsat_size)
        
        # Add image collection list to list
        landsat_all.append(landsat_list)
        
        # Add image size to list
        landsat_sizes.append(landsat_size)
    
    # Print total number of cloud-free images
    print(f"Total available images: {sum(landsat_sizes)}")
    
    # Return based on return_sizes flag
    if return_sizes:
        return landsat_all, landsat_sizes
    else:
        return landsat_all

# Identify which months have the most cloud-free data
monthly_dates = []

for month in range(1,13):
    dates_month = date_list(month)
    monthly_dates.append(dates_month)

monthly_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                 "Oct", "Nov", "Dec"]

monthly_sizes = []
monthly_sums = []
for dates, name in zip(monthly_dates, monthly_names):
    test_list, test_sizes = landsat_collect(name, dates, ee_aoi, return_sizes = True)
    monthly_sizes.append(test_sizes)
    monthly_sums.append(sum(test_sizes))
    
# Create date range for january composites
dates_jan = date_list(1)

# Create date range for february composites
dates_feb = date_list(2)

# Create date range for march composites
dates_mar = date_list(3)

# Create date range for november composites
dates_nov = date_list(11)

# Create date range for december composites
dates_dec = date_list(12)

# Extract landsat images for january
landsat_jan = landsat_collect("January", dates_jan, ee_aoi)

# Extract landsat images for february
landsat_feb = landsat_collect("February", dates_feb, ee_aoi)

# Extract landsat images for march
landsat_mar = landsat_collect("March", dates_mar, ee_aoi)

# Extract landsat images for november
landsat_nov = landsat_collect("November", dates_nov, ee_aoi)

# Extract landsat images for december
landsat_dec = landsat_collect("December", dates_dec, ee_aoi)
        


############################################################################


# PLOT AVAILABLE LANDSAT IMAGERY


############################################################################
# Define function for plotting available imagery
def img_plt(x_data, y_data, line_data, colors):
    
    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Initialize bottom position for stacking
    bottom = np.zeros(len(monthly_names))

    # Iterate over each month and year
    for category_data, label, color in zip(y_data, years, colors):
        
        # Create bar with month data stacked by year
        ax.bar(x_data, category_data, bottom=bottom, label=label, color=color)
        
        # Update bottom position
        bottom += category_data  

    # Plot total images line
    ax.plot(x_data, line_data, color='#b10026', lw=2, 
            label='Total Available Images')

    # Add x tickmarks and names
    ax.set_xticks(x_data)
    ax.set_xticklabels(monthly_names)

    # Set axes labels
    ax.set_ylabel('Number of Available Images with <30% Cloud Cover')
    ax.set_xlabel('Years')

    # Set title
    ax.set_title('Data availability per month and year')

    # Add legend
    ax.legend(title = "Legend", ncol=6, loc='upper center')
    
    # Add gridlines
    ax.grid(True, linestyle="--")

    # Show the plot
    plt.tight_layout()
    plt.show()

# Define color palette
colors = [
    "#08306b",  # Dark Blue
    "#08519c",  # Medium Dark Blue
    "#2171b5",  # Moderate Blue
    "#4292c6",  # Light Blue
    "#6baed6",  # Pale Blue
    "#9ecae1",  # Very Light Blue
    "#c6dbef",  # Lightest Blue
    "#eff3ff",  # Almost White Blue
    "#bdd7e7",  # Soft Blue
    "#9ec1d9",  # Slate Blue
    "#6c8ebf",  # Steel Blue
    "#39588c",  # Midnight Blue
]

# Define x and y data
x_data = np.arange(len(years))
y_data = np.array(monthly_sizes).T

# Plot data
img_plt(x_data, y_data, monthly_sums, colors)



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
month_imgs(landsat_jan, bands, "GEE_Landsat_Jan")

# Download february images
month_imgs(landsat_feb, bands, "GEE_Landsat_Feb")

# Download march images
month_imgs(landsat_mar, bands, "GEE_Landsat_Mar")

# Download november images
month_imgs(landsat_nov, bands, "GEE_Landsat_Nov")

# Download december images
month_imgs(landsat_dec, bands, "GEE_Landsat_Dec")




