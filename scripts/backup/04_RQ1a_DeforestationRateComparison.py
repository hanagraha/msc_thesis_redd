# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:56:10 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from rasterstats import zonal_stats
import matplotlib.pyplot as plt



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



############################################################################


# IMPORT DATA


############################################################################

### RASTER DATA
gfc_lossyear_paths = [f"data/intermediate/gfc_lossyear_{year}.tif" for 
                      year in range(2013, 2024)]
tmf_deforyear_paths = [f"data/intermediate/tmf_DeforestationYear_{year}.tif" 
                       for year in range(2013, 2024)]
tmf_degrayear_paths = [f"data/intermediate/tmf_DegradationYear_{year}.tif" 
                       for year in range(2013, 2024)]
tmf_annual_paths = [f"data/jrc_preprocessed/tmf_AnnualChange_{year}_fm.tif" for 
                    year in range(2013,2024)]

### POLYGON DATA
villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']])

# Calculate REDD+ and non-REDD+ area (m2? ha?)
areas = villages.groupby('grnp_4k')['area'].sum()
redd_area = areas.get(1,0)
nonredd_area = areas.get(0,0)

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']]
villages = villages.dissolve(by='grnp_4k')
villages = villages.reset_index()



############################################################################


# EXPLORE ANNUAL CHANGE DATASET


############################################################################
years = list(range(2013, 2024))

# Create a dictionary to store the pixel counts for each value (1-6) per year
annual_pixels = {value: [] for value in range(1, 7)}

# Iterate over each .tif file and count pixel values from 1-6
for file in tmf_annual_paths:
    with rasterio.open(file) as src:
        raster_data = src.read(1)
        
        # Count the number of pixels for each value (1 to 6)
        for value in range(1, 7):
            pixel_count = np.sum(raster_data == value)
            annual_pixels[value].append(pixel_count)

# Convert pixel counts to a DataFrame
tmf_annual_df = pd.DataFrame(annual_pixels, index=years)

# Define colors for each pixel value
colors = {
    1: (0/255, 90/255, 0/255),      # Undisturbed tropical moist forest
    2: (100/255, 155/255, 35/255),  # Degraded tropical moist forest
    3: (255/255, 135/255, 15/255),  # Deforested land
    4: (210/255, 250/255, 60/255),  # Tropical moist forest regrowth
    5: (0/255, 140/255, 190/255),   # Permanent and seasonal water
    6: (211/255, 211/255, 211/255)   # Other land cover
}

# Define labels for pixel values
labels = {
    1: "Undisturbed tropical moist forest",
    2: "Degraded tropical moist forest",
    3: "Deforested land",
    4: "Tropical moist forest regrowth",
    5: "Permanent and seasonal water",
    6: "Other land cover"
}

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each pixel value as a stack in the bar chart
bottom = np.zeros(len(years))
for value in range(1, 7):
    ax.bar(years, tmf_annual_df[value], bottom=bottom, color=colors[value], 
           label=labels[value])
    bottom += tmf_annual_df[value]

# Customize the plot
ax.set_xlabel('Year')
ax.set_ylabel('Number of Pixels')
ax.set_title('TMF Annual Change from 2013-2023 in Gola REDD+ AOI')

# Set x-ticks to show every year
ax.set_xticks(years)  # Set tick marks for each year
ax.set_xticklabels(years)  # Label them with the year values

ax.legend(title='Land Cover Types', loc='lower left')

# Show the plot
plt.tight_layout()
plt.show()



############################################################################


# EXPLORE TRANSITION MAP DATASET


############################################################################
### READ FILE UNIQUE VALUES
tmf_trans_file = "data/jrc_preprocessed/tmf_TransitionMap_MainClasses_fm.tif"
with rasterio.open(tmf_trans_file) as tmf:
    tmf_trans = tmf.read(1)

unique_values, pixel_counts = np.unique(tmf_trans, return_counts=True)

"""
NOTE: the value "0" exists in the transition map. This value is present in the 
raw data and is not described in the TMF data manual. Because it has a limited 
coverage, the value "0" pixels will excluded from the following analysis
"""

# Exclude the value "0" and NA
unique_values = unique_values[1:10]
pixel_counts = pixel_counts[1:10]


# List class labels (from the tmf manual)
tmf_trans_labels = ['Undisturbed tropical moist forest', 
                    'Degraded tropical moist forest',
                    'Forest regrowth',
                    'Deforested land - plantations',
                    'Deforested land - water bodies',
                    'Deforested land - other',
                    'Ongoing deforestation/degradation',
                    'Permanent and seasonal water',
                    'Other land cover']


### CALCULATE PIXEL CLASS DISTRIBUTIONS
total_pixels = np.sum(pixel_counts)
pixel_percent = (pixel_counts / total_pixels)*100

# Store in a dataframe
tmf_trans_summary = pd.DataFrame({
    'Value': unique_values,
    'Label': tmf_trans_labels,
    'Pixel_Count': pixel_counts,
    'Pixel_Percent': pixel_percent
})

# Sort classes in descending percent order
tmf_trans_sorted = tmf_trans_summary.sort_values(by='Pixel_Percent', 
                                                 ascending=False)


### PLOT AS BAR CHART
plt.figure(figsize=(10, 6))
bars = plt.bar(tmf_trans_sorted['Label'], tmf_trans_sorted['Pixel_Percent'])

# Add labels and title
plt.xlabel('Land Cover Classes')
plt.ylabel('Pixel Percent (%)')
plt.title('Pixel Percentage by Land Cover Class (Sorted)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()



############################################################################


# EXTRACT DEFORESTATION PER YEAR FOR WHOLE AOI


############################################################################
# Geodataframes for statistics
gfc_loss_aoi = aoi.copy()
tmf_defor_aoi = aoi.copy()
tmf_degra_aoi = aoi.copy()

### EXTRACT DEFORESTATION PIXELS
# GFC lossyear
for year, raster_path in zip(range(2013, 2024), gfc_lossyear_paths):
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        
        # Extract deforestation per polygon
        deforestation_pixels = np.sum(raster_data == year)
        
    gfc_loss_aoi[f"{year}"] = deforestation_pixels    
    
print("Extracted gfc deforestation in AOI")

# TMF Deforestation Year
for year, raster_path in zip(range(2013, 2024), tmf_deforyear_paths):
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        
        # Extract deforestation per polygon
        deforestation_pixels = np.sum(raster_data == year)
        
    tmf_defor_aoi[f"{year}"] = deforestation_pixels    
    
print("Extracted tmf deforestation in AOI")

# TMF Degradation Year
for year, raster_path in zip(range(2013, 2024), tmf_degrayear_paths):
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        
        # Extract deforestation per polygon
        deforestation_pixels = np.sum(raster_data == year)
        
    tmf_degra_aoi[f"{year}"] = deforestation_pixels    
    
print("Extracted tmf degradation in AOI")


### PLOT RESULTS
# Assuming gfc_loss_aoi, tmf_defor_aoi, and tmf_degra_aoi have columns for each year
years = list(range(2013, 2024))

# Extracting deforestation pixels per year
gfc_loss_pixels = [gfc_loss_aoi[str(year)].values[0] for year in years]
tmf_defor_pixels = [tmf_defor_aoi[str(year)].values[0] for year in years]
tmf_degra_pixels = [tmf_degra_aoi[str(year)].values[0] for year in years]

# Combine tmf_defor and tmf_degra for the line plot
tmf_combined_pixels = [defor + degra for defor, degra in zip(tmf_defor_pixels, tmf_degra_pixels)]

# Set bar width
bar_width = 0.4
index = np.arange(len(years))

# Color Palette
tmf_col2 = "#6BAF4B"  # Bright Olive Green
gfc_col2 = "#800020"  # Burgundy (distinct)
tmf_col1 = "#A1D8D5"  # Light Teal
gfc_col1 = "#2E5B2E"  # Dark Green
tmf_col3 = "#FF7F50"  # Coral (distinct)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot GFC Loss AOI bars
bars1 = ax.bar(index - bar_width/2, gfc_loss_pixels, bar_width, 
               label='GFC Tree Cover Loss', color=gfc_col1)

# Plot TMF Deforestation and Degradation stacked bars
bars2 = ax.bar(index + bar_width/2, tmf_defor_pixels, bar_width, 
               label='TMF Deforestation', color=tmf_col1)
bars3 = ax.bar(index + bar_width/2, tmf_degra_pixels, bar_width, 
               label='TMF Degradation', bottom=tmf_defor_pixels, color=tmf_col2)

# Add lines through the top of GFC Loss AOI bars
ax.plot(index - bar_width/2, gfc_loss_pixels, color=gfc_col2, 
        label='GFC Tree Cover Loss (line)')

# Add lines through the top of TMF Combined bars
ax.plot(index + bar_width/2, tmf_combined_pixels, color=tmf_col3,
        label='TMF Combined (line)')

# Customize the plot
ax.set_xlabel('Year')
ax.set_ylabel('Number of Pixels')
ax.set_title('GFC Tree Cover Loss vs TMF Deforestation + Degradation in the Gola REDD+ AOI')
ax.set_xticks(index)
ax.set_xticklabels(years)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()



############################################################################


# EXTRACT DEFORESTATION PER YEAR PER REDD+ / NON-REDD+ AREA


############################################################################
# Geodataframes for statistics
gfc_loss_stats = villages.copy()
tmf_defor_stats = villages.copy()
tmf_degra_stats = villages.copy()

### EXTRACT DEFORESTATION PIXELS
# GFC lossyear
for year, raster_path in zip(range(2013, 2024), gfc_lossyear_paths):
    
    with rasterio.open(raster_path) as src:
        
        # Extract deforestation per polygon
        deforestation_stats = zonal_stats(gfc_loss_stats, raster_path, 
                                          stats="count", nodata=nodata_val)
    
    # Store stats in a column
    gfc_loss_stats[f"{year}"] = [stat['count'] for stat in deforestation_stats]
    
print("Extracted gfc deforestation in villages")

# TMF deforestation year
for year, raster_path in zip(range(2013, 2024), tmf_deforyear_paths):
    
    with rasterio.open(raster_path) as src:
        
        # Extract deforestation per polygon
        deforestation_stats = zonal_stats(tmf_defor_stats, raster_path, 
                                          stats="count", nodata=nodata_val)
    
    # Store stats in a column
    tmf_defor_stats[f"{year}"] = [stat['count'] for stat in deforestation_stats]
    
print("Extracted tmf deforestation in villages")

# TMF degradation year
for year, raster_path in zip(range(2013, 2024), tmf_degrayear_paths):
    
    with rasterio.open(raster_path) as src:
        
        # Extract deforestation per polygon
        deforestation_stats = zonal_stats(tmf_degra_stats, raster_path, 
                                          stats="count", nodata=nodata_val)
    
    # Store stats in a column
    tmf_degra_stats[f"{year}"] = [stat['count'] for stat in deforestation_stats]
    
print("Extracted tmf degradation in villages")


### CALCULATE DEFORESTATION AS PROPORTION OF AREA
# Extract the columns for years 2013-2023
years = gfc_loss_stats.columns[2:]

gfc_redd_pixels = gfc_loss_stats.loc[1, years].values
gfc_nonredd_pixels = gfc_loss_stats.loc[0, years].values

tmf_defor_redd_pixels = tmf_defor_stats.loc[1, years].values
tmf_defor_nonredd_pixels = tmf_defor_stats.loc[0, years].values

tmf_degra_redd_pixels = tmf_degra_stats.loc[1, years].values
tmf_degra_nonredd_pixels = tmf_degra_stats.loc[0, years].values

tmf_redd_pixels = [defor + degra for defor, degra in zip(
    tmf_defor_redd_pixels, tmf_degra_redd_pixels)]
tmf_redd_pixels = np.array(tmf_redd_pixels, dtype=object)

tmf_nonredd_pixels = [defor + degra for defor, degra in zip(
    tmf_defor_nonredd_pixels, tmf_degra_nonredd_pixels)]
tmf_nonredd_pixels = np.array(tmf_nonredd_pixels, dtype=object)

# Convert deforestation pixels to area proportion
pixel_area = 0.09

gfc_redd_perc = ((gfc_redd_pixels * pixel_area) / redd_area)*100
gfc_nonredd_perc = ((gfc_nonredd_pixels * pixel_area) / nonredd_area)*100

tmf_redd_perc = ((tmf_redd_pixels * pixel_area) / redd_area)*100
tmf_nonredd_perc = ((tmf_nonredd_pixels * pixel_area) / nonredd_area)*100


### PLOT RESULTS
# Color palatte
redd_col1 = "#8B0000"  # Dark Red
redd_col2 = "#D2691E"  # Chocolate

nonredd_col1 = "#228B22"  # Forest Green
nonredd_col2 = "#4682B4"  # Steel Blue

plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_redd_perc, color=redd_col1,
         label='GFC Deforestation in REDD+ Villages')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, gfc_nonredd_perc, color=nonredd_col1, 
         label='GFC Deforestation in Non-REDD+ Villages')

# Plot the pixel values for REDD+ villages
plt.plot(years, tmf_redd_perc, color=redd_col2,
         label='TMF Deforestation and Degradation in REDD+ Villages')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, tmf_nonredd_perc, color=nonredd_col2,
         label='TMF Deforestation and Degradation in Non-REDD+ Villages')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('% of Deforestation Pixels Per REDD+/Non-REDD+ Area')
plt.title('Deforestation in REDD+ vs Non-REDD+ Villages (2013-2023)')

# Add legend
plt.legend()

# Show the grid and plot
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-ticks for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()








