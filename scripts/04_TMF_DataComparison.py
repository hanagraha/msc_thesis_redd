# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:56:10 2024

@author: hanna

This file compares data from TMF's transition map, annual change map, 
deforestation year, and degradation year to GFC's lossyear data. 

Expected execution time: <1min
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

# Define year range
years = list(range(2013, 2024))

# Color Palatte (TMF Defor + Degra)
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
brown = '#48202a'
beige = '#ddcc77'
bluecols = [blue1, blue2, blue3]

# Color Palette (GFC vs TMF - redd vs nonredd)
redd_col1 = "#820300"  # Darker Red
redd_col2 = "#820300"  # Lighter Red - darker
nonredd_col1 = "#4682B4"  # Darker Blue - lighter
nonredd_col2 = "#4682B4"  # Lighter Blue



############################################################################


# IMPORT DATA


############################################################################
# Define gfc lossyear paths
gfc_lossyear_paths = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif" 
                      for year in range(2013, 2024)]

# Define tmf deforestation year paths
tmf_deforyear_paths = [f"data/jrc_preprocessed/tmf_DeforestationYear_fm_{year}.tif" 
                       for year in range(2013, 2024)]

# Define tmf degradation year paths
tmf_degrayear_paths = [f"data/jrc_preprocessed/tmf_DegradationYear_fm_{year}.tif" 
                       for year in range(2013, 2024)]

# Define tmf annual change paths
tmf_annual_paths = [f"data/jrc_preprocessed/tmf_AnnualChange_{year}_fm.tif" for 
                    year in range(2013,2024)]

# Define tmf transition map path
tmf_trans_file = "data/jrc_preprocessed/tmf_TransitionMap_MainClasses_fm.tif"

# Read tmf transition map
with rasterio.open(tmf_trans_file) as tmf:
    tmf_trans = tmf.read(1)

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Read grnp data
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create aoi
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


# %%
############################################################################


# EXPLORE/PLOT ANNUAL CHANGE MAP


############################################################################
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


# %%
############################################################################


# EXPLORE/PLOT TRANSITION MAP


############################################################################
# Extract unique values
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

# Calculate total number of pixels
total_pixels = np.sum(pixel_counts)

# Calculate proportional pixel counts
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

# Initialize figure
plt.figure(figsize=(10, 8))

# Add bar data
bars = plt.bar(tmf_trans_sorted['Label'], tmf_trans_sorted['Pixel_Percent'])

# Add axes labels
plt.xlabel('Land Cover Classes', fontsize = 16)
plt.ylabel('Pixel Percent (%)', fontsize = 16)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize = 16)

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# EXPLORE TMF DEFORESTATION AND DEGRADATION MAPS (# PIXELS)


############################################################################
# Geodataframes for statistics
gfc_loss_aoi = aoi.copy()
tmf_defor_aoi = aoi.copy()
tmf_degra_aoi = aoi.copy()

# Extract years from GFC lossyear
for year, raster_path in zip(range(2013, 2024), gfc_lossyear_paths):
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        
        # Extract deforestation per polygon
        deforestation_pixels = np.sum(raster_data == year)
        
    gfc_loss_aoi[f"{year}"] = deforestation_pixels    
    
print("Extracted gfc deforestation in AOI")

# Extract years from TMF Deforestation Year
for year, raster_path in zip(range(2013, 2024), tmf_deforyear_paths):
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        
        # Extract deforestation per polygon
        deforestation_pixels = np.sum(raster_data == year)
        
    tmf_defor_aoi[f"{year}"] = deforestation_pixels    
    
print("Extracted tmf deforestation in AOI")

# Extract years from TMF Degradation Year
for year, raster_path in zip(range(2013, 2024), tmf_degrayear_paths):
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        
        # Extract deforestation per polygon
        deforestation_pixels = np.sum(raster_data == year)
        
    tmf_degra_aoi[f"{year}"] = deforestation_pixels    
    
print("Extracted tmf degradation in AOI")

# Extract deforestation pixels per year
gfc_loss_pixels = [gfc_loss_aoi[str(year)].values[0] for year in years]
tmf_defor_pixels = [tmf_defor_aoi[str(year)].values[0] for year in years]
tmf_degra_pixels = [tmf_degra_aoi[str(year)].values[0] for year in years]

# Extract deforestation ha per year
gfc_loss_ha = [x * 0.09 for x in gfc_loss_pixels]
tmf_defor_ha = [x * 0.09 for x in tmf_defor_pixels]
tmf_degra_ha = [x * 0.09 for x in tmf_degra_pixels]

# Combine tmf_defor and tmf_degra for the line plot
tmf_combined_pixels = [defor + degra for defor, degra in zip(tmf_defor_pixels, tmf_degra_pixels)]
tmf_combined_ha = [defor + degra for defor, degra in zip(tmf_defor_ha, tmf_degra_ha)]


# %%
############################################################################


# PLOT TMF DEFORESTATION AND DEGRADATION MAPS (HA)


############################################################################
# Set bar width
bar_width = 0.4
index = np.arange(len(years))

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot GFC Loss AOI bars
ax.bar(index - bar_width/2, gfc_loss_ha, bar_width, color = beige,
       label = 'GFC Tree Cover Loss')

# Plot TMF Deforestation and Degradation stacked bars
ax.bar(index + bar_width/2, tmf_defor_ha, bar_width, color = bluecols[0],
       label = 'TMF Deforestation')

ax.bar(index + bar_width/2, tmf_degra_ha, bar_width, color = bluecols[1],
       label = 'TMF Degradation', bottom = tmf_defor_ha)

# Add lines through the top of GFC Loss AOI bars
ax.plot(index - bar_width/2, gfc_loss_ha, color = brown, 
        label = 'GFC Tree Cover Loss (line)')

# Add lines through the top of TMF Combined bars
ax.plot(index + bar_width/2, tmf_combined_ha, color = brown, linestyle = '--',
        label = 'TMF Combined (line)')

# Customize the plot
ax.set_xlabel('Year', fontsize=16)
ax.set_ylabel('Deforestation Area (ha)', fontsize=16)
ax.set_xticks(index)
ax.set_xticklabels(years, fontsize=16)
ax.tick_params(axis='y', labelsize=16)  
ax.legend(fontsize=16)

# Add gridlines
ax.grid(True, linestyle = '--', alpha = 0.6)

# Display the plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# EXPLORE TMF DEFORESTATION AND DEGRADATION PER REDD+ / NON-REDD+ AREA


############################################################################
# Geodataframes for statistics
gfc_loss_stats = villages.copy()
tmf_defor_stats = villages.copy()
tmf_degra_stats = villages.copy()

# Extract years from GFC lossyear
for year, raster_path in zip(range(2013, 2024), gfc_lossyear_paths):
    
    with rasterio.open(raster_path) as src:
        
        # Extract deforestation per polygon
        deforestation_stats = zonal_stats(gfc_loss_stats, raster_path, 
                                          stats="count", nodata=nodata_val)
    
    # Store stats in a column
    gfc_loss_stats[f"{year}"] = [stat['count'] for stat in deforestation_stats]
    
print("Extracted gfc deforestation in villages")

# Extract years from TMF deforestation
for year, raster_path in zip(range(2013, 2024), tmf_deforyear_paths):
    
    with rasterio.open(raster_path) as src:
        
        # Extract deforestation per polygon
        deforestation_stats = zonal_stats(tmf_defor_stats, raster_path, 
                                          stats="count", nodata=nodata_val)
    
    # Store stats in a column
    tmf_defor_stats[f"{year}"] = [stat['count'] for stat in deforestation_stats]
    
print("Extracted tmf deforestation in villages")

# Extract years from TMF degradation
for year, raster_path in zip(range(2013, 2024), tmf_degrayear_paths):
    
    with rasterio.open(raster_path) as src:
        
        # Extract deforestation per polygon
        deforestation_stats = zonal_stats(tmf_degra_stats, raster_path, 
                                          stats="count", nodata=nodata_val)
    
    # Store stats in a column
    tmf_degra_stats[f"{year}"] = [stat['count'] for stat in deforestation_stats]
    
print("Extracted tmf degradation in villages")

# Extract the columns for years 2013-2023
years = gfc_loss_stats.columns[2:]

# Extract gfc redd statistics
gfc_redd_pixels = gfc_loss_stats.loc[1, years].values

# Extract gfc nonredd statistics
gfc_nonredd_pixels = gfc_loss_stats.loc[0, years].values

# Extract tmf deforestation in redd
tmf_defor_redd_pixels = tmf_defor_stats.loc[1, years].values

# Extract tmf deforestation in nonredd
tmf_defor_nonredd_pixels = tmf_defor_stats.loc[0, years].values

# Extract tmf degradation in redd
tmf_degra_redd_pixels = tmf_degra_stats.loc[1, years].values

# Extract tmf degradation in nonredd
tmf_degra_nonredd_pixels = tmf_degra_stats.loc[0, years].values

# Combine tmf deforestation and degradation in redd
tmf_redd_pixels = [defor + degra for defor, degra in zip(
    tmf_defor_redd_pixels, tmf_degra_redd_pixels)]

# Convert tmf defordegra pixels to array
tmf_redd_pixels = np.array(tmf_redd_pixels, dtype=object)

# Combine tmf deforestation and degradation in nonredd
tmf_nonredd_pixels = [defor + degra for defor, degra in zip(
    tmf_defor_nonredd_pixels, tmf_degra_nonredd_pixels)]

# Convert tmf defordegra pixels to array
tmf_nonredd_pixels = np.array(tmf_nonredd_pixels, dtype=object)

# Convert gfc redd predictions to area proportion
gfc_redd_perc = ((gfc_redd_pixels * 0.09) / redd_area)*100

# Convert gfc nonredd predictions to area proportion
gfc_nonredd_perc = ((gfc_nonredd_pixels * 0.09) / nonredd_area)*100

# Convert tmf redd predictions to area proportion
tmf_redd_perc = ((tmf_redd_pixels * 0.09) / redd_area)*100

# Convert tmf nonredd predictions to area proportion
tmf_nonredd_perc = ((tmf_nonredd_pixels * 0.09) / nonredd_area)*100


# %%
############################################################################


# PLOT TMF DEFORESTATION AND DEGRADATION PER REDD+ / NON-REDD+ AREA


############################################################################
# Initiate figure
plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_redd_perc, color=redd_col1,
         label='GFC Deforestation in REDD+ Villages')

# Plot the pixel values for REDD+ villages
plt.plot(years, tmf_redd_perc, color=redd_col2,
         label='TMF Deforestation and Degradation in REDD+ Villages', 
         linestyle = '--')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, gfc_nonredd_perc, color=nonredd_col1, 
         label='GFC Deforestation in Non-REDD+ Villages')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, tmf_nonredd_perc, color=nonredd_col2,
         label='TMF Deforestation and Degradation in Non-REDD+ Villages',
         linestyle = '--')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('% of Deforestation Pixels Per REDD+/Non-REDD+ Area')

# Add legend
plt.legend()

# Add gridlines
plt.grid(True)

# Adjust tickmarks
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()


































