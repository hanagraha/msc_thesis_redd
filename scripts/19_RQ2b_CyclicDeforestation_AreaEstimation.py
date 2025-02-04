# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:17:56 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import rasterio



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

# Set output directory
val_dir = os.path.join('data', 'validation')

# Set year range
years = range(2013, 2024)

# Define color palatte
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read files in subfolder
def folder_files(folder, suffix):
    
    # Define folder path
    folderpath = os.path.join(val_dir, folder)
    
    # Create empty list to store files
    paths = []

    # Iterate over every item in folder
    for file in os.listdir(folderpath):
        
        # Check if file ends in suffix
        if file.endswith(suffix):
            
            # Create path for file
            filepath = os.path.join(folderpath, file)
            
            # Add file to list
            paths.append(filepath)
    
    return paths

# Define function to read files from list
def list_read(pathlist, suffix, filt = False):
    
    # Create empty dictionary to store outputs
    files = {}
    
    # Iterate over each file in list
    for path in pathlist:
        
        # Read file
        data = pd.read_csv(path)
        
        # Extract file name
        filename = os.path.basename(path)
        
        # Remove suffix from filename
        var = filename.replace(suffix, "")
        
        # Add data to dictionary
        files[var] = data
    
    # If filter is true
    if filt == True:
        
        # Iterate over each read file
        for key in files:
            
            # Subset to only keep years 2013-2023
            files[key] = files[key][(files[key]['year'] >= 2013) & \
                                    (files[key]['year'] <= 2023)]
            
            # Reset index
            files[key] = files[key].reset_index(drop = True)
    
    return files

# Define function to extract area from map
def map_area(path):
    
    # Read raster
    with rasterio.open(path) as rast:
        
        # Read data
        data = rast.read()
        
        # Calculate number of non-na pixels
        pixels = np.sum(data != nodata_val)
        
        # Convert pixels to ha
        ha = pixels * 0.09
        
    return data, pixels, ha

# Define function to read csv validation data
def csv_read(datapath):
    
    # Read validation data
    data = pd.read_csv(datapath, delimiter = ",", index_col = 0)
    
    # Convert csv geometry to WKT
    data['geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
    
    # Convert dataframe to geodataframe
    data = gpd.GeoDataFrame(data, geometry = 'geometry', crs="EPSG:32629")
    
    return data

# Read validation data
valdata = csv_read("data/validation/validation_datasets/validation_points_2013_2023_780_nobuffer.csv")

# Read protocol c statistics (no buffer)
protc_statpaths = folder_files("val_protc", "stehmanstats.csv")
protc_stats = list_read(protc_statpaths, "_stehmanstats.csv", filt = True)

# Calculate aoi map area
stratmap, total_pix, total_ha = map_area("data/validation/stratification_maps/stratification_layer_nogrnp.tif")

# Calculate redd+ map area
strat_redd, redd_pix, redd_ha = map_area("data/validation/stratification_maps/stratification_layer_redd.tif")

# Calculate nonredd+ map area
strat_nonredd, nonredd_pix, nonredd_ha = map_area("data/validation/stratification_maps/stratification_layer_nonredd.tif")

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k').reset_index()

# Create redd+ polygon
redd_union = gpd.GeoSeries(villages.loc[1].geometry).unary_union

# Create nonredd+ polygon
nonredd_union = gpd.GeoSeries(villages.loc[0].geometry).unary_union

# Filter points within REDD+ multipolygon
points_redd = valdata[valdata.geometry.within(redd_union)]

# Filter points within non-REDD+ multipolygon
points_nonredd = valdata[valdata.geometry.within(nonredd_union)]


# %%
############################################################################


# CALCULATE PROPORTIONAL AREAS (AOI)


############################################################################
"""
Y = (strata_size(proportion of class in strata) + ....) / map size

eg. [40,000(0.20) + 30,000(0.00) + 20,000(0.50) + 10,000(0.20)]/100,000

where strata 1, 2, 3, 4 have sizes 40,0000, 30,000, 20,000, and 10,000
sample size per strata = 10 points
occurence of class C in strata 1 = 2, 2 = 0, 3 = 5, 4 = 2
"""

# Define function to calculate proportional areas
def steh_area(valdata, stratmap, deforlist, pix, ha):
    
    # Calculate number of pixels per strata
    pixvals, pixcounts = np.unique(stratmap, return_counts = True)

    # Create dataframe
    strata_size = pd.DataFrame({'strata': pixvals[:-1],
                                'size': pixcounts[:-1]})

    # Create empty list to hold deforestation area 
    year_defor = pd.DataFrame(index=strata_size['strata'])

    # Iterate over each year
    for year in years:
        
        # Create empty list to hold year deforestation area
        strata_defor = []
        
        # Iterate over each strata
        for idx, row in strata_size.iterrows():
            
            # Extract strata number
            strata = row['strata']
            
            # Extract strata size
            size = row['size']
            
            # Subset validation data for that strata
            data = valdata[valdata['strata'] == strata]
            
            # Count sum of deforestation in that year
            defor = data[deforlist].eq(year).sum().sum()
            
            # Calculate class proportion in that strata
            cp = defor / len(data)
            
            # Multiply by strata size
            area = (cp * size) / pix
            
            # Add deforestation area to list
            strata_defor.append(area)
            
        # Add the list as a column in the DataFrame
        year_defor[year] = strata_defor

    # Take the sum per year
    total_defor = year_defor.sum(axis=0) * ha
    
    return total_defor

# Estimate deforestation for the first year
first_defor = steh_area(valdata, stratmap, ['defor1'], total_pix, total_ha)

# Estimate deforestation for second year
second_defor = steh_area(valdata, stratmap, ['defor2'], total_pix, total_ha)

# Estimate deforestation for third year
third_defor = steh_area(valdata, stratmap, ['defor3'], total_pix, total_ha)

# Estimate deforestation for all years
all_defor = steh_area(valdata, stratmap, ['defor1', 'defor2', 'defor3'], 
                      total_pix, total_ha)

# Extract area calculated by stehman
stehman_area = protc_stats['protc_gfc']['area'] * total_ha
stehman_area.index = pd.Index(years)


# %%
############################################################################


# CALCULATE PROPORTIONAL AREAS (REDD)


############################################################################
# Calculate redd area deforestation (first)
first_defor_redd = steh_area(points_redd, strat_redd, ['defor1'], redd_pix, redd_ha)

# Calculate redd area deforestation (all)
all_defor_redd = steh_area(points_redd, strat_redd, ['defor1', 'defor2', 'defor3'], 
                           redd_pix, redd_ha)

# Calculate nonredd area deforestation (first)
first_defor_nonredd = steh_area(points_nonredd, strat_nonredd, ['defor1'], 
                                nonredd_pix, nonredd_ha)

# Calculate nonredd area deforestation (all)
all_defor_nonredd = steh_area(points_nonredd, strat_nonredd, ['defor1', 'defor2', 'defor3'], 
                              nonredd_pix, nonredd_ha)

# Extract redd area calculated by stehman
steh_redd = protc_stats['protc_gfc_redd']['area'] * redd_ha
steh_redd.index = pd.Index(years)

# Extract nonredd area calculated by stehman
steh_nonredd = protc_stats['protc_gfc_nonredd']['area'] * nonredd_ha
steh_nonredd.index = pd.Index(years)


# %%
############################################################################


# CALCULATE STANDARD ERROR (STEHMAN, FIRST DEFORESTATION)


############################################################################
def calc_eea(data_dict):
    
    # Create a copy of the input dictionary
    eea_dict = data_dict.copy()
    
    # Iterate over each dictionary iem
    for key, value in eea_dict.items():
        
        # If the key is for nonredd areas
        if "nonredd" in key:
            
            # Calculate error adjsuted area
            area = value['area'] * nonredd_ha
            
            # Calculate area standard error
            error = value['se_a'] * nonredd_ha
            
        # If the key is for redd areas
        elif "redd" in key:
            
            # Calculate error adjusted area
            area = value['area'] * redd_ha
            
            # Calculate area standard error
            error = value['se_a'] * redd_ha 
            
        # If the key is for the whole area
        else: 
    
            # Calculate error adjusted area
            area = value['area'] * total_ha
            
            # Extract area standard error
            error = value['se_a'] * total_ha
        
        # Calculate 95% confidence interval
        ci95 = 1.96 * error
        
        # Calculate 50% confidence interval
        ci50 = 0.67 * error
        
        # Add error adjusted area to df
        value['eea'] = area
        
        # Add 95ci to df
        value['ci95'] = ci95
        
        # Add 50ci to df
        value['ci50'] = ci50
    
    return eea_dict

# Calculate eea and ci for prot b
protc_eea = calc_eea(protc_stats)


# %%
############################################################################


# PLOT REDD DEFORESTATION


############################################################################
# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize = (18,6))

# Calculate the common y-axis limits
min_y = min((protc_eea["protc_gfc_redd"]['eea'] - 
             protc_eea["protc_gfc_redd"]['ci50']).min(), 
            (protc_eea["protc_gfc_nonredd"]['eea'] - 
             protc_eea["protc_gfc_nonredd"]['ci50']).min())
max_y = max((protc_eea["protc_gfc_redd"]['eea'] + 
             protc_eea["protc_gfc_redd"]['ci50']).max(), 
            (protc_eea["protc_gfc_nonredd"]['eea'] + 
             protc_eea["protc_gfc_nonredd"]['ci50']).max())

# Add padding for better visualization
padding = 0.05 * (max_y - min_y)
min_y -= padding
max_y += padding

# Set y-axis limits for both axes
axes[0].set_ylim(min_y, max_y)
axes[1].set_ylim(min_y, max_y)

# PLOT 1: REDD DEFOR AREA

# Create 95% ci rectangle
axes[0].fill_between(
    years, 
    protc_eea["protc_gfc_redd"]['eea'][0] - protc_eea["protc_gfc_redd"]['ci95'][0],
    protc_eea["protc_gfc_redd"]['eea'][0] + protc_eea["protc_gfc_redd"]['ci95'][0],
    color = bluecols[1],
    alpha = 0.2,
    label = "95% confidence interval"
    )

# Create 50% ci rectangle
axes[0].fill_between(
    years, 
    protc_eea["protc_gfc_redd"]['eea'][0] - protc_eea["protc_gfc_redd"]['ci50'][0],
    protc_eea["protc_gfc_redd"]['eea'][0] + protc_eea["protc_gfc_redd"]['ci50'][0],
    color = bluecols[1],
    alpha = 0.3,
    label = "50% confidence interval"
    )

# Plot redd defor
axes[0].errorbar(
    years,
    protc_eea["protc_gfc_redd"]['eea'],
    yerr = protc_eea["protc_gfc_redd"]['ci50'],
    fmt="-o",
    capsize = 5,
    color = bluecols[0],
    label = "REDD+ Deforestation"
)

# Add all deforestation data
axes[0].plot(years, all_defor_redd, linestyle='--', color = bluecols[2], 
         label='All Deforestation in REDD+')

# Add x-axis tick marks
axes[0].set_xticks(years)

# Adjust fontsize of tick labels
axes[0].tick_params(axis='both', which='major', labelsize=14)

# Add axes labels
axes[0].set_xlabel("Year", fontsize=14)
axes[0].set_ylabel("Error-Adjusted Deforestation Area (ha)", fontsize=14)

# Add a title and legend
axes[0].legend(fontsize=16, loc = "upper right")

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# PLOT 2: NONREDD DEFOR AREA

# Create 95% ci rectangle
axes[1].fill_between(
    years, 
    protc_eea["protc_gfc_nonredd"]['eea'][0] - protc_eea["protc_gfc_nonredd"]['ci95'][0],
    protc_eea["protc_gfc_nonredd"]['eea'][0] + protc_eea["protc_gfc_nonredd"]['ci95'][0],
    color = bluecols[1],
    alpha = 0.2,
    label = "95% confidence interval"
    )

# Create 50% ci rectangle
axes[1].fill_between(
    years, 
    protc_eea["protc_gfc_nonredd"]['eea'][0] - protc_eea["protc_gfc_nonredd"]['ci50'][0],
    protc_eea["protc_gfc_nonredd"]['eea'][0] + protc_eea["protc_gfc_nonredd"]['ci50'][0],
    color = bluecols[1],
    alpha = 0.3,
    label = "50% confidence interval"
    )

# Plot nonredd defor
axes[1].errorbar(
    years,
    protc_eea["protc_gfc_nonredd"]['eea'],
    yerr = protc_eea["protc_gfc_nonredd"]['ci50'],
    fmt="-o",
    capsize = 5,
    color = bluecols[0],
    label = "Non-REDD+ Deforestation"
)

# Add all deforestation data
axes[1].plot(years, all_defor_nonredd, linestyle='--', color = bluecols[2], 
         label='All Deforestation in Non-REDD+')

# Add x-axis tick marks
axes[1].set_xticks(years)

# Adjust font size of tick labels
axes[1].tick_params(axis='both', which='major', labelsize=14)

# Add axes labels
axes[1].set_xlabel("Year", fontsize=14)
# axes[1].set_ylabel("Error-Adjusted Deforestation Area (ha)", fontsize=14)

# Remove y axis
axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)

# Add a title and legend
axes[1].legend(fontsize=16, loc = "upper right")

# Add gridlines
axes[1].grid(linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()



# %%
############################################################################


# PLOT MULTIPLE DEFORESTATION EVENTS


############################################################################
# Initialize figure
plt.figure(figsize=(10, 6))

# Add all deforestation data
plt.plot(years, all_defor, linestyle='--', color = bluecols[2], 
         label='All Deforestation')

# Add first deforestation data
plt.errorbar(
    years,
    protc_eea['protc_gfc']['eea'],
    yerr = protc_eea['protc_gfc']['ci50'],
    fmt="-o",
    capsize = 5,
    color = bluecols[0],
    label = "First Deforestation"
)

# Create 95% ci rectangle
plt.fill_between(
    years, 
    protc_eea['protc_gfc']['eea'][0] - protc_eea['protc_gfc']['ci95'][0],
    protc_eea['protc_gfc']['eea'][0] + protc_eea['protc_gfc']['ci95'][0],
    color = bluecols[1],
    alpha = 0.2,
    label = "95% confidence interval"
    )

# Create 95% ci rectangle
plt.fill_between(
    years, 
    protc_eea['protc_gfc']['eea'][0] - protc_eea['protc_gfc']['ci50'][0],
    protc_eea['protc_gfc']['eea'][0] + protc_eea['protc_gfc']['ci50'][0],
    color = bluecols[1],
    alpha = 0.3,
    label = "50% confidence interval"
    )

# Add axes labels
plt.xlabel("Year", fontsize = 16)
plt.ylabel("Deforestation Area (ha)", fontsize = 16)

# Get handles if plot items
handles, labels = plt.gca().get_legend_handles_labels()

# Define order of legend items
order = [0, 3, 1, 2] 

# Add legend with manual item ordering
plt.legend([handles[i] for i in order], [labels[i] for i in order], 
           fontsize=14, loc="upper right")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Adjust tickmarks
plt.xticks(years, fontsize = 14)
plt.yticks(fontsize = 14)

# Show the plot
plt.show()


# %%
############################################################################


# CALCULATE STANDARD ERROR FOR AREA ESTIMATION


############################################################################
"""
V(Y) = (1/N2) * (strata_size^2 * ((1-proportion of class in strata / sample size)))
"""
# Extract gfc area errors
gfc_errors = protd_stats['protd_gfc']['se_a']

# Extract tmf area errors
tmf_errors = protd_stats['protd_tmf']['se_a']

# Extract se area errors
se_errors = protd_stats['protd_se']['se_a']


# Calculate sample variance

# Calculate number of pixels per strata
pixvals, pixcounts = np.unique(stratmap, return_counts = True)

# Create dataframe
strata_size = pd.DataFrame({'strata': pixvals[:-1],
                            'size': pixcounts[:-1]})

# Iterate over each strata
for idx, row in strata_size.iterrows():
    
    # Extract strata number
    strata = row['strata']
    
    # Extract strata size
    size = row['size']
    
    # Subset validation data for that strata
    data = valdata[valdata['strata'] == strata]
    
    # Separate all deforestation events
    defor = 
    
    # Calculate indicator row (correct/incorrect classification)
    if data['gfc'] == data['defor1']
    
    # Define sample mean of strata
    mean = 
    
    # Iterate over each point in strata
    for point in data:
        
        # Define y_u indicator variable
        y_u = 






