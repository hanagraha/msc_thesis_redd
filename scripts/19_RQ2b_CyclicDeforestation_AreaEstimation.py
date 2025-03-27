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
from matplotlib.pyplot import savefig
import numpy as np
import rasterio
from rasterio.mask import mask



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
gfc_col = "#820300"  # Darker Red
tmf_col = "#4682B4"  # Darker Blue - lighter

# Define pixel area
pixel_area = 0.09



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
valdata = csv_read("data/validation/validation_datasets/validation_points_780.csv")

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

# Define multiyear gfc lossyear path
gfc_path = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Define multiyear tmf defordegra path
tmf_path = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# Define multiyear se path
se_path = "data/intermediate/gfc_tmf_sensitive_early.tif"


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

# Calculate missed deforestation
miss_defor = all_defor - first_defor

# %%
############################################################################


# CALCULATE PROPORTIONAL AREAS (REDD)


############################################################################
# Calculate redd area deforestation (first)
first_defor_redd = steh_area(points_redd, strat_redd, ['defor1'], redd_pix, redd_ha)

# Calculate redd area deforestation (all)
all_defor_redd = steh_area(points_redd, strat_redd, ['defor1', 'defor2', 'defor3'], 
                           redd_pix, redd_ha)

recur_defor_redd = steh_area(points_redd, strat_redd, ['defor2', 'defor3'], 
                             redd_pix, redd_ha)

# Calculate nonredd area deforestation (first)
first_defor_nonredd = steh_area(points_nonredd, strat_nonredd, ['defor1'], 
                                nonredd_pix, nonredd_ha)

# Calculate nonredd area deforestation (all)
all_defor_nonredd = steh_area(points_nonredd, strat_nonredd, ['defor1', 'defor2', 'defor3'], 
                              nonredd_pix, nonredd_ha)

recur_defor_nonredd = steh_area(points_nonredd, strat_nonredd, ['defor2', 'defor3'], 
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


# PLOT REDD DEFORESTATION (HA)


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
    label = "REDD+ Deforestation (First)",
    linewidth = 2
)

# Add all deforestation data
axes[0].plot(years, all_defor_redd, linestyle='--', color = bluecols[2], 
         label='REDD+ Deforestation (All)', linewidth = 2)

# Add x-axis tick marks
axes[0].set_xticks(years)

# Adjust fontsize of tick labels
axes[0].tick_params(axis='both', which='major', labelsize=16)

# Add axes labels
axes[0].set_xlabel("Year", fontsize=14)
axes[0].set_ylabel("Error-Adjusted Deforestation Area (ha)", fontsize=16)

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
    label = "Non-REDD+ Deforestation (First)",
    linewidth = 2
)

# Add all deforestation data
axes[1].plot(years, all_defor_nonredd, linestyle='--', color = bluecols[2], 
         label='Non-REDD+ Deforestation (All)', linewidth = 2)

# Add x-axis tick marks
axes[1].set_xticks(years)

# Adjust font size of tick labels
axes[1].tick_params(axis='both', which='major', labelsize=16)

# Add axes labels
axes[1].set_xlabel("Year", fontsize=16)
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


# PLOT REDD DEFORESTATION (%)


############################################################################
# Define proportional error estimations (stehman)
redd_eea_prop = (protc_eea["protc_gfc_redd"]['eea'] / redd_ha)*100
redd_ci50_prop = (protc_eea["protc_gfc_redd"]['ci50'] / redd_ha)*100
redd_ci95_prop = (protc_eea["protc_gfc_redd"]['ci95'] / redd_ha)*100
nonredd_eea_prop = (protc_eea["protc_gfc_nonredd"]['eea'] / nonredd_ha)*100
nonredd_ci95_prop = (protc_eea["protc_gfc_nonredd"]['ci95'] / nonredd_ha)*100
nonredd_ci50_prop = (protc_eea["protc_gfc_nonredd"]['ci50'] / nonredd_ha)*100

# Define proportional error estimates (all defor)
all_defor_redd_prop = (all_defor_redd / redd_ha)*100
all_defor_nonredd_prop = (all_defor_nonredd / nonredd_ha)*100

# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize = (18,6))

# Calculate the common y-axis limits
min_y = min((redd_eea_prop - redd_ci50_prop).min(), 
            (nonredd_eea_prop - nonredd_ci50_prop).min())
max_y = max((redd_eea_prop + redd_ci50_prop).max(), 
            (nonredd_eea_prop + nonredd_ci50_prop).max())

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
    redd_eea_prop[0] - redd_ci95_prop[0],
    redd_eea_prop[0] + redd_ci95_prop[0],
    color = bluecols[1],
    alpha = 0.2,
    label = "95% confidence interval"
    )

# Create 50% ci rectangle
axes[0].fill_between(
    years, 
    redd_eea_prop[0] - redd_ci50_prop[0],
    redd_eea_prop[0] + redd_ci50_prop[0],
    color = bluecols[1],
    alpha = 0.3,
    label = "50% confidence interval"
    )

# Plot redd defor
axes[0].errorbar(
    years,
    redd_eea_prop,
    yerr = redd_ci50_prop,
    fmt="-o",
    capsize = 5,
    color = bluecols[0],
    label = "REDD+ Deforestation (First)",
    linewidth = 2
)

# Add all deforestation data
axes[0].plot(years, all_defor_redd_prop, linestyle='--', color = bluecols[2], 
         label='REDD+ Deforestation (All)', linewidth = 2)

# Add x-axis tick marks
axes[0].set_xticks(years)

# Adjust fontsize of tick labels
axes[0].tick_params(axis='both', which='major', labelsize=14)

# Add axes labels
axes[0].set_xlabel("Year", fontsize=14)
axes[0].set_ylabel("Error-Adjusted Deforestation Area (%)", fontsize=16)

# Add a title and legend
axes[0].legend(fontsize=16, loc = "upper right")

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# PLOT 2: NONREDD DEFOR AREA

# Create 95% ci rectangle
axes[1].fill_between(
    years, 
    nonredd_eea_prop[0] - nonredd_ci95_prop[0],
    nonredd_eea_prop[0] + nonredd_ci95_prop[0],
    color = bluecols[1],
    alpha = 0.2,
    label = "95% confidence interval"
    )

# Create 50% ci rectangle
axes[1].fill_between(
    years, 
    nonredd_eea_prop[0] - nonredd_ci50_prop[0],
    nonredd_eea_prop[0] + nonredd_ci50_prop[0],
    color = bluecols[1],
    alpha = 0.3,
    label = "50% confidence interval"
    )

# Plot nonredd defor
axes[1].errorbar(
    years,
    nonredd_eea_prop,
    yerr = nonredd_ci50_prop,
    fmt="-o",
    capsize = 5,
    color = bluecols[0],
    label = "Non-REDD+ Deforestation (First)",
    linewidth = 2
)

# Add all deforestation data
axes[1].plot(years, all_defor_nonredd_prop, linestyle='--', color = bluecols[2], 
         label='Non-REDD+ Deforestation (All)', linewidth = 2)

# Add x-axis tick marks
axes[1].set_xticks(years)

# Adjust font size of tick labels
axes[1].tick_params(axis='both', which='major', labelsize=14)

# Add axes labels
axes[1].set_xlabel("Year", fontsize=14)

# Remove y axis
axes[1].tick_params(axis='y', which='both', left=False, labelleft=False)

# Add a title and legend
axes[1].legend(fontsize=16, loc = "upper right")

# Add gridlines
axes[1].grid(linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()

# Define plot path
savepath = "data/plots/Presentation Plots/rq4_redd_deforarea.png"

# Download plot
savefig(savepath, transparent=True)

plt.show()


# %%
############################################################################


# PLOT VALIDATION AND PREDICTED DEFORESTATION


############################################################################
# Define function to extract multiyear statistics
def multiyear_atts(rastpath, yearrange, perc = False):
    
    # Read array
    with rasterio.open(rastpath) as rast:
        
        # Mask to redd geometry
        redd_image, redd_transform = mask(rast, [redd_union], crop = True)
        
        # Mask to nonredd geometry
        nonredd_image, nonredd_transform = mask(rast, [nonredd_union], crop = True)
    
    # Extract unique values and pixel counts (redd)
    redd_values, redd_counts = np.unique(redd_image, return_counts=True)
    
    # Extract unique values and pixel counts (nonredd)
    nonredd_values, nonredd_counts = np.unique(nonredd_image, return_counts=True)
    
    # Create dataframe with unique values and pixel counts
    attributes = pd.DataFrame({"Year": redd_values, 
                               "REDD+": redd_counts / redd_pix,
                               "Non-REDD+": nonredd_counts / nonredd_pix})
    
    # Filter only for attributes within yearrange
    filt_atts = attributes[(attributes['Year'] >= min(yearrange)) & \
                           (attributes['Year'] <= max(yearrange))]
    
    # Reset index
    filt_atts = filt_atts.reset_index(drop=True)
    
    # If precentage set to true
    if perc == True:
        
        # Convert proportions to percentages
        filt_atts["REDD+"] = filt_atts["REDD+"] * 100
        filt_atts["Non-REDD+"] = filt_atts["Non-REDD+"] * 100
    
    return filt_atts

# Extract annual deforestation data from gfc
gfc_zonal = multiyear_atts(gfc_path, years, perc = True)
print("GFC REDD+ Change:", gfc_zonal['REDD+'][10]-gfc_zonal['REDD+'][0])
print("GFC Non-REDD+ Change:", gfc_zonal['Non-REDD+'][10]-gfc_zonal['Non-REDD+'][0])
print("GFC Diff in Diff:", (gfc_zonal['REDD+'][10]-gfc_zonal['REDD+'][0]) - (gfc_zonal['Non-REDD+'][10]-gfc_zonal['Non-REDD+'][0]))
print()

# Extract annual deforestation data from tmf
tmf_zonal = multiyear_atts(tmf_path, years, perc = True)
print("TMF REDD+ Change:", tmf_zonal['REDD+'][10]-tmf_zonal['REDD+'][0])
print("TMF Non-REDD+ Change:", tmf_zonal['Non-REDD+'][10]-tmf_zonal['Non-REDD+'][0])
print("TMF Diff in Diff:", (tmf_zonal['REDD+'][10]-tmf_zonal['REDD+'][0]) - (tmf_zonal['Non-REDD+'][10]-tmf_zonal['Non-REDD+'][0]))
print()

# Extract annual deforestation data from se
se_zonal = multiyear_atts(se_path, years, perc = True)
print("SE REDD+ Change:", se_zonal['REDD+'][10]-se_zonal['REDD+'][0])
print("SE Non-REDD+ Change:", se_zonal['Non-REDD+'][10]-se_zonal['Non-REDD+'][0])
print("SE Diff in Diff:", (se_zonal['REDD+'][10]-se_zonal['REDD+'][0]) - (se_zonal['Non-REDD+'][10]-se_zonal['Non-REDD+'][0]))
print()

# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize = (18,6))

min_y = 0
max_y = 12

# Set y-axis limits for both axes
axes[0].set_ylim(min_y, max_y)
axes[1].set_ylim(min_y, max_y)

# PLOT 1: REDD DEFOR AREA

# Plot redd defor
axes[0].errorbar(
    years,
    redd_eea_prop,
    yerr = redd_ci50_prop,
    fmt="-o",
    capsize = 5,
    color = bluecols[2],
    label = "Error-Adjusted REDD+ Deforestation (First)", 
    linewidth = 2
)

# Add gfc deforestation data
axes[0].plot(years, gfc_zonal['REDD+'], linestyle='-', color = bluecols[0], 
         label='GFC REDD+ Deforestation', linewidth = 2)

# Add tmf deforestation data
axes[0].plot(years, tmf_zonal['REDD+'], linestyle='-', color = bluecols[1], 
         label='TMF REDD+ Deforestation', linewidth = 2)

# Add se deforestation data
axes[0].plot(years, se_zonal['REDD+'], linestyle='-', color = "#A0C878", 
         label='SE REDD+ Deforestation', linewidth = 2)

# Add x-axis tick marks
axes[0].set_xticks(years)

# Adjust fontsize of tick labels
axes[0].tick_params(axis='both', which='major', labelsize=14)

# Add axes labels
axes[0].set_xlabel("Year", fontsize=14)
axes[0].set_ylabel("Proportional Deforestation Area (%)", fontsize=16)

# Add a title and legend
axes[0].legend(fontsize=16, loc = "upper right")

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# PLOT 2: NONREDD DEFOR AREA

# Plot nonredd defor
axes[1].errorbar(
    years,
    nonredd_eea_prop,
    yerr = nonredd_ci50_prop,
    fmt="-o",
    capsize = 5,
    color = bluecols[2],
    label = "Error-Adjusted Non-REDD+ Deforestation (First)", 
    linewidth = 2
)

# Add gfc deforestation data
axes[1].plot(years, gfc_zonal['Non-REDD+'], linestyle='-', color = bluecols[0], 
         label='GFC Non-REDD+ Deforestation', linewidth = 2)

# Add tmf deforestation data
axes[1].plot(years, tmf_zonal['Non-REDD+'], linestyle='-', color = bluecols[1], 
         label='TMF Non-REDD+ Deforestation', linewidth = 2)

# Add se deforestation data
axes[1].plot(years, se_zonal['Non-REDD+'], linestyle='-', color = "#A0C878", 
         label='SE Non-REDD+ Deforestation', linewidth = 2)

# Add x-axis tick marks
axes[1].set_xticks(years)

# Adjust font size of tick labels
axes[1].tick_params(axis='both', which='major', labelsize=14)

# Add axes labels
axes[1].set_xlabel("Year", fontsize=14)

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


# PLOT MULTIPLE VALIDATION AND PREDICTED DEFORESTATION


############################################################################
# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize = (18,6))

min_y = 0
max_y = 12

# Set y-axis limits for both axes
axes[0].set_ylim(min_y, max_y)
axes[1].set_ylim(min_y, max_y)

# PLOT 1: REDD DEFOR AREA

# Add all deforestation data
axes[0].plot(years, all_defor_redd_prop, linestyle='--', color = bluecols[2], 
         label='Error-Adjusted REDD+ Deforestation (All)', linewidth = 2)

# Add gfc deforestation data
axes[0].plot(years, gfc_zonal['REDD+'], linestyle='-', color = bluecols[0], 
         label='GFC REDD+ Deforestation', linewidth = 2)

# Add tmf deforestation data
axes[0].plot(years, tmf_zonal['REDD+'], linestyle='-', color = bluecols[1], 
         label='TMF REDD+ Deforestation', linewidth = 2)

# Add se deforestation data
axes[0].plot(years, se_zonal['REDD+'], linestyle='-', color = "#A0C878", 
         label='SE REDD+ Deforestation', linewidth = 2)

# Add x-axis tick marks
axes[0].set_xticks(years)

# Adjust fontsize of tick labels
axes[0].tick_params(axis='both', which='major', labelsize=16)

# Add axes labels
axes[0].set_xlabel("Year", fontsize=16)
axes[0].set_ylabel("Proportional Deforestation Area (%)", fontsize=16)

# Add a title and legend
axes[0].legend(fontsize=16, loc = "upper right")

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# PLOT 2: NONREDD DEFOR AREA

# Add all deforestation data
axes[1].plot(years, all_defor_nonredd_prop, linestyle='--', color = bluecols[2], 
         label='Error-Adjusted Non-REDD+ Deforestation (All)', linewidth = 2)

# Add gfc deforestation data
axes[1].plot(years, gfc_zonal['Non-REDD+'], linestyle='-', color = bluecols[0], 
         label='GFC Non-REDD+ Deforestation', linewidth = 2)

# Add tmf deforestation data
axes[1].plot(years, tmf_zonal['Non-REDD+'], linestyle='-', color = bluecols[1], 
         label='TMF Non-REDD+ Deforestation', linewidth = 2)

# Add se deforestation data
axes[1].plot(years, se_zonal['Non-REDD+'], linestyle='-', color = "#A0C878", 
         label='SE Non-REDD+ Deforestation', linewidth = 2)

# Add x-axis tick marks
axes[1].set_xticks(years)

# Adjust font size of tick labels
axes[1].tick_params(axis='both', which='major', labelsize=16)

# Add axes labels
axes[1].set_xlabel("Year", fontsize=16)

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


# PLOT MULTIPLE VALIDATION AND PREDICTED DEFORESTATION


############################################################################
# Initialize figure
plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_zonal['REDD+'], color=bluecols[0], linewidth = 2,
         label='GFC Deforestation in REDD+ Villages')

# Plot the pixel values for REDD+ villages
plt.plot(years, tmf_zonal['REDD+'], color=bluecols[1], linewidth = 2,
         label='TMF Deforestation and Degradation in REDD+ Villages')

# Plot the multiple deforestation REDD+
plt.plot(years, all_defor_redd_prop, linewidth = 2, color = bluecols[2], 
         label='Error-Adjusted REDD+ Deforestation')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, gfc_zonal['Non-REDD+'], color=bluecols[0], linewidth = 2, 
         label='GFC Deforestation in Non-REDD+ Villages', linestyle = '--')

# Plot the pixel values for non-REDD+ villages
plt.plot(years, tmf_zonal['Non-REDD+'], color=bluecols[1], linewidth = 2,
         label='TMF Deforestation and Degradation in Non-REDD+ Villages',
         linestyle = '--')

# Plot the multiple deforestation non-REDD+
plt.plot(years, all_defor_nonredd_prop, linestyle='--', color = bluecols[2], 
         label='Error-Adjusted Non-REDD+ Deforestation', linewidth = 2)

# Add labels and title
plt.xlabel('Year', fontsize = 14)
plt.ylabel('Proportional Deforestation Area (%)', fontsize = 14)

# Add x tickmarks
plt.xticks(years, fontsize = 14)

# Edit y tickmark fontsize
plt.yticks(fontsize = 14)

# Add legend
plt.legend(fontsize=13)

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

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

# Define plot path
savepath = "data/plots/Presentation Plots/rq3_recur_defor.png"

# Download plot
savefig(savepath, transparent=True)

# Show the plot
plt.show()


# %%
############################################################################


# PLOT GFC AND EAD FOR CANVA


############################################################################
# Calculate difference in redd-nonredd gfc
gfc_rdif = gfc_zonal['REDD+'] - gfc_zonal['Non-REDD+']

# Calculate difference in redd-nonredd ead
ead_rdif = all_defor_redd_prop - all_defor_nonredd_prop

# Calculate redd 95% confidence interval
redd_ci95 = protc_stats['protc_gfc_redd']['se_a'] * 100 * 1.96

# Calculate nonredd 95% confidence interval
nonredd_ci95 = protc_stats['protc_gfc_nonredd']['se_a'] * 100 * 1.96

# Calculate redd 50% confidence interval
redd_ci50 = protc_stats['protc_gfc_redd']['se_a'] * 100 * 0.67

# Calculate nonredd 50% confidence interval
nonredd_ci50 = protc_stats['protc_gfc_nonredd']['se_a'] * 100 * 0.67


# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot 1: line of redd/nonredd, gfc/tmf defor

# Plot the pixel values for REDD+ villages
axes[0].plot(years, gfc_zonal['REDD+'], color = tmf_col, linewidth = 2, 
             label = 'GFC Deforestation in REDD+ Villages')

# Plot the multiple deforestation REDD+
# axes[0].plot(years, all_defor_redd_prop, color = gfc_col, linewidth = 2, 
#              label = 'Error-Adjusted REDD+ Deforestation')

axes[0].errorbar(years, all_defor_redd_prop, yerr = redd_ci50, fmt="-o", 
                 capsize = 5, color = gfc_col, label = 
                 "Error-Adjusted REDD+ Deforestation")

# Plot the pixel values for non-REDD+ villages
axes[0].plot(years, gfc_zonal['Non-REDD+'], color = tmf_col, linewidth = 2, 
             label='GFC Deforestation in Non-REDD+ Villages', linestyle = '--')

# Plot the multiple deforestation non-REDD+
# axes[0].plot(years, all_defor_nonredd_prop, color = gfc_col, linewidth = 2,
#              label='Error-Adjusted Non-REDD+ Deforestation',  linestyle='--')

axes[0].errorbar(years, all_defor_nonredd_prop, yerr = nonredd_ci50, fmt="-o", 
                 capsize = 5, color = gfc_col, label = 
                 "Error-Adjusted Non-REDD+ Deforestation", linestyle = "--")

# Add x axis label
axes[0].set_xlabel('Year', fontsize=17)

# Add y axis label
axes[0].set_ylabel('Proportional Deforestation Area (%)', fontsize=17)

# Add tickmarks
axes[0].set_xticks(years)
axes[0].tick_params(axis='both', labelsize=16)

# Add legend
axes[0].legend(fontsize = 14, loc = 'upper right')

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# Plot 2: Difference in redd/nonredd

# Define bar width
bar_width = 0.4

# Define x values for bar chart
x = range(len(years))  # Assuming `x` represents years as indices

# Plot gfc redd differences
axes[1].bar(x, gfc_rdif*100, width = bar_width, label = 'GFC Deforestation', 
            color = tmf_col, align = 'center')

# Plot multiple deforestation redd differences
axes[1].bar([i + bar_width for i in x], ead_rdif*100, width = bar_width, 
            label = 'Error-Adjusted Deforestation (All)', color = gfc_col, 
            align = 'center')

# Add x axis label
axes[1].set_xlabel('Year', fontsize = 17)

# Add y axis label
axes[1].set_ylabel('Deforestation Area Difference (%)', fontsize=17)

# Add x tickmarks
axes[1].set_xticks([i + bar_width / 2 for i in x])  
axes[1].set_xticklabels(years)

# Edit ticklabel fontsize
axes[1].tick_params(axis='both', labelsize = 16)

# Add gridlines
axes[1].grid(axis='y', which='major', linestyle='--', alpha = 0.6)
axes[1].grid(axis='x', linestyle = "--", alpha = 0.6)

# Add legend
axes[1].legend(fontsize=16)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# PLOT GFC AND EAD FOR CANVA (SINGLE)


############################################################################
# Initialize figure with subplots
plt.figure(figsize=(10, 6))

# Plot the pixel values for REDD+ villages
plt.plot(years, gfc_zonal['REDD+'], color = tmf_col, linewidth = 2, 
             label = 'GFC Deforestation in REDD+ Villages')

# Plot the multiple deforestation REDD+
plt.errorbar(years, all_defor_redd_prop, yerr = redd_ci50, fmt="-o", 
                 capsize = 5, color = gfc_col, label = 
                 "Error-Adjusted REDD+ Deforestation")

# Plot the pixel values for non-REDD+ villages
plt.plot(years, gfc_zonal['Non-REDD+'], color = tmf_col, linewidth = 2, 
             label='GFC Deforestation in Non-REDD+ Villages', linestyle = '--')

# Plot the multiple deforestation non-REDD+
plt.errorbar(years, all_defor_nonredd_prop, yerr = nonredd_ci50, fmt="-o", 
                 capsize = 5, color = gfc_col, label = 
                 "Error-Adjusted Non-REDD+ Deforestation", linestyle = "--")

# Add x axis label
plt.xlabel('Year', fontsize=17)

# Add y axis label
plt.ylabel('Proportional Deforestation Area (%)', fontsize=17)

# Add tickmarks
plt.xticks(years)
plt.tick_params(axis='both', labelsize=16)

# Add legend
plt.legend(fontsize = 14, loc = 'upper right')

# Add gridlines
plt.grid(linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()






