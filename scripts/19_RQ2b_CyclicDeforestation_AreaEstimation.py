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
def list_read(pathlist, suffix):
    
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
        
    return files

# Read validationd data
valdata = pd.read_csv("data/validation/validation_datasets/validation_points_2013_2023_780_nobuffer.csv", 
                       delimiter=",", index_col=0)

# Convert csv geometry to WKT
valdata['geometry'] = gpd.GeoSeries.from_wkt(valdata['geometry'])

# Convert dataframe to geodataframe
valdata = gpd.GeoDataFrame(valdata, geometry='geometry', crs="EPSG:32629")

# Read stratification map
with rasterio.open("data/intermediate/stratification_layer_nogrnp.tif") as rast:
    
    # Read data
    stratmap = rast.read()
    
    # Get profile
    profile = rast.profile

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k').reset_index()

# Extract redd+ geometry
redd = villages.loc[1].geometry

# Combine multipolygon into one
redd_union = gpd.GeoSeries(redd).unary_union

# Extract non-redd+ geometry
nonredd = villages.loc[0].geometry

# Combine multipolygon into one
nonredd_union = gpd.GeoSeries(nonredd).unary_union 

# Define protocol d filepaths
protd_statpaths = folder_files("val_protd_780nobuff", "stehmanstats.csv")
protd_cmpaths = folder_files("val_protd_780nobuff", "confmatrix.csv")

# Read protocol d statistics
protd_stats = list_read(protd_statpaths, "_stehmanstats.csv")
protd_cm = list_read(protd_cmpaths, "_confmatrix.csv")

# Subset to only keep years 2013-2023 (statistics)
for key in protd_stats:    
    protd_stats[key] = protd_stats[key][(protd_stats[key]['year'] >= 2013) & \
                                    (protd_stats[key]['year'] <= 2023)]
    protd_stats[key] = protd_stats[key].reset_index(drop = True)

# Define gfc lossyear filepath
gfc_lossyear_file = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Read gfc deforestation data
with rasterio.open(gfc_lossyear_file) as gfc:
    gfc_defor = gfc.read(1)
    gfc_profile = gfc.profile

# Calculate total map pixels
total_pix = np.sum(gfc_defor != np.nan)

# Convert map pixels to map area (ha)
total_ha = total_pix * 0.09


# %%
############################################################################


# CALCULATE RECURRING AREA ESTIMATION (WHOLE AREA)


############################################################################\
# Define function to calculate deforestation area per year
def defor_area(valpoints, stratamap, defor1 = True, defor2 = True, defor3 = True):
    
    # Extract strata
    strata = valpoints['strata']
    
    # Calculate number of pixels per strata
    pixvals, pixcounts = np.unique(stratamap, return_counts = True)
    
    # Calculate number of points per strata
    strata_points = strata.value_counts().sort_index()
    
    # Caclulate representative area of each point
    point_area = pixcounts[:-1] / strata_points
    
    # Convert pixels to ha
    point_ha = point_area * 0.09
    
    # Assign point area for each validation point
    valpoints_ha = pd.Series([point_ha.get(x, np.nan) for x in strata], 
                             index = pd.Series(strata).index)
    
    # Create new validation database
    valarea = valpoints.copy()
    
    # Add deforestation count column
    valarea['defor_count'] = ((valarea[['defor1', 'defor2', 'defor3']] != 0).sum(axis=1))
    
    # Add point area column
    valarea['point_area'] = valpoints_ha
    
    # Create dataframe for the results
    defor_annual = pd.DataFrame([[0] * len(years)], columns = years)
    
    # Set row index
    defor_annual.index = ['defor_area']
    
    # Iterate over each point
    for idx, row in valarea.iterrows():
        
        # Calculate further if including first deforestation
        if defor1 == True:
        
            # Extract deforestation 1 year
            year1 = row['defor1']
            
            # If there is deforestation in year1
            if year1 in years:
                
                # Add corresponding area to dataframe (summing it up)
                defor_annual[year1] = defor_annual[year1] + row['point_area']
                
        # Calculate further if including second deforestation
        if defor2 == True:
        
            # Extract deforestation 2 year
            year2 = row['defor2'] 
            
            # If there is deforestation in year2
            if year2 in years:
                
                # Add corresponding area to dataframe (summing it up)
                defor_annual[year2] = defor_annual[year2] + row['point_area']
            
        # Calculate further if including third deforestation
        if defor3 == True:
            
            # Extract deforestation 3 year
            year3 = row['defor3'] 
            
            # If there is deforestation in year3
            if year3 in years:
                
                # Add corresponding area to dataframe (summing it up)
                defor_annual[year3] = defor_annual[year3] + row['point_area']
            
    return defor_annual

# Calculate area for all deforestation events
defor3_area = defor_area(valdata, stratmap)

# Calculate area for first deforestation event
defor1_area = defor_area(valdata, stratmap, defor1 = True, defor2 = False, defor3 = False)

# Extract area calculated by stehman
stehman_area = protd_stats['protd_gfc']['area'] * total_ha
stehman_area.index = pd.Index(years)
    

# %%




















