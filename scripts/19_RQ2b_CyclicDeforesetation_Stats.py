# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:26:48 2024

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
import seaborn as sns



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
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Set year range
years = range(2013, 2024)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Read validation data (unprocessed)
val_data = pd.read_csv("data/validation/validation_points_labelled.csv", 
                       delimiter=";", index_col=0)

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

# Read validation data (preprocessed)
val_data_proc = pd.read_csv("data/validation/validation_points_preprocessed2.csv")

# Convert csv geometry to WKT
val_data_proc['geometry'] = gpd.GeoSeries.from_wkt(val_data_proc['geometry'])

# Convert dataframe to geodataframe
val_data_proc = gpd.GeoDataFrame(val_data_proc, geometry='geometry', crs="EPSG:32629") 



############################################################################


# PLOTTING FUNCTIONS


############################################################################
# Define function to create strata line plot
def strata_plt(strat_data, y_label):
    
    # Initialize figure
    plt.figure(figsize=(10, 6))

    # Plot line data
    plt.plot(range(1, 24), strat_data, linestyle='-', color='#1E2A5E')

    # Add axes labels
    plt.xlabel('Strata', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Add x ticks for every strata
    plt.gca().set_xticks(range(1,24))

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.show()
    
# Define function to create annual line plot
def annual_plt(strat_data, y_label):
    
    # Initialize figure
    plt.figure(figsize=(10, 6))

    # Plot line data
    plt.plot(years, strat_data, linestyle='-', color='#1E2A5E')

    # Add axes labels
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Add x ticks for every strata
    plt.gca().set_xticks(years)

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.show()



############################################################################


# CALCULATE AVERAGE CONFIDENCE PER STRATA


############################################################################
# Create empty list to hold average confidence per strata
strat_confs = []

# Iterate over each strata
for strata in range(1, 24):
    
    # Extract strata data
    strata_data = val_data[val_data['strata'] == strata]
    
    # Extract confidence per deforestation event
    conf_data = pd.DataFrame({
        'conf1': strata_data['conf1'],
        'conf2': strata_data['conf2'],
        'conf3': strata_data['conf3']})

    # Replace 0 confidence with na values
    valid_confs = conf_data.replace(0, pd.NA)
    
    # Calculate average confidence
    avg_confs = valid_confs.mean(axis=1, skipna=True).mean()
    
    # Add average confidence to list
    strat_confs.append(avg_confs)
    
# Plot average confidence
strata_plt(strat_confs, "Average Confidence")
   


############################################################################


# CALCULATE REGROWTH COUNTS


############################################################################
# Create dataframe with regrowth counts
regr_vals = pd.concat([
    
    # Count unique values for first regrowth event
    val_data['regr1'].value_counts(),
    
    # Count unique values for second regrowth event
    val_data['regr2'].value_counts(),
    
    # Count unique values for third regrowth event
    val_data['regr3'].value_counts()
    
# Get sum of value counts over different years
], axis=1).sum(axis=1)

# Sort regrowth results by year
regr_vals = regr_vals.sort_index()

# Plot annual regrowth
annual_plt(regr_vals[1:12], "Regrowth Pixels per Strata")



############################################################################


# CALCULATE REGROWTH DURATION


############################################################################
# Create empty list to hold average deforestation duration
regr_time = []

# Iterate over each year
for year in years:
    
    # Extract year data for first defor event
    defor1 = val_data[val_data['defor1'] == year]
    
    # Calculate defor duration
    dur1 = defor1['regr1'].replace(0, 2024) - defor1['defor1']
    
    # Extract year data for second defor event
    defor2 = val_data[val_data['defor2'] == year]
    
    # Calculate defor duration
    dur2 = defor2['regr2'].replace(0, 2024) - defor2['defor2']
    
    # Extract year data for third defor event
    defor3 = val_data[val_data['defor3'] == year]
    
    # Calculate defor duration
    dur3 = defor3['regr3'].replace(0, 2024) - defor3['defor3']
    
    # Combine all durations
    dur = pd.concat([dur1, dur2, dur3], ignore_index=True)
    
    # Calculate average duration
    avg_dur = dur.mean()
    
    # Add average duration to list
    regr_time.append(avg_dur)
    




















