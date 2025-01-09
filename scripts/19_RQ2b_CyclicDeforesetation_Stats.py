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
import statistics



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

# Define color palatte
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



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
    
# Define function to create stacked bar chart
def stacked_bar(bardata, linedata, linelabel):
    
    # Initialize figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Add bar chart data
    bars = bardata.plot(kind="bar", stacked=True, ax=ax1, color=bluecols)
    
    # Add line data
    ax1.plot(range(len(linedata)), linedata, color = "crimson", label = linelabel)
    
    # Add x tickmarks
    ax1.set_xticks(range(len(linedata)))
    
    # Add x labels
    ax1.set_xticklabels(bardata.index, rotation = 0)
    
    # Add axes labels
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Deforestation Count (# of Pixels)")
    
    # Add gridlines
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    ax1.legend(loc="upper left")
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
# Define function to create double bar chart
def double_bar(bardata1, label1, bardata2, label2, linedata, label3):
    
    # Initialize figure
    fig, ax = plt.subplots(figsize = (10, 6))
    
    # Define bar width
    bar_width = 0.4
    
    # Plot bar data 1
    ax.bar(years, bardata1, width = bar_width, label = label1, color = bluecols[0])
    
    # Plot bar data 2
    ax.bar([i + bar_width for i in years], bardata2, width = bar_width, 
           label = label2, color = bluecols[1])
    
    # Add line data
    ax.plot([i + bar_width / 2 for i in years], linedata, color = "crimson", label = label3)
    
    # Add x tickmarks
    ax.set_xticks([i + bar_width / 2 for i in years])
    
    # Add x labels
    ax.set_xticklabels(years, rotation = 0)
    
    # Add gridlines
    ax.grid(True, linestyle = "--", alpha = 0.6)
    
    # Add axes labels
    ax.set_xlabel("Year")
    ax.set_ylabel("Pixel Count")
    
    # Add legend
    ax.legend()
    
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
strata_plt(strat_confs, "Average Deforestation Label Confidence (1-10)")
   


############################################################################


# CALCULATE REGROWTH COUNTS


############################################################################
# Create dataframe with regrowth counts (per year)
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
annual_plt(regr_vals[1:12], "Total Labelled Regrowth Points")

# Calculate only the first regrowth event
regr_first = val_data['regr1'].value_counts().sort_index()

# Replace 2013 regrowth with 0
regr_first[2013] = 0



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
    
# Plot average regrowth time per year
annual_plt(regr_time, "Average Regrowth Duration (Years)")



############################################################################


# CALCULATE MULTIPLE DEFORESTATION COUNTS


############################################################################
# Create dataframe with regrowth counts (per year)
defor_vals = pd.concat([
    
    # Count unique values for first regrowth event
    val_data['defor1'].value_counts(),
    
    # Count unique values for second regrowth event
    val_data['defor2'].value_counts(),
    
    # Count unique values for third regrowth event
    val_data['defor3'].value_counts()
    
# Get sum of value counts over different years
], axis=1).sort_index()

# Add column names for clarity
defor_vals.columns = ["Deforestation 1", "Deforestation 2", "Deforestation 3"]

# Sort regrowth results by year
defor_vals = defor_vals.sort_index()

# Fill na values with 0
defor_vals = defor_vals.fillna(0)

# Calculate total deforestation per year
total_defor = defor_vals.sum(axis=1)

# Plot deforestation counts
stacked_bar(defor_vals[2:], total_defor[2:], "Total Deforestation")



############################################################################


# CALCULATE MULTIPLE DEFORESTATION PROPORTIONS


############################################################################
# Add counts of deforestation 2 and 3
defor_after = defor_vals["Deforestation 2"] + defor_vals["Deforestation 3"]

# Calculate first deforestation
defor_first = defor_vals["Deforestation 1"]

# Calculate deforestation proportions
defor_props = defor_after / defor_first

# Plot total additional defor and regrowth (all)
double_bar(defor_after[2:], "Additional Deforestation", regr_vals[1:12],
           "Forest Regrowth (All Deforestation Events)", defor_first[2:],
           "Total Deforestation (First Detection)")

# Plot total additional defor and regrowth (first)
double_bar(defor_after[2:], "Additional Deforestation", regr_first[1:12], 
           "Forest Regrowth (After First Deforestation Event)", defor_first[2:],
           "First Detected Deforestation")



############################################################################


# CALCULATE DEFORESTATION EVENT GAPS


############################################################################
# Create empty list to hold average deforestation duration
defor_time_avg = []

# Create empty list to hold mode deforestation duration
defor_time_mode = []

# Iterate over each year
for year in years:
    
    # Extract year data for first defor event
    defor1 = val_data[val_data['defor1'] == year]
    
    # Calculate defor duration
    dur1 = defor1['defor2'] - defor1['defor1']
    
    # Replace invalid duration counts
    dur1 = dur1.replace(-year, np.nan)
    
    # Extract year data for second defor event
    defor2 = val_data[val_data['defor2'] == year]
    
    # Calculate defor duration
    dur2 = defor2['defor3'] - defor2['defor2']
    
    # Replace invalid duration counts
    dur2 = dur2.replace(-year, np.nan)
    
    # Combine all durations
    dur = pd.concat([dur1, dur2], ignore_index=True)
    
    # Calculate average duration
    avg_dur = dur.mean()
    
    # Calculate mode
    mode = statistics.mode(dur)
    
    # Add average duration to list
    defor_time_avg.append(avg_dur)
    
    # Add mode duration to list
    defor_time_mode.append(mode)
    
# Plot average regrowth time per year
annual_plt(defor_time_avg, "Average Time Between Deforestation Events (Years)")

# Plot mode regrowth time per year
annual_plt(defor_time_mode, "Most Frequent Time Between Deforestation Events (Years)")



############################################################################


# CONVERT VALIDATION POINTS FOR HEATMAPPING


############################################################################
"""
Heat map can use intensity based on confidence and multiple deforestation
potentially also: area proportion of each class??
"""

# Copy val_data
val_data_heat = val_data.copy()

# Create heat column out of number of deforestation events
val_data_heat['heat'] = (val_data_heat[["defor1", "defor2", "defor3"]] != 0).sum(axis=1)

# Export to shapefile
val_data_heat.to_file("data/validation/valdata_heat.shp")



































