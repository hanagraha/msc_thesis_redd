# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:51:42 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import pandas as pd
import os
import numpy as np
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

# Set year range
years = range(2013, 2024)

# Define pixel area
pixel_area = 0.09

# Set output directory
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read multiple rasters
def read_files(pathlist):
    
    # Create empty list to hold arrays
    arrlist = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read file
        with rasterio.open(path) as rast:
            data = rast.read(1)
            profile = rast.profile
            
            # Add array to list
            arrlist.append(data)
            
    return arrlist, profile

# Define file paths for annual tmf rasters
tmf_defordegra_paths = [f"data/intermediate/tmf_defordegra_{year}.tif" for 
                      year in years]

# Define file path for tmf transition raster
tmf_trans_file = "data/jrc_preprocessed/tmf_TransitionMap_MainClasses_fm.tif"

# Define file path for tmf annual change rasters
tmf_annual_paths = [f"data/jrc_preprocessed/tmf_AnnualChange_{year}_fm.tif" for 
                    year in years]

# Read defordegra rasters
tmf_defordegra_arrs, profile = read_files(tmf_defordegra_paths)

# Read annual change rasters
tmf_annual_arrs, profile2 = read_files(tmf_annual_paths)

# Read tmf transition map
with rasterio.open(tmf_trans_file) as tmf:
    tmf_trans = tmf.read(1)



############################################################################


# EXPLORE TMF TRANSITION MAP


############################################################################
# Define function to create attribute table
def att_table(arr, expected_classes=None):
    unique_values, pixel_counts = np.unique(arr, return_counts=True)
    
    # Create a DataFrame with unique values and pixel counts
    attributes = pd.DataFrame({"Class": unique_values, 
                               "Frequency": pixel_counts})
    
    # If expected_classes is provided, run the following:
    if expected_classes is not None:
        
        # Reindex DataFrame to include all expected_classes
        attributes = attributes.set_index("Class").reindex(expected_classes, fill_value=0)
        
        # Reset index to have Class as a column again
        attributes.reset_index(inplace=True)

    attributes = attributes.transpose()
    
    return attributes

# Identify unique values and counts
trans_attributes = att_table(tmf_trans)

"""
NOTE: the value "0" exists in the transition map. This value is present in the 
raw data and is not described in the TMF data manual. Because it has a limited 
coverage, the value "0" pixels will excluded from the following analysis
"""

# Exclude 0 values
trans_attributes = trans_attributes.drop(columns = trans_attributes.columns[
    trans_attributes.loc['Class'] == 0])

# TMF class labels from data manual
tmf_trans_labels = ['Undisturbed tropical moist forest', 
                    'Degraded tropical moist forest',
                    'Forest regrowth',
                    'Deforested land - plantations',
                    'Deforested land - water bodies',
                    'Deforested land - other',
                    'Ongoing deforestation/degradation',
                    'Permanent and seasonal water',
                    'Other land cover']



############################################################################


# RECLASSIFY ANNUAL DEFORDEGRA RASTERS


############################################################################
# Define function to write a list of arrays to file
def filestack_write(arraylist, yearrange, dtype, fileprefix):
    
    # Create empty list to store output filepaths
    filelist = []
    
    # Save each array to drive
    for var, year in zip(arraylist, yearrange):
        
        # Adapt file datatype
        data = var.astype(dtype)
        
        # Define file name and path
        output_filename = f"{fileprefix}_{year}.tif"
        output_filepath = os.path.join(out_dir, output_filename)
        
        # Update profile with dtype string
        profile['dtype'] = data.dtype.name
        
        # Write array to file
        with rasterio.open(output_filepath, "w", **profile) as dst:
            dst.write(data, 1)
            
        # Append filepath to list
        filelist.append(output_filepath)
        
        print(f"{output_filename} saved to file")
    
    return filelist

# Create empty list to store new arrays
defordegra_reclass = []

# Iterate over each array
for arr, year in zip(tmf_defordegra_arrs, years):
    
    # Copy array
    reclass_arr = np.copy(arr)
    
    # Create mask for defordegra pixels in that year
    mask = reclass_arr == year
    
    # Replace defordegra pixels with transition map values
    reclass_arr[mask] = tmf_trans[mask]
    
    # Add reclassified array to list
    defordegra_reclass.append(reclass_arr)
    
# Write reclassified arrays to file
tmf_annual_trans_files = filestack_write(defordegra_reclass, years, 
                                         rasterio.uint8, "annual_trans")
    
# Expected list of classes
class_list = [20.0, 30.0, 41.0, 43.0, 50.0, 255.0]

# Create empty list to store attribute tables
annual_trans_atts = []

# Iterate over each reclassified array
for arr in defordegra_reclass:
    
    # Create attribute table
    att = att_table(arr, class_list)
    
    # Add attribute table to list
    annual_trans_atts.append(att)
    


############################################################################


# CREATE TIME SERIES DATA FOR PLOTTING


############################################################################
# Define function to create time series per class
def class_ts(attributes_list, yearrange):
    
    # Identify unique classes
    unique_classes = attributes_list[0].loc['Class'].values
    
    # Initialize a dictionary to store frequency lists for each class
    class_frequencies = {}
    
    # Initialize an empty list for each class in the dictionary
    for class_value in unique_classes:
        if class_value != nodata_val:
            class_frequencies[class_value] = []

    # Iterate over each attribute table
    for df in attributes_list:
        
        # Iterate over classes
        for class_value in unique_classes:
            
            # Skip nodata class
            if class_value != nodata_val:
                
                # Find index of class
                class_index = df.loc['Class'] == class_value
                
                # Find frequency of that index
                frequency_value = df.loc['Frequency', class_index].values[0]
                
                # Add frequency to list
                class_frequencies[class_value].append(frequency_value)
    
    # Convert dictionary to dataframe
    ts = pd.DataFrame(class_frequencies)
    
    # Set index to years
    ts.index = yearrange

    return ts

# Create time series of annual transition attributes
trans_ts = class_ts(annual_trans_atts, years)

# Create empty list to store attribute tables
annual_atts = []

# Iterate over each annual change array
for arr in tmf_annual_arrs:
    
    # Create attribute table
    att = att_table(arr)
    
    # Add attribute table to list
    annual_atts.append(att)
    
# Create time series of annual change attributes
annual_ts = class_ts(annual_atts, years)



############################################################################


# PLOT ANNUAL TRANSITION RASTERS


############################################################################
# Define function to plot deforestation rates
def trans_ts_plot(ts, yearrange, colors, labels, title):
    
    # Initialize plot figure
    plt.figure(figsize=(10, 6))

    # Iterate over data in list
    for val, col, lab in zip(ts.columns, colors, labels):
        plt.plot(yearrange, ts[val], color = col, label = lab)

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('# of Deforestation Pixels')
    
    # Add title
    plt.title(title)
    
    # Add legend
    plt.legend()

    # Add gridlines
    plt.grid(linestyle='--')
    
    # Rotate x ticks for readability
    plt.xticks(years, rotation=45)
    
    # Adjust layout for spacing
    plt.tight_layout()
    
    plt.show()

            
# Define reusable plotting parameters
colors = ['#A0522D', '#6B8E23', '#B8860B', '#CD853F', '#708090', '#708090']
labels= ["Degraded TMF", "Forest Regrowth", "Deforested Land - Plantations", 
         "Deforested Land - Other", "Ongoing Deforestation/Degradation (2020-2023)", "other"]

# # Example plot
# plt.bar([1, 2, 3, 4, 5], [10, 20, 15, 30, 25], color=colors)
# plt.show()

# Plot transition map classes
trans_ts_plot(trans_ts, years, colors, labels, 
              "Deforestation Rates for Yearly Transition Map Data")

trans_ts_plot(annual_ts, years, colors, labels, 
              "Deforestation Rates for Annual Change Data")

# Assuming annual_ts and trans_ts are your two DataFrames

# Concatenate the two DataFrames along the columns
combined_df = pd.concat([annual_ts, trans_ts], axis=1)

# Optionally, rename columns to avoid confusion
combined_df.columns = list(annual_ts.columns) + list(trans_ts.columns)

# Display the combined DataFrame
print(combined_df)


############################################################################


# PLOT ANNUAL CHANGE RASTERS


############################################################################

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









