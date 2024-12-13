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

# Define file paths for tmf deforestation and degradation rasters
tmf_defordegra_paths = [f"data/intermediate/tmf_defordegra_{year}.tif" for 
                      year in years]

# Define file path for tmf annual change rasters
tmf_annual_paths = [f"data/jrc_preprocessed/tmf_AnnualChange_{year}_fm.tif" for 
                    year in years]

# Define file path for gfc lossyear
gfc_lossyear_paths = [f"data/intermediate/gfc_lossyear_{year}.tif" for 
                      year in years]

# Define file path for tmf transition raster
tmf_trans_file = "data/jrc_preprocessed/tmf_TransitionMap_MainClasses_fm.tif"

# Read defordegra rasters
tmf_defordegra_arrs, profile = read_files(tmf_defordegra_paths)

# Read annual change rasters
tmf_annual_arrs, profile2 = read_files(tmf_annual_paths)

# Read lossyear rasters
gfc_lossyear_arrs, profile3 = read_files(gfc_lossyear_paths)

# Read tmf transition map
with rasterio.open(tmf_trans_file) as tmf:
    tmf_trans = tmf.read(1)
    
# Dictionary of tmf annual change classes (from tmf data manual)
annchange_dict = {
    1: "Undisturbed tropical moist forest",
    2: "Degraded tropical moist forest",
    3: "Deforested land",
    4: "Tropical moist forest regrowth",
    5: "Permanent and seasonal water",
    6: "Other land cover"
}
    
# Dictionary of tmf transition map classes (from tmf data manual)
transmap_dict = {
    10: "Undisturbed tropical moist forest", 
    20: "Degraded tropical moist forest", 
    30: "Forest regrowth", 
    41: "Deforested land - plantations", 
    42: "Deforested land - water bodies", 
    43: "Deforested land - other", 
    50: "Deforestation/degradation ongoing (2021-2023)", 
    60: "Permanent and seasonal water", 
    70: "Other land cover"
}



############################################################################


# COMVERT TMF TRANSITION RASTER TO ANNUAL DATA


############################################################################
# Define function to create attribute table
def att_table(arr, expected_classes=None):
    
    # Extract unique values and pixel counts
    unique_values, pixel_counts = np.unique(arr, return_counts=True)
    
    # Create a DataFrame with unique values and pixel counts
    attributes = pd.DataFrame({"Class": unique_values, 
                               "Frequency": pixel_counts})
    
    # If expected_classes is provided, run the following:
    if expected_classes is not None:
        
        # Reindex DataFrame to include all expected_classes
        attributes = attributes.set_index("Class").reindex(expected_classes, 
                                                           fill_value=0)
        
        # Reset index to have Class as a column again
        attributes.reset_index(inplace=True)
        
    # Switch rows and columns of dataframe
    attributes = attributes.transpose()
    
    return attributes

# Define function to create annual data from a single-layer array
def arr2ann(annual_arrlist, ref_arr, yearrange):
    
    # Create empty list to hold reclassified annual arrays
    ann_arrs = []
    
    # Iterate over annual arrays and years
    for arr, year in zip(annual_arrlist, yearrange):
        
        # Copy array
        reclass_arr = np.copy(arr)
        
        # Create mask for defordegra pixels in that year
        mask = reclass_arr == year
        
        # Replace defordegra pixels with transition map values
        reclass_arr[mask] = ref_arr[mask]
        
        # Add reclassified array to list
        ann_arrs.append(reclass_arr)
        
    return ann_arrs

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

# Reclassify transition map to annual data
trans_annual = arr2ann(tmf_defordegra_arrs, tmf_trans, years)
    
# Write reclassified arrays to file
trans_annual_files = filestack_write(trans_annual, years, 
                                         rasterio.uint8, "annual_trans")
    


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

# Expected list of classes
class_list = [20.0, 30.0, 41.0, 43.0, 50.0, 255.0]

# Create empty list to store annual transition attributes
annual_trans_atts = []

# Iterate over each annual transition array
for arr in trans_annual:
    
    # Create attribute table
    att = att_table(arr, class_list)
    
    # Add attribute table to list
    annual_trans_atts.append(att)

# Create time series of annual transition attributes
trans_ts = class_ts(annual_trans_atts, years)

# Create empty list to store annual change attributes
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
def multiclass_ts_plot(ts, yearrange, colors, class_dict, title):
    
    # Initialize plot figure
    plt.figure(figsize=(10, 6))

    # Iterate over data in list
    for val, col in zip(ts.columns, colors):
        label = class_dict.get(val)
        plt.plot(yearrange, ts[val], color = col, label = label)

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

# Plot annual transition map classes
multiclass_ts_plot(trans_ts, years, colors, transmap_dict,
              "Deforestation Rates for Yearly TMF Transition Map Data")

# Plot annual change map
multiclass_ts_plot(annual_ts, years, colors, annchange_dict, 
              "Deforestation Rates for Annual Change Data")



############################################################################


# CONVERT GFC LOSSYEAR AND TMF DEFORDEGRA TO TIME SERIES DATA


############################################################################
# Define function to create time series data from gfc lossyear and tmf defordegra
def multiyear_ts(arr_list, yearrange):
    
    # Create empty dictionary
    year_defor = {}
    
    # Iterate over each year
    for year in yearrange:
        
        # Create dictionary to store deforestation pixels per year
        year_defor[year] = []
    
    # Iterate over each array and year
    for arr, year in zip(arr_list, yearrange):
        
        # Extract number of year pixels
        defor_pixels = np.sum(arr == year)
        
        # Add pixel count to list
        year_defor[year].append(defor_pixels)
        
    # Convert dictionary to dataframe
    ts = pd.DataFrame(year_defor).transpose()
    
    return ts

# Create time series from gfc lossyear
lossyear_ts = multiyear_ts(gfc_lossyear_arrs, years)

# Create time series from tmf defordegra
defordegra_ts = multiyear_ts(tmf_defordegra_arrs, years)
    

############################################################################


# RECLASSIFY TRANSITION AND ANNUAL CHANGE CLASSES FOR COMPARABILITY


############################################################################
# Reclassify transition map time series
"""
Trans combination strategy 1: All deforestation and degradation classes will 
be summed together and regrowth will be subtracted.
"""
trans_defor = trans_ts[20.0] + trans_ts[41.0] + trans_ts[43.0] + \
    trans_ts[50.0] - trans_ts[30.0]
    
# Convert series to dataframe
trans_defor = pd.DataFrame(trans_defor)
    
"""
Trans combination strategy 2: Don't subtract regrowth
"""
trans_defor2 = trans_ts[20.0] + trans_ts[41.0] + trans_ts[43.0] + \
    trans_ts[50.0]

# Convert series to dataframe
trans_defor2 = pd.DataFrame(trans_defor2)

"""
Trans combination strategy 3: add regrowth
"""
trans_defor3 = trans_ts[20.0] + trans_ts[41.0] + trans_ts[43.0] + \
    trans_ts[50.0] + trans_ts[30.0]

# Convert series to dataframe
trans_defor3 = pd.DataFrame(trans_defor3)

# Reclassify annual change time series
"""
The degraded and deforested tropical moist forest classes will be summed. The
regrowth class will be subtracted. Other classes will be ignored. 
"""
ann_defor = annual_ts[3]

# Convert series to dataframe
ann_defor = pd.DataFrame(ann_defor)



############################################################################


# PLOT DEFORESTATION RATES


############################################################################
# Define function to plot deforestation rates
def defor_ts_plot(ts_list, yearrange, colors, labels):
    
    # Initialize plot figure
    plt.figure(figsize=(12, 6))
    
    # Iterate over data in list
    for ts, col, lab in zip(ts_list, colors, labels):
        plt.plot(yearrange, ts, color = col, label = lab)

    # Add axes labels
    plt.xlabel('Year')
    plt.ylabel('# of Deforestation Pixels')
    
    # Add title
    plt.title("Deforestation Rates from TMF and GFC Datasets")
    
    # Add legend
    plt.legend()

    # Add gridlines
    plt.grid(linestyle='--')
    
    # Rotate x ticks for readability
    plt.xticks(yearrange, rotation=45)
    
    # Adjust layout for spacing
    plt.tight_layout()
    
    plt.show()

# Define reusable parameters for plotting
ts_list = [ann_defor, trans_defor, defordegra_ts, lossyear_ts]
colors = ['#A0522D', '#6B8E23', '#B8860B', '#708090']
labels = ["TMF Annual Change", "TMF Transition - Main Classes", 
          "TMF Deforestation and Degradation Year", "GFC Lossyear"]

# Plot GFC lossyear and TMF defordegra
defor_ts_plot(ts_list[2:], years, colors[2:], labels[2:])

# Plot deforestation for all TMF datasets
defor_ts_plot(ts_list[:3], years, colors[:3], labels[:3])

# Plot TMF and GFC datasets
defor_ts_plot(ts_list[1:], years, colors[1:], labels[1:])

# Define new list
ts_list = [trans_defor, trans_defor2, trans_defor3, defordegra_ts, lossyear_ts]
colors = ['#8B4513', '#D2B48C', '#556B2F', '#CD5C5C', '#4682B4']
labels = ["TMF transition map (regrowth subtracted)", 
          "TMF transition map (regrowth ignored)", 
          "TMF transition map (regrowth added)",
          "TMF deforestation and degradation year", "GFC lossyear"]

# Plot combinations for GFC comparability
defor_ts_plot(ts_list, years, colors, labels)








