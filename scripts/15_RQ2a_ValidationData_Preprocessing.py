# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:52:36 2024

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



############################################################################


# SET UP DIRECTORY AND NODATA


############################################################################
# Define function to create new folders (if necessary)
def newfolder(folderlist, parfolder):
    
    # Iterate over each folder name
    for folder in folderlist:
        
        # Define folder path
        path = os.path.join(parfolder, folder)
        
        # Check if folder does not exist
        if not os.path.exists(path):
            
            # Create folder
            os.makedirs(path)
            
            # Print statement
            print(f"{path} created")
          
        # If folder already exists
        else:
            
            # Print statement
            print(f"{path} already exists")

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

# Define dataset names
datanames = ["gfc", "tmf", "se"]

# Define new folder names for validation protocols
protocol_folders = ["val_prota", "val_protb", "val_protc", "val_prota_buff", 
                    "val_protb_buff", "val_protc_buff"]

# Create new folders (if necessary)
newfolder(protocol_folders, val_dir)

# Set default plotting colors
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read csv validation data
def csv_read(datapath):
    
    # Read validation data
    data = pd.read_csv(datapath, delimiter = ",", index_col = 0)
    
    # Convert csv geometry to WKT
    data['geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
    
    # Convert dataframe to geodataframe
    data = gpd.GeoDataFrame(data, geometry = 'geometry', crs="EPSG:32629")
    
    return data

# Read no buffer validation data
val_data = csv_read("data/validation/validation_datasets/validation_points_2013_2023_780_nobuffer.csv")

# Read buffered validation data
val_data_buff = csv_read("data/validation/validation_datasets/validation_points_2013_2023_780_buffer.csv")

# Define stratification map path
strat_path = "data/intermediate/stratification_layer_nogrnp.tif"

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k').reset_index()

# Create redd+ polygon
redd_union = gpd.GeoSeries(villages.loc[1].geometry).unary_union

# Create nonredd+ polygon
nonredd_union = gpd.GeoSeries(villages.loc[0].geometry).unary_union


# %%
############################################################################


# MAP: RATIO OF REDD+, NON-REDD+


############################################################################
# Filter points within REDD+ multipolygon
points_redd = val_data[val_data.geometry.within(redd_union)]

# Filter points within non-REDD+ multipolygon
points_nonredd = val_data[val_data.geometry.within(nonredd_union)]

# Check for points outside polygons (just in case)
points_outside = val_data[
    ~val_data.geometry.within(redd_union) &
    ~val_data.geometry.within(nonredd_union)
]

# Output results
print("Points within REDD+ polygon:", len(points_redd))
print("Points within non-REDD+ polygon:", len(points_nonredd))
print("Points outside both polygons:", len(points_outside))

# Check strata counts in redd points
redd_strata = np.unique(points_redd['strata'], return_counts = True)

# Check strata counts in nonredd points
nonredd_strata = np.unique(points_nonredd['strata'], return_counts = True)

# Plot in bar chart
plt.figure(figsize=(10, 6))

# Define bar width
width = 0.4

# Add redd+ bars
plt.bar(redd_strata[0]- width/2, redd_strata[1], width, label = 
        "REDD+ Villages", color = bluecols[0])

# Add nonredd+ bars
plt.bar(nonredd_strata[0] + width/2, nonredd_strata[1], width, label = 
        "Non-REDD+ Villages", color = bluecols[1])

# Add axes tiitles
plt.xlabel("Strata", fontsize = 12)
plt.ylabel("Point Count", fontsize = 12)

# Add tickmarks
plt.xticks(redd_strata[0])

# Add gridlines
plt.grid(True, linestyle = "--")

# Add legend
plt.legend(fontsize = 12)

# Display the plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# PROTOCOL A: IF DEFORESTATION IS DETECTED


############################################################################
"""
Definition of Agreement: time insensitive. mark agreement if deforestation
is labeled in validation and detected in prediction. or if undisturbed is
labeled in validation and detected in prediction.
"""
# Define function to manipulate with protocol A
def prot_a(valdata, col, keepcols):
    
    # Copy input validation data
    val_data = valdata.copy()

    # Iterate over each row in validation dataset
    for idx, row in val_data.iterrows():
        
        # If deforestation IS detected in validation dataset
        if row['defor1'] != 0:
            
            # Label deforestation
            val_data.loc[idx, 'prot_a_val'] = 1
        
        # If deforestation is NOT detected in validation dataset
        else: 
            
            # Label undeforested
            val_data.loc[idx, 'prot_a_val'] = 0
            
        # If deforestation IS detected in the prediction dataset
        if row[col] != 0:
            
            # Mark deforestation
            val_data.loc[idx, 'prot_a_pred'] = 1 
            
        # If deforestation is NOT detected in the prediction dataset
        else: 
            
            # Mark undeforested
            val_data.loc[idx, 'prot_a_pred'] = 0
    
    # Add data name to list
    cols = keepcols + [col, 'prot_a_val', 'prot_a_pred']
    
    # Only keep relevant columns
    val_data = val_data[cols]
    
    return val_data

# Define columns of interest
keepcols = ["strata", "geometry", "defor1", "defor2", "defor3"]
    
# Run protocol a for gfc 
prota_gfc = prot_a(val_data, "gfc", keepcols)

# Run protocol a for tmf
prota_tmf = prot_a(val_data, "tmf", keepcols)

# Run protocol a for se 
prota_se = prot_a(val_data, "se", keepcols)

# Create list of all protocol a data
prota_data = [prota_gfc, prota_tmf, prota_se]

# Run protocol a for gfc (buffered)
prota_gfc_buff = prot_a(val_data_buff, "gfc", keepcols)

# Run protocol a for tmf (buffered)
prota_tmf_buff = prot_a(val_data_buff, "tmf", keepcols)

# Run protocol a for se (buffered)
prota_se_buff = prot_a(val_data_buff, "se", keepcols)

# Create list of all protocol a data (buffered)
prota_data_buff = [prota_gfc_buff, prota_tmf_buff, prota_se_buff]


# %%
############################################################################


# PROTOCOL B: IF FIRST YEAR OF FIRST DEFORESTATION YEAR IS DETECTED
# WITH 1 YEAR BUFFER


############################################################################
"""
Definition of Agreement: time sensitive. mark agreement if predicted defor
year is within one year of validation defor year of the first observed defor 
event
"""
# Define function to create new column prioritizing a certain dataset
def prot_b(valdata, col, keepcols):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Create mask where any defor year matches dataset
    mask = val_data[col].between(val_data['defor1'] - 1, val_data['defor1'] + 1)

    # Assign dataset year where mask is true, otherwise first defor year
    val_data['prot_b'] = np.where(mask, val_data[col], val_data['defor1'])
    
    # Add data name to list
    cols = keepcols + [col, 'prot_b']
    
    # Only keep relevant columns
    val_data = val_data[cols]
    
    return val_data

# Define columns of interest
keepcols = ["strata", "geometry"]

# Run protocol b for gfc 
protb_gfc = prot_b(val_data, 'gfc', keepcols)

# Run protocol b for tmf 
protb_tmf = prot_b(val_data, 'tmf', keepcols)

# Run protocol c for se 
protb_se = prot_b(val_data, 'se', keepcols)

# Create list of all protocol d data
protb_data = [protb_gfc, protb_tmf, protb_se]

# Run protocol b for gfc (buffered)
protb_gfc_buff = prot_b(val_data_buff, 'gfc', keepcols)

# Run protocol b for tmf (buffered)
protb_tmf_buff = prot_b(val_data_buff, 'tmf', keepcols)

# Run protocol c for se (buffered)
protb_se_buff = prot_b(val_data_buff, 'se', keepcols)

# Create list of all protocol d data (buffered)
protb_data_buff = [protb_gfc_buff, protb_tmf_buff, protb_se_buff]


# %%
############################################################################


# PROTOCOL C: IF FIRST YEAR OF FIRST DEFORESTATION YEAR IS DETECTED


############################################################################
"""
Definition of Agreement: time sensitive. mark agreement if predicted defor
year matches validation defor year of the first observed defor event
"""
# Define function to create new column prioritizing a certain dataset
def prot_c(valdata, col, keepcols):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Re-name deforestation column
    val_data['prot_c'] = val_data['defor1']
    
    # Rename confidence column
    val_data['prot_c_conf'] = val_data['conf1']
    
    # Add data name to list
    cols = keepcols[:2] + [col, 'prot_c', 'prot_c_conf']
    
    # Only keep relevant columns
    val_data = val_data[cols]
    
    return val_data

# Define columns of interest
keepcols = ["strata", "geometry"]

# Run protocol b for gfc
protc_gfc = prot_c(val_data, 'gfc', keepcols)

# Run protocol b for tmf
protc_tmf = prot_c(val_data, 'tmf', keepcols)

# Run protocol c for se
protc_se = prot_c(val_data, 'se', keepcols)

# Create list of all protocol d data
protc_data = [protc_gfc, protc_tmf, protc_se]

# Run protocol b for gfc (buffered)
protc_gfc_buff = prot_c(val_data_buff, 'gfc', keepcols)

# Run protocol b for tmf (buffered)
protc_tmf_buff = prot_c(val_data_buff, 'tmf', keepcols)

# Run protocol c for se (buffered)
protc_se_buff = prot_c(val_data_buff, 'se', keepcols)

# Create list of all protocol d data (buffered)
protc_data_buff = [protc_gfc_buff, protc_tmf_buff, protc_se_buff]


# %%
############################################################################


# SPLIT PRE-PROCESSED FILES BY REDD / NONREDD


############################################################################
# Define function to split by redd and nonredd polygons
def reddsplit(valdata_list, datanames):
    
    # Create empty dictionary to hold redd and nonredd valdata
    redd_data = {}
    nonredd_data = {}
    
    # Iterate over each dataset in list
    for valdata, name in zip(valdata_list, datanames):
    
        # Filter points within redd+ multipolygon
        points_redd = valdata[valdata.geometry.within(redd_union)]
        
        # Filter points within nonredd+ multipolygon
        points_nonredd = valdata[valdata.geometry.within(nonredd_union)]
        
        # Add redd gdf to redd dictionary
        redd_data[name] = points_redd
        
        # Add nonredd gdf to nonredd dictionary
        nonredd_data[name] = points_nonredd
    
    return redd_data, nonredd_data

# Split protocol a datasets
prota_redd, prota_nonredd = reddsplit(prota_data, datanames)

# Split protocol b datasets
protb_redd, protb_nonredd = reddsplit(protb_data, datanames)

# Split protocol c datasets
protc_redd, protc_nonredd = reddsplit(protc_data, datanames)

# Split protocol a datasets (buffered)
prota_redd_buff, prota_nonredd_buff = reddsplit(prota_data_buff, datanames)

# Split protocol b datasets (buffered)
protb_redd_buff, protb_nonredd_buff = reddsplit(protb_data_buff, datanames)

# Split protocol c datasets (buffered)
protc_redd_buff, protc_nonredd_buff = reddsplit(protc_data_buff, datanames)


# %%
############################################################################


# WRITE PRE-PROCESSED FILES TO DISK


############################################################################
# Define function to write list of gdfs
def write_list(datalist, datanames, protname, ext = False):
    
    # Iterate over each item in list
    for data, name in zip(datalist, datanames):
        
        # If the extension is provided
        if ext != False:
            
            # Define output folder
            outfolder = os.path.join(val_dir, f"val_{protname}_{ext}")
        
        # If the extension is not provided
        else: 
            
            # Define output folder
            outfolder = os.path.join(val_dir, f"val_{protname}")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{name}.csv")
        
        # Write to csv
        data.to_csv(outfilepath, index = False)    
        
        # Print statement
        print(f"{outfilepath} saved to file")
        
# Define function to write dictionary of gdfs
def write_dic(protdics, protname, polyname, ext = False):
    
    # Iterate over each item in dictionary
    for key, value in protdics.items():
        
        # If the extension is provided
        if ext != False:
            
            # Define output folder
            outfolder = os.path.join(val_dir, f"val_{protname}_{ext}")
            
        # If the extension is not provided
        else:
        
            # Define output folder
            outfolder = os.path.join(val_dir, f"val_{protname}")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{key}_{polyname}.csv")
        
        # Write to csv
        value.to_csv(outfilepath, index = False)
        
        # Print statement
        print(f"{outfilepath} saved to file")

# Write prota data to folder
write_list(prota_data, datanames, "prota")

# Write protb data to folder
write_list(protb_data, datanames, "protb")

# Write protc data to folder
write_list(protc_data, datanames, "protc")

# write redd prota data
write_dic(prota_redd, "prota", "redd")

# write nonredd prota data
write_dic(prota_nonredd, "prota", "nonredd")

# write redd protb data
write_dic(protb_redd, "protb", "redd")

# write nonredd protb data
write_dic(protb_nonredd, "protb", "nonredd")

# write redd protc data
write_dic(protc_redd, "protc", "redd")

# write nonredd protc data
write_dic(protc_nonredd, "protc", "nonredd")

# Write prota data to folder (buffered)
write_list(prota_data_buff, datanames, "prota", "buff")

# Write protb data to folder (buffered)
write_list(protb_data, datanames, "protb", "buff")

# Write protc data to folder (buffered)
write_list(protc_data, datanames, "protc", "buff")

# write redd prota data (buffered)
write_dic(prota_redd, "prota", "redd", "buff")

# write nonredd prota data (buffered)
write_dic(prota_nonredd, "prota", "nonredd", "buff")

# write redd protb data (buffered)
write_dic(protb_redd, "protb", "redd", "buff")

# write nonredd protb data (buffered)
write_dic(protb_nonredd, "protb", "nonredd", "buff")

# write redd protc data (buffered)
write_dic(protc_redd, "protc", "redd", "buff")

# write nonredd protc data (buffered)
write_dic(protc_nonredd, "protc", "nonredd", "buff")










