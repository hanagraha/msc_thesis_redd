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
import rasterio
from rasterio.mask import mask



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


# SPLIT STRATIFICATION MAP INTO REDD+ / NONREDD+ AREAS


############################################################################
# Define function to crop to geometry
def geom_crop(geometry, output_path):
    
    # Read stratification map
    with rasterio.open(strat_path) as rast:
        
        # Mask to geometry
        out_image, out_transform = mask(rast, [geometry], crop = True)
        
        # Extract metadata
        meta = rast.meta.copy()
        
        # Update metadata
        meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
            })
        
        # Save clipped raster
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(out_image)
            
    return out_image

# Define function to calculate strata size
def incl_prob(stratmap, valdata):
    
    # Calculate number of pixels per strata
    pixvals, pixcounts = np.unique(stratmap, return_counts = True)

    # Create dataframe
    strata_size = pd.DataFrame({'strata': pixvals[:-1],
                                'size': pixcounts[:-1]})
    
    # Calculate number of samples per strata
    sampvals, sampcounts = np.unique(valdata['strata'], return_counts = True)
    
    # Create dataframe
    samples = pd.DataFrame({'strata': sampvals,
                            'samples': sampcounts})
    
    # Calculate inclusion probability
    prob = pd.DataFrame({'strata': sampvals,
                         'incl_prob': (samples['samples'] / strata_size['size'])
                         })
    
    return prob

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
            
# Define output path for redd+ geometry
redd_path = os.path.join("data", "intermediate", "stratification_layer_redd.tif")

# Define output path for nonredd+ geometry
nonredd_path = os.path.join("data", "intermediate", "stratification_layer_nonredd.tif")

# Clip to redd geometry
redd_strat = geom_crop(redd_union, redd_path)

# Clip to nonredd geometry
nonredd_strat = geom_crop(nonredd_union, nonredd_path)

# Calculate inclusion probability for redd
redd_prob = incl_prob(redd_strat, points_redd)

# Calculate inclusion probability for nonredd
nonredd_prob = incl_prob(nonredd_strat, points_nonredd)
            
        
# %%        
############################################################################


# MAP: RATIO OF REDD+, NON-REDD+


############################################################################ 
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
            
            # If deforestation IS detected in gfc dataset
            if row[col] != 0:
                
                # Mark agreement
                val_data.loc[idx, 'prot_a'] = 1 
            
            # If deforestation is NOT detected in gfc dataset
            else: 
                
                # Mark disagreement
                val_data.loc[idx, 'prot_a'] = 0
                
        # If deforestation is NOT detected in validation dataset
        else:
            
            # If deforestation is NOT detected in gfc dataset
            if row[col] == 0:
                
                # Mark agreement
                val_data.loc[idx, 'prot_a'] = 1 
                
            # If deforestation IS detected in gfc dataset
            else:
            
                # Mark disagreement
                val_data.loc[idx, 'prot_a'] = 0
    
    # Add data name to list
    cols = keepcols[:2] + [col] + keepcols[2:]
    
    # Only keep relevant columns
    val_data = val_data[cols]
    
    return val_data

# Define NEW!! function to manipulate with protocol A
def prot_aa(valdata, col, keepcols):
    
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
    
# Run protocol a for gfc (all)
prota_gfc = prot_aa(val_data, "gfc", keepcols)

# Run protocol a for tmf (all)
prota_tmf = prot_aa(val_data, "tmf", keepcols)

# Run protocol a for se (all)
prota_se = prot_aa(val_data, "se", keepcols)

# Create list of all protocol a data
prota_data = [prota_gfc, prota_tmf, prota_se]


# %%
############################################################################


# PROTOCOL B: IF FIRST YEAR OF FIRST DEFORESTATION YEAR IS DETECTED
# WITH 1 YEAR BUFFER


############################################################################
"""
Same as protocol D but with a 1 year buffer
"""
# Define function to create new column prioritizing a certain dataset
def prot_b(valdata, col, keepcols):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Create mask where any defor year matches dataset
    mask = val_data[col].between(val_data['defor1'] - 1, val_data['defor1'] + 1)

    # Assign dataset year where mask is true, otherwise first defor year
    val_data['prot_e'] = np.where(mask, val_data[col], val_data['defor1'])
    
    # Add data name to list
    cols = keepcols + [col, 'prot_e']
    
    # Only keep relevant columns
    val_data = val_data[cols]
    
    return val_data

# Define columns of interest
keepcols = ["strata", "geometry"]

# Run protocol b for gfc (all)
protb_gfc = prot_b(val_data, 'gfc', keepcols)

# Run protocol b for tmf (all)
protb_tmf = prot_b(val_data, 'tmf', keepcols)

# Run protocol c for se (all)
protb_se = prot_b(val_data, 'se', keepcols)

# Create list of all protocol d data
protb_data = [protb_gfc, protb_tmf, protb_se]


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
    val_data['prot_d'] = val_data['defor1']
    
    # Rename confidence column
    val_data['prot_d_conf'] = val_data['conf1']
    
    # Add data name to list
    cols = keepcols[:2] + [col, 'prot_d', 'prot_d_conf']
    
    # Only keep relevant columns
    val_data = val_data[cols]
    
    return val_data

# Define columns of interest
keepcols = ["strata", "geometry"]

# Run protocol b for gfc (all)
protc_gfc = prot_c(val_data, 'gfc', keepcols)

# Run protocol b for tmf (all)
protc_tmf = prot_c(val_data, 'tmf', keepcols)

# Run protocol c for se (all)
protc_se = prot_c(val_data, 'se', keepcols)

# Create list of all protocol d data
protc_data = [protc_gfc, protc_tmf, protc_se]


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


# %%
############################################################################


# WRITE PRE-PROCESSED FILES TO DISK


############################################################################
# Define function to write list of gdfs
def write_list(datalist, datanames, protname):
    
    # Iterate over each item in list
    for data, name in zip(datalist, datanames):
        
        # Define output folder
        outfolder = os.path.join(val_dir, f"val_{protname}")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{name}.csv")
        
        # Write to csv
        data.to_csv(outfilepath, index = False)    
        
        # Print statement
        print(f"{outfilepath} saved to file")
        
# Define function to write dictionary of gdfs
def write_dic(protdics, protname, polyname):
    
    # Iterate over each item in dictionary
    for key, value in protdics.items():
        
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












