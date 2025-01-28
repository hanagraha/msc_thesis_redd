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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import seaborn as sns
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

# Define dataset labels
yearlabs = [0] + list(years)

# Define new folder names for validation protocols
protocol_folders = ["val_prota", "val_protb", "val_protc", "val_protd", "val_prote"]

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
# Read validation data
# val_data = pd.read_csv("data/validation/validation_points_labelled.csv", 
#                        delimiter=",", index_col=0)

# val_data = pd.read_csv("data/validation/validation_points_with_buffer_labelled.csv", 
#                        delimiter=",", index_col=0)

val_data = pd.read_csv("data/validation/validation_datasets/validation_points_2013_2023_780.csv", 
                       delimiter=",", index_col=0)

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

# Define dataset names
datanames = ["gfc", "tmf", "se"]

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


# %%
############################################################################


# CREATE CLASSIC CONFUSION MATRICES


############################################################################
# Define function to plot three confusion matrices side by side
def matrix_plt(matrices, names, fmt):
    
    # Initiate figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Iterate over each confusion matrix
    for i in range(0, len(matrices)):
        
        # Create heatmap
        sns.heatmap(matrices[i], annot=True, fmt=fmt, cmap='Blues', ax=axes[i],
                    xticklabels=yearlabs, yticklabels=yearlabs)
        
        # Add title
        axes[i].set_title(f'{names[i]} Confusion Matrix')
        axes[i].set_xlabel(f'{names[i]} Predicted Labels')
        axes[i].set_ylabel('Validation Labels')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
# Define function to calculate confusion matrices for gfc, tmf, se
def cm_calcplot(valdata, valcolumn):
    
    # Calculate confusion matrix for gfc
    gfc_matrix = confusion_matrix(valdata[valcolumn], valdata['gfc'])

    # Calculate confusion matrix for tmf
    tmf_matrix = confusion_matrix(valdata[valcolumn], valdata['tmf'])

    # Calculate confusion matrix for sensitive early combination
    se_matrix = confusion_matrix(valdata[valcolumn], valdata['se'])

    # Plot calculated matrices
    matrix_plt([gfc_matrix, tmf_matrix, se_matrix], datanames, 'd')
    
# Calculate confusion matrix for gfc, tmf, and se
cm_calcplot(val_data, 'defor1')



############################################################################


# CALCULATE ACCURACY AND PRECISION


############################################################################    
# Define function to print overall accuracy for gfc, tmf, and se
def tripacc(title, dataset, col, col2=None, col3=None, sw1=None, sw2=None, sw3=None):
    
    # Copy col for col2 and col3 if not provided
    col2 = col2 if col2 is not None else col
    col3 = col3 if col3 is not None else col
    
    # If only sw1 provided, set sw2 and sw3 equal to sw1
    if sw1 is not None:
        sw2 = sw2 if sw2 is not None else sw1
        sw3 = sw3 if sw3 is not None else sw1
        
    # Retrieve sample weights from the dataset columns if provided
    sw1_col = dataset[sw1] if sw1 is not None else None
    sw2_col = dataset[sw2] if sw2 is not None else None
    sw3_col = dataset[sw3] if sw3 is not None else None
    
    # Calculate accuracy for gfc
    gfc_acc = accuracy_score(dataset[col], dataset['gfc'], sample_weight = sw1_col)
    
    # Calculate accuracy for tmf
    tmf_acc = accuracy_score(dataset[col2], dataset['tmf'], sample_weight = sw2_col)
    
    # Calculate accuracy for se
    se_acc = accuracy_score(dataset[col3], dataset['se'], sample_weight = sw3_col)
    
    # Print statement
    print(f"{title}:\n"
          f"gfc accuracy: {gfc_acc}\n"
          f"tmf accuracy: {tmf_acc}\n"
          f"sensitive early accuracy: {se_acc}\n")
    
# Define function to print precision score for gfc, tmf, and se
def tripprec(title, dataset, col, col2=None, col3=None, sw1=None, sw2=None, sw3=None):
    
    # Copy col for col2 and col3 if not provided
    col2 = col2 if col2 is not None else col
    col3 = col3 if col3 is not None else col
    
    # If only sw1 provided, set sw2 and sw3 equal to sw1
    if sw1 is not None:
        sw2 = sw2 if sw2 is not None else sw1
        sw3 = sw3 if sw3 is not None else sw1
        
    # Retrieve sample weights from the dataset columns if provided
    sw1_col = dataset[sw1] if sw1 is not None else None
    sw2_col = dataset[sw2] if sw2 is not None else None
    sw3_col = dataset[sw3] if sw3 is not None else None
    
    # Calculate accuracy for gfc
    gfc_acc = precision_score(dataset[col], dataset['gfc'], average = 'macro', sample_weight = sw1_col)
    
    # Calculate accuracy for tmf
    tmf_acc = precision_score(dataset[col2], dataset['tmf'], average = 'macro', sample_weight = sw2_col)
    
    # Calculate accuracy for se
    se_acc = precision_score(dataset[col3], dataset['se'], average = 'macro', sample_weight = sw3_col)
    
    # Print statement
    print(f"{title}:\n"
          f"gfc precision: {gfc_acc}\n"
          f"tmf precision: {tmf_acc}\n"
          f"sensitive early precision: {se_acc}\n")

# Calculate overall accuracy of raw validation data
tripacc("Raw validation data", val_data, 'defor1')

# Plot confusion matrices for gfc, tmf, se with processed validation data
cm_calcplot(val_data, 'defor1')

# Calculate overall accuracy
tripacc("Weighted validation accuracy", val_data, col='defor1', sw1='conf1')

# Calculate overall precision
tripprec("Weighted validation precision", val_data, col='defor1', sw1='conf1')


# %%
############################################################################


# CHECK: RATIO OF REDD+, NON-REDD+


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


# PROTOCOL B: IF DEFORESTED YEARS OF ANY DEFORESTED EVENT IS DETECTED


############################################################################
"""
Definition of Agreement: mark agreement if prediction defor year matches
any year between observed defor and regr, for any deforestation event
"""
# Define function to create annual deforestation and confidence dataframes
def val_expand(dataset):
    
    # Create yearly validation deforestation data
    yearly_val = pd.DataFrame(0, index=dataset.index, columns=yearlabs)
    
    # Create yearly validation confidence data
    yearly_conf = pd.DataFrame(0, index=dataset.index, columns=yearlabs)
    
    # Iterate through each row of the second DataFrame
    for idx, row in dataset.iterrows():
        
        # Iterate over each deforestation event pair
        for i in range(1, 4):  
            
            # Define deforestation column
            defor_col = f"defor{i}"
            
            # Define regrowth column
            regr_col = f"regr{i}"
            
            # Define confidence column
            conf_col = f"conf{i}"
            
            # Select deforestation cell
            defor_year = row[defor_col]
            
            # Select regrowth cell
            regr_year = row[regr_col]

            # Select confidence value
            conf_value = row[conf_col]
            
            # Only for the first defroestation event
            if i == 1:
                
                # If no deforestation is detected
                if defor_year == 0:
                    
                    # Assign year 0 with value 1
                    yearly_val.loc[idx, 0] = 1
                    
                    # Assign respective confidence value
                    yearly_conf.loc[idx, 0] = conf_value
            
            # If there was detected deforestation
            if defor_year != 0:
                
                # If there detected regrowth
                if regr_year != 0:
                    
                    # Fill cells between deforestation and regrowth with value 1
                    yearly_val.loc[idx, defor_year:regr_year - 1] = 1
                    
                    # Fill cells with respective confidence value
                    yearly_conf.loc[idx, defor_year:regr_year - 1] = conf_value
                
                # If there is no detected regrowth
                else:
                    
                    # Fill all cells after deforestation with value 1
                    yearly_val.loc[idx, defor_year:] = 1
                    
                    # Fill all cells with respective confidence value
                    yearly_conf.loc[idx, defor_year:] = conf_value    
        
    return yearly_val, yearly_conf

# Define function to align validation with test data
def prot_b(valdata, col, keepcols):
    
    # Create a copy of the validation data
    val_data = valdata.copy()
    
    # Create yearly validation data
    yearly_val, yearly_conf = val_expand(val_data)
    
    # Extract dataset deforestation
    defor = valdata[col]
    
    # Create empty lists to hold validation year and confidence
    val = []
    conf = []
    
    # Iterate over each row (validation point)
    for idx, row in yearly_val.iterrows():
        
        # Extract deforestation year
        deforyear = defor[idx]
        
        # If deforestation year is detected in validation dataset
        if row[deforyear] == 1: 
            
            # Assign year to validation column
            val_year = deforyear
        
        # If deforestation year is NOT detected in validation dataset
        else:
        
            # Assign first detected validation deforestation
            val_year = yearly_val.columns[row == 1].min()
            
            # Assign 0 if no deforestation detected 2013-2023 (for 2012 case)
            if pd.isna(val_year):
                val_year = 0
            
        # Assign respective confidence
        val_conf = yearly_conf.loc[idx, val_year]

        # Append to the lists
        val.append(val_year)
        conf.append(val_conf)
    
    # Add new validation year
    val_data['prot_b'] = val
    
    # Add new validation confidence
    val_data['prot_b_conf'] = conf
    
    # Update columns of interest
    cols = keepcols + [col, 'prot_b', 'prot_b_conf'] 
    
    # Only keep columns of interest
    val_data = val_data[cols]
    
    return val_data

# Define columns of interest
keepcols = ["strata", "geometry"]

# Create aligned valiadtion data for gfc
protb_gfc = prot_b(val_data, "gfc", keepcols)

# Create aligned validation data for tmf
protb_tmf = prot_b(val_data, "tmf", keepcols)

# Create aligned validation data for se
protb_se = prot_b(val_data, "se", keepcols)

# Create list of all protocol b data
protb_data = [protb_gfc, protb_tmf, protb_se]


# %%
############################################################################


# PROTOCOL C: IF FIRST YEAR OF ANY DEFORESTATION YEAR IS DETECTED


############################################################################
"""
Definition of Agreement: time sensitive. mark agreement if predicted defor
year matches validation defor year of the first, second, or third observed 
defor event
"""
# Define function to create new column prioritizing a certain dataset
def prot_c(valdata, col, keepcols):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Create mask where any defor year matches dataset
    mask1 = val_data[col] == val_data['defor1']
    mask2 = (val_data[col] == val_data['defor2']) & (val_data['defor2'] != 0)
    mask3 = (val_data[col] == val_data['defor3']) & (val_data['defor3'] != 0)

    # Assign dataset year where mask is true, otherwise the first defor year
    val_data['prot_c'] = np.where(mask1, val_data['defor1'], 
                                  np.where(mask2, val_data['defor2'], 
                                           np.where(mask3, val_data['defor3'], 
                                                    val_data['defor1'])))

    # Assign corresponding confidence value
    val_data['prot_c_conf'] = np.where(mask1, val_data['conf1'], 
                                       np.where(mask2, val_data['conf2'], 
                                                np.where(mask3, val_data['conf3'], 
                                                         val_data['conf1'])))
    
    # Add data name to list
    cols = keepcols + [col, 'prot_c', 'prot_c_conf']
    
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

# Create list of all protocol c data
protc_data = [protc_gfc, protc_tmf, protc_se]


# %%
############################################################################


# PROTOCOL D: IF FIRST YEAR OF FIRST DEFORESTATION YEAR IS DETECTED


############################################################################
"""
Definition of Agreement: time sensitive. mark agreement if predicted defor
year matches validation defor year of the first observed defor event
"""
# Define function to create new column prioritizing a certain dataset
def prot_d(valdata, col, keepcols):
    
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
protd_gfc = prot_d(val_data, 'gfc', keepcols)

# Run protocol b for tmf (all)
protd_tmf = prot_d(val_data, 'tmf', keepcols)

# Run protocol c for se (all)
protd_se = prot_d(val_data, 'se', keepcols)

# Create list of all protocol d data
protd_data = [protd_gfc, protd_tmf, protd_se]


# %%
############################################################################


# PROTOCOL E: IF FIRST YEAR OF FIRST DEFORESTATION YEAR IS DETECTED
# WITH 1 YEAR BUFFER


############################################################################
"""
Same as protocol D but with a 1 year buffer
"""
# Define function to create new column prioritizing a certain dataset
def prot_e(valdata, col, keepcols):
    
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
prote_gfc = prot_e(val_data, 'gfc', keepcols)

# Run protocol b for tmf (all)
prote_tmf = prot_e(val_data, 'tmf', keepcols)

# Run protocol c for se (all)
prote_se = prot_e(val_data, 'se', keepcols)

# Create list of all protocol d data
prote_data = [prote_gfc, prote_tmf, prote_se]


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

# Split protocol d datasets
protd_redd, protd_nonredd = reddsplit(protd_data, datanames)

# Split protocol e datasets
prote_redd, prote_nonredd = reddsplit(prote_data, datanames)



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

# Write protd data to folder
write_list(protd_data, datanames, "protd")

# Write prote data to folder
write_list(prote_data, datanames, "prote")

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

# write redd protd data
write_dic(protd_redd, "protd", "redd")

# write nonredd protd data
write_dic(protd_nonredd, "protd", "nonredd")

# Write redd prote data
write_dic(prote_redd, "prote", "redd")

# Write nonredd prote data
write_dic(prote_nonredd, "prote", "nonredd")


# %%
############################################################################


# WRITE PRE-PROCESSED FILES TO DISK (WITH BUFFER)


############################################################################
# Define function to write list of gdfs
def write_list_buff(datalist, datanames, protname):
    
    # Iterate over each item in list
    for data, name in zip(datalist, datanames):
        
        # Define output folder
        outfolder = os.path.join(val_dir, f"val_{protname}_buffered")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{name}.csv")
        
        # Write to csv
        data.to_csv(outfilepath, index = False)    
        
        # Print statement
        print(f"{outfilepath} saved to file")
        
# Define function to write dictionary of gdfs
def write_dic_buff(protdics, protname, polyname):
    
    # Iterate over each item in dictionary
    for key, value in protdics.items():
        
        # Define output folder
        outfolder = os.path.join(val_dir, f"val_{protname}_buffered")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{key}_{polyname}.csv")
        
        # Write to csv
        value.to_csv(outfilepath, index = False)
        
        # Print statement
        print(f"{outfilepath} saved to file")

# Write prota data to folder
write_list_buff(prota_data, datanames, "prota")

# Write protb data to folder
write_list_buff(protb_data, datanames, "protb")

# Write protc data to folder
write_list_buff(protc_data, datanames, "protc")

# Write protd data to folder
write_list_buff(protd_data, datanames, "protd")

# write redd prota data
write_dic_buff(prota_redd, "prota", "redd")

# write nonredd prota data
write_dic_buff(prota_nonredd, "prota", "nonredd")

# write redd protb data
write_dic_buff(protb_redd, "protb", "redd")

# write nonredd protb data
write_dic_buff(protb_nonredd, "protb", "nonredd")

# write redd protc data
write_dic_buff(protc_redd, "protc", "redd")

# write nonredd protc data
write_dic_buff(protc_nonredd, "protc", "nonredd")

# write redd protd data
write_dic_buff(protd_redd, "protd", "redd")

# write nonredd protd data
write_dic_buff(protd_nonredd, "protd", "nonredd")


# %%
############################################################################


# WRITE PRE-PROCESSED FILES TO DISK (780 SUBSET)


############################################################################
# Define function to write list of gdfs
def write_list_sub(datalist, datanames, protname):
    
    # Iterate over each item in list
    for data, name in zip(datalist, datanames):
        
        # Define output folder
        outfolder = os.path.join(val_dir, f"val_{protname}_sub")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{name}.csv")
        
        # Write to csv
        data.to_csv(outfilepath, index = False)    
        
        # Print statement
        print(f"{outfilepath} saved to file")
        
# Define function to write dictionary of gdfs
def write_dic_sub(protdics, protname, polyname):
    
    # Iterate over each item in dictionary
    for key, value in protdics.items():
        
        # Define output folder
        outfolder = os.path.join(val_dir, f"val_{protname}_sub")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{key}_{polyname}.csv")
        
        # Write to csv
        value.to_csv(outfilepath, index = False)
        
        # Print statement
        print(f"{outfilepath} saved to file")

# Write prota data to folder
write_list_sub(prota_data, datanames, "prota")

# Write protb data to folder
write_list_sub(protb_data, datanames, "protb")

# Write protc data to folder
write_list_sub(protc_data, datanames, "protc")

# Write protd data to folder
write_list_sub(protd_data, datanames, "protd")

# Write prote data to folder
write_list_sub(prote_data, datanames, "prote")

# write redd prota data
write_dic_sub(prota_redd, "prota", "redd")

# write nonredd prota data
write_dic_sub(prota_nonredd, "prota", "nonredd")

# write redd protb data
write_dic_sub(protb_redd, "protb", "redd")

# write nonredd protb data
write_dic_sub(protb_nonredd, "protb", "nonredd")

# write redd protc data
write_dic_sub(protc_redd, "protc", "redd")

# write nonredd protc data
write_dic_sub(protc_nonredd, "protc", "nonredd")

# write redd protd data
write_dic_sub(protd_redd, "protd", "redd")

# write nonredd protd data
write_dic_sub(protd_nonredd, "protd", "nonredd")

# Write redd prote data
write_dic_sub(prote_redd, "prote", "redd")

# Write nonredd prote data
write_dic_sub(prote_nonredd, "prote", "nonredd")


# %%
############################################################################


# FILTER BASED ON CONFIDENCE


############################################################################
# Filter to keep only points with confidence > 6
valdata_filt = val_data[val_data['conf1'] > 6]

# Count how many points per strata
valdata_filt_strat = np.unique(valdata_filt['strata'], return_counts = True)

# Convert to dataframe
valdata_filt_strat = pd.DataFrame({'strata': valdata_filt_strat[0],
                                  'count': valdata_filt_strat[1]})

# Plot in bar chart
plt.figure(figsize=(10, 6))

# Define bar width
width = 0.4

# Add redd+ bars
plt.bar(valdata_filt_strat['strata']- width/2, valdata_filt_strat['count'], 
        width, label = "REDD+ Villages", color = bluecols[0])

# Add axes tiitles
plt.xlabel("Strata", fontsize = 12)
plt.ylabel("Sample Count", fontsize = 12)

# Add tickmarks
plt.xticks(valdata_filt_strat['strata'])

# Add gridlines
plt.grid(True, linestyle = "--")

# Add legend
plt.legend(fontsize = 12)

# Display the plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# RE-RUN PROTOCOLS BASED ON FILTERED DATASET


############################################################################        
# Define columns of interest
keepcols = ["strata", "geometry", "defor1", "defor2", "defor3"]
    
# Run protocol a for gfc (all)
prota_gfc_filt = prot_aa(valdata_filt, "gfc", keepcols)

# Run protocol a for tmf (all)
prota_tmf_filt = prot_aa(valdata_filt, "tmf", keepcols)

# Run protocol a for se (all)
prota_se_filt = prot_aa(valdata_filt, "se", keepcols)

# Create list of all protocol a data
prota_data_filt = [prota_gfc_filt, prota_tmf_filt, prota_se_filt]

# Define columns of interest
keepcols = ["strata", "geometry"]

# Create aligned valiadtion data for gfc
protb_gfc_filt = prot_b(valdata_filt, "gfc", keepcols)

# Create aligned validation data for tmf
protb_tmf_filt = prot_b(valdata_filt, "tmf", keepcols)

# Create aligned validation data for se
protb_se_filt = prot_b(valdata_filt, "se", keepcols)

# Create list of all protocol b data
protb_data_filt = [protb_gfc_filt, protb_tmf_filt, protb_se_filt]

# Define columns of interest
keepcols = ["strata", "geometry"]

# Run protocol b for gfc (all)
protc_gfc_filt = prot_c(valdata_filt, 'gfc', keepcols)

# Run protocol b for tmf (all)
protc_tmf_filt = prot_c(valdata_filt, 'tmf', keepcols)

# Run protocol c for se (all)
protc_se_filt = prot_c(valdata_filt, 'se', keepcols)

# Create list of all protocol c data
protc_data_filt = [protc_gfc_filt, protc_tmf_filt, protc_se_filt]

# Define columns of interest
keepcols = ["strata", "geometry"]

# Run protocol d for gfc (all)
protd_gfc_filt = prot_d(valdata_filt, 'gfc', keepcols)

# Run protocol d for tmf (all)
protd_tmf_filt = prot_d(valdata_filt, 'tmf', keepcols)

# Run protocol d for se (all)
protd_se_filt = prot_d(valdata_filt, 'se', keepcols)

# Create list of all protocol d data
protd_data_filt = [protd_gfc_filt, protd_tmf_filt, protd_se_filt]

# Split filtered protocol a datasets
prota_redd_filt, prota_nonredd_filt = reddsplit(prota_data_filt, datanames)

# Split filtered protocol b datasets
protb_redd_filt, protb_nonredd_filt = reddsplit(protb_data_filt, datanames)

# Split filtered protocol v datasets
protc_redd_filt, protc_nonredd_filt = reddsplit(protc_data_filt, datanames)

# Split filtered protocol d datasets
protd_redd_filt, protd_nonredd_filt = reddsplit(protd_data_filt, datanames)


# %%
############################################################################


# WRITE FILTERED DATASET


############################################################################
# Define function to write list of gdfs
def write_list_filt(datalist, datanames, protname):
    
    # Iterate over each item in list
    for data, name in zip(datalist, datanames):
        
        # Define output folder
        outfolder = os.path.join(val_dir, f"val_{protname}_filt")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{name}.csv")
        
        # Write to csv
        data.to_csv(outfilepath, index = False)    
        
        # Print statement
        print(f"{outfilepath} saved to file")
        
# Define function to write dictionary of gdfs
def write_dic_filt(protdics, protname, polyname):
    
    # Iterate over each item in dictionary
    for key, value in protdics.items():
        
        # Define output folder
        outfolder = os.path.join(val_dir, f"val_{protname}_filt")
        
        # Define output filename
        outfilepath = os.path.join(outfolder, f"{protname}_{key}_{polyname}.csv")
        
        # Write to csv
        value.to_csv(outfilepath, index = False)
        
        # Print statement
        print(f"{outfilepath} saved to file")
        
# Write prota data to folder
write_list_filt(prota_data_filt, datanames, "prota")

# Write protb data to folder
write_list_filt(protb_data_filt, datanames, "protb")

# Write protc data to folder
write_list_filt(protc_data_filt, datanames, "protc")
        
# Write protd data to folder
write_list_filt(protd_data_filt, datanames, "protd")

# write redd prota data
write_dic_filt(prota_redd_filt, "prota", "redd")

# write nonredd prota data
write_dic_filt(prota_nonredd_filt, "prota", "nonredd")

# write redd protb data
write_dic_filt(protb_redd_filt, "protb", "redd")

# write nonredd protb data
write_dic_filt(protb_nonredd_filt, "protb", "nonredd")

# write redd protc data
write_dic_filt(protc_redd_filt, "protc", "redd")

# write nonredd protc data
write_dic_filt(protc_nonredd_filt, "protc", "nonredd")

# write redd protd data
write_dic_filt(protd_redd_filt, "protd", "redd")

# write nonredd protd data
write_dic_filt(protd_nonredd_filt, "protd", "nonredd")







