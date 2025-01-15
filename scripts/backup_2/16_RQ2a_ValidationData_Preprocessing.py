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

# Define dataset labels
yearlabs = [0] + list(years)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Read validation data
# val_data = pd.read_csv("data/validation/validation_points_labelled.csv", 
#                        delimiter=";", index_col=0)
val_data = pd.read_csv("data/validation/validation_points_1380.csv", 
                       delimiter=",", index_col=0)

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

datanames = ["GFC", "TMF", "Sensitive Early"]



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



############################################################################


# VALIDATION MANIPULATION: PRIORITIZE AGREEING DEFOR YEAR (PREV METHOD)


############################################################################
# Define function to create new column prioritizing a certain dataset
def agvalcol(dataset, col, newcol, confcol):
    
    # Create mask where any defor year matches dataset
    mask1 = dataset[col] == dataset['defor1']
    mask2 = (dataset[col] == dataset['defor2']) & (dataset['defor2'] != 0)
    mask3 = (dataset[col] == dataset['defor3']) & (dataset['defor3'] != 0)

    # Assign dataset year where mask is true, otherwise the first defor year
    dataset[newcol] = np.where(mask1, dataset['defor1'], 
                               np.where(mask2, dataset['defor2'], 
                                        np.where(mask3, dataset['defor3'], 
                                                 dataset['defor1'])))

    # Assign corresponding confidence value
    dataset[confcol] = np.where(mask1, dataset['conf1'], 
                                np.where(mask2, dataset['conf2'], 
                                         np.where(mask3, dataset['conf3'], 
                                                  dataset['conf1'])))
    
    return dataset

# Copy validation data
proc_valdata = val_data.copy()

# Create new column to prioritize gfc matches
proc_valdata = agvalcol(proc_valdata, 'gfc', 'val_gfc', 'val_gfc_conf')

# Create new column to prioritize tmf matches
proc_valdata = agvalcol(proc_valdata, 'tmf', 'val_tmf', 'val_tmf_conf')

# Create new colun to priotize se matches
proc_valdata = agvalcol(proc_valdata, 'se', 'val_se', 'val_se_conf')

# Plot confusion matrices for gfc, tmf, se with processed validation data
cm_calcplot(proc_valdata, 'val_gfc')

# Calculate accuracy for processed validation data
tripacc("Weighted and matched validation data", proc_valdata, 'val_gfc', 
        'val_tmf', 'val_se', 'val_gfc_conf', 'val_tmf_conf', 'val_se_conf')

# Export pre-processed validation data
proc_valdata.to_csv("data/validation/validation_points_preprocessed.csv", 
                    index=False)



############################################################################


# VALIDATION MANIPULATION: PRIORITIZE AGREEING DEFOR YEAR (NEW METHOD)


############################################################################
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
def val_align(validation, confidence, product_data, label):
    
    # Create empty lists to hold validation year and confidence
    val = []
    conf = []
    
    # Iterate over each row (validation point)
    for row in range(len(product_data)):
        
        # Extract deforestation year
        year = product_data.iloc[row]
        
        # If deforestation year is detected in validation dataset
        if validation.iloc[row][year] == 1: 
            
            # Assign year to validation column
            val_year = year
        
        # If deforestation year is NOT detected in validation dataset
        else:
        
            # Assign first detected validation deforestation
            val_year = validation.columns[validation.iloc[row] == 1].min()
            
            # Assign 0 if no deforestation detected 2013-2023 (for 2012 case)
            if pd.isna(val_year):
                val_year = 0
            
        #Assign respective confidence
        val_conf = confidence.loc[row, val_year]

        # Append to the lists
        val.append(val_year)
        conf.append(val_conf)
    
    # Create dataframe with validation results
    val_results = pd.DataFrame({
        label: product_data,
        f"{label}_val": val,
        f"{label}_conf": conf})
    
    return val_results

# Create yearly validation data
yearly_val, yearly_conf = val_expand(val_data)

# Create aligned valiadtion data for gfc
gfc_val = val_align(yearly_val, yearly_conf, val_data['gfc'], "gfc")

# Create aligned validation data for tmf
tmf_val = val_align(yearly_val, yearly_conf, val_data['tmf'], "tmf")

# Create aligned validation data for se
se_val = val_align(yearly_val, yearly_conf, val_data['se'], "se")

# Combine datasets
comb_val = pd.concat([val_data.iloc[:,:2], gfc_val, tmf_val, se_val], axis = 1)

# Plot confusion matrices for gfc, tmf, se with processed validation data
cm_calcplot(comb_val, 'gfc_val')

# Calculate accuracy for processed validation data
tripacc("Weighted and matched validation data", comb_val, 'gfc_val', 
        'tmf_val', 'se_val', 'gfc_conf', 'tmf_conf', 'se_conf')

# Write combine dataset to csv
# comb_val.to_csv('data/validation/validation_points_preprocessed2.csv', index=False)
comb_val.to_csv('data/validation/validation_points_1380_preprocessed.csv', index=False)



