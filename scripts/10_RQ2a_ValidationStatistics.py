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
from sklearn.metrics import confusion_matrix, accuracy_score
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

# Set default plotting colors
defaultblue = "#4682B4"
reddcol = "brown"
nonreddcol = "dodgerblue"
grnpcol = "darkgreen"

# blue colors
blue1 = "#1E2A5E"
blue2 = "#55679C"
blue3 = "#83B4FF"
blue4 = "#87C4FF"
bluecols = [blue1, blue2, blue3]

# Define dataset labels
yearlabs = [0] + list(years)




############################################################################


# IMPORT AND READ DATA


############################################################################
# Read validation data
val_data = pd.read_csv("data/validation/validation_points_labelled.csv", delimiter=";")
val_data_st7 = pd.read_csv("data/validation/validation_points_labelled_minstrata7.csv", delimiter=";")

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])
val_data_st7['geometry'] = gpd.GeoSeries.from_wkt(val_data_st7['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 
val_data_st7 = gpd.GeoDataFrame(val_data_st7, geometry='geometry', crs="EPSG:32629") 

# Read gfc stehman statistic data (calculated in R)
gfc_stehman = pd.read_csv("data/validation/gfc_stehmanstats.csv", delimiter=",")
gfc_stehman_cm = pd.read_csv("data/validation/gfc_confmatrix.csv", delimiter=",")

# Read tmf stehman statistic data (calculated in R)
tmf_stehman = pd.read_csv("data/validation/tmf_stehmanstats.csv", delimiter=",")
tmf_stehman_cm = pd.read_csv("data/validation/tmf_confmatrix.csv", delimiter=",")

# Read se stehman statistic data (calculated in R)
se_stehman = pd.read_csv("data/validation/se_stehmanstats.csv", delimiter=",")
se_stehman_cm = pd.read_csv("data/validation/se_confmatrix.csv", delimiter=",")

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


# VALIDATION MANIPULATION: SUBTRACT 1 


############################################################################
# Define function to subtract from all non-0 deforestation and regrowth years
def col_sub1(dataset, col_list):
    
    # Iterate over each column
    for col in col_list:
        
        # Subtract 1 from cells with non-0 data
        dataset[col] = np.where(dataset[col] != 0, dataset[col] - 1, 
                                dataset[col])
        
        # Convert cells with value 2012 to 0
        dataset[col] = np.where(dataset[col] == 2012, 0, dataset[col])
        
    return dataset
    
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

# Copy validation data
proc_valdata = val_data.copy()
proc_valdata_st7 = val_data_st7.copy()

# Subtract 1 from deforestation and regrowth years
proc_valdata = col_sub1(proc_valdata, ['defor1', 'regr1', 'defor2', 'regr2',
                                       'defor3', 'regr3'])

proc_valdata_st7 = col_sub1(proc_valdata_st7, ['defor1', 'regr1', 'defor2', 
                                               'regr2', 'defor3', 'regr3'])

# Calculate overall accuracy of raw validation data
tripacc("Raw validation data", val_data, 'defor1')

# Calculate overall accuracy of processed validation data
tripacc("Subtracted 1 from validation data", proc_valdata, 'defor1')
tripacc("Subtracted 1 from validation data", proc_valdata_st7, 'defor1')

# Plot confusion matrices for gfc, tmf, se with processed validation data
cm_calcplot(proc_valdata, 'defor1')
cm_calcplot(proc_valdata_st7, 'defor1')

# Weighted overall accuracy of processed validation data
tripacc("Subtracted 1 and weighted validation data", proc_valdata, 
        col='defor1', sw1='conf1')
tripacc("Subtracted 1 and weighted validation data", proc_valdata_st7, 
        col='defor1', sw1='conf1')



############################################################################


# VALIDATION MANIPULATION: PRIORITIZE AGREEING DEFOR YEAR


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



############################################################################


# VALIDATION MANIPULATION: ONLY YEARS 2017-2023


############################################################################
# Remove strata 1-8 (agreement/disagreement 2013-2016)
subset_valdata = proc_valdata[(proc_valdata['strata'] > 8)]

# Calculate accuracy for processed validation data
tripacc("Validation data only from 2017-2023)", subset_valdata, 'val_gfc', 
        'val_tmf', 'val_se')



############################################################################


# PLOT STEHMAN CONFUSION MATRICES


############################################################################
# Define function to formate stehman confusion matrix
def steh_cm(stehman_cm, deci):
    
    # Exclude the sum row and indices
    cm = stehman_cm.iloc[:-1, 1:-1]
    
    # Fill na values with 0
    cm = cm.fillna(0)
    
    # Convert dataframe to array
    cm = cm.to_numpy()
    
    # Round to (deci) number of decimals
    cm = np.round(cm, deci)
    
    return cm

# Format stehman's gfc confusion matrix
gfc_scm = steh_cm(gfc_stehman_cm, 3)

# Format stehman's tmf confusion matrix
tmf_scm = steh_cm(tmf_stehman_cm, 3)

# Format stehman's se confusion matrix
se_scm = steh_cm(se_stehman_cm, 3)

# Plot confusion matrices
matrix_plt([gfc_scm, tmf_scm, se_scm], datanames, '.2f')



############################################################################


# MANIPULATE STEHMAN STATISTICS


############################################################################
# Define function to plot stehman stats
def steh_lineplt(datalist, stat, se, ylab):
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    
    # Iterate over each dataset
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add lines with error bars
        plt.errorbar(data['year'], data[stat], yerr=data[se], fmt='-o', 
                     color=col, capsize=5, elinewidth=1, ecolor=col, 
                     label=name)
    
    # Add gridlines
    plt.grid(True, linestyle = "--")
    
    # Add tickmarks
    plt.xticks(years)
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel(ylab)
    plt.title(f"{ylab} for GFC, TMF, and Sensitive Early Datasets")
    
    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# Remove gfc statistics for year 0
gfc_stehman0 = gfc_stehman[1:]

# Remove tmf statistics for year 0
tmf_stehman0 = tmf_stehman[1:]

# Remove se statistics for year 0
se_stehman0 = se_stehman[1:]

# Plot user's accuracy
steh_lineplt([gfc_stehman0, tmf_stehman0, se_stehman0], 'ua', 'se_ua', 
             "User's Accuracy")

# Plot producer's accuracy
steh_lineplt([gfc_stehman0, tmf_stehman0, se_stehman0], 'pa', 'se_pa', 
             "Producer's Accuracy")

# Plot area estimates
steh_lineplt([gfc_stehman0, tmf_stehman0, se_stehman0], 'area', 'se_a', 
             "Deforestation Area Estimates")




















