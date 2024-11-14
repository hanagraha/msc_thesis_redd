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
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
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



############################################################################


# IMPORT AND READ DATA


############################################################################
# Read validation data
val_data = pd.read_csv("data/validation/validation_points_labelled.csv", delimiter=";")

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

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
# Define dataset labels
yearlabs = [0] + list(years)

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

# Extract data from validation dataset
gfc = val_data['gfc']
tmf = val_data['tmf']
se = val_data['se']
val_first = val_data['defor1']

# Calculate confusion matrix for gfc
gfc_matrix = confusion_matrix(val_first, gfc)

# Calculate confusion matrix for tmf
tmf_matrix = confusion_matrix(val_first, tmf)

# Calculate confusion matrix for sensitive early combination
se_matrix = confusion_matrix(val_first, se)

# Plot calculated matrices
matrix_plt([gfc_matrix, tmf_matrix, se_matrix], datanames, 'd')



############################################################################


# CREATE CONFUSION MATRICES PER YEAR


############################################################################    
# Define annual labels
annlabs = ["Undisturbed"] + list(years)

# Define function to plot annual confusion matrices
def mlmatrix_plt(matrices, dataset):
    
    # Set up the figure and axes
    fig, axes = plt.subplots(3, 4, figsize=(15, 8))
    axes = axes.flatten()

    # Plot each confusion matrix
    for matrix, lab, i in zip(matrices, annlabs, range(0,12)):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(lab)
        axes[i].set_xlabel(f'{dataset} Predicted Labels')
        axes[i].set_ylabel('Validation Labels')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Calculate annual gfc confusion matrices
gfc_mlmatrix = multilabel_confusion_matrix(val_first, gfc, labels = yearlabs)

# Calculate annual tmf confusion matrices
tmf_mlmatrix = multilabel_confusion_matrix(val_first, tmf, labels = yearlabs)

# Calculate annual se confusion matrices
se_mlmatrix = multilabel_confusion_matrix(val_first, se, labels = yearlabs)

# Plot annual gfc confusion matrices
mlmatrix_plt(gfc_mlmatrix, "GFC")

# Plot annual tmf confusion matrices
mlmatrix_plt(tmf_mlmatrix, "TMF")

# Plot annual se confusion matrices
mlmatrix_plt(se_mlmatrix, "Sensitive Early")



############################################################################


# MANIPULATE STEHMAN CONFUSION MATRICES


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
    for data, name in zip(datalist, datanames):
        
        # Add lines with error bars
        plt.errorbar(data['year'], data[stat], yerr=data[se], fmt='-o', 
                     color=defaultblue, capsize=5, elinewidth=1, ecolor='r', 
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




















