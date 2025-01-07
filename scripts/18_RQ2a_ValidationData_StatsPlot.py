# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:43:31 2024

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

# Set default plotting colors
blue1 = "#1E2A5E"
blue2 = "#55679C"
blue3 = "#83B4FF"
blue4 = "#87C4FF"
bluecols = [blue1, blue2, blue3]

# other colors test
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]

# Define dataset labels
yearlabs = [0] + list(years)

# Define dataset names
datanames = ["GFC", "TMF", "Sensitive Early"]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Read validation data
val_data = pd.read_csv("data/validation/validation_points_preprocessed2.csv")

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

# Read gfc stehman statistic data (calculated in R, unprocessed)
gfc_stats = pd.read_csv("data/validation/gfc_stehmanstats.csv", delimiter=",")
gfc_cm = pd.read_csv("data/validation/gfc_confmatrix.csv", delimiter=",")

# Read gfc stehman statistic data (calculated in R, pre-processed)
proc_gfc_stats = pd.read_csv("data/validation/proc_gfc_stehmanstats.csv", delimiter=",")
proc_gfc_cm = pd.read_csv("data/validation/proc_gfc_confmatrix.csv", delimiter=",")

# Read gfc stehman statistic data (calculated in R, pre-processed 2)
proc2_gfc_stats = pd.read_csv("data/validation/proc2_gfc_stehmanstats.csv", delimiter=",")

# Read tmf stehman statistic data (calculated in R, unprocessed)
tmf_stats = pd.read_csv("data/validation/tmf_stehmanstats.csv", delimiter=",")
tmf_cm = pd.read_csv("data/validation/tmf_confmatrix.csv", delimiter=",")

# Read tmf stehman statistic data (calculated in R, pre-processed)
proc_tmf_stats = pd.read_csv("data/validation/proc_tmf_stehmanstats.csv", delimiter=",")
proc_tmf_cm = pd.read_csv("data/validation/proc_tmf_confmatrix.csv", delimiter=",")

# Read tmf stehman statistic data (calculated in R, pre-processed 2)
proc2_tmf_stats = pd.read_csv("data/validation/proc2_tmf_stehmanstats.csv", delimiter=",")

# Read se stehman statistic data (calculated in R, unprocessed)
se_stats = pd.read_csv("data/validation/se_stehmanstats.csv", delimiter=",")
se_cm = pd.read_csv("data/validation/se_confmatrix.csv", delimiter=",")

# Read se stehman statistic data (calculated in R, pre-processed)
proc_se_stats = pd.read_csv("data/validation/proc_se_stehmanstats.csv", delimiter=",")
proc_se_cm = pd.read_csv("data/validation/proc_se_confmatrix.csv", delimiter=",")

# Read se stehman statistic data (calculated in R, pre-processed 2)
proc2_se_stats = pd.read_csv("data/validation/proc2_se_stehmanstats.csv", delimiter=",")



############################################################################


# PLOT STEHMAN CONFUSION MATRICES (UNPROCESSED)


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

# Format stehman's gfc confusion matrix
gfc_scm = steh_cm(gfc_cm, 3)

# Format stehman's tmf confusion matrix
tmf_scm = steh_cm(tmf_cm, 3)

# Format stehman's se confusion matrix
se_scm = steh_cm(se_cm, 3)

# Plot confusion matrices
matrix_plt([gfc_scm, tmf_scm, se_scm], datanames, '.2f')



############################################################################


# PLOT STEHMAN CONFUSION MATRICES (PRE-PROCESSED)


############################################################################
# Format stehman's gfc confusion matrix
proc_gfc_scm = steh_cm(proc_gfc_cm, 3)

# Format stehman's tmf confusion matrix
proc_tmf_scm = steh_cm(proc_tmf_cm, 3)

# Format stehman's se confusion matrix
proc_se_scm = steh_cm(proc_se_cm, 3)

# Plot confusion matrices
matrix_plt([proc_gfc_scm, proc_tmf_scm, proc_se_scm], datanames, '.2f')



############################################################################


# STEHMAN STATISTICS PLOTTING FUNCTIONS


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
    
# Define function to plot user's and producer's accuracies side by side
def steh_dual_lineplt(datalist, datanames):
    
    # Initialize figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Subplot for User's Accuracy
    axes[0].set_title("User's Accuracy")
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add data and error bars
        axes[0].errorbar(data['year'], data['ua'], yerr=data['se_ua'], fmt='-o', 
                         color=col, capsize=5, elinewidth=1, ecolor=col, 
                         label=name, linewidth = 2)
    
    # Add gridlines
    axes[0].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[0].set_xticks(years)
    
    # Add axes labels
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel("Accuracy")
    
    # Add legend
    axes[0].legend()

    # Subplot for Producer's Accuracy
    axes[1].set_title("Producer's Accuracy")
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add data and error bars
        axes[1].errorbar(data['year'], data['pa'], yerr=data['se_pa'], fmt='-o', 
                         color=col, capsize=5, elinewidth=1, ecolor=col, 
                         label=name, linewidth = 2)
    
    # Add gridlines
    axes[1].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[1].set_xticks(years)
    
    # Add x labels
    axes[1].set_xlabel('Year')
    
    # Add legend
    axes[1].legend()

    # Add tight layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()
    


############################################################################


# PLOT STEHMAN STATISTICS (UNPROCESSED)


############################################################################
# Remove gfc statistics for year 0
gfc_stats0 = gfc_stats[2:]

# Remove tmf statistics for year 0
tmf_stats0 = tmf_stats[2:]

# Remove se statistics for year 0
se_stats0 = se_stats[2:]

# Plot user's accuracy
steh_lineplt([gfc_stats0, tmf_stats0, se_stats0], 'ua', 'se_ua', 
             "User's Accuracy")

# Plot producer's accuracy
steh_lineplt([gfc_stats0, tmf_stats0, se_stats0], 'pa', 'se_pa', 
             "Producer's Accuracy")

# Plot area estimates
steh_lineplt([gfc_stats0, tmf_stats0, se_stats0], 'area', 'se_a', 
             "Deforestation Area Estimates")

# Plot user's and producer's accuracy side by side
steh_dual_lineplt([gfc_stats0, tmf_stats0, se_stats0], datanames)



############################################################################


# PLOT STEHMAN STATISTICS (PRE-PROCESSED)


############################################################################
# Remove gfc statistics for year 0
proc_gfc_stats0 = proc_gfc_stats[2:]

# Remove tmf statistics for year 0
proc_tmf_stats0 = proc_tmf_stats[2:]

# Remove se statistics for year 0
proc_se_stats0 = proc_se_stats[2:]

# Plot user's accuracy
steh_lineplt([proc_gfc_stats0, proc_tmf_stats0, proc_se_stats0], 'ua', 'se_ua', 
             "User's Accuracy")

# Plot producer's accuracy
steh_lineplt([proc_gfc_stats0, proc_tmf_stats0, proc_se_stats0], 'pa', 'se_pa', 
             "Producer's Accuracy")

# Plot area estimates
steh_lineplt([proc_gfc_stats0, proc_tmf_stats0, proc_se_stats0], 'area', 'se_a', 
             "Deforestation Area Estimates")

# Plot user's and producer's accuracy side by side
steh_dual_lineplt([proc_gfc_stats0, proc_tmf_stats0, proc_se_stats0], datanames)



############################################################################


# PLOT STEHMAN STATISTICS (PRE-PROCESSED 2)


############################################################################
# Remove gfc statistics for year 0
proc2_gfc_stats0 = proc2_gfc_stats[1:]

# Remove tmf statistics for year 0
proc2_tmf_stats0 = proc2_tmf_stats[1:]

# Remove se statistics for year 0
proc2_se_stats0 = proc2_se_stats[1:]

# Plot user's accuracy
steh_lineplt([proc2_gfc_stats0, proc2_tmf_stats0, proc2_se_stats0], 'ua', 'se_ua', 
             "User's Accuracy")

# Plot producer's accuracy
steh_lineplt([proc2_gfc_stats0, proc2_tmf_stats0, proc2_se_stats0], 'pa', 'se_pa', 
             "Producer's Accuracy")

# Plot area estimates
steh_lineplt([proc2_gfc_stats0, proc2_tmf_stats0, proc2_se_stats0], 'area', 'se_a', 
             "Deforestation Area Estimates")

# Plot user's and producer's accuracy side by side
steh_dual_lineplt([proc2_gfc_stats0, proc2_tmf_stats0, proc2_se_stats0], datanames)



############################################################################


# PLOT COMMISSION / OMISSION ERROR


############################################################################
"""
Producer's Accuracy = 100%-Omission Error
User's Accuracy = 100%-Commission Error
source: https://gsp.humboldt.edu/olm/courses/GSP_216/lessons/accuracy/metrics.html
"""
# Define function to calculate commission and ommission error
def comom_err(dataset):
    
    # Extract producers accuracy
    pa = dataset['pa']
    
    # Extract users accuracy
    ua = dataset['ua']
    
    # Calculate ommission error
    oe = 1 - pa
    
    # Calculate commission error
    ce = 1 - ua
    
    # Combine ommission and commission error
    errors = pd.DataFrame({
        'year': dataset['year'],
        'oe': oe,
        'ce': ce})
    
    return errors

# Define function to plot user's and producer's accuracies side by side
def errors_lineplt(datalist, datanames):
    
    # Initialize figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Subplot for User's Accuracy
    axes[0].set_title("Ommission Error")
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add ommission error data
        axes[0].plot(data['year'], data['oe'], color=col, label=name, linewidth=2)
    
    # Add gridlines
    axes[0].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[0].set_xticks(data['year'])
    
    # Add axes labels
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel("Error")
    
    # Add legend
    axes[0].legend()

    # Subplot for Producer's Accuracy
    axes[1].set_title("Commission Error")
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add commission error data
        axes[1].plot(data['year'], data['ce'], color=col, label=name, linewidth=2)
    
    # Add gridlines
    axes[1].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[1].set_xticks(data['year'])
    
    # Add x labels
    axes[1].set_xlabel('Year')
    
    # Add legend
    axes[1].legend()

    # Add tight layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()

# Calculate commission and ommission errors for gfc
gfc_errors = comom_err(proc2_gfc_stats)

# Calculate commission and ommission errors for tmf
tmf_errors = comom_err(proc2_tmf_stats)

# Calculate commission and ommission errors for se
se_errors = comom_err(proc2_se_stats)

# Plot error data
errors_lineplt([gfc_errors[1:], tmf_errors[1:], se_errors[1:]], datanames)









