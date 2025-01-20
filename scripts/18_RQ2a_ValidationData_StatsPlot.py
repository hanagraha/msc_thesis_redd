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
val_dir = os.path.join('data', 'validation')

# Set year range
years = range(2013, 2024)

# Set default plotting colors
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]

# Define dataset labels
yearlabs = [0] + list(years)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read files in subfolder
def folder_files(folder, suffix):
    
    # Define folder path
    folderpath = os.path.join(val_dir, folder)
    
    # Create empty list to store files
    paths = []

    # Iterate over every item in folder
    for file in os.listdir(folderpath):
        
        # Check if file ends in suffix
        if file.endswith(suffix):
            
            # Create path for file
            filepath = os.path.join(folderpath, file)
            
            # Add file to list
            paths.append(filepath)
    
    return paths

# Define function to read files from list
def list_read(pathlist, suffix):
    
    # Create empty dictionary to store outputs
    files = {}
    
    # Iterate over each file in list
    for path in pathlist:
        
        # Read file
        data = pd.read_csv(path)
        
        # Extract file name
        filename = os.path.basename(path)
        
        # Remove suffix from filename
        var = filename.replace(suffix, "")
        
        # Add data to dictionary
        files[var] = data
        
    return files
        
# Read validation data
val_data = pd.read_csv("data/validation/validation_points.csv")

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

# Read protocol a data
prota_filepaths = folder_files("val_prota", ".csv")
prota_files = list_read(prota_filepaths, ".csv")

# Read protocol b statistics
protb_statpaths = folder_files("val_protb", "stehmanstats.csv")
protb_stats = list_read(protb_statpaths, "_stehmanstats.csv")

# Read protocol c statistics
protc_statpaths = folder_files("val_protc", "stehmanstats.csv")
protc_stats = list_read(protc_statpaths, "_stehmanstats.csv")

# Read protocol d statistics
protd_statpaths = folder_files("val_protd", "stehmanstats.csv")
protd_stats = list_read(protd_statpaths, "_stehmanstats.csv")

# Read protocol b confusion matrices
protb_cmpaths = folder_files("val_protb", "confmatrix.csv")
protb_cm = list_read(protb_cmpaths, "_confmatrix.csv")

# Read protocol c confusion matrices
protc_cmpaths = folder_files("val_protc", "confmatrix.csv")
protc_cm = list_read(protc_cmpaths, "_confmatrix.csv")

# Read protocol d confusion matrices
protd_cmpaths = folder_files("val_protd", "confmatrix.csv")
protd_cm = list_read(protd_cmpaths, "_confmatrix.csv")


# %%
############################################################################


# DEFINE CONFUSION MATRICES PLOTTING FUNCTIONS


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



############################################################################


# DEFINE STEHMAN STATISTICS PLOTTING FUNCTIONS


############################################################################
"""
Producer's Accuracy = 100%-Omission Error
User's Accuracy = 100%-Commission Error
source: https://gsp.humboldt.edu/olm/courses/GSP_216/lessons/accuracy/metrics.html
"""
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
    

# %%
############################################################################


# PROTOCOL A: CALCULATE STATISTICS


############################################################################
# Define function to calculate accuracy
def prota_acc(datavals):
    
    # Extract number of agreements
    val1 = (datavals['prot_a'] == 1).sum()
    
    # Extract number of disagreements
    val0 = (datavals['prot_a'] == 0).sum()
    
    # Calculate accuracy
    acc = val1 / (val1 + val0)
    
    return acc

# Define function to calculate accuracy
def prota_accs(datadic):
    
    # Create empty dictionary to store information
    accuracies = {}
    
    # Iterate over each item in the dictionary
    for key, data in datadic.items():
        
        # Extract number of agreements
        val1 = (data['prot_a'] == 1).sum()
        
        # Extract number of disagreements
        val2 = (data['prot_a'] == 0).sum()
        
        # Calculate accuracy
        acc = val1 / (val1 + val2)
        
        # Add accuracy to dictionary
        accuracies[key] = acc
    
    return accuracies

# Calculate protocol a accuracies
prota_accuracies = prota_accs(prota_files)


# %%
############################################################################


# PROTOCOL B: PLOT CONFUSION MATRICES


############################################################################
# Create protocol b gfc cm
protb_gfc_cm = steh_cm(protb_cm["protb_gfc"], 2)

# Create protocol b tmf cm
protb_tmf_cm = steh_cm(protb_cm["protb_tmf"], 2)

# Create protocol b se cm
protb_se_cm = steh_cm(protb_cm["protb_se"], 2)

# Define dataset names
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot confusion matrices
matrix_plt([protb_gfc_cm, protb_tmf_cm, protb_se_cm], datanames, '.2f')


# %%
############################################################################


# PROTOCOL B: PLOT STEHMAN STATISTICS


############################################################################
"""
Definition of Agreement: mark agreement if prediction defor year matches
any year between observed defor and regr, for any deforestation event
"""
# Define datanames
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot gfc, tmf, and se accuracies
steh_dual_lineplt([protb_stats["protb_gfc"][1:], 
                   protb_stats["protb_tmf"][1:], 
                   protb_stats["protb_se"][1:]], datanames)

# Plot gfc, tmf, and se errors
errors_lineplt([comom_err(protb_stats["protb_gfc"][1:]),
                comom_err(protb_stats["protb_tmf"][1:]),
                comom_err(protb_stats["protb_se"][1:])], datanames)

# Define dataset names
datanames = ["GFC REDD+", "GFC Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats["protb_gfc_redd"][1:],
                   protb_stats["protb_gfc_nonredd"][1:]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats["protb_gfc_redd"][1:]),
                comom_err(protb_stats["protb_gfc_nonredd"][1:])], datanames)

# Define dataset names
datanames = ["TMF REDD+", "TMF Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats["protb_tmf_redd"][1:],
                   protb_stats["protb_tmf_nonredd"][1:]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats["protb_tmf_redd"][1:]),
                comom_err(protb_stats["protb_tmf_nonredd"][1:])], datanames)

# Define dataset names
datanames = ["SE REDD+", "SE Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats["protb_se_redd"][1:],
                   protb_stats["protb_se_nonredd"][1:]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats["protb_se_redd"][1:]),
                comom_err(protb_stats["protb_se_nonredd"][1:])], datanames)


# %%
############################################################################


# PROTOCOL C: PLOT STEHMAN STATISTICS


############################################################################
"""
Definition of Agreement: time sensitive. mark agreement if predicted defor
year matches validation defor year of the first, second, or third observed 
defor event
"""
# Define datanames
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot gfc, tmf, and se accuracies
steh_dual_lineplt([protc_stats["protc_gfc"][2:13], 
                   protc_stats["protc_tmf"][2:13], 
                   protc_stats["protc_se"][2:13]], datanames)

# Plot gfc, tmf, and se errors
errors_lineplt([comom_err(protc_stats["protc_gfc"][2:13]),
                comom_err(protc_stats["protc_tmf"][2:13]),
                comom_err(protc_stats["protc_se"][2:13])], datanames)

# Define dataset names
datanames = ["GFC REDD+", "GFC Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats["protc_gfc_redd"][2:13],
                   protc_stats["protc_gfc_nonredd"][2:13]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats["protc_gfc_redd"][2:13]),
                comom_err(protc_stats["protc_gfc_nonredd"][2:13])], datanames)

# Define dataset names
datanames = ["TMF REDD+", "TMF Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats["protc_tmf_redd"][2:13],
                   protc_stats["protc_tmf_nonredd"][2:13]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats["protc_gfc_redd"][2:13]),
                comom_err(protc_stats["protc_gfc_nonredd"][2:13])], datanames)

# Define dataset names
datanames = ["SE REDD+", "SE Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats["protc_se_redd"][2:13],
                   protc_stats["protc_se_nonredd"][2:13]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats["protc_gfc_redd"][2:13]),
                comom_err(protc_stats["protc_gfc_nonredd"][2:13])], datanames)


# %%
############################################################################


# PROTOCOL D: PLOT STEHMAN STATISTICS


############################################################################
"""
Definition of Agreement: time sensitive. mark agreement if predicted defor
year matches validation defor year of the first observed defor event
"""
# Define datanames
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot gfc, tmf, and se accuracies
steh_dual_lineplt([protd_stats["protd_gfc"][2:13], 
                   protd_stats["protd_tmf"][2:13], 
                   protd_stats["protd_se"][2:13]], datanames)

# Plot gfc, tmf, and se errors
errors_lineplt([comom_err(protd_stats["protd_gfc"][2:13]),
                comom_err(protd_stats["protd_tmf"][2:13]),
                comom_err(protd_stats["protd_se"][2:13])], datanames)

# Define dataset names
datanames = ["GFC REDD+", "GFC Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protd_stats["protd_gfc_redd"][2:13],
                   protd_stats["protd_gfc_nonredd"][2:13]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protd_stats["protd_gfc_redd"][2:13]),
                comom_err(protd_stats["protd_gfc_nonredd"][2:13])], datanames)

# Define dataset names
datanames = ["TMF REDD+", "TMF Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protd_stats["protd_tmf_redd"][2:13],
                   protd_stats["protd_tmf_nonredd"][2:13]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protd_stats["protd_gfc_redd"][2:13]),
                comom_err(protd_stats["protd_gfc_nonredd"][2:13])], datanames)

# Define dataset names
datanames = ["SE REDD+", "SE Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protd_stats["protd_se_redd"][2:13],
                   protd_stats["protd_se_nonredd"][2:13]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protd_stats["protd_gfc_redd"][2:13]),
                comom_err(protd_stats["protd_gfc_nonredd"][2:13])], datanames)














