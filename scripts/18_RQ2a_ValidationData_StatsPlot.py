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
yearlabs = [0] + list(years) + ['sum']

# Generate year strings for 2013-2023
year_strings = [str(y) for y in yearlabs]



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
def list_read(pathlist, suffix, filt = False, cmfilt = False):
    
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
    
    # If filter is true
    if filt == True:
        
        # Iterate over each read file
        for key in files:
            
            # Subset to only keep years 2013-2023
            files[key] = files[key][(files[key]['year'] >= 2013) & (files[key]['year'] <= 2023)]
            
            # Reset index
            files[key] = files[key].reset_index(drop = True)
    
    # If confusion matrix filter is true
    if cmfilt == True:
        
        # Iterate over each read file
        for key in files:
            
            # Define the rows to keep
            keeprows = files[key]['Unnamed: 0'].isin(year_strings)
        
            # Define the columns to keep
            keepcols = year_strings
            
            # Subset the DataFrame
            files[key] = files[key].loc[keeprows, keepcols]
            
            # Set index of dataframe
            files[key].index = keepcols
    
    return files
    
# Read protocol a data
prota_filepaths = folder_files("val_prota", "stehmanstats.csv")
prota_files = list_read(prota_filepaths, "_stehmanstats.csv")

# Read protocol b statistics
protb_statpaths = folder_files("val_protb", "stehmanstats.csv")
protb_stats = list_read(protb_statpaths, "_stehmanstats.csv", filt = True)

# Read protocol c statistics
protc_statpaths = folder_files("val_protc", "stehmanstats.csv")
protc_stats = list_read(protc_statpaths, "_stehmanstats.csv", filt = True)

# Read protocol a data (buffered)
prota_filepaths_buff = folder_files("val_prota_buff", "stehmanstats.csv")
prota_files_buff = list_read(prota_filepaths_buff, "_stehmanstats.csv")

# Read protocol b statistics (buffered)
protb_statpaths_buff = folder_files("val_protb_buff", "stehmanstats.csv")
protb_stats_buff = list_read(protb_statpaths_buff, "_stehmanstats.csv", filt = True)

# Read protocol c statistics (buffered)
protc_statpaths_buff = folder_files("val_protc_buff", "stehmanstats.csv")
protc_stats_buff = list_read(protc_statpaths_buff, "_stehmanstats.csv", filt = True)

# Read protocol a confusion matrices
prota_cmpaths = folder_files("val_prota", "confmatrix.csv")
prota_cm = list_read(prota_cmpaths, "_confmatrix.csv")

# Read protocol b confusion matrices
protb_cmpaths = folder_files("val_protb", "confmatrix.csv")
protb_cm = list_read(protb_cmpaths, "_confmatrix.csv", cmfilt = True)

# Read protocol c confusion matrices
protc_cmpaths = folder_files("val_protc", "confmatrix.csv")
protc_cm = list_read(protc_cmpaths, "_confmatrix.csv", cmfilt = True)

# Read protocol a confusion matrices (buffered)
prota_cmpaths_buff = folder_files("val_prota_buff", "confmatrix.csv")
prota_cm_buff = list_read(prota_cmpaths_buff, "_confmatrix.csv")

# Read protocol b confusion matrices (buffered)
protb_cmpaths_buff = folder_files("val_protb_buff", "confmatrix.csv")
protb_cm_buff = list_read(protb_cmpaths_buff, "_confmatrix.csv", cmfilt = True)

# Read protocol c confusion matrices (buffered)
protc_cmpaths_buff = folder_files("val_protc_buff", "confmatrix.csv")
protc_cm_buff = list_read(protc_cmpaths_buff, "_confmatrix.csv", cmfilt = True)


#%%
############################################################################


# DEFINE CONFUSION MATRICES PLOTTING FUNCTIONS


############################################################################
# Define function to formate stehman confusion matrix
def steh_cm(stehman_cm, deci):
    
    # Exclude the sum row and 0 year
    cm = stehman_cm.iloc[1:-1, 1:-1]
    
    # Fill na values with 0
    cm = cm.fillna(0)
    
    # Convert dataframe to array
    cm = cm.to_numpy()
    
    # Multiply by 100 to convert to %
    cm = cm * 100
    
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
                    xticklabels=years, yticklabels=years, vmax = 6)
        
        # Add title
        # axes[i].set_title(f'{names[i]} Confusion Matrix', fontsize = 16)
        axes[i].set_xlabel('Validation Labels', fontsize = 16)
        axes[i].set_ylabel(f'{names[i]} Predicted Labels', fontsize = 16)
    
        # Adjust tick labels font size
        axes[i].tick_params(axis='both', labelsize=14)
    
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
    axes[0].set_title("User's Accuracy", fontsize = 16)
    
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
    
    # Adjust tickmark font size
    axes[0].tick_params(axis='both', labelsize = 14)
    
    # Add axes labels
    axes[0].set_xlabel('Year', fontsize = 14)
    axes[0].set_ylabel("Accuracy", fontsize = 14)
    
    # Add legend
    axes[0].legend(fontsize = 16)

    # Subplot for Producer's Accuracy
    axes[1].set_title("Producer's Accuracy", fontsize = 16)
    
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
    
    # Adjust tickmark font size
    axes[1].tick_params(axis='both', labelsize = 14)
    
    # Add x labels
    axes[1].set_xlabel('Year', fontsize = 14)
    
    # Add legend
    axes[1].legend(fontsize = 16)

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
    axes[0].set_title("Commission Error", fontsize = 16)
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add ommission error data
        axes[0].plot(data['year'], data['ce'], color=col, label=name, linewidth=2)
    
    # Add gridlines
    axes[0].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[0].set_xticks(data['year'])
    
    # Adjust tickmark font size
    axes[0].tick_params(axis='both', labelsize = 14)
    
    # Add axes labels
    axes[0].set_xlabel('Year', fontsize = 14)
    axes[0].set_ylabel("Error", fontsize = 14)
    
    # Add legend
    axes[0].legend(fontsize = 16)

    # Subplot for Producer's Accuracy
    axes[1].set_title("Omission Error", fontsize = 16)
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add commission error data
        axes[1].plot(data['year'], data['oe'], color=col, label=name, linewidth=2)
    
    # Add gridlines
    axes[1].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[1].set_xticks(data['year'])
    
    # Adjust tickmark font size
    axes[1].tick_params(axis='both', labelsize = 14)
    
    # Add x labels
    axes[1].set_xlabel('Year', fontsize = 14)
    
    # Add legend
    axes[1].legend(fontsize = 16)

    # Add tight layout for better spacing
    plt.tight_layout()

    # Display the plot
    plt.show()


# %%
############################################################################


# PLOT CONFUSION MATRICES (PROTOCOL B AND C)


############################################################################
# Create protocol b gfc cm
protb_gfc_cm = steh_cm(protb_cm["protb_gfc"], 2)

# Create protocol b tmf cm
protb_tmf_cm = steh_cm(protb_cm["protb_tmf"], 2)

# Create protocol b se cm
protb_se_cm = steh_cm(protb_cm["protb_se"], 2)

# Define dataset names
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot confusion matrices (protocol b)
matrix_plt([protb_gfc_cm, protb_tmf_cm, protb_se_cm], datanames, '.2f')

# Create protocol b gfc cm (buffered)
protb_gfc_cm_buff = steh_cm(protb_cm_buff["protb_gfc"], 2)

# Create protocol b tmf cm (buffered)
protb_tmf_cm_buff = steh_cm(protb_cm_buff["protb_tmf"], 2)

# Create protocol b se cm (buffered)
protb_se_cm_buff = steh_cm(protb_cm["protb_se"], 2)

# Plot confusion matrices (protocol b buff)
matrix_plt([protb_gfc_cm_buff, protb_tmf_cm_buff, protb_se_cm_buff], datanames, '.2f')

# Create protocol c gfc cm
protc_gfc_cm = steh_cm(protc_cm["protc_gfc"], 2)

# Create protocol c tmf cm
protc_tmf_cm = steh_cm(protc_cm["protc_tmf"], 2)

# Create protocol c se cm
protc_se_cm = steh_cm(protc_cm["protc_se"], 2)

# Plot confusion matrices (protocol c)
matrix_plt([protc_gfc_cm, protc_tmf_cm, protc_se_cm], datanames, '.2f')

# Create protocol c gfc cm (buffered)
protc_gfc_cm_buff = steh_cm(protc_cm_buff["protc_gfc"], 2)

# Create protocol c tmf cm (buffered)
protc_tmf_cm_buff = steh_cm(protc_cm_buff["protc_tmf"], 2)

# Create protocol c se cm (buffered)
protc_se_cm_buff = steh_cm(protc_cm_buff["protc_se"], 2)

# Plot confusion matrices (protocol c buff)
matrix_plt([protc_gfc_cm_buff, protc_tmf_cm_buff, protc_se_cm_buff], datanames, '.2f')


# %%
############################################################################


# PROTOCOL B: PLOT STEHMAN STATISTICS (NO BUFFER)


############################################################################   
# Define datanames
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot gfc, tmf, and se accuracies
steh_dual_lineplt([protb_stats["protb_gfc"], 
                   protb_stats["protb_tmf"], 
                   protb_stats["protb_se"]], datanames)

# Plot gfc, tmf, and se errors
errors_lineplt([comom_err(protb_stats["protb_gfc"]),
                comom_err(protb_stats["protb_tmf"]),
                comom_err(protb_stats["protb_se"])], datanames)

# Define dataset names
datanames = ["GFC REDD+", "GFC Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats["protb_gfc_redd"],
                   protb_stats["protb_gfc_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats["protb_gfc_redd"]),
                comom_err(protb_stats["protb_gfc_nonredd"])], datanames)

# Define dataset names
datanames = ["TMF REDD+", "TMF Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats["protb_tmf_redd"],
                   protb_stats["protb_tmf_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats["protb_tmf_redd"]),
                comom_err(protb_stats["protb_tmf_nonredd"])], datanames)

# Define dataset names
datanames = ["SE REDD+", "SE Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats["protb_se_redd"],
                   protb_stats["protb_se_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats["protb_se_redd"]),
                comom_err(protb_stats["protb_se_nonredd"])], datanames)


# %%
############################################################################


# PROTOCOL B: PLOT STEHMAN STATISTICS (WITH BUFFER)


############################################################################
# Define datanames
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot gfc, tmf, and se accuracies
steh_dual_lineplt([protb_stats_buff["protb_gfc"], 
                   protb_stats_buff["protb_tmf"], 
                   protb_stats_buff["protb_se"]], datanames)

# Plot gfc, tmf, and se errors
errors_lineplt([comom_err(protb_stats_buff["protb_gfc"]),
                comom_err(protb_stats_buff["protb_tmf"]),
                comom_err(protb_stats_buff["protb_se"])], datanames)

# Define dataset names
datanames = ["GFC REDD+", "GFC Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats_buff["protb_gfc_redd"],
                   protb_stats_buff["protb_gfc_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats_buff["protb_gfc_redd"]),
                comom_err(protb_stats_buff["protb_gfc_nonredd"])], datanames)

# Define dataset names
datanames = ["TMF REDD+", "TMF Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats_buff["protb_tmf_redd"],
                   protb_stats_buff["protb_tmf_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats_buff["protb_tmf_redd"]),
                comom_err(protb_stats_buff["protb_tmf_nonredd"])], datanames)

# Define dataset names
datanames = ["SE REDD+", "SE Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protb_stats_buff["protb_se_redd"],
                   protb_stats_buff["protb_se_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protb_stats_buff["protb_se_redd"]),
                comom_err(protb_stats_buff["protb_se_nonredd"])], datanames)


# %%
############################################################################


# PROTOCOL C: PLOT STEHMAN STATISTICS


############################################################################
# Define datanames
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot gfc, tmf, and se accuracies
steh_dual_lineplt([protc_stats["protc_gfc"], 
                   protc_stats["protc_tmf"], 
                   protc_stats["protc_se"]], datanames)

# Plot gfc, tmf, and se errors
errors_lineplt([comom_err(protc_stats["protc_gfc"]),
                comom_err(protc_stats["protc_tmf"]),
                comom_err(protc_stats["protc_se"])], datanames)

# Define dataset names
datanames = ["GFC REDD+", "GFC Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats["protc_gfc_redd"],
                   protc_stats["protc_gfc_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats["protc_gfc_redd"]),
                comom_err(protc_stats["protc_gfc_nonredd"])], datanames)

# Define dataset names
datanames = ["TMF REDD+", "TMF Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats["protc_tmf_redd"],
                   protc_stats["protc_tmf_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats["protc_tmf_redd"]),
                comom_err(protc_stats["protc_tmf_nonredd"])], datanames)

# Define dataset names
datanames = ["SE REDD+", "SE Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats["protc_se_redd"],
                   protc_stats["protc_se_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats["protc_se_redd"]),
                comom_err(protc_stats["protc_se_nonredd"])], datanames)


# %%
############################################################################


# PROTOCOL C: PLOT STEHMAN STATISTICS (WITH BUFFER)


############################################################################
# Define datanames
datanames = ["GFC", "TMF", "Sensitive Early"]

# Plot gfc, tmf, and se accuracies
steh_dual_lineplt([protc_stats_buff["protc_gfc"], 
                   protc_stats_buff["protc_tmf"], 
                   protc_stats_buff["protc_se"]], datanames)

# Plot gfc, tmf, and se errors
errors_lineplt([comom_err(protc_stats_buff["protc_gfc"]),
                comom_err(protc_stats_buff["protc_tmf"]),
                comom_err(protc_stats_buff["protc_se"])], datanames)

# Define dataset names
datanames = ["GFC REDD+", "GFC Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats_buff["protc_gfc_redd"],
                   protc_stats_buff["protc_gfc_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats_buff["protc_gfc_redd"]),
                comom_err(protc_stats_buff["protc_gfc_nonredd"])], datanames)

# Define dataset names
datanames = ["TMF REDD+", "TMF Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats_buff["protc_tmf_redd"],
                   protc_stats_buff["protc_tmf_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats_buff["protc_tmf_redd"]),
                comom_err(protc_stats_buff["protc_tmf_nonredd"])], datanames)

# Define dataset names
datanames = ["SE REDD+", "SE Non-REDD+"]

# Plot redd+ and non-redd+ accuracies
steh_dual_lineplt([protc_stats_buff["protc_se_redd"],
                   protc_stats_buff["protc_se_nonredd"]], datanames)

# Plot redd+ and nonredd+ errors
errors_lineplt([comom_err(protc_stats_buff["protc_se_redd"]),
                comom_err(protc_stats_buff["protc_se_nonredd"])], datanames)






