# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"Z:\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# READ DATA
# -------------------------------------------------------------------------
# Define year strings
years = range(2013, 2024)
yearlabs = [0] + list(years) + ['sum']
year_strings = [str(y) for y in yearlabs]

# Define color palette
bluecols = ["#1E2A5E", "#83B4FF"]

# Define data columns
datacols = ["year", "ua", "pa", "area", "se_ua", "se_pa", "se_a"]

# Define function to list files in subfolder
def listfiles(folder, suffixes=('_cm', '_stats')):

    # Define folder path
    folderpath = os.path.join('native_validation', folder)
    
    # Create empty list to store files
    paths = []

    # Iterate over items in folder
    for file in os.listdir(folderpath):
        
        # Remove extension before checking
        name, ext = os.path.splitext(file)

        if name.endswith(suffixes):
            filepath = os.path.join(folderpath, file)
            paths.append(filepath)

    return paths

# Define function to read files from list
def list_read(pathlist, suffix, subset=False):
    
    # Create empty dictionary to store outputs
    files = {}
    
    # Iterate over each file in list
    for path in pathlist:
        
        # Read file
        data = pd.read_csv(path)

        # Subset for rows
        if subset: 
            data = data.loc[data["year"].between(2013, 2023),
                            datacols]
        
        # Extract file name
        filename = os.path.basename(path)
        
        # Remove suffix from filename
        var = filename.replace(suffix, "")
        
        # Add data to dictionary
        files[var] = data
    
    return files

# Read time insensitive data
timeinsensitive_stats = list_read(listfiles("timeinsensitive", suffixes=('_stats')), suffix="_stats.csv")

# Read time sensitive data
anyyear_stats = list_read(listfiles("anyyear", suffixes=('_stats')), suffix="_stats.csv", subset=True)
anyyear_cm = list_read(listfiles("anyyear", suffixes=('_cm')), suffix="_cm.csv")
firstyear_stats = list_read(listfiles("firstyear", suffixes=('_stats')), suffix="_stats.csv", subset=True)
firstyear_cm = list_read(listfiles("firstyear", suffixes=('_stats')), suffix="_cm.csv")


# -------------------------------------------------------------------------
# DEFINE CONFUSION MATRICES PLOTTING FUNCTIONS
# -------------------------------------------------------------------------
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Iterate over each confusion matrix
    for i in range(0, len(matrices)):
        
        # Create heatmap
        sns.heatmap(matrices[i], annot=True, fmt=fmt, cmap='Blues', ax=axes[i],
                    xticklabels=years, yticklabels=years, vmax = 6)
        
        # Add title
        axes[i].set_xlabel('Validation Labels', fontsize = 16)
        axes[i].set_ylabel(f'{names[i]} Predicted Labels', fontsize = 16)
    
        # Adjust tick labels font size
        axes[i].tick_params(axis='both', labelsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# DEFINE STEHMAN STATISTICS PLOTTING FUNCTIONS
# -------------------------------------------------------------------------
# Define function to plot user's and producer's accuracies side by side
def steh_dual_lineplt(datalist, filename, datanames=["GFC", "TMF"]):
    
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

    # Define output filepath
    filepath = f"figs/native_validation/{filename}.png"

    # Save plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)

    # Print save confirmation
    print(f"Plot saved as: {filepath}")
    
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
        'ce': ce,
        'oe_se': dataset['se_pa'],
        'ce_se': dataset['se_ua']})
    
    return errors

# Define function to plot user's and producer's accuracies side by side
def errors_lineplt(datalist, filename, datanames=["GFC", "TMF"]):
    
    # Initialize figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Subplot for User's Accuracy
    axes[0].set_title("Commission Error", fontsize = 16)
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add ommission error data
        axes[0].errorbar(data['year'], data['ce'], yerr=data['ce_se'], fmt='-o', 
                         color = col, capsize = 5, elinewidth = 1, label=name, 
                         ecolor = col, linewidth=2)
    
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
        axes[1].errorbar(data['year'], data['oe'], yerr=data['oe_se'], fmt='-o', 
                         color = col, capsize = 5, elinewidth = 1, label=name, 
                         ecolor = col, linewidth=2)
    
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

    # Tight layout
    plt.tight_layout()

    # Define output filepath
    filepath = f"figs/native_validation/{filename}.png"

    # Save plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)

    # Print save confirmation
    print(f"Plot saved as: {filepath}")

    # Show plot
    plt.show()


# -------------------------------------------------------------------------
# DEFINE STEHMAN STATISTICS PLOTTING FUNCTIONS
# -------------------------------------------------------------------------
# Plot accuracies
steh_dual_lineplt([anyyear_stats["gfc_lossyear_anyyear"], 
                   anyyear_stats["tmf_dist_anyyear"]], 
                   "anyyear_tmfdist_accuracies")

steh_dual_lineplt([firstyear_stats["gfc_lossyear_firstyear"], 
                   firstyear_stats["tmf_dist_firstyear"]], 
                   "firstyear_tmfdist_accuracies")

# Plot errors
errors_lineplt([comom_err(anyyear_stats["gfc_lossyear_anyyear"]),
                comom_err(anyyear_stats["tmf_dist_anyyear"])],
                "anyyear_tmfdist_errors")

errors_lineplt([comom_err(firstyear_stats["gfc_lossyear_firstyear"]),
                comom_err(firstyear_stats["tmf_dist_firstyear"])],
                "firstyear_tmfdist_errors")