# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:46:54 2024

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
years = range(2016, 2024)

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
val_data = pd.read_csv("data/validation/validation_points_labelled_minstrata7.csv", 
                       delimiter=";")

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

# Define dataset names
datanames = ["GFC", "TMF", "Sensitive Early"]

# Read gfc stehman statistic data (calculated in R)
gfc_stehman = pd.read_csv("data/validation/gfc_stehman_minstrata7.csv", delimiter=",")

# Read tmf stehman statistic data (calculated in R)
tmf_stehman = pd.read_csv("data/validation/tmf_stehman_minstrata7.csv", delimiter=",")

# Read se stehman statistic data (calculated in R)
se_stehman = pd.read_csv("data/validation/se_stehman_minstrata7.csv", delimiter=",")



############################################################################


# CLEAN UP VALIDATION DATASET


############################################################################
# Define function to subtract from all non-0 deforestation and regrowth years
def col_sub1(dataset, col_list):
    
    # Copy dataset
    sub_dataset = dataset.copy()
    
    # Iterate over each column
    for col in col_list:
        
        # Subtract 1 from cells with non-0 data
        sub_dataset[col] = np.where(sub_dataset[col] != 0, sub_dataset[col] - 1, 
                                    sub_dataset[col])
        
    return sub_dataset
    
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

# Define function to subset gfc, tmf, and se data for only 2016-2023
def subset_yrs(dataset, interest_years):
    
    # Create copy of dataset
    sub_dataset = dataset.copy()
    
    # Replace years outside interest range with 0
    sub_dataset = np.where(sub_dataset.isin(interest_years), sub_dataset, 0)
    
    # Create dataframe
    sub_dataset = pd.Series(sub_dataset)
    
    return sub_dataset

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

# Subtract 1 from validation data years
val_data_proc = col_sub1(val_data, ['defor1', 'regr1', 'defor2', 'regr2', 
                                    'defor3', 'regr3'])

# Create yearly validation data
yearly_val, yearly_conf = val_expand(val_data_proc)

# Subset gfc data
gfc_sub = subset_yrs(val_data_proc['gfc'], yearlabs)

# Subset tmf data
tmf_sub = subset_yrs(val_data_proc['tmf'], yearlabs)

# Subset se data
se_sub = subset_yrs(val_data_proc['se'], yearlabs)
               
# Create aligned valiadtion data for gfc
gfc_val = val_align(yearly_val, yearly_conf, gfc_sub, "gfc")

# Create aligned validation data for tmf
tmf_val = val_align(yearly_val, yearly_conf, tmf_sub, "tmf")

# Create aligned validation data for se
se_val = val_align(yearly_val, yearly_conf, se_sub, "se")

# Combine datasets
comb_val = pd.concat([val_data_proc.iloc[:,1:3], gfc_val, tmf_val, se_val], 
                     axis = 1)

# Write combine dataset to csv
comb_val.to_csv('data/validation/valdata_minstrata7_proc.csv', index=False)



############################################################################


# CALCULATE CONFUSION MATRICES AND OVERALL ACCURACY


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

# Calculate confusion matrix for gfc
gfc_matrix = confusion_matrix(gfc_val['gfc_val'], gfc_val['gfc'], labels = yearlabs,
                              sample_weight = gfc_val['gfc_conf'])

# Calculate confusion matrix for tmf
tmf_matrix = confusion_matrix(tmf_val['tmf_val'], tmf_val['tmf'], labels = yearlabs,
                              sample_weight = tmf_val['tmf_conf'])

# Calculate confusion matrix for se
se_matrix = confusion_matrix(se_val['se_val'], se_val['se'], labels = yearlabs,
                              sample_weight = se_val['se_conf'])

# Plot calculated matrices
matrix_plt([gfc_matrix, tmf_matrix, se_matrix], datanames, 'd')

# Calculate accuracy for gfc
gfc_acc = accuracy_score(gfc_val['gfc_val'], gfc_val['gfc'], sample_weight = gfc_val['gfc_conf'])

# Calculate accuracy for tmf
tmf_acc = accuracy_score(tmf_val['tmf_val'], tmf_val['tmf'], sample_weight = tmf_val['tmf_conf'])

# Calculate accuracy for se
se_acc = accuracy_score(se_val['se_val'], se_val['se'], sample_weight = se_val['se_conf'])

# Print statement
print(f"gfc accuracy: {gfc_acc}\n"
      f"tmf accuracy: {tmf_acc}\n"
      f"sensitive early accuracy: {se_acc}\n")



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
                         label=name)
    
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
                         label=name)
    
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

# Plot user's and producer's accuracy side by side
steh_dual_lineplt([gfc_stehman0, tmf_stehman0, se_stehman0], datanames)


