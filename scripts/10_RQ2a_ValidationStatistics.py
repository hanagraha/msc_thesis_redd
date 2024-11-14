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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt



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



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define filepath for validation data
val_data = pd.read_csv("data/validation/validation_points_labelled.csv", delimiter=";")

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 



############################################################################


# CREATE CLASSIC CONFUSION MATRICES


############################################################################
# Select the two columns you want to analyze
gfc = val_data['gfc']
tmf = val_data['tmf']
se = val_data['se']
val_first = val_data['defor1']

# Define dataset labels
yearlabs = [0] + list(years)

from sklearn.metrics import multilabel_confusion_matrix

# Calculate confusion matrix for gfc
gfc_matrix = confusion_matrix(val_first, gfc)

# Calculate confusion matrix for tmf
tmf_matrix = confusion_matrix(val_first, tmf)

# Calculate confusion matrix for sensitive early combination
se_matrix = confusion_matrix(val_first, se)

# Define figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot GFC confusion matrix
sns.heatmap(gfc_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=yearlabs, yticklabels=yearlabs)
axes[0].set_title('GFC Confusion Matrix')
axes[0].set_xlabel('GFC Predicted Labels')
axes[0].set_ylabel('Validation Labels')

# Plot TMF confusion matrix
sns.heatmap(tmf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=yearlabs, yticklabels=yearlabs)
axes[1].set_title('TMF Confusion Matrix')
axes[1].set_xlabel('TMF Predicted Labels')
axes[1].set_ylabel('Validation Labels')

# Plot Sensitive Early Combination confusion matrix
sns.heatmap(se_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=yearlabs, yticklabels=yearlabs)
axes[2].set_title('Sensitive Early Combination Confusion Matrix')
axes[2].set_xlabel('Sensitive Early Predicted Labels')
axes[2].set_ylabel('Validation Labels')

# Adjust layout
plt.tight_layout()
plt.show()



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


# CALCULATE STATISTICS


############################################################################
# Define function to calculate group of statistics
def statlist(true, pred):
    
    # Calculate accuracy
    acc = accuracy_score(true, pred)
    
    # Calculate precision
    
# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  # use 'macro' for multiclass
recall = recall_score(y_true, y_pred, average='binary')        # use 'macro' for multiclass
f1 = f1_score(y_true, y_pred, average='binary')                # use 'macro' for multiclass

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


























