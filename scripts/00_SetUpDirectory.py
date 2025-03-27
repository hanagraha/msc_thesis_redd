# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:24:08 2024

@author: hanna

This file creates subfolders to store and organize data. 

Expected runtime: <1min
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os



############################################################################


# SET UP DIRECTORY


############################################################################
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# %%
############################################################################


# CREATE NECESSARY SUBFOLDERS IN DIRECTORY


############################################################################
# Define main directory
main_dir = os.path.join(os.getcwd(), 'data')

# List of subfolders that should exist
required_folders = ['hansen_raw', 'hansen_preprocessed', 'jrc_raw', 
                    'jrc_preprocessed', 'intermediate', 'plots', 'test', 
                    'cc_composites', 'validation', 'planet_raw']

# Loop through the list and check if each folder exists
for folder in required_folders:
    folder_path = os.path.join(main_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder}' created.")
    else:
        print(f"Folder '{folder}' already exists.")