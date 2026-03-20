"""
Created on Mar 6 2026

@author: hanna

This file preprocesses Global Forest Change (GFC) and Tropical Moist Forests (TMF) data for 2013-2023.

Expected runtime XX min
"""

# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterstats import zonal_stats


from shapely.geometry import shape
import json


# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"Z:\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# DEFINE CONSTANTS
# -------------------------------------------------------------------------
# Define annual change classes
tmf_annualchange_dict = {
    1: 'Undisturbed TMF',
    2: 'Degraded TMF',
    3: 'Deforested TMF',
    4: 'TMF Regrowth',
    5: 'Water',
    6: 'Other',
    0: 'No data'
}

# Define main transition map classes
tmf_maintrans_dict = {
    10: 'Undisturbed TMF',
    20: 'Degraded TMF', 
    30: 'TMF Regrowth',
    41: 'Deforested - Tree Plantations',
    42: 'Deforested - Water',  
    43: 'Deforested - Other',
    50: 'Ongoing Deforestation/Degradation',
    60: 'Water',
    70: 'Other'
}

# Define years
years = list(range(2013, 2024))

# Define color palatte 
bluecols = ['brown', "#1E2A5E", "#83B4FF"]


# -------------------------------------------------------------------------
# READ SAMPLE DATA
# -------------------------------------------------------------------------
# Define function to read csv as geodataframe
def csv_read(datapath):
    
    # Read validation data
    data = pd.read_csv(datapath, delimiter=",")
    
    # Parse GeoJSON strings to Shapely geometries
    data['geometry'] = data['.geo'].apply(lambda x: shape(json.loads(x)))

    # Drop unnecessary columns
    data = data.drop(columns=['.geo', 'system:index'])
    
    # Convert dataframe to geodataframe
    data = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:32629")

    # Convert all data columns (excl. geometry) to integer
    data.iloc[:, :-1] = data.iloc[:, :-1].fillna(255).astype(int)
    
    return data.reset_index(drop=True)

# Read samples with exact pixel map data
valdata = csv_read("ee_processed/sampledata/valdata_map.csv")
valdata_fm = csv_read("ee_processed/sampledata/valdata_map_fm.csv")
valdata_tm = csv_read("ee_processed/sampledata/valdata_map_tm.csv")

# Read samples with 3x3 window map data (mode)
valdata_mapwindow = csv_read("ee_processed/sampledata/valdata_mapwindow.csv")
valdata_fm_window = csv_read("ee_processed/sampledata/valdata_mapwindow_fm.csv")
valdata_tm_window = csv_read("ee_processed/sampledata/valdata_mapwindow_tm.csv")

# Convert gfc to years
valdata['gfc_lossyear'] = np.where(valdata['gfc_lossyear'] == 0, 0, valdata['gfc_lossyear'] + 2000)
valdata_fm['gfc_lossyear_fm'] = np.where(valdata_fm['gfc_lossyear_fm'] == 0, 0, valdata_fm['gfc_lossyear_fm'] + 2000)
valdata_tm['gfc_lossyear_tm'] = np.where(valdata_tm['gfc_lossyear_tm'] == 0, 0, valdata_tm['gfc_lossyear_tm'] + 2000)
valdata_mapwindow['gfc_lossyear_mode'] = np.where(valdata_mapwindow['gfc_lossyear_mode'] == 0, 0, valdata_mapwindow['gfc_lossyear_mode'] + 2000)
valdata_fm_window['gfc_lossyear_fm_mode'] = np.where(valdata_fm_window['gfc_lossyear_fm_mode'] == 0, 0, valdata_fm_window['gfc_lossyear_fm_mode'] + 2000)
valdata_tm_window['gfc_lossyear_tm_mode'] = np.where(valdata_tm_window['gfc_lossyear_tm_mode'] == 0, 0, valdata_tm_window['gfc_lossyear_tm_mode'] + 2000)


# -------------------------------------------------------------------------
# READ RASTER DATA
# -------------------------------------------------------------------------
# Define function to read raster stack
def read_raster(filepath):
    
    with rasterio.open(filepath) as src:
        
        # Check CRS and shape
        print(f"CRS: {src.crs}")
        print(f"Shape (bands, rows, cols): {src.count, src.height, src.width}")
        print(f"Pixel size (x, y): {src.res}")

        # Extract band names
        bandnames = src.descriptions

        # Initialize empty dictionary
        bands = {}

        # Iterate over each band
        for i, name in enumerate(bandnames, start=1):
            
            # Read band data to dictionary
            bands[name] = src.read(i)
    
    return bands

# Read unmasked data
datastack = read_raster("ee_processed/unmasked/datastack.tif")
datastack_fm = read_raster("ee_processed/masked-undisturbed/rasters/datastack_fm.tif")
datastack_tm = read_raster("ee_processed/masked-forest/rasters/datastack_tm.tif")


# -------------------------------------------------------------------------
# EXTRACT DISTURBANCE YEARS AS LIST
# -------------------------------------------------------------------------
# Extract all possible disturbance years
def unqiueyears(row):

    # Extract unique values
    values = row.unique().astype(int)

    # Get valid disturbance years
    validdist = values[(values != 0) & (values >= 2013) & (values <= 2023)].astype(int)

    return validdist.tolist() if len(validdist) > 0 else [0]

# Extract tmf disturbance years as list
valdata['tmf_distlist'] = valdata[['tmf_dist1', 'tmf_dist2', 'tmf_dist3']].apply(unqiueyears, axis=1)
valdata_fm['tmf_distlist'] = valdata_fm[['tmf_dist1_fm', 'tmf_dist2_fm', 'tmf_dist3_fm']].apply(unqiueyears, axis=1)
valdata_tm['tmf_distlist'] = valdata_tm[['tmf_dist1_tm', 'tmf_dist2_tm', 'tmf_dist3_tm']].apply(unqiueyears, axis=1)
valdata_mapwindow['tmf_distlist'] = valdata_mapwindow[['tmf_dist1_mode', 'tmf_dist2_mode', 'tmf_dist3_mode']].apply(unqiueyears, axis=1)
valdata_fm_window['tmf_distlist'] = valdata_fm_window[['tmf_dist1_fm_mode', 'tmf_dist2_fm_mode', 'tmf_dist3_fm_mode']].apply(unqiueyears, axis=1)
valdata_tm_window['tmf_distlist'] = valdata_tm_window[['tmf_dist1_tm_mode', 'tmf_dist2_tm_mode', 'tmf_dist3_tm_mode']].apply(unqiueyears, axis=1) 

# Extract gfc disturbance years as list
valdata['gfc_losslist'] = valdata[['gfc_lossyear']].apply(unqiueyears, axis=1)
valdata_fm['gfc_losslist'] = valdata_fm[['gfc_lossyear_fm']].apply(unqiueyears, axis=1)
valdata_tm['gfc_losslist'] = valdata_tm[['gfc_lossyear_tm']].apply(unqiueyears, axis=1)
valdata_mapwindow['gfc_losslist'] = valdata_mapwindow[['gfc_lossyear_mode']].apply(unqiueyears, axis=1)
valdata_fm_window['gfc_losslist'] = valdata_fm_window[['gfc_lossyear_fm_mode']].apply(unqiueyears, axis=1)
valdata_tm_window['gfc_losslist'] = valdata_tm_window[['gfc_lossyear_tm_mode']].apply(unqiueyears, axis=1)


# -------------------------------------------------------------------------
# PREPROCESS DATA: TIME INSENSITIVE 
# -------------------------------------------------------------------------
# Define function for processing time insensitive map-reference matches
def timeinsensitive(valdata, col, filename=False):

    # Copy input
    val_data = valdata.copy()
    
    # Iterate over each row in validation dataset
    for idx, row in val_data.iterrows():

        # Assign binary reference value
        ref_val = 1 if row['defor1'] != 0 else 0
        val_data.loc[idx, 'ref'] = ref_val

        # If list is empty, assign 0
        if row[col] == [0]:
            val_data.loc[idx, 'map'] = 0
        
        # If list and reference contain 0, assign 0
        elif ref_val == 0 and 0 in row[col]:
            val_data.loc[idx, 'map'] = 0
        
        # If list contains disturbance years, assign 1
        else:
            val_data.loc[idx, 'map'] = 1

    # Keep only relevant columns
    val_data_exp = val_data[['strata', 'geometry', 'ref', 'map']]
    
    # Optional export
    if filename:
        val_data_exp.to_csv(f'ee_processed/timeinsensitive/{filename}.csv', index=False)
        print(f"Exported ee_processed/timeinsensitive/{filename}.csv")
    
    return val_data_exp

# Process for time insenstive map-reference matches (unmasked)
gfc_timeinsensitive = timeinsensitive(valdata, 'gfc_losslist', filename='gfc_timeinsensitive')
tmf_timeinsensitive = timeinsensitive(valdata, 'tmf_distlist', filename='tmf_dist_timeinsensitive')

# Process for time insenstive map-reference matches (undisturbed forest mask)
gfc_timeinsensitive_fm = timeinsensitive(valdata_fm, 'gfc_losslist', filename='gfc_timeinsensitive_fm')
tmf_timeinsensitive_fm = timeinsensitive(valdata_fm, 'tmf_distlist', filename='tmf_dist_timeinsensitive_fm')

# Process for time insenstive map-reference matches (forest mask)
gfc_timeinsensitive_tm = timeinsensitive(valdata_tm, 'gfc_losslist', filename='gfc_timeinsensitive_tm')
tmf_timeinsensitive_tm = timeinsensitive(valdata_tm, 'tmf_distlist', filename='tmf_dist_timeinsensitive_tm')

# Process for time insenstive map-reference matches (3x3 window mode, unmasked)
gfc_timeinsensitive_window = timeinsensitive(valdata_mapwindow, 'gfc_losslist', filename='gfc_timeinsensitive_window')
tmf_timeinsensitive_window = timeinsensitive(valdata_mapwindow, 'tmf_distlist', filename='tmf_dist_timeinsensitive_window')

# Process for time insenstive map-reference matches (3x3 window mode, undisturbed forest mask)
gfc_timeinsensitive_fm_window = timeinsensitive(valdata_fm_window, 'gfc_losslist', filename='gfc_timeinsensitive_fm_window')
tmf_timeinsensitive_fm_window = timeinsensitive(valdata_fm_window, 'tmf_distlist', filename='tmf_dist_timeinsensitive_fm_window')

# Process for time insenstive map-reference matches (3x3 window mode, forest mask)
gfc_timeinsensitive_tm_window = timeinsensitive(valdata_tm_window, 'gfc_losslist', filename='gfc_timeinsensitive_tm_window')
tmf_timeinsensitive_tm_window = timeinsensitive(valdata_tm_window, 'tmf_distlist', filename='tmf_dist_timeinsensitive_tm_window')


# -------------------------------------------------------------------------
# PREPROCESS DATA: TIME SENSITIVE
# -------------------------------------------------------------------------
# Define function for processing time sensitive map-reference matches
def timesensitive(valdata, map_col, ref_col, folder=False):

    # Copy input
    val_data = valdata.copy()

    # Define function to compare map and reference values
    def matches(map_value, ref_value):
        
        # Convert map values to array
        map_array = np.array(map_value, dtype=int) if isinstance(
            map_value, (list, np.ndarray)
        ) else np.array([map_value], dtype=int)

        # Convert ref values to array
        ref_array = np.array(ref_value, dtype=int) if isinstance(
            ref_value, (list, np.ndarray)
        ) else np.array([ref_value], dtype=int)

        # Compute pairwise differences
        diff = np.abs(map_array[:, None] - ref_array[None, :])

        # Find first matching reference year (with one year buffer)
        match_indices = np.where(diff <= 1)

        # If there is a match, return match year from ref array
        if match_indices[0].size > 0:
            map_idx = match_indices[0][0]
            ref_idx = match_indices[1][0]

            matched_map = int(map_array[map_idx])
            matched_ref = int(ref_array[ref_idx])

            # Force equality if diff == 1
            if abs(matched_map - matched_ref) == 1:
                matched_map = matched_ref

            return matched_ref, matched_map

        return int(ref_array[0]), int(map_array[0])

    # Apply and unpack both values
    val_data[["ref", "map"]] = val_data.apply(
        lambda row: pd.Series(matches(row[map_col], row[ref_col])),
        axis=1
    )

    val_data_exp = val_data[['strata', 'geometry', 'ref', 'map']]

    if folder:
        val_data_exp.to_csv(
            f'ee_processed/{folder}/{map_col}_{folder}.csv',
            index=False
        )
        print(f"Exported ee_processed/{folder}/{map_col}_{folder}.csv")

    return val_data_exp


# -------------------------------------------------------------------------
# Process for detecting first disturbance between 2013-2023



# Unmasked
gfc_first = timesensitive(valdata, 'gfc_losslist', 'defor1', folder='firstyear')
tmf_first = timesensitive(valdata, 'tmf_distlist', 'defor1', folder='firstyear')

# Process for time insenstive map-reference matches (undisturbed forest mask)
gfc_timeinsensitive_fm = timesensitive(valdata_fm, 'gfc_losslist', filename='gfc_timeinsensitive_fm')
tmf_timeinsensitive_fm = timesensitive(valdata_fm, 'tmf_distlist', filename='tmf_dist_timeinsensitive_fm')

# Process for time insenstive map-reference matches (forest mask)
gfc_timeinsensitive_tm = timesensitive(valdata_tm, 'gfc_losslist', filename='gfc_timeinsensitive_tm')
tmf_timeinsensitive_tm = timesensitive(valdata_tm, 'tmf_distlist', filename='tmf_dist_timeinsensitive_tm')

# Process for time insenstive map-reference matches (3x3 window mode, unmasked)
gfc_timeinsensitive_window = timesensitive(valdata_mapwindow, 'gfc_losslist', filename='gfc_timeinsensitive_window')
tmf_timeinsensitive_window = timesensitive(valdata_mapwindow, 'tmf_distlist', filename='tmf_dist_timeinsensitive_window')

# Process for time insenstive map-reference matches (3x3 window mode, undisturbed forest mask)
gfc_timeinsensitive_fm_window = timesensitive(valdata_fm_window, 'gfc_losslist', filename='gfc_timeinsensitive_fm_window')
tmf_timeinsensitive_fm_window = timesensitive(valdata_fm_window, 'tmf_distlist', filename='tmf_dist_timeinsensitive_fm_window')

# Process for time insenstive map-reference matches (3x3 window mode, forest mask)
gfc_timeinsensitive_tm_window = timesensitive(valdata_tm_window, 'gfc_losslist', filename='gfc_timeinsensitive_tm_window')
tmf_timeinsensitive_tm_window = timesensitive(valdata_tm_window, 'tmf_distlist', filename='tmf_dist_timeinsensitive_tm_window')





