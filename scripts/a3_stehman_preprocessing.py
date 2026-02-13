
"""
Option A: Time insensitive
Option B: One year buffer to any disturbance year
Option C: One year buffer to first disturbance year
"""

# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import ast

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"Z:\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# READ DATA
# -------------------------------------------------------------------------
# Define function to read csv validation data
def csv_read(datapath):
    
    # Read validation data
    data = pd.read_csv(datapath, delimiter = ",", index_col = 0)
    
    # Convert csv geometry to WKT
    data['geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
    
    # Convert dataframe to geodataframe
    data = gpd.GeoDataFrame(data, geometry = 'geometry', crs="EPSG:32629")
    
    return data.reset_index()

# Read sample data (exact location)
valdata = csv_read("native_validation/validation_mapdata.csv")

# Read sample data (buffered)
valdata_buff = csv_read("native_validation/validation_mapdata_buffered.csv")

# Convert string lists to lists
for col in valdata.columns[-3:]:
    valdata[col] = valdata[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

for col in valdata_buff.columns[4:]:
    valdata_buff[col] = valdata_buff[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )


# -------------------------------------------------------------------------
# PREPROCESS DATA: TIME INSENSITIVE (EXACT PIXEL)
# -------------------------------------------------------------------------
# Define function to manipulate with protocol A
def optA(valdata, col_list, filename=False):
    
    # Copy input validation data
    val_data = valdata.copy()

    # Iterate over each row in validation dataset
    for idx, row in val_data.iterrows():
        
        # If deforestation IS detected in validation dataset
        if row['defor1'] != 0:
            
            # Label deforestation
            val_data.loc[idx, 'ref'] = 1
        
        # If deforestation is NOT detected in validation dataset
        else: 
            
            # Label undeforested
            val_data.loc[idx, 'ref'] = 0
        
        # Check if any of the columns indicate disturbance
        if any((row[col] != 0 and row[col] != 255) for col in col_list):
                val_data.loc[idx, 'map'] = 1

        else:
            val_data.loc[idx, 'map'] = 0

    # Subset columns
    val_data_exp = val_data[['strata', 'geometry', 'ref', 'map']]

    # If filename is provided, export the data
    if filename:
        val_data_exp.to_csv(f'native_validation/timeinsensitive/{filename}.csv', index=True)
        print(f"Exported native_validation/timeinsensitive/{filename}.csv")
            
    return val_data_exp
    
# Run protocol a for gfc 
gfc_optA = optA(valdata, ["gfc_lossyear"], filename="gfc_timeinsensitive")

# Run protocol a for tmf disturbances
tmf_optA_dist = optA(valdata, ["tmf_deforyear", "tmf_degrayear", 
    "tmfac_defor1", "tmfac_defor2", "tmfac_degra1", "tmfac_degra2"], 
    filename = "tmf_dist_timeinsensitive")

# Run protocol a for tmf deforestation
tmf_optA_defor = optA(valdata, ["tmf_deforyear", "tmfac_defor1", "tmfac_defor2"], 
    filename = "tmf_defor_timeinsensitive")


# -------------------------------------------------------------------------
# PREPROCESS DATA: TIME INSENSITIVE (BUFFERED PIXEL)
# -------------------------------------------------------------------------
# Define option a function for buffered preditions
def optA_buff(valdata, col, filename=False):

    # Copy input
    val_data = valdata.copy()
    
    # Iterate over each row in validation dataset
    for idx, row in val_data.iterrows():

        # Assign binary reference value
        ref_val = 1 if row['defor1'] != 0 else 0
        val_data.loc[idx, 'ref'] = ref_val

        # If list is empty, assign 0
        if not row[col]:
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
        val_data_exp.to_csv(f'native_validation/timeinsensitive/{filename}.csv', index=False)
        print(f"Exported native_validation/timeinsensitive/{filename}.csv")
    
    return val_data_exp

# Run protocol a for gfc disturbances
gfc_buff_optA = optA_buff(valdata_buff, 'gfc_lossyear_buff', 
                          filename='gfc_timeinsensitive_buffered')   

# Run protocol a for tmf disturbances
tmf_distbuff_optA = optA_buff(valdata_buff, 'tmfac_dist_buff', 
                              filename='tmf_timeinsensitive_dist_buffered')

# Run protocol a for tmf deforestation
tmf_deforbuff_optA = optA_buff(valdata_buff, 'tmfac_defor_buff', 
                               filename='tmf_timeinsensitive_defor_buffered')


# -------------------------------------------------------------------------
# PREPROCESS DATA: ANY DISTURBANCE YEAR
# -------------------------------------------------------------------------
# Define function for any year match + one year buffer
def optB(valdata, map_col, ref_col, filename=False):
    
    # Copy input validation data
    val_data = valdata.copy()

    # Define function to check for matches within tolerance
    def matches(map_value, ref_value):

        # Convert list to float array
        map_array = np.array(map_value, dtype=int)

        # Check if year exists within tolerance
        if np.any(np.abs(map_array - ref_value) <= 1):
            return ref_value
        
        # Return first value if no match
        else:
            return map_array[0] 

    # Apply matching function to each row
    val_data["map"] = val_data.apply(
        lambda row: matches(row[map_col], row[ref_col]),
        axis=1
    )

    # Create reference column
    val_data["ref"] = val_data[ref_col]

    # Keep only relevant columns
    val_data_exp = val_data[['strata', 'geometry', 'ref', 'map']]
    
    # Optional export
    if filename:
        val_data_exp.to_csv(f'native_validation/anyyear/{filename}.csv', index=False)
        print(f"Exported native_validation/anyyear/{filename}.csv")

    return val_data_exp

# Run year matching for gfc and tmf (exact)
tmf_defor_optB = optB(valdata, "tmfac_defor", "defor1", filename = "tmf_yearmatch_defor")
tmf_dist_optB = optB(valdata, "tmfac_dist", "defor1", filename = "tmf_yearmatch_dist")

# Run year matching for gfc and tmf (buffered)
gfc_buff_optB = optB(valdata_buff, 'gfc_lossyear_buff', 'defor1',
                filename='gfc_yearmatch_buffered')
tmf_deforbuff_optB = optB(valdata_buff, 'tmfac_defor_buff', 'defor1', 
                filename='tmf_yearmatch_defor_buffered')
tmf_distbuff_optB = optB(valdata_buff, 'tmfac_dist_buff', 'defor1', 
                filename='tmf_yearmatch_dist_buffered')


# -------------------------------------------------------------------------
# PREPROCESS DATA: TIME SENSITIVE
# -------------------------------------------------------------------------
# Define function for any year match + one year buffer
def time_sensitive(valdata, map_col, ref_col, folder = False):
    
    # Copy input validation data
    val_data = valdata.copy()

    # Define function to check for matches within tolerance
    def matches(map_value, ref_value):

        # Convert map values to array
        map_array = np.array(map_value, dtype=int) if isinstance(map_value,
            (list, np.ndarray)) else np.array([map_value], dtype=int)

        # Convert ref values to array
        ref_array = np.array(ref_value, dtype=int) if isinstance(ref_value, 
            (list, np.ndarray)) else np.array([ref_value], dtype=int)

        # Compute pairwise differences
        diff = np.abs(map_array[:, None] - ref_array[None, :])

        # Find first matching reference year (with one year buffer)
        match_indices = np.where(diff <= 1)

        # If there is a match, return match year from ref array
        if match_indices[1].size > 0:
            return int(ref_array[match_indices[1][0]])
        
        # If no match, return first map year
        else:
            return int(map_array[0])
        
    # Apply matching function to each row
    val_data["map"] = val_data.apply(
        lambda row: matches(row[map_col], row[ref_col]),
        axis=1
    )

    # Create reference column (take first element if list)
    val_data["ref"] = val_data[ref_col].apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
    )

    # Keep only relevant columns
    val_data_exp = val_data[['strata', 'geometry', 'ref', 'map']]
    
    # Optional export
    if folder:
        val_data_exp.to_csv(f'native_validation/{folder}/{map_col}_{folder}.csv', index=False)
        print(f"Exported native_validation/{folder}/{map_col}_{folder}.csv")

    return val_data_exp

# Preprocess gfc data (pixel)
gfc_lossyear_any = time_sensitive(valdata, "gfc_lossyear", "ref_years", folder="anyyear")
gfc_lossyear_first = time_sensitive(valdata, "gfc_lossyear", "defor1", folder='firstyear')

# Preprocess tmf defor data (pixel)
tmf_defor_any = time_sensitive(valdata, 'tmf_defor', 'ref_years', folder="anyyear")
tmf_defor_first = time_sensitive(valdata, 'tmf_defor', 'defor1', folder='firstyear')

# Preprocess tmf dist data (pixel)
tmf_dist_any = time_sensitive(valdata, 'tmf_dist', 'ref_years', folder="anyyear")
tmf_dist_first = time_sensitive(valdata, 'tmf_dist', 'defor1', folder="firstyear")

# Preprocess gfc data (buffered)
gfc_lossyear_any_buff = time_sensitive(valdata_buff, 'gfc_lossyear_buff', 'ref_years', folder='anyyear')
gfc_lossyear_first_buff = time_sensitive(valdata_buff, 'gfc_lossyear_buff', 'defor1', folder='firstyear')

# Preprocess tmf defor data (pixel)
tmf_defor_any_buff = time_sensitive(valdata_buff, 'tmf_defor_buff', 'ref_years', folder="anyyear")
tmf_defor_first_buff = time_sensitive(valdata_buff, 'tmf_defor_buff', 'defor1', folder='firstyear')

# Preprocess tmf dist data (pixel)
tmf_dist_any_buff = time_sensitive(valdata_buff, 'tmf_dist_buff', 'ref_years', folder="anyyear")
tmf_dist_first_buff = time_sensitive(valdata_buff, 'tmf_dist_buff', 'defor1', folder="firstyear")



