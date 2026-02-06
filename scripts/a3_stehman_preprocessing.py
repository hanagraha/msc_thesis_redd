
"""
Option A: Time insensitive
Option B: Any year match
Option C: 1 year buffer to deforestation year
Option D: Exact year match
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
for col in valdata.columns[-2:]:
    valdata[col] = valdata[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

for col in valdata_buff.columns[4:]:
    valdata_buff[col] = valdata_buff[col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )


# -------------------------------------------------------------------------
# PREPROCESS DATA: TIME INSENSITIVE, EXACT
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
# PREPROCESS DATA: TIME INSENSITIVE, BUFFERED
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
# PREPROCESS DATA: YEAR MATCHING + YEAR BUFFER
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
        val_data_exp.to_csv(f'native_validation/yearmatch/{filename}.csv', index=False)
        print(f"Exported native_validation/yearmatch/{filename}.csv")

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
# PREPROCESS DATA: EXACT
# -------------------------------------------------------------------------
# Define function for exact year
def optC(valdata, col):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Create mask where any defor year matches dataset
    mask = val_data[col].between(val_data['defor1'] - 1, val_data['defor1'] + 1)

    # Assign dataset year where mask is true, otherwise first defor year
    val_data['prot_b'] = np.where(mask, val_data[col], val_data['defor1'])

    # Create reference column
    val_data["ref"] = val_data[ref_col]

    # Keep only relevant columns
    val_data_exp = val_data[['strata', 'geometry', 'ref', 'map']]
    
    return val_data

# Run protocol b for gfc 
protb_gfc = optB(valdata, 'gfc')

# Run protocol b for tmf 
protb_tmf = optB(valdata, 'tmf')