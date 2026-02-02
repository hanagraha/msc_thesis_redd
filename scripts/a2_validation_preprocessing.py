# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from rasterstats import zonal_stats

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"Z:\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# GEOMETRY DATA
# -------------------------------------------------------------------------
# Read geometries
grnp = gpd.read_file("gola gazetted polygon/Gola_Gazetted_Polygon.shp")
villages = gpd.read_file("village polygons/VillagePolygons.geojson")
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()

# Define years
years = list(range(2013, 2024))


# -------------------------------------------------------------------------
# REFERENCE DATA
# -------------------------------------------------------------------------
# Define function to read csv validation data
def csv_read(datapath):
    
    # Read validation data
    data = pd.read_csv(datapath, delimiter = ",", index_col = 0)
    
    # Convert csv geometry to WKT
    data['geometry'] = gpd.GeoSeries.from_wkt(data['geometry'])
    
    # Convert dataframe to geodataframe
    data = gpd.GeoDataFrame(data, geometry = 'geometry', crs="EPSG:32629")
    
    return data

# Read validation data
valdata = csv_read("validation/validation_datasets/validation_points_780.csv")


# -------------------------------------------------------------------------
# STRATIFICATION MAPS
# -------------------------------------------------------------------------
# Define function to extract area from map
def map_area(path):

    # Read raster data
    with rasterio.open(path) as rast:
        data = rast.read()

    # Calculate area (hectares)
    pixels = np.sum(data != 255)
    ha = pixels * 0.09
        
    return data, pixels, ha

# Calculate aoi map area
stratmap, total_pix, total_ha = map_area("validation/stratification_maps/stratification_layer_nogrnp.tif")

# Calculate redd+ map area
strat_redd, redd_pix, redd_ha = map_area("validation/stratification_maps/stratification_layer_redd.tif")

# Calculate nonredd+ map area
strat_nonredd, nonredd_pix, nonredd_ha = map_area("validation/stratification_maps/stratification_layer_nonredd.tif")


# -------------------------------------------------------------------------
# PREDICTION DATA
# -------------------------------------------------------------------------
# Define dictionary of prediction datasets
rasters = {
    'gfc_lossyear': 'native_validation/gfc_lossyear_native.tif',
    'tmf_deforyear': 'native_validation/tmf_deforyear_native.tif',
    'tmf_degrayear': 'native_validation/tmf_degrayear_native.tif',
    'tmfac_defor1': 'native_validation/tmfac_firstdeforyear_native.tif',
    'tmfac_defor2': 'native_validation/tmfac_seconddeforyear_native.tif',
    'tmfac_degra1': 'native_validation/tmfac_firstdegrayear_native.tif',
    'tmfac_degra2': 'native_validation/tmfac_seconddegrayear_native.tif'
}


# -------------------------------------------------------------------------
# EXTRACT SAMPLE VALUES
# -------------------------------------------------------------------------
# Create copy of valdata
valdata_expanded = valdata[['strata', 'geometry', 'defor1', 'defor2', 'defor3']].copy()

# Define function to extract raster values
def raster_extract(rasterpath, samples):

    # Extract sample coordinates
    coords = [(geom.x, geom.y) for geom in samples.geometry]

    # Read raster
    with rasterio.open(rasterpath) as src:

        # Collect values at sample points
        values = np.array([val[0] for val in src.sample(coords)])

    return values

# Extract values from all prediction datasets
for name, path in rasters.items():
    valdata_expanded[name] = raster_extract(path, valdata_expanded)

# Export expanded validation data
valdata_expanded.to_csv("native_validation/validation_mapdata.csv", index=False)

# Create 45m buffers around each point
valdata_buffered = valdata[['strata', 'geometry', 'defor1', 'defor2', 'defor3']].copy()
valdata_buffered["geometry"] = valdata_buffered.geometry.buffer(45)

# Define function to extract maximum disturbance year within buffer
def raster_buffextract(rasterpath, samples):

    # Calcuate zonal statistics
    stats = zonal_stats(samples, rasterpath, stats=None, categorical=True,
        geojson_out=False)

    # Extract unique values per polygon
    unique_vals = [list(d.keys()) if d is not None else [] for d in stats]

    return unique_vals

# Extract buffered maximum values for all prediction datasets
for name, path in rasters.items():
    valdata_buffered[f'{name}_buff'] = raster_buffextract(path, valdata_buffered)

# Combine tmf deforestation years
valdata_buffered["tmfac_defor_buff"] = valdata_buffered["tmfac_defor1_buff"] + \
    valdata_buffered["tmfac_defor2_buff"] + valdata_buffered["tmf_deforyear_buff"]
valdata_buffered["tmfac_dist_buff"] = valdata_buffered["tmfac_defor1_buff"] + \
    valdata_buffered["tmfac_defor2_buff"] + valdata_buffered["tmfac_degra1_buff"] + \
    valdata_buffered["tmfac_degra2_buff"] + valdata_buffered["tmf_deforyear_buff"] + \
    valdata_buffered["tmf_degrayear_buff"]

# Define function to clean up valid years
def valid_years(list):

    # Filter for years between 2013-2023, else set to 0
    valid = [v if 2013 <= v <= 2023 else 0 for v in list]

    return np.unique(valid).tolist()

# Apply to tmfac_dist_buff
valdata_buffered["tmfac_defor_buff_clean"] = valdata_buffered["tmfac_defor_buff"].apply(valid_years)
valdata_buffered["tmfac_dist_buff_clean"] = valdata_buffered["tmfac_dist_buff"].apply(valid_years)
valdata_buffered["gfc_lossyear_buff_clean"] = valdata_buffered["gfc_lossyear_buff"].apply(valid_years)

# Export buffered validation data
valdata_buffered.to_csv("native_validation/validation_mapdata_buffered.csv", index=False)


# -------------------------------------------------------------------------
# PREPROCESS DATA FOR VALIDATION
# -------------------------------------------------------------------------
"""
Option A: Time insensitive
Option B: Any year match
Option C: 1 year buffer to deforestation year
Option D: Exact year match
"""

# ---------------- OPTION A: TIME INSENSITIVE ----------------
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
        val_data_exp.to_csv(f'native_validation/timeinsensitive/{filename}.csv', index=False)
        print(f"Exported native_validation/timeinsensitive/{filename}.csv")
            
    return val_data_exp
    
# Run protocol a for gfc 
gfc_optA = optA(valdata_expanded, ["gfc_lossyear"], filename="gfc_timeinsensitive")

# Run protocol a for tmf
tmf_optA = optA(valdata_expanded, ["tmf_deforyear", "tmf_degrayear", 
    "tmfac_defor1", "tmfac_defor2", "tmfac_degra1", "tmfac_degra2"], 
    filename = "tmf_timeinsensitive")

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

# Run time insensitive on buffered predictions
gfc_buff_optA = optA_buff(valdata_buffered, 'gfc_lossyear_buff_clean', filename='gfc_timeinsensitive_buffered')   
tmf_distbuff_optA = optA_buff(valdata_buffered, 'tmfac_dist_buff_clean', filename='tmf_timeinsensitive_dist_buffered')
tmf_deforbuff_optA = optA_buff(valdata_buffered, 'tmfac_defor_buff_clean', filename='tmf_timeinsensitive_defor_buffered')


# ---------------- OPTION B: ANY YEAR ----------------
# Define function for any year match + one year buffer
def optB(valdata, col):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Create mask where any defor year matches dataset
    mask = val_data[col].between(val_data['defor1'] - 1, val_data['defor1'] + 1)

    # Assign dataset year where mask is true, otherwise first defor year
    val_data['prot_b'] = np.where(mask, val_data[col], val_data['defor1'])
    
    return val_data

def buffer_matches(valdata, buffer_col, ref_col, tol=1):

    df = valdata.copy()

    def get_optB(buffer_values, ref_value):
        buffer_array = np.array(buffer_values, dtype=float)
        # Check if any value is within ±tol of ref_value
        if np.any(np.abs(buffer_array - ref_value) <= tol):
            return ref_value
        else:
            return buffer_array[0]  # first value if no match

    df["optB"] = df.apply(
        lambda row: get_optB(row[buffer_col], row[ref_col]),
        axis=1
    )

    return df



# ---------------- OPTION C: ONE YEAR BUFFER ----------------
# Define function for one year buffer
def optC(valdata, col):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Create mask where any defor year matches dataset
    mask = val_data[col].between(val_data['defor1'] - 1, val_data['defor1'] + 1)

    # Assign dataset year where mask is true, otherwise first defor year
    val_data['prot_b'] = np.where(mask, val_data[col], val_data['defor1'])
    
    return val_data

# Run protocol b for gfc 
protb_gfc = optB(valdata, 'gfc')

# Run protocol b for tmf 
protb_tmf = optB(valdata, 'tmf')


# -------------------------------------------------------------------------
# STEHMAN AREA ESTIMATION
# -------------------------------------------------------------------------
# Define function to calculate proportional areas
def steh_area(valdata, stratmap, deforlist, pix, ha):
    
    # Calculate number of pixels per strata
    pixvals, pixcounts = np.unique(stratmap, return_counts = True)

    # Create dataframe
    strata_size = pd.DataFrame({'strata': pixvals[:-1],
                                'size': pixcounts[:-1]})

    # Create empty list to hold deforestation area 
    year_defor = pd.DataFrame(index=strata_size['strata'])

    # Iterate over each year
    for year in years:
        
        # Create empty list to hold year deforestation area
        strata_defor = []
        
        # Iterate over each strata
        for idx, row in strata_size.iterrows():
            
            # Extract strata number
            strata = row['strata']
            
            # Extract strata size
            size = row['size']
            
            # Subset validation data for that strata
            data = valdata[valdata['strata'] == strata]
            
            # Count sum of deforestation in that year
            defor = data[deforlist].eq(year).sum().sum()
            
            # Calculate class proportion in that strata
            cp = defor / len(data)
            
            # Multiply by strata size
            area = (cp * size) / pix
            
            # Add deforestation area to list
            strata_defor.append(area)
            
        # Add the list as a column in the DataFrame
        year_defor[year] = strata_defor

    # Take the sum per year
    total_defor = year_defor.sum(axis=0) * ha
    
    return total_defor

# Estimate deforestation for the first year
first_defor = steh_area(valdata, stratmap, ['defor1'], total_pix, total_ha)

# Estimate deforestation for second year
second_defor = steh_area(valdata, stratmap, ['defor2'], total_pix, total_ha)

# Estimate deforestation for third year
third_defor = steh_area(valdata, stratmap, ['defor3'], total_pix, total_ha)

# Estimate deforestation for all years
all_defor = steh_area(valdata, stratmap, ['defor1', 'defor2', 'defor3'], 
                      total_pix, total_ha)

# Extract area calculated by stehman
stehman_area = protc_stats['protc_gfc']['area'] * total_ha
stehman_area.index = pd.Index(years)

# Calculate missed deforestation
miss_defor = all_defor - first_defor