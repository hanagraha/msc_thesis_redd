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
# EXTRACT SAMPLE VALUES (EXACT)
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
        values = np.array([v[0] for v in src.sample(coords, masked=False)])
        nodata = src.nodata

    # Only replace nodata if you truly want to
    if nodata is not None:
        values = np.where(values == nodata, 0, values)

    return values

# Extract values from all prediction datasets
for name, path in rasters.items():
    valdata_expanded[name] = raster_extract(path, valdata_expanded)

# Define function to combine
def combine_cols(df, collist): 

    # Create copy
    df_copy = df.copy()

    # Combine valid years from each column into list
    combined = df_copy.apply(lambda row: [int(v) for v in sorted(np.unique(
        [row[c] for c in collist]) ) if v == 0 or 2013 <= v <= 2023], axis=1)

    return combined

# Define deforestation columns
tmf_deforcols = ['tmfac_defor1', 'tmfac_defor2', 'tmf_deforyear']
tmf_distcols = tmf_deforcols + ['tmfac_degra1', 'tmfac_degra2', 'tmf_degrayear']

# Add valid and unique years to dataframe
valdata_expanded["tmfac_defor"] = combine_cols(valdata_expanded, tmf_deforcols)
valdata_expanded["tmfac_dist"] = combine_cols(valdata_expanded, tmf_distcols)

# Export expanded validation data
valdata_expanded.to_csv("native_validation/validation_mapdata.csv", index=False)


# -------------------------------------------------------------------------
# EXTRACT SAMPLE VALUES (45M BUFFER)
# -------------------------------------------------------------------------
# Create 45m buffers around each point
valdata_buffered = valdata[['strata', 'geometry', 'defor1', 'defor2', 'defor3']].copy()
valdata_buffered["geometry"] = valdata_buffered.geometry.buffer(45)

# Define function to extract maximum disturbance year within buffer
def raster_buffextract(rasterpath, samples):

    # Calcuate zonal statistics
    stats = zonal_stats(samples, rasterpath, stats=None, categorical=True,
        nodata=None, geojson_out=False)

    # Extract unique values per polygon
    unique_vals = [list(d.keys()) if d is not None else [] for d in stats]

    # Replace 255 with 0
    unique_vals = [[0 if v == 255 else v for v in vals] for vals in unique_vals]

    # Print statement
    print(f"Extracted data from {rasterpath}")

    return unique_vals

# Extract buffered maximum values for all prediction datasets
for name, path in rasters.items():
    valdata_buffered[f'{name}_buff'] = raster_buffextract(path, valdata_buffered)

# Define function to clean up valid years
def valid_years(list):

    # Filter for years between 2013-2023, else set to 0
    valid = [v if 2013 <= v <= 2023 else 0 for v in list]

    return np.unique(valid).tolist()

# Combine tmf deforestation years
tmf_defor = valdata_buffered["tmfac_defor1_buff"] + \
    valdata_buffered["tmfac_defor2_buff"] + valdata_buffered["tmf_deforyear_buff"]

# Combine tmf disturbance years
tmf_dist = tmf_defor + valdata_buffered["tmfac_degra1_buff"] + valdata_buffered["tmfac_degra2_buff"] \
    + valdata_buffered["tmf_degrayear_buff"]

# Add valid and unique years to dataframe
valdata_buffered["tmfac_defor_buff"] = tmf_defor.apply(valid_years)
valdata_buffered["tmfac_dist_buff"] = tmf_dist.apply(valid_years)

# Export buffered validation data
valdata_buffered.to_csv("native_validation/validation_mapdata_buffered.csv", index=False)
