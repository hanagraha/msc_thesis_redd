# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np

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
# GFC lossyear
with rasterio.open('native_validation/gfc_lossyear_native.tif') as rast:     
    gfc_lossyear = rast.read(1)

# TMF deforyear
with rasterio.open('native_validation/tmf_deforyear_native.tif') as rast:     
    tmf_deforyear = rast.read(1)

# TMF degra year
with rasterio.open('native_validation/tmf_degrayear_native.tif') as rast:     
    tmf_degrayear = rast.read(1)

# TMF aannual change first deforestation
with rasterio.open('native_validation/tmfac_firstdeforyear_native.tif') as rast:     
    tmfac_defor1 = rast.read(1)

# TMF aannual change second deforestation
with rasterio.open('native_validation/tmfac_seconddeforyear_native.tif') as rast:     
    tmfac_defor2 = rast.read(1)

# TMF aannual change first degradation
with rasterio.open('native_validation/tmfac_firstdegrayear_native.tif') as rast:     
    tmfac_degra1 = rast.read(1)

# TMF aannual change second degradation
with rasterio.open('native_validation/tmfac_seconddegrayear_native.tif') as rast:     
    tmfac_degra2 = rast.read(1)
    

# -------------------------------------------------------------------------
# PREPROCESS DATA FOR VALIDATION
# -------------------------------------------------------------------------
"""
Option A: One year buffer to any of the years
Option B: First disturbance year
"""

# Define function for one year buffer
def prot_b(valdata, col, keepcols=['strata', 'geometry']):
    
    # Copy input validation data
    val_data = valdata.copy()
    
    # Create mask where any defor year matches dataset
    mask = val_data[col].between(val_data['defor1'] - 1, val_data['defor1'] + 1)

    # Assign dataset year where mask is true, otherwise first defor year
    val_data['prot_b'] = np.where(mask, val_data[col], val_data['defor1'])
    
    # Add data name to list
    cols = keepcols + [col, 'prot_b']
    
    # Only keep relevant columns
    val_data = val_data[cols]
    
    return val_data

# Run protocol b for gfc 
protb_gfc = prot_b(valdata, 'gfc')

# Run protocol b for tmf 
protb_tmf = prot_b(valdata, 'tmf')


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