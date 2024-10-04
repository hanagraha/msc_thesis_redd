# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:18:38 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
from rasterstats import zonal_stats
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch



############################################################################


# SET UP DIRECTORY


############################################################################
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())



############################################################################


# IMPORT DATA


############################################################################
gfc_lossyear_paths = [f"data/intermediate/gfc_lossyear_{year}.tif" for 
                      year in range(2013, 2024)]
tmf_deforyear_paths = [f"data/intermediate/tmf_DeforestationYear_{year}.tif" 
                       for year in range(2013, 2024)]
tmf_degrayear_paths = [f"data/intermediate/tmf_DegradationYear_{year}.tif" 
                       for year in range(2013, 2024)]
tmf_annual_paths = [f"data/jrc_preprocessed/tmf_AnnualChange_{year}_fm.tif" for 
                    year in range(2013,2024)]

# Set nodata value
nodata_val = 255


### POLYGON DATA
villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create AOI
aoi = (gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True))
       .dissolve()[['geometry']])



############################################################################


# READ DATA


############################################################################
# Read each GFC lossyear tif and assign to unique variables
gfc_vars = []
for path in gfc_lossyear_paths:
    year = path.split('_')[-1].split('.')[0]  # Extract year from filename
    var_name = f"gfc_{year}"
    
    with rasterio.open(path) as gfc:
        locals()[var_name] = gfc.read(1)
        
        gfc_vars.append(var_name)
        print(f"Loaded {var_name}")

# Read each TMF deforestation year tif and assign to unique variables
tmf_defor_vars = []

for path in tmf_deforyear_paths:
    year = path.split('_')[-1].split('.')[0]  
    var_name = f"tmf_defor{year}"
    
    with rasterio.open(path) as tmf:
        locals()[var_name] = tmf.read(1)
        
        tmf_defor_vars.append(var_name)
        print(f"Loaded {var_name}")

# Read each TMF degradation year tif and assign to unique variables
tmf_degra_vars = []

for path in tmf_degrayear_paths:
    year = path.split('_')[-1].split('.')[0]
    var_name = f"tmf_degra{year}"
    
    with rasterio.open(path) as tmf:
        locals()[var_name] = tmf.read(1)
        tmf_profile = tmf.profile
        
        tmf_degra_vars.append(var_name)
        print(f"Loaded {var_name}")



############################################################################


# COMBINE TMF DEFORESTATION AND DEGRADATION


############################################################################
"""
Taking the maximum of the two rasters is equivalent to a union combination of
the raster datasets becuase the NAN value is 255, and all values of interest
range from 2013-2023.
"""

### COMBINE DEFORESTATION AND DEGRADATION
# Loop over each year's deforestation and degradation data
years = range(2013, 2024)
tmf_vars =  []

for defor, degra, year in zip(tmf_defor_vars, tmf_degra_vars, years):
    defor_data = locals()[defor]
    degra_data = locals()[degra]
    
    var_name = f"tmf_{year}"
    
    combined_data = np.maximum(degra_data, defor_data)
    locals()[var_name] = combined_data
    tmf_vars.append(var_name)
    
    print(f"Combined {defor} and {degra}")

# Check pixel statistics to see if it worked (use 2013 as example)
defor_vals, defor_counts = np.unique(locals()[tmf_defor_vars[0]], return_counts=True)
degra_vals, degra_counts = np.unique(locals()[tmf_degra_vars[0]], return_counts=True)
union_vals, union_counts = np.unique(locals()[tmf_vars[0]], return_counts=True)

union_counts == defor_counts + degra_counts

"""
The nodata values are unequal because they are overriden by pixels with 2013, 
if present. The sum of deforestation and degradation pixels in 2013 are equal
to the total union counts
"""

### WRITE TO FILE
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

for var, year in zip(tmf_vars, years):
    tmf_data = locals()[var]
    output_filename = f"tmf_defordegra_{year}.tif"
    output_filepath = os.path.join(out_dir, output_filename)
    
    with rasterio.open(output_filepath, 'w', **tmf_profile) as dst:
        dst.write(tmf_data.astype(rasterio.float32), 1)
    
    print(f"Data for {var} saved to file")
        


############################################################################


# CREATE BINARY GFC AND TMF LAYERS


############################################################################
# Reclassify GFC
gfc_binary_vars = []

for var, year in zip(gfc_vars, years):
    gfc_data = locals()[var]
    binary_data = np.where(gfc_data == 255, 0, 1)
    var_name = f"gfc_binary_{year}"
    locals()[var_name] = binary_data
    
    gfc_binary_vars.append(var_name)
    print(f"Reclassified {var} to binary codes")
    
# Check codes are binary (use 2013 as example)
binary_vals = np.unique(locals()[gfc_binary_vars[1]])
orig_vals = np.unique(locals()[gfc_vars[0]])

if np.array_equal(binary_vals, orig_vals):
    print(f"Binary reclassification for GFC failed. Original values are \
          {orig_vals} and new values are {binary_vals}")
else:
    print(f"Binary reclassification for GFC succeeded. Original values are \
          {orig_vals} and new values are {binary_vals}")


# Reclassify TMF
tmf_binary_vars = []

for var, year in zip(tmf_vars, years):
    tmf_data = locals()[var]
    binary_data = np.where(tmf_data == 255, 0, 1)
    var_name = f"tmf_binary_{year}"
    locals()[var_name] = binary_data
    
    tmf_binary_vars.append(var_name)
    print(f"Reclassified {var} to binary codes")
    
# Check codes are binary (use 2013 as example)
binary_vals = np.unique(locals()[tmf_binary_vars[1]])
orig_vals = np.unique(locals()[tmf_vars[0]])

if np.array_equal(binary_vals, orig_vals):
    print(f"Binary reclassification for TMF failed. Original values are \
          {orig_vals} and new values are {binary_vals}")
else:
    print(f"Binary reclassification for TMF succeeded. Original values are \
          {orig_vals} and new values are {binary_vals}")



############################################################################


# CREATE SPATIAL AGREEMENT LAYER


############################################################################
# Create empty list to store agreement layers
agreements = []

# Add binary GFC and TMF layers
for gfc, tmf, year in zip(gfc_binary_vars, tmf_binary_vars, years):
    gfc_data = locals()[gfc]
    tmf_data = locals()[tmf]
    var_name = f"agreemeent_{year}"
    
    agreement = gfc_data + tmf_data
    locals()[var_name] = agreement
    
    agreements.append(var_name)
    print(f"Spatial agreement map created for {year}")
    
# Check values for spatial agreement (should be 0, 1, 2)
agree_vals = np.unique(locals()[agreements[1]])
print(f"Values in agreement map are {agree_vals}")

# Save maps to file
tmf_profile.update(dtype=rasterio.uint8)  

for var, year in zip(agreements, years):
    data = locals()[var]
    data = data.astype(np.uint8)
    output_filename = f"agreement_gfc_combtmf_{year}.tif"
    output_filepath = os.path.join(out_dir, output_filename)
    
    with rasterio.open(output_filepath, 'w', **tmf_profile) as dst:
        dst.write(data.astype(rasterio.uint8), 1)
    
    print(f"Data for {var} saved to file")



############################################################################


# CALCULATE SPATIAL AGREEMENT STATISTICS


############################################################################



