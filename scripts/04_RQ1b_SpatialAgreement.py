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
from rasterio.mask import mask
from statsmodels.stats.contingency_tables import mcnemar 
import matplotlib.pyplot as plt



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
aoi_geom = aoi.geometry


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
    var_name = f"agreement_{year}"
    
    agreement = gfc_data + tmf_data
    locals()[var_name] = agreement
    
    agreements.append(var_name)
    print(f"Spatial agreement map created for {year}")
    
# Check values for spatial agreement (should be 0, 1, 2)
agree_vals = np.unique(locals()[agreements[1]])
print(f"Values in agreement map are {agree_vals}")


# Save maps to file
agreement_files = []

for var, year in zip(agreements, years):
    data = locals()[var]
    data = data.astype(np.uint8)
    output_filename = f"agreement_gfc_combtmf_{year}.tif"
    output_filepath = os.path.join(out_dir, output_filename)
    
    with rasterio.open(output_filepath, 'w', **tmf_profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    
    agreement_files.append(output_filepath)
    print(f"Data for {var} saved to file")
    

# Clip again and save to remove agreement on pixels outside the AOI
def clip_raster(raster_path, aoi_geom, nodata_val):
    with rasterio.open(raster_path) as rast:
        raster_clip, out_transform = mask(rast, aoi_geom, crop=True, 
                                          nodata=nodata_val)
        
        out_meta = rast.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'count': 1,
            'height': raster_clip.shape[1],
            'width': raster_clip.shape[2],
            'transform': out_transform,
            'nodata': nodata_val})
        
    raster_clip = raster_clip.astype('float32')
    
    # Save as the same name to override the unclipped file
    with rasterio.open(raster_path, 'w', **out_meta) as dest:
        dest.write(raster_clip)
    
    return raster_clip

# Clip each spatial agreement map and override old files
for file, var in zip(agreement_files, agreements):
    clipped_agreement = clip_raster(file, aoi_geom, nodata_val)
    
    # This overrides the old agreement variables
    locals()[var] = clipped_agreement
    
    print(f"Saved clipped data for {var}")

# Check values in clipped file (should be 0, 1, 2, 255)    
agree_vals = np.unique(locals()[agreements[1]])
print(f"Values in agreement map are {agree_vals}")



############################################################################


# CALCULATE SPATIAL AGREEMENT STATISTICS


############################################################################
def calc_agreement_ratios(image):
    # Mask out NoData (255) values
    valid_pixels = image[image != 255]
    
    # Count pixels with values 0, 1, and 2
    total_pixels = valid_pixels.size
    count_0 = np.sum(valid_pixels == 0)
    count_1 = np.sum(valid_pixels == 1)
    count_2 = np.sum(valid_pixels == 2)
    
    # Calculate ratios
    perc_0 = (count_0 / total_pixels)*100
    perc_1 = (count_1 / total_pixels)*100
    perc_2 = (count_2 / total_pixels)*100
    
    return perc_0, perc_1, perc_2


agree_stats = None

# Process each image
for var, year in zip(agreements, years):
    data = locals()[var]
    perc_0, perc_1, perc_2 = calc_agreement_ratios(data)
    
    # Append results to the DataFrame
    temp_df = pd.DataFrame({
        'Year': [year],
        'Percent_0': [perc_0],
        'Percent_1': [perc_1],
        'Percent_2': [perc_2]
    })
    
    if agree_stats is None:
        agree_stats = temp_df
    else:
        agree_stats = pd.concat([agree_stats, temp_df], ignore_index=True)

# Show the DataFrame
print(agree_stats)



############################################################################


# PLOT STATISTICS


############################################################################
"""
The following is created with help from ChatGPT, especially the diagonal lines
"""

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                               gridspec_kw={'height_ratios': [0.8, 1]})

# Plot data in the upper subplot (for the range 90% and above)
agree_stats.plot(x='Year', y=['Percent_0', 'Percent_1', 'Percent_2'], 
                 kind='line', ax=ax1, legend=False)
ax1.set_ylim(90, 100)  # Upper range
ax1.grid(True, linestyle='--')
ax1.set_ylabel('Percentage of Pixels in AOI (%)')
ax1.spines['bottom'].set_visible(False)  # Hide the bottom spine (axis line)

# Plot data in the lower subplot (for the range below 5%)
agree_stats.plot(x='Year', y=['Percent_0', 'Percent_1', 'Percent_2'], 
                 kind='line', ax=ax2, legend=False)
ax2.set_ylim(0, 5)  # Lower range
ax2.grid(True, linestyle='--')
ax2.set_xlabel('Year')
ax2.set_ylabel('Percentage of Pixels in AOI (%)')
ax2.spines['top'].set_visible(False)  # Hide the top spine (axis line)

# Add diagonal lines to indicate the break between axes
d = .015  # Diagonal line size
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  # Upper left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Upper right diagonal

kwargs.update(transform=ax2.transAxes)  # Switch to lower subplot
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Lower left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Lower right diagonal

# Create a single legend for both subplots, positioned below the x-axis
lines, _ = ax1.get_legend_handles_labels()

# Custom labels for the legend
custom_labels = ['Agreement on Not Deforested', 'Disagreement on Deforested', 
                 'Agreement on Deforested']

# Add the legend
fig.legend(lines, custom_labels, loc='upper center', 
           bbox_to_anchor=(0.5, +0.01), ncol=3)

# Set X axis ticks
ax2.set_xticks(agree_stats['Year'])  # Set ticks to each year
ax2.set_xticklabels(agree_stats['Year'], rotation=45)  # Set labels and rotate for better visibility

plt.tight_layout(rect=[0, 0.01, 1, 1])
plt.show()



############################################################################


# MCNEMAR'S TEST


############################################################################
"""
McNemar's test requires a square contingency table. To achieve this, a 
different reclassification scheme (still binary) and agreement are calculated

Examples on conducting McNemar's test with Python found in:
https://www.geeksforgeeks.org/how-to-perform-mcnemars-test-in-python/

"""

# Reclassify GFC
"""
GFC will be reclassified to be 1 for no deforestation, 2 for deforestation
"""
gfc_binary_vars = []

for var, year in zip(gfc_vars, years):
    gfc_data = locals()[var]
    binary_data = np.where(gfc_data == 255, 1, 2)
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
"""
TMF will be reclassisified to be 4 for no deforestation, 6 for deforestation
"""
tmf_binary_vars = []

for var, year in zip(tmf_vars, years):
    tmf_data = locals()[var]
    binary_data = np.where(tmf_data == 255, 4, 6)
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


# Create empty list to store agreement layers
agreements = []

# Add binary GFC and TMF layers
for gfc, tmf, year in zip(gfc_binary_vars, tmf_binary_vars, years):
    gfc_data = locals()[gfc]
    tmf_data = locals()[tmf]
    var_name = f"agreement_{year}"
    
    agreement = gfc_data + tmf_data
    locals()[var_name] = agreement
    
    agreements.append(var_name)
    print(f"Spatial agreement map created for {year}")
    
# Check values for spatial agreement (should be 5, 6, 7, 8)
agree_vals = np.unique(locals()[agreements[1]])
print(f"Values in agreement map are {agree_vals}")

# Save maps to file
agreement_files = []

for var, year in zip(agreements, years):
    data = locals()[var]
    data = data.astype(np.uint8)
    output_filename = f"agreement_mcnemar_matrix_{year}.tif"
    output_filepath = os.path.join(out_dir, output_filename)
    
    with rasterio.open(output_filepath, 'w', **tmf_profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    
    agreement_files.append(output_filepath)
    print(f"Data for {var} saved to file")
    
# Clip each spatial agreement map and override old files
for file, var in zip(agreement_files, agreements):
    clipped_agreement = clip_raster(file, aoi_geom, nodata_val)
    
    # This overrides the old agreement variables
    locals()[var] = clipped_agreement
    
    print(f"Saved clipped data for {var}")

# Check values in clipped file (should be 0, 1, 2, 255)    
agree_vals = np.unique(locals()[agreements[1]])
print(f"Values in agreement map are {agree_vals}")

def agreement_matrix(image):
    # Mask out NoData (255) values
    valid_pixels = image[image != 255]
    
    # Count pixels with values 0, 1, and 2
    count_5 = np.sum(valid_pixels == 5) # agreement not deforestation
    count_6 = np.sum(valid_pixels == 6) # only GFC says deforestation
    count_7 = np.sum(valid_pixels == 7) # only TMF says deforestation
    count_8 = np.sum(valid_pixels == 8) # agreement on deforestation
    
    # Create contingency matrix
    matrix = [[count_5, count_7], 
              [count_6, count_8]]

    return matrix


agree_matrices = []

# Process each image
for var, year in zip(agreements, years):
    data = locals()[var]
    matrix = agreement_matrix(data)
    agree_matrices.append(matrix)

# Show the DataFrame
print(agree_matrices)


# Run McNemar's test on every contingency matrix
mcnemar_results = []

# Perform McNemar test for each matrix and store the results
for matrix in agree_matrices:
    result = mcnemar(matrix)
    mcnemar_results.append({'statistic': result.statistic, 
                            'pvalue': result.pvalue})
    
# Save in a dataframe
mcnemar_df = pd.DataFrame(mcnemar_results, index=years)
