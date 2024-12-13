# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:25:56 2024

@author: hanna
"""

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
from sklearn.metrics import cohen_kappa_score



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



############################################################################


# IMPORT DATA


############################################################################
gfc_lossyear_file = "data/hansen_preprocessed/gfc_lossyear_fm.tif"
tmf_defordegra_file = "data/intermediate/gfc_tmf_combyear.tif"

villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")


# Create REDD+ and non-REDD+ polygons
villages = villages[['grnp_4k', 'geometry']]
villages = villages.dissolve(by='grnp_4k')
villages = villages.reset_index()



############################################################################


# READ DATA


############################################################################
with rasterio.open(gfc_lossyear_file) as gfc:
    gfc_lossyear = gfc.read(1)
    
with rasterio.open(tmf_defordegra_file) as tmf:
    tmf_defordegra = tmf.read(1)
    profile = tmf.profile



############################################################################


# CREATE BINARY GFC AND TMF LAYERS


############################################################################
years = range(2013, 2024)

# Reclassify GFC (where undisturbed = 1, deforested = 2)
gfc_binary_vars = []
for year in years:
    binary_data = np.where(gfc_lossyear == year, 2, np.where(
                            gfc_lossyear == nodata_val, nodata_val, 1))
    varname = f"gfc_binary_{year}"
    locals()[varname] = binary_data
    gfc_binary_vars.append(varname)
    
    print(f"Reclassified {varname} to binary codes")

# Reclassify TMF (where undisturbed = 4, deforested = 6)
tmf_binary_vars = []
for year in years:
    binary_data = np.where(tmf_defordegra == year, 6, np.where(
                            tmf_defordegra == nodata_val, nodata_val, 4))
    varname = f"tmf_binary_{year}"
    locals()[varname] = binary_data
    tmf_binary_vars.append(varname)
    
    print(f"Reclassified {varname} to binary codes")



############################################################################


# CREATE SPATIAL AGREEMENT LAYERS


############################################################################
# Create empty list to store agreement layers
agreements = []

# Add binary GFC and TMF layers
for gfc, tmf, year in zip(gfc_binary_vars, tmf_binary_vars, years):
    gfc_data = locals()[gfc]
    tmf_data = locals()[tmf]
    var_name = f"agreement_{year}"
    
    agreement = np.where((gfc_data == 255) | (tmf_data == 255), 255, 
                          gfc_data + tmf_data)
    locals()[var_name] = agreement
    
    agreements.append(var_name)
    print(f"Spatial agreement map created for {year}")
    
# Check values for spatial agreement (should be 5, 6, 7, 8)
agree_vals = np.unique(locals()[agreements[1]])

print(f"Values in agreement map are {agree_vals}")


# Save maps to file
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')
agreement_files = []

for var, year in zip(agreements, years):
    data = locals()[var]
    data = data.astype(np.uint8)
    output_filename = f"agreement_gfc_combtmf_{year}.tif"
    output_filepath = os.path.join(out_dir, output_filename)
    
    with rasterio.open(output_filepath, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)
    
    agreement_files.append(output_filepath)
    print(f"Data for {var} saved to file")



############################################################################


# CALCULATE SPATIAL AGREEMENT STATISTICS FOR AOI


############################################################################

def calc_agreement_ratios(image):
    # Mask out NoData (255) values
    valid_pixels = image[image != 255]
    
    # Count pixels with values 5, 6, 7, and 8
    total_pixels = valid_pixels.size
    count_5 = np.sum(valid_pixels == 5) # agreement undisturbed
    count_6 = np.sum(valid_pixels == 6) # only gfc detects deforested
    count_7 = np.sum(valid_pixels == 7) # only tmf detects deforested
    count_8 = np.sum(valid_pixels == 8) # agreement deforested
    
    # Reclassify counts to agreement and disagreement
    agreement_undisturbed = count_5
    disagreement = count_6 + count_7
    agreement_deforested = count_8
    
    # Calculate ratios
    perc_agree_undisturbed = (agreement_undisturbed / total_pixels)*100
    perc_disagree = (disagreement / total_pixels)*100
    perc_agree_deforested = (agreement_deforested / total_pixels)*100
    
    return perc_agree_undisturbed, perc_disagree, perc_agree_deforested


agree_stats = None

# Calculate agreement stats for each agreement image
for var, year in zip(agreements, years):
    data = locals()[var]
    perc_5, perc_67, perc_8 = calc_agreement_ratios(data)
    
    # Append results to the DataFrame
    temp_df = pd.DataFrame({
        'Year': [year],
        'Agree_Undisturbed': [perc_5],
        'Disagree': [perc_67],
        'Agree_Deforested': [perc_8]
    })
    
    if agree_stats is None:
        agree_stats = temp_df
    else:
        agree_stats = pd.concat([agree_stats, temp_df], ignore_index=True)

# Show the DataFrame
print(agree_stats)



############################################################################


# CALCULATE SPATIAL AGREEMENT STATISTICS FOR REDD+/NON-REDD+ VILLAGES


############################################################################
# Create REDD+ and non-REDD+ geometries
redd_geom = villages.loc[1].geometry
nonredd_geom = villages.loc[0, 'geometry']

def filestack_clip(arraylist, yearrange, geometry, nodataval):
    clipped_list = []
    for file, year, in zip(arraylist, yearrange):
        with rasterio.open(file) as rast:
            agree_clip, agree_trans = mask(rast, geometry, crop=True,
                                            nodata=nodataval)
        clipped_list.append(agree_clip)
    filenum = len(clipped_list)
    print(f"Clipping complete for {filenum} files")
    return clipped_list

# Clip to REDD+ area
redd_agreement = []
for file, year in zip(agreement_files, years):
    with rasterio.open(file) as rast:
        agree_clip, agree_trans = mask(rast, [redd_geom], crop=True, 
                                       nodata=nodata_val) 
    redd_agreement.append(agree_clip)
    print(f"Clipped {year} agreement to REDD+ villages")

# Check values for spatial agreement (should be 5, 6, 7, 8, 255)
agree_vals = np.unique(redd_agreement[1])

print(f"Values in agreement map are {agree_vals}") 
    
# Clip to non-REDD+ area
nonredd_agreement = []
for file, year in zip(agreement_files, years):
    with rasterio.open(file) as rast:
        agree_clip, agree_trans = mask(rast, [nonredd_geom], crop=True, 
                                       nodata=nodata_val) 
    nonredd_agreement.append(agree_clip)
    print(f"Clipped {year} agreement to non-REDD+ villages")
    
# Check values for spatial agreement (should be 5, 6, 7, 8, 255)
agree_vals = np.unique(nonredd_agreement[1])

print(f"Values in agreement map are {agree_vals}") 


### REDD+ AGREEMENT STATISTICS
redd_agree_stats = None

# Calculate agreement stats for each agreement image
for var, year in zip(redd_agreement, years):
    perc_5, perc_67, perc_8 = calc_agreement_ratios(var)
    
    # Append results to the DataFrame
    temp_df = pd.DataFrame({
        'Year': [year],
        'Agree_Undisturbed': [perc_5],
        'Disagree': [perc_67],
        'Agree_Deforested': [perc_8]
    })
    
    if redd_agree_stats is None:
        redd_agree_stats = temp_df
    else:
        redd_agree_stats = pd.concat([redd_agree_stats, temp_df], 
                                     ignore_index=True)

# Show the DataFrame
print(redd_agree_stats)


### NON-REDD+ AGREEMENT STATISTICS
nonredd_agree_stats = None

# Calculate agreement stats for each agreement image
for var, year in zip(nonredd_agreement, years):
    perc_5, perc_67, perc_8 = calc_agreement_ratios(var)
    
    # Append results to the DataFrame
    temp_df = pd.DataFrame({
        'Year': [year],
        'Agree_Undisturbed': [perc_5],
        'Disagree': [perc_67],
        'Agree_Deforested': [perc_8]
    })
    
    if nonredd_agree_stats is None:
        nonredd_agree_stats = temp_df
    else:
        nonredd_agree_stats = pd.concat([nonredd_agree_stats, temp_df], 
                                        ignore_index=True)

# Show the DataFrame
print(nonredd_agree_stats)



############################################################################


# CALCULATE SPATIAL AGREEMENT STATISTICS FOR GRNP


############################################################################
# Create GRNP geometry
grnp_geom = grnp.geometry

# Clip to GRNP area
grnp_agreement = []
for file, year in zip(agreement_files, years):
    with rasterio.open(file) as rast:
        agree_clip, agree_trans = mask(rast, grnp_geom, crop=True, 
                                       nodata=nodata_val) 
    grnp_agreement.append(agree_clip)
    print(f"Clipped {year} agreement to the GRNP")

# Check values for spatial agreement (should be 5, 6, 7, 8, 255)
agree_vals = np.unique(grnp_agreement[1])

print(f"Values in agreement map are {agree_vals}") 

### GRNP AGREEMENT STATISTICS
grnp_agree_stats = None

# Calculate agreement stats for each agreement image
for var, year in zip(grnp_agreement, years):
    perc_5, perc_67, perc_8 = calc_agreement_ratios(var)
    
    # Append results to the DataFrame
    temp_df = pd.DataFrame({
        'Year': [year],
        'Agree_Undisturbed': [perc_5],
        'Disagree': [perc_67],
        'Agree_Deforested': [perc_8]
    })
    
    if grnp_agree_stats is None:
        grnp_agree_stats = temp_df
    else:
        grnp_agree_stats = pd.concat([grnp_agree_stats, temp_df], 
                                     ignore_index=True)

# Show the DataFrame
print(grnp_agree_stats)


############################################################################


# PLOT STATISTICS FOR AGREEMENT IN AOI


############################################################################
"""
The following is created with help from ChatGPT, especially the diagonal lines
"""

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                               gridspec_kw={'height_ratios': [0.8, 1]})

# Plot data in the upper subplot (for the range 90% and above)
agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 'Agree_Deforested'], 
                 kind='line', ax=ax1, legend=False)
ax1.set_ylim(94, 100)  # Upper range
ax1.grid(True, linestyle='--')
ax1.set_ylabel('Percentage of Pixels in AOI (%)')
ax1.spines['bottom'].set_visible(False)  # Hide the bottom spine (axis line)

# Plot data in the lower subplot (for the range below 5%)
agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 'Agree_Deforested'], 
                 kind='line', ax=ax2, legend=False)
ax2.set_ylim(0, 3)  # Lower range
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


# PLOT STATISTICS FOR AGREEMENT IN REDD+ AND NON-REDD+ VILLAGES


############################################################################
"""
The following is created with help from ChatGPT, especially the diagonal lines
"""

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                               gridspec_kw={'height_ratios': [0.8, 1]})

# Plot REDD+ data in the upper subplot (for the range 90% and above)
redd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                   'Agree_Deforested'], 
                      kind='line', ax=ax1, legend=False, color=['blue', 
                                        'orange', 'red'], linestyle='-')
# Plot non-REDD+ data in the upper subplot
nonredd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                      'Agree_Deforested'], 
                         kind='line', ax=ax1, legend=False, color=['green', 
                                        'orange', 'red'], linestyle='--')

ax1.set_ylim(90, 100) 
ax1.grid(True, linestyle='--')
ax1.set_ylabel('Percentage of Pixels in AOI (%)')
ax1.spines['bottom'].set_visible(False) 

# Plot REDD+ data in the lower subplot (for the range below 5%)
redd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                   'Agree_Deforested'], 
                      kind='line', ax=ax2, legend=False, color=['green', 
                                        'orange', 'red'], linestyle='-')
# Plot non-REDD+ data in the lower subplot
nonredd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                      'Agree_Deforested'], 
                         kind='line', ax=ax2, legend=False, color=['green', 
                                        'orange', 'red'], linestyle='--')

ax2.set_ylim(0, 5)  # Lower range
ax2.grid(True, linestyle='--')
ax2.set_xlabel('Year')
ax2.set_ylabel('Percentage of Pixels in AOI (%)')
ax2.spines['top'].set_visible(False)

# Add diagonal lines to indicate the break between axes
d = .015  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs) 

kwargs.update(transform=ax2.transAxes) 
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

# Create a single legend for both subplots, positioned below the x-axis
lines, _ = ax1.get_legend_handles_labels()

# Custom labels for the legend
custom_labels = ['REDD+ Agreement on Not Deforested', 
                 'REDD+ Disagreement on Deforested', 
                 'REDD+ Agreement on Deforested',
                 'Non-REDD+ Agreement on Not Deforested', 
                 'Non-REDD+ Disagreement on Deforested', 
                 'Non-REDD+ Agreement on Deforested']

# Add the legend
fig.legend(lines, custom_labels, loc='upper center', 
           bbox_to_anchor=(0.5, +0.01), ncol=3)

# Set X axis ticks
ax2.set_xticks(redd_agree_stats['Year']) 
ax2.set_xticklabels(redd_agree_stats['Year'], rotation=45)

plt.tight_layout(rect=[0, 0.01, 1, 1])
plt.show()



############################################################################


# PLOT STATISTICS FOR AGREEMENT IN REDD+, NON-REDD+, AND GRNP (FOR REFERENCE)


############################################################################
# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                               gridspec_kw={'height_ratios': [0.8, 1]})

# Plot REDD+ data in the upper subplot (for the range 90% and above)
redd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                   'Agree_Deforested'], 
                      kind='line', ax=ax1, legend=False, color=['green', 
                                            'orange', 'red'], linestyle='-')
# Plot non-REDD+ data in the upper subplot
nonredd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                      'Agree_Deforested'], 
                         kind='line', ax=ax1, legend=False, color=['green', 
                                            'orange', 'red'], linestyle='--')
# Plot grnp_agree_stats data in the upper subplot
grnp_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                   'Agree_Deforested'], 
                      kind='line', ax=ax1, legend=False, color=['#1f77b4', 
                                        '#4f83cc', '#86a8e7'], linestyle='-')

ax1.set_ylim(90, 100) 
ax1.grid(True, linestyle='--')
ax1.set_ylabel('Percentage of Pixels in AOI (%)')
ax1.spines['bottom'].set_visible(False)

# Plot REDD+ data in the lower subplot (for the range below 5%)
redd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                   'Agree_Deforested'], 
                      kind='line', ax=ax2, legend=False, color=['green', 
                                            'orange', 'red'], linestyle='-')
# Plot non-REDD+ data in the lower subplot
nonredd_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                      'Agree_Deforested'], 
                         kind='line', ax=ax2, legend=False, color=['green', 
                                            'orange', 'red'], linestyle='--')
# Plot grnp_agree_stats data in the lower subplot
grnp_agree_stats.plot(x='Year', y=['Agree_Undisturbed', 'Disagree', 
                                   'Agree_Deforested'], 
                      kind='line', ax=ax2, legend=False, color=['#1f77b4', 
                                        '#4f83cc', '#86a8e7'], linestyle='-')

ax2.set_ylim(0, 5) 
ax2.grid(True, linestyle='--')
ax2.set_xlabel('Year')
ax2.set_ylabel('Percentage of Pixels in AOI (%)')
ax2.spines['top'].set_visible(False) 

# Add diagonal lines to indicate the break between axes
d = .015  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs) 

kwargs.update(transform=ax2.transAxes) 
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs) 
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

# Create a single legend for all datasets, positioned below the x-axis
lines, _ = ax1.get_legend_handles_labels()

# Custom labels for the legend
custom_labels = ['REDD+ Agreement on Not Deforested', 
                 'REDD+ Disagreement on Deforested', 
                 'REDD+ Agreement on Deforested',
                 'Non-REDD+ Agreement on Not Deforested', 
                 'Non-REDD+ Disagreement on Deforested', 
                 'Non-REDD+ Agreement on Deforested',
                 'GRNP Agreement on Not Deforested', 
                 'GRNP Disagreement on Deforested', 
                 'GRNP Agreement on Deforested']

# Add the legend
fig.legend(lines, custom_labels, loc='upper center', 
           bbox_to_anchor=(0.5, +0.01), ncol=3)

# Set X axis ticks
ax2.set_xticks(redd_agree_stats['Year'])
ax2.set_xticklabels(redd_agree_stats['Year'], rotation=45)

plt.tight_layout(rect=[0, 0.01, 1, 1])
plt.show()



############################################################################


# MCNEMAR'S TEST


############################################################################
"""
McNemar's test requires a square contingency table. This format should be:
    
                               [TMF Deforestation]   [TMF Not Deforestation]      
    [GFC Deforestation]                a                        b
    [GFC Not Deforesetation]           c                        d
    
Examples on conducting McNemar's test with Python found in:
https://www.geeksforgeeks.org/how-to-perform-mcnemars-test-in-python/

"""
# Create contingency tables
def contingency_table(image):
    # Mask out NoData (255) values
    valid_pixels = image[image != 255]
    
    # Count pixels with values 0, 1, and 2
    count_5 = np.sum(valid_pixels == 5) # agreement not deforestation
    count_6 = np.sum(valid_pixels == 6) # only GFC says deforestation
    count_7 = np.sum(valid_pixels == 7) # only TMF says deforestation
    count_8 = np.sum(valid_pixels == 8) # agreement on deforestation
    
    # Create contingency matrix
    matrix = [[count_8, count_6], 
              [count_7, count_5]]

    return matrix


### MCNEMAR'S FOR AOI AGREEMENT
# Create contingency matrices for AOI agreement
agree_matrices = []

for var, year in zip(agreements, years):
    data = locals()[var]
    matrix = contingency_table(data)
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
print(mcnemar_df)


### MCNEMAR'S FOR REDD+ AGREEMENT
# Create contingency matrices for AOI agreement
redd_agree_matrices = []

for var, year in zip(redd_agreement, years):
    matrix = contingency_table(var)
    redd_agree_matrices.append(matrix)

# Show the DataFrame
print(redd_agree_matrices)

# Run McNemar's test on every contingency matrix
redd_mcnemar_results = []

# Perform McNemar test for each matrix and store the results
for matrix in redd_agree_matrices:
    result = mcnemar(matrix)
    redd_mcnemar_results.append({'statistic': result.statistic, 
                                 'pvalue': result.pvalue})
    
# Save in a dataframe
redd_mcnemar_df = pd.DataFrame(redd_mcnemar_results, index=years)
print(redd_mcnemar_df)


### MCNEMAR'S FOR NON-REDD+ AGREEMENT
# Create contingency matrices for AOI agreement
nonredd_agree_matrices = []

for var, year in zip(nonredd_agreement, years):
    matrix = contingency_table(var)
    nonredd_agree_matrices.append(matrix)

# Show the DataFrame
print(nonredd_agree_matrices)

# Run McNemar's test on every contingency matrix
nonredd_mcnemar_results = []

# Perform McNemar test for each matrix and store the results
for matrix in nonredd_agree_matrices:
    result = mcnemar(matrix)
    nonredd_mcnemar_results.append({'statistic': result.statistic, 
                                    'pvalue': result.pvalue})
    
# Save in a dataframe
nonredd_mcnemar_df = pd.DataFrame(nonredd_mcnemar_results, index=years)
print(nonredd_mcnemar_df)



############################################################################


# PLOT MCNEMAR'S TEST RESULTS


############################################################################
### RESULTS FOR THE WHOLE AOI
plt.figure(figsize=(10, 6))
plt.plot(mcnemar_df.index, mcnemar_df['statistic'], linestyle='-', 
         color='#5B9BD5')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('McNemar Statistic')
plt.xticks(mcnemar_df.index)  # Show all years as ticks
plt.grid(linestyle='--')  # Add gridlines
plt.show()


### RESULTS FOR REDD+ / NON-REDD+ VILLAGES
plt.figure(figsize=(10, 6))

# Plot REDD+ McNemar statistics
plt.plot(redd_mcnemar_df.index, redd_mcnemar_df['statistic'], linestyle='-', 
         color='#5B9BD5', label='REDD+')

# Plot non-REDD+ McNemar statistics
plt.plot(nonredd_mcnemar_df.index, nonredd_mcnemar_df['statistic'], 
         linestyle='--', color='#ED7D31', label='Non-REDD+')

# Highlight the first point in non-REDD+ with a red dot
plt.plot(nonredd_mcnemar_df.index[0], nonredd_mcnemar_df['statistic'].iloc[0], 
         'ro', label='Statistically insignificant McNemar results')

# Adding labels, title, and legend
plt.xlabel('Year')
plt.ylabel('McNemar Statistic')
plt.title('McNemar Statistic Comparison: REDD+ vs Non-REDD+')
plt.xticks(redd_mcnemar_df.index)
plt.grid(linestyle='--')

# Add legend to differentiate the lines
plt.legend()

# Display the plot
plt.show()



############################################################################


# COHEN'S KAPPA


############################################################################
"""
"The kappa statistic, which is a number between -1 and 1. The maximum value 
means complete agreement; zero or lower means chance agreement."
From: https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.cohen_kappa_score.html

"""

# Create copies of the original arrays
gfc_copy = np.copy(gfc_binary_2013).astype(float)
tmf_copy = np.copy(tmf_binary_2013).astype(float)

# Replace 255 with NaN in gfc_copy
gfc_copy[gfc_copy == 255] = np.nan

# Replace 255 with NaN in tmf_copy
tmf_copy[tmf_copy == 255] = np.nan

# Convert 4 to 1 and 6 to 2 in tmf_copy
tmf_copy[tmf_copy == 4] = 1
tmf_copy[tmf_copy == 6] = 2

# Remove NaNs for Kappa calculation
mask = ~np.isnan(gfc_copy) & ~np.isnan(tmf_copy)
filtered_gfc = gfc_copy[mask]
filtered_tmf = tmf_copy[mask]

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(filtered_gfc, filtered_tmf)

print(f"Cohen's Kappa: {kappa}")


############################################################################


# CREATE COMBINED AGREEMENT MAPS


############################################################################
"""
The sensitive early combination is defined by Bos et al. (2019) as recording 
the earliest deforestation year between two datasets, regardless of the other
product's detection
"""

### SENSITIVE EARLY
# Read relevant datasets
gfc_lossyear = "data/hansen_preprocessed/gfc_lossyear_fm.tif"
tmf_defordegra = "data/intermediate/tmf_defordegra_year.tif"

# Combine TMF and GFC maps
with rasterio.open(gfc_lossyear) as src1, rasterio.open(tmf_defordegra) as src2:
    gfc = src1.read(1)  
    tmf = src2.read(1) 
    profile = src1.profile

    gfc_mask = gfc == nodata_val
    tmf_mask = tmf == nodata_val

    # Take minimum value of both datasets, or the value of the other dataset
    # if pixel value is 255 (nodata)
    combined_data = np.where(tmf_mask, gfc, np.where(gfc_mask, 
                             tmf, np.minimum(tmf, gfc)))
    
    # Define output filename
    gfc_tmf_outfile = "data/intermediate/gfc_tmf_sensitive_early.tif"

    # Write the output raster
    with rasterio.open(gfc_tmf_outfile, 'w', **profile) as dst:
        dst.write(combined_data, 1)

print(f"Combined raster saved to {gfc_tmf_outfile}")

# View unique values to check
comb_vals = np.unique(combined_data)
print(f"Values in agreement map are {comb_vals}")

























