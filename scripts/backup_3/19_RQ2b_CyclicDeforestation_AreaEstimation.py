# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:17:56 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import rasterio



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

# Set output directory
val_dir = os.path.join('data', 'validation')

# Set year range
years = range(2013, 2024)

# Define color palatte
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read files in subfolder
def folder_files(folder, suffix):
    
    # Define folder path
    folderpath = os.path.join(val_dir, folder)
    
    # Create empty list to store files
    paths = []

    # Iterate over every item in folder
    for file in os.listdir(folderpath):
        
        # Check if file ends in suffix
        if file.endswith(suffix):
            
            # Create path for file
            filepath = os.path.join(folderpath, file)
            
            # Add file to list
            paths.append(filepath)
    
    return paths

# Define function to read files from list
def list_read(pathlist, suffix):
    
    # Create empty dictionary to store outputs
    files = {}
    
    # Iterate over each file in list
    for path in pathlist:
        
        # Read file
        data = pd.read_csv(path)
        
        # Extract file name
        filename = os.path.basename(path)
        
        # Remove suffix from filename
        var = filename.replace(suffix, "")
        
        # Add data to dictionary
        files[var] = data
        
    return files

# Read validationd data
valdata = pd.read_csv("data/validation/validation_datasets/validation_points_2013_2023_780_nobuffer.csv", 
                       delimiter=",", index_col=0)

# Convert csv geometry to WKT
valdata['geometry'] = gpd.GeoSeries.from_wkt(valdata['geometry'])

# Convert dataframe to geodataframe
valdata = gpd.GeoDataFrame(valdata, geometry='geometry', crs="EPSG:32629")

# Read stratification map
with rasterio.open("data/intermediate/stratification_layer_nogrnp.tif") as rast:
    
    # Read data
    stratmap = rast.read()
    
    # Get profile
    profile = rast.profile

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k').reset_index()

# Extract redd+ polygon area (ha)
redd_ha = villages.loc[1].geometry.area / 10000

# Extract non-redd+ polygon area (ha)
nonredd_ha = villages.loc[0].geometry.area / 10000

# Define protocol d filepaths
protd_statpaths = folder_files("val_protd_780nobuff", "stehmanstats.csv")
protd_cmpaths = folder_files("val_protd_780nobuff", "confmatrix.csv")

# Read protocol d statistics
protd_stats = list_read(protd_statpaths, "_stehmanstats.csv")
protd_cm = list_read(protd_cmpaths, "_confmatrix.csv")

# Subset to only keep years 2013-2023 (statistics)
for key in protd_stats:    
    protd_stats[key] = protd_stats[key][(protd_stats[key]['year'] >= 2013) & \
                                    (protd_stats[key]['year'] <= 2023)]
    protd_stats[key] = protd_stats[key].reset_index(drop = True)

# Calculate total map pixels
total_pix = np.sum(stratmap != 255)

# Convert map pixels to map area (ha)
total_ha = total_pix * 0.09

# Read stratification map (redd)
with rasterio.open("data/intermediate/stratification_layer_redd.tif") as rast:
    
    # Read data
    strat_redd = rast.read()
    
    # Get profile 
    profile = rast.profile
    
# Read stratification map (nonredd)
with rasterio.open("data/intermediate/stratification_layer_nonredd.tif") as rast:
    
    # Read data
    strat_nonredd = rast.read()
    
    # Get profile
    profile = rast.profile
    
# Calculate redd pixels
redd_pix = np.sum(strat_redd != 255)

# Calculate nonredd pixels
nonredd_pix = np.sum(strat_nonredd != 255)


# %%
############################################################################


# CALCULATE RECURRING AREA ESTIMATION (WHOLE AREA)


############################################################################
# Define function to calculate deforestation area per year
def defor_area(valpoints, stratamap, defor1 = True, defor2 = True, defor3 = True):
    
    # Extract strata
    strata = valpoints['strata']
    
    # Calculate number of pixels per strata
    pixvals, pixcounts = np.unique(stratamap, return_counts = True)
    
    # Calculate number of points per strata
    strata_points = strata.value_counts().sort_index()
    
    # Caclulate representative area of each point
    point_area = pixcounts[:-1] / strata_points
    
    # Convert pixels to ha
    point_ha = point_area * 0.09
    
    # Assign point area for each validation point
    valpoints_ha = pd.Series([point_ha.get(x, np.nan) for x in strata], 
                             index = pd.Series(strata).index)
    
    # Create new validation database
    valarea = valpoints.copy()
    
    # Add deforestation count column
    valarea['defor_count'] = ((valarea[['defor1', 'defor2', 'defor3']] != 0).sum(axis=1))
    
    # Add point area column
    valarea['point_area'] = valpoints_ha
    
    # Create dataframe for the results
    defor_annual = pd.DataFrame([[0] * len(years)], columns = years)
    
    # Set row index
    defor_annual.index = ['defor_area']
    
    # Iterate over each point
    for idx, row in valarea.iterrows():
        
        # Calculate further if including first deforestation
        if defor1 == True:
        
            # Extract deforestation 1 year
            year1 = row['defor1']
            
            # If there is deforestation in year1
            if year1 in years:
                
                # Add corresponding area to dataframe (summing it up)
                defor_annual[year1] = defor_annual[year1] + row['point_area']
                
        # Calculate further if including second deforestation
        if defor2 == True:
        
            # Extract deforestation 2 year
            year2 = row['defor2'] 
            
            # If there is deforestation in year2
            if year2 in years:
                
                # Add corresponding area to dataframe (summing it up)
                defor_annual[year2] = defor_annual[year2] + row['point_area']
            
        # Calculate further if including third deforestation
        if defor3 == True:
            
            # Extract deforestation 3 year
            year3 = row['defor3'] 
            
            # If there is deforestation in year3
            if year3 in years:
                
                # Add corresponding area to dataframe (summing it up)
                defor_annual[year3] = defor_annual[year3] + row['point_area']
            
    return defor_annual

# Calculate area for all deforestation events
defor3_area = defor_area(valdata, stratmap)

# Calculate area for first deforestation event
defor1_area = defor_area(valdata, stratmap, defor1 = True, defor2 = False, defor3 = False)

# Extract area calculated by stehman
stehman_area = protd_stats['protd_gfc']['area'] * total_ha
stehman_area.index = pd.Index(years)
    

# %%
############################################################################


# PROPORTIONAL AREAS MANUAL


############################################################################
"""
Y = (strata_size(proportion of class in strata) + ....) / map size

eg. [40,000(0.20) + 30,000(0.00) + 20,000(0.50) + 10,000(0.20)]/100,000

where strata 1, 2, 3, 4 have sizes 40,0000, 30,000, 20,000, and 10,000
sample size per strata = 10 points
occurence of class C in strata 1 = 2, 2 = 0, 3 = 5, 4 = 2
"""
# Define function to calculate proportional areas
def steh_area(valdata, stratmap, deforlist):
    
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
            area = (cp * size) / total_pix
            
            # Add deforestation area to list
            strata_defor.append(area)
            
        # Add the list as a column in the DataFrame
        year_defor[year] = strata_defor

    # Take the sum per year
    total_defor = year_defor.sum(axis=0) 
    
    return total_defor

# Estimate deforestation for the first year
first_defor = steh_area(valdata, stratmap, ['defor1'])

# Estimate deforestation for second year
second_defor = steh_area(valdata, stratmap, ['defor2'])

# Estimate deforestation for third year
third_defor = steh_area(valdata, stratmap, ['defor3'])

# Estimate deforestation for all years
all_defor = steh_area(valdata, stratmap, ['defor1', 'defor2', 'defor3'])


# %%
############################################################################


# CALCULATE STANDARD ERROR


############################################################################
def calc_eea(data_dict):
    
    # Create a copy of the input dictionary
    eea_dict = data_dict.copy()
    
    # Iterate over each dictionary iem
    for key, value in eea_dict.items():
        
        # If the key is for nonredd areas
        if "nonredd" in key:
            
            # Calculate error adjsuted area
            area = value['area'] * nonredd_ha
            
            # Calculate area standard error
            error = value['se_a'] * nonredd_ha
            
        # If the key is for redd areas
        elif "redd" in key:
            
            # Calculate error adjusted area
            area = value['area'] * redd_ha
            
            # Calculate area standard error
            error = value['se_a'] * redd_ha 
            
        # If the key is for the whole area
        else: 
    
            # Calculate error adjusted area
            area = value['area'] * total_ha
            
            # Extract area standard error
            error = value['se_a'] * total_ha
        
        # Calculate 95% confidence interval
        ci95 = 1.96 * error
        
        # Calculate 50% confidence interval
        ci50 = 0.67 * error
        
        # Add error adjusted area to df
        value['eea'] = area
        
        # Add 95ci to df
        value['ci95'] = ci95
        
        # Add 50ci to df
        value['ci50'] = ci50
    
    return eea_dict

# Calculate eea and ci for prot b
protd_eea = calc_eea(protd_stats)


# %%
############################################################################


# PLOT MULTIPLE DEFORESTATION EVENTS


############################################################################
# Initialize figure
plt.figure(figsize=(10, 6))

# Add all deforestation data
plt.plot(years, all_defor * total_ha, linestyle='--', color = bluecols[2], 
         label='All Deforestation')

# Add first deforestation data
plt.errorbar(
    years,
    protd_eea['protd_gfc']['eea'],
    yerr = protd_eea['protd_gfc']['ci50'],
    fmt="-o",
    capsize = 5,
    color = bluecols[0],
    label = "First Deforestation"
)

# Create 95% ci rectangle
plt.fill_between(
    years, 
    protd_eea['protd_gfc']['eea'][0] - protd_eea['protd_gfc']['ci95'][0],
    protd_eea['protd_gfc']['eea'][0] + protd_eea['protd_gfc']['ci95'][0],
    color = bluecols[1],
    alpha = 0.2,
    label = "95% confidence interval"
    )

# Create 95% ci rectangle
plt.fill_between(
    years, 
    protd_eea['protd_gfc']['eea'][0] - protd_eea['protd_gfc']['ci50'][0],
    protd_eea['protd_gfc']['eea'][0] + protd_eea['protd_gfc']['ci50'][0],
    color = bluecols[1],
    alpha = 0.3,
    label = "50% confidence interval"
    )

# Add axes labels
plt.xlabel("Year", fontsize = 16)
plt.ylabel("Deforestation Area (ha)", fontsize = 16)

# Get handles if plot items
handles, labels = plt.gca().get_legend_handles_labels()

# Define order of legend items
order = [0, 3, 1, 2] 

# Add legend with manual item ordering
plt.legend([handles[i] for i in order], [labels[i] for i in order], 
           fontsize=14, loc="upper right")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Adjust tickmarks
plt.xticks(years, fontsize = 14)
plt.yticks(fontsize = 14)

# Show the plot
plt.show()


# %%
############################################################################


# CALCULATE STANDARD ERROR FOR AREA ESTIMATION


############################################################################
"""
V(Y) = (1/N2) * (strata_size^2 * ((1-proportion of class in strata / sample size)))
"""
# Extract gfc area errors
gfc_errors = protd_stats['protd_gfc']['se_a']

# Extract tmf area errors
tmf_errors = protd_stats['protd_tmf']['se_a']

# Extract se area errors
se_errors = protd_stats['protd_se']['se_a']


# Calculate sample variance

# Calculate number of pixels per strata
pixvals, pixcounts = np.unique(stratmap, return_counts = True)

# Create dataframe
strata_size = pd.DataFrame({'strata': pixvals[:-1],
                            'size': pixcounts[:-1]})

# Iterate over each strata
for idx, row in strata_size.iterrows():
    
    # Extract strata number
    strata = row['strata']
    
    # Extract strata size
    size = row['size']
    
    # Subset validation data for that strata
    data = valdata[valdata['strata'] == strata]
    
    # Separate all deforestation events
    defor = 
    
    # Calculate indicator row (correct/incorrect classification)
    if data['gfc'] == data['defor1']
    
    # Define sample mean of strata
    mean = 
    
    # Iterate over each point in strata
    for point in data:
        
        # Define y_u indicator variable
        y_u = 






