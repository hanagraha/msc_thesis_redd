# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:10:35 2025

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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

# Set year range
years = range(2013, 2024)

# Set output directory
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Set default plotting colors
defaultblue = "#4682B4"
reddcol = "brown"
nonreddcol = "dodgerblue"
grnpcol = "darkgreen"

# Define color palatte (2 colors)
gfc_col = "#820300"  # Darker Red
tmf_col = "#4682B4"  # Darker Blue - lighter

# Define Color Palatte (3 colors)
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read multiple rasters
def read_files(pathlist):
    
    # Create empty list to hold arrays
    arrlist = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read file
        with rasterio.open(path) as rast:
            data = rast.read(1)
            profile = rast.profile
            
            # Add array to list
            arrlist.append(data)
            
    return arrlist, profile

# Define file paths for clustered gfc rasters
gfc_forclust_paths = [f"data/intermediate/gfc_forclust_{year}.tif" for 
                      year in years]

# Define file paths for clustered tmf rasters
tmf_forclust_paths = [f"data/intermediate/tmf_forclust_{year}.tif" for 
                      year in years]

# Define file paths for ratio rasters
agratio_forclust_paths = [f"data/intermediate/disag_ratio_{year}.tif" for year 
                        in years]

# Define file paths for size rasters
agsize_forclust_paths = [f"data/intermediate/disag_size_{year}.tif" for year 
                       in years]

# Read gfc rasters 
gfc_forclust_arrs, profile = read_files(gfc_forclust_paths)

# Read tmf rasters
tmf_forclust_arrs, profile = read_files(tmf_forclust_paths)

# Read ratio rasters
agratio_forclust_arrs, ratprofile = read_files(agratio_forclust_paths)

# Read size rasters
agsize_forclust_arrs, sizeprofile = read_files(agsize_forclust_paths)


# %%
############################################################################


# CALCULATE PATCH SIZE FREQUENCY BY PATCH COUNT


############################################################################
# Define function to create patch size attribute table
def patch_att(arrlist):
    
    # Create empty dataframe
    atts = pd.DataFrame()
    
    # Iterate over each array
    for arr, year in zip(arrlist, years):
        
        # Extract pixel counts in unique patch sizes
        vals, pixels = np.unique(arr[arr != nodata_val], return_counts = True)
        
        # Convert pixel counts to patch counts
        patches = pixels / vals
        
        # Create attribute table
        att = pd.DataFrame({"Year": year,
                            "Patch Size": vals,
                            "Patch Count": patches})
        
        # Add attribute table to parent dataframe
        atts = pd.concat([atts, att], ignore_index = True)
        
    return atts

# Define function to convert patch attributes to boxplot data
def patch_boxdata(att_df, dataname):
    
    # Create empty list to hold data
    data = []
    
    # Iterate over each year
    for year in years:
        
        # Extract patch sizes present in that year
        sizes = att_df[att_df["Year"] == year]["Patch Size"]
        
        # Extract patch counts present in that year
        counts = att_df[att_df["Year"] == year]["Patch Count"]
        
        # Iterate over each size-count pair
        for size, count in zip(sizes, counts):
            
            # Multiply size row by number of counts
            data.extend([(size, year, dataname)] * int(count))
            
    # Convert list to dataframe
    boxplot_data = pd.DataFrame(data, columns = ["Patch Size", "Year", "Dataset"])
    
    return boxplot_data

# Define function to calculate mean patch size
def patch_statsum(boxdata):
    
    # Create empty list to hold means and maximums
    means = []
    maxs = []
    
    # Iterate over each year
    for year in years:
        
        # Calculate mean patch size for that year
        yearmean = boxdata[boxdata["Year"] == year]["Patch Size"].mean() * 0.09
        
        # Add mean to list
        means.append(yearmean)
        
        # Calculate max patch size for that year
        yearmax = boxdata[boxdata["Year"] == year]["Patch Size"].max() * 0.09
        
        # Add max to list
        maxs.append(yearmax)
        
    # Create dataframe from list
    stats_df = pd.DataFrame({"Year": years, 
                             "Mean Patch Size": means, 
                             "Max Patch Size": maxs})
        
    return stats_df

# Create gfc patch attribute table
gfc_patch_atts = patch_att(gfc_forclust_arrs)

# Create tmf patch attribute table
tmf_patch_atts = patch_att(tmf_forclust_arrs)

# Create spatial agreement size attribute table
agsize_patch_atts = patch_att(agsize_forclust_arrs)

# Convert gfc attribute table to boxplot data
gfc_boxdata = patch_boxdata(gfc_patch_atts, "GFC Deforestation")

# Convert tmf attribute table to boxplot data
tmf_boxdata = patch_boxdata(tmf_patch_atts, "TMF Deforestation")

# Convert size attribute table to boxplot data
agsize_boxdata = patch_boxdata(agsize_patch_atts, "Potential Deforestation") 

# Combine gfc and tmf boxplot data
gfctmf_boxdata = pd.concat([gfc_boxdata, tmf_boxdata], ignore_index=True)

# Calculate mean and max gfc patch size
gfc_patchstats = patch_statsum(gfc_boxdata)

# Calculate mean and max tmf patch size
tmf_patchstats = patch_statsum(tmf_boxdata)

# Create copy of boxplot data
gfctmf_boxdata_ha = gfctmf_boxdata.copy()

# Convert patch size from pixels to ha
gfctmf_boxdata_ha['Patch Size']*=0.09


# %%
############################################################################


# PLOT PATCH BOX PLOT


############################################################################
# Initialize figure
plt.figure(figsize=(10, 6))

# Define dataset colors for seaborne
data_cols = {"GFC Deforestation": bluecols[0], "TMF Deforestation": bluecols[1]}

# Plot boxplot data
sns.boxplot(x='Year', y='Patch Size', hue='Dataset', data=gfctmf_boxdata_ha, 
            width = 0.7, showfliers=False, palette = data_cols)

# Edit tickmark fontsizes
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)

# Add y major gridlines
plt.grid(axis='y', which='major', color='gray', linestyle='--', alpha = 0.6)

# Add axes labels
plt.xlabel('Year', fontsize = 16)
plt.ylabel('Deforestation Patch Size (ha)', fontsize = 16)

# Adapt legend fontsize
plt.legend(fontsize = 16)

# Show plot
plt.show()


# %%
############################################################################


# PLOT MAXIMUM PATCH SIZES


############################################################################
# Initialize figure
plt.figure(figsize = (10, 6))

# Add gfc patch statistics
plt.plot(years, gfc_patchstats["Max Patch Size"], color = bluecols[0], 
         linewidth = 2, label = "GFC Deforestation Patches")

# Add tmf patch statsitics
plt.plot(years, tmf_patchstats["Max Patch Size"], color = bluecols[1],
         linewidth = 2, label = "TMF Deforestation Patches")

# Add axes labels
plt.xlabel("Year", fontsize = 16)
plt.ylabel("Maximum Patch Area (ha)", fontsize = 16)

# Plot years as x axis
plt.xticks(years)

# Adjust font size of tick labels
plt.tick_params(axis='both', which='major', labelsize = 14)

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Add legend
plt.legend(fontsize = 16)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# PLOT MEAN PATCH SIZES


############################################################################
# Initialize figure
plt.figure(figsize = (10, 6))

# Add gfc patch statistics
plt.plot(years, gfc_patchstats["Mean Patch Size"], color = bluecols[0], 
         linewidth = 2, label = "GFC Deforestation Patches")

# Add tmf patch statsitics
plt.plot(years, tmf_patchstats["Mean Patch Size"], color = bluecols[1],
         linewidth = 2, label = "TMF Deforestation Patches")

# Add axes labels
plt.xlabel("Year", fontsize = 16)
plt.ylabel("Mean Patch Area (ha)", fontsize = 16)

# Plot years as x axis
plt.xticks(years)

# Adjust font size of tick labels
plt.tick_params(axis='both', which='major', labelsize = 14)

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Add legend
plt.legend(fontsize = 16)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# COMBINE AGREEMENT RATIO AND CLUSTER SIZE


############################################################################
# Flatten agreement ratio rasters
agratio_forclust_flat = [arr.flatten() for arr in agratio_forclust_arrs]

# Flatten size rasters
agsize_forclust_flat = [arr.flatten() for arr in agsize_forclust_arrs]

# Remove nodata values
datamasks = [(arr != 255) for arr in agratio_forclust_flat]

# Create empty dictionary to hold data
ratsize = {}

# Iterate over each array and mask
for ratio, size, mask, year in zip(agratio_forclust_flat, agsize_forclust_flat, 
                                   datamasks, years):
    
    # Create dataframe
    df = pd.DataFrame({
        "Agreement Ratio": ratio[mask],
        "Patch Size": size[mask]
        })
    
    # Add dataframe to dictionary
    ratsize[year] = df

# Extract list of patch sizes
allsizes = [np.unique(size) for size in agsize_forclust_arrs]

# Test with 2013
sizelist2013 = allsizes[0]
size2013 = agsize_forclust_arrs[0]
ratio2013 = agratio_forclust_arrs[0]
year = 2013

patchratios = {}

# Iterate over each array
for i, year in enumerate(years):
    
    # Extract patch size array
    sizearr = agsize_forclust_arrs[i]
    
    # Extract patch ratio array
    ratioarr = agratio_forclust_arrs[i]
    
    # Create list of unique patch sizes
    sizelist = np.unique(sizearr)

    # Create empty lists to hold patch size statistics
    sizes = []
    pixels = []
    patches = []
    ratios = []
    
    # Iterate over each patch size
    for size in sizelist:
        
        # Create patch size mask
        patchmask = (sizearr == size) & (ratioarr != nodata_val)
                
        # Convert patch size to ha
        size_ha = size * 0.09
        
        # Add to list
        sizes.append(size_ha)
        
        # Calculate number of pixels in mask
        patch_pix = np.sum(patchmask)
        
        # Add to list
        pixels.append(patch_pix)
        
        # Calculate number of patches with that size
        patch_count = patch_pix / size
        
        # Add to list
        patches.append(patch_count)
        
        # Extract mean agreement of patches
        ratio = np.mean(ratioarr[patchmask])
        
        # Add to list
        ratios.append(ratio)
        
    # Create dataframe from patch statistics
    df = pd.DataFrame({
        "Patch Size": sizes,
        "Pixel Count": pixels,
        "Patch Count": patches,
        "Agreement Ratio": ratios})
    
    # Add dataframe to dictionary
    patchratios[year] = df
    
    # Print statement
    print(f"Calculated patch size statistics for {year}")


# %%
############################################################################


# PLOT RATIO AND SIZE SCATTERPLOT


############################################################################
# Define color palatte
palette = sns.color_palette("viridis", n_colors=len(years))

# Initialize figure
plt.figure(figsize=(10, 6))

# Iterate over dictionary entries (per year)
for i, (year, df) in enumerate(patchratios.items()):
    
    # Add scatterplot data
    sns.scatterplot(data=df, x='Patch Size', y='Agreement Ratio', label=year, 
                    color=palette[i])

# Add axes labels
plt.xlabel("Deforestation Patch Size (ha)", fontsize=16)
plt.ylabel("Mean Deforestation Agreement Ratio", fontsize=16)

# Adjust font size of tick labels
plt.tick_params(axis='both', which='major', labelsize=14)

# Add gridlines
plt.grid(linestyle="--", alpha=0.6)

# Add legend
plt.legend(title="Year", fontsize=16, title_fontsize = 16)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# BIN SPATIAL AGREEMENT PATCH SIZES


############################################################################
# Define bins
bins = [0, 0.5, 1, 5, 10, float('inf')]

# Define bin labels
labels = ['0-0.5', '0.5-1', '1-5', '5-10', '>10']

# Create empty dictionary to store aggregated data
patchratios_agg = {}

# Iterate over each dictionary item
for year, df in patchratios.items():

    # Create patch size column
    df['Patch Size Bin'] = pd.cut(df['Patch Size'], bins=bins, labels=labels, 
                                  right=False)
    
    # Aggregate data by patch size bin
    binned_df = df.groupby('Patch Size Bin', observed = False).agg({
        'Pixel Count': 'sum',
        'Patch Count': 'sum',
        'Agreement Ratio': 'mean'
    }).reset_index()

    patchratios_agg[year] = binned_df


# %%
############################################################################


# PLOT BINNED DATA


############################################################################
# Convert the dictionary into a DataFrame
df = pd.DataFrame()

for year, year_df in patchratios_agg.items():
    year_df['Year'] = year  # Add the year column to each DataFrame
    df = pd.concat([df, year_df])

# Create color palette
palette = sns.color_palette("deep", 5)

# Initialize figure
plt.figure(figsize=(10, 6))

# Add lineplot data
sns.lineplot(data=df, x='Year', y='Agreement Ratio', hue='Patch Size Bin', 
             linewidth = 2, palette = palette)

# Adjust axes labels
plt.xlabel('Year', fontsize=16)
plt.ylabel('Mean Proportional Deforestation Agreement', fontsize=16)

# Adjust font size of tick labels
plt.tick_params(axis='both', which='major', labelsize=14)

# Add x ticks for every year
plt.xticks(years)

# Add gridlines
plt.grid(linestyle='--', alpha=0.6)

# Add legend
plt.legend(title="Patch Size (ha)", fontsize=16, title_fontsize = 16)

# Show plot
plt.tight_layout()
plt.show()


