# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:45:46 2025

@author: hanna
"""
############################################################################


# IMPORT PACKAGES


############################################################################

import os
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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

# Define output directory
out_dir = os.path.join('data', 'intermediate')

# Define output directory
val_dir = os.path.join("data", "validation")

# Set year range
years = range(2013, 2024)

# Set random seed for reproducibility
np.random.seed(42)

# Define color palatte
blue1 = "#1E2A5E"
blue2 = "#83B4FF"
blue3 = "brown"
bluecols = [blue1, blue2, blue3]



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read list of paths
def read_files(pathlist):
    
    # Create empty list to hold arrays
    arrlist = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read file
        with rasterio.open(path) as rast:
            data = rast.read(1)
            profile = rast.profile
            transform = rast.transform
            
            # Add array to list
            arrlist.append(data)
            
    return arrlist, profile, transform

# Define agreement filepaths
agreement_filepaths = [f"data/intermediate/agreement_gfc_combtmf_{year}.tif"
                       for year in years]

# Define gfc lossyear filepath
gfc_lossyear_path = "data/hansen_preprocessed/gfc_lossyear_fm.tif"

# Define tmf defordegra filepath
tmf_defordegra_path = "data/jrc_preprocessed/tmf_defordegrayear_fm.tif"

# Define sensitive early filepath 
sensitive_early_path = "data/intermediate/gfc_tmf_sensitive_early.tif"

# Read agreement files
agreement_arrs, profile, transform = read_files(agreement_filepaths)

# Read GRNP vector data
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create GRNP geometry
grnp_geom = grnp.geometry

# Read vector data
villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create REDD+ and non-REDD+ polygons
villages = villages[['grnp_4k', 'geometry']]
villages = villages.dissolve(by='grnp_4k')
villages = villages.reset_index()

# Create REDD+ and non-REDD+ geometries
redd_geom = villages.loc[1, 'geometry']
nonredd_geom = villages.loc[0, 'geometry']

# Read stratification array 
with rasterio.open("data/intermediate/stratification_layer_nogrnp.tif") as rast:
    nogrnp_stratarr = rast.read(1)


# %%
############################################################################


# CREATE BUFFER STRATA


############################################################################
# Define function to write file to drive
def rast_write(array, filename, out_dir, profile):
    
    # Define output filepath
    output_filepath = os.path.join(out_dir, filename)
    
    # Save to file
    with rasterio.open(output_filepath, "w", **profile) as dst:
        dst.write(array, 1)

# Create a undisturbed forest mask (strata 23)
forest = (nogrnp_stratarr == 23)

# Define the structure for erosion
buffer_structure = np.ones((3, 3))

# Perform binary erosion to shrink clusters
buffer = ndimage.binary_erosion(forest, structure = buffer_structure)

# Copy stratified array to store results
strat_buffed = np.copy(nogrnp_stratarr)

# Assign buffer forest to straat 23
strat_buffed[forest & ~buffer] = 23

# Assign undisturbed forest to strata 24
strat_buffed[buffer] = 24

# Write stratified layer to file
rast_write(strat_buffed, "stratification_layer_buffered.tif", out_dir, profile)


# %%
############################################################################


# EXPLORE STRATA SIZE


############################################################################
# Extract values and ocunts of each strata
values, counts = np.unique(strat_buffed, return_counts = True)

# Drop the value, count pair for nodata val
values = values[:24]
counts = counts[:24]

# Define colors for deforested strata bars
colors = [bluecols[0] if i % 2 else bluecols[1] for i in values[:-2]]  

# Define colors for undisturbed strata bars
colors.extend(['green', 'green'])

# Initialize figure
plt.figure(figsize = (10, 6))

# Add bar data
bars = plt.bar(values, counts, color=colors)

# Add axes labels
plt.xlabel('Strata', fontsize = 12)
plt.ylabel('Number of Pixels', fontsize = 12)

# Add gridlines
plt.grid(True, linestyle = "--", alpha = 0.6)

# Add x axes tickmarks
plt.gca().set_xticks(values)

# Create manual legend
legend_elements = [
    Patch(facecolor = colors[0], edgecolor = colors[0], label = 
          'Deforestation Disagreement Strata'),
    Patch(facecolor = colors[1], edgecolor = colors[1], label = 
          'Deforestation Agreement Strata'),
    Patch(facecolor = colors[23], edgecolor = colors[23], label = 
          'Undisturbed Forest Agreement Strata')
]

# Add legend
plt.legend(handles = legend_elements, loc = 'upper left', fontsize = 12)

# Adjust layout and display
plt.tight_layout()
plt.show()


# %%
############################################################################


# SAMPLE POINTS FROM STRATA


############################################################################
# Define function to sample points from array
def strat_sample(array, sample_size, transform, profile, min_strata = None, 
                 max_strata = None, random_state=None):
    
    # Set local random generator with given random state (for reproducibility)
    rng = np.random.default_rng(random_state)
    
    # Identify number of clasess
    classnum = len(np.unique(array))
    
    # Select strata of interest (if defined)
    if min_strata is not None or max_strata is not None:
        
        # Replace values outside range with nodata
        array = np.where(
            (array >= (min_strata if min_strata is not None else array.min())) &
            (array <= (max_strata if max_strata is not None else array.max())),
            array, nodata_val
            )
    
    # Create empty list to hold point coordinates
    sample_points = []
    
    # Iterate over each class
    for strata in range(np.min(array), classnum):
        
        # Find indices of pixels in class
        pixel_indices = np.argwhere(array == strata)
        
        # If pixels in class < sample size
        if len(pixel_indices) <= sample_size:
            
            # Sample all pixels
            selected_indices = pixel_indices
            
        # If pixels in class > sample size
        else:
            
            # Take random sample
            selected_indices = pixel_indices[rng.choice(
                pixel_indices.shape[0], sample_size, replace=False)]
            
        # Iterate over each pixel index
        for idx in selected_indices:
            
            # Extract row and column for each pixel
            row, col = idx
            
            # Use transform information convert indices to coordinates
            x, y = rasterio.transform.xy(transform, row, col)
            
            # Append strata and Point geometry (x, y) to list
            sample_points.append({"strata": strata, "geometry": Point(x, y)})
            
    # Convert sample point list to geodataframe
    sample_point_gdf = gpd.GeoDataFrame(sample_points, crs=profile['crs'])
    
    return sample_point_gdf

# Define function to write points to file
def shp_write(gdf, filename, out_dir):
    
    # Define output filepath
    output_filepath = os.path.join(out_dir, filename)
    
    # Write to file
    gdf.to_file(output_filepath)

# Create sample points with 60 points per strata
points_60 = strat_sample(strat_buffed, 60, transform, profile, 
                         random_state = 33)

# Write sample points to file
shp_write(points_60, "validation_points_buffer.shp", val_dir)



############################################################################


# EXTRACT DATASET VALUES PER SAMPLE POINT


############################################################################
# Define function to extract raster values per point
def extract_val(points_gdf, tiflist, tifnames):
    
    # Copy points gdf
    gdf = points_gdf.copy()
    
    # Iterate over each tif file
    for tif, name in zip(tiflist, tifnames):
        
        # Create empty list to store pixel values
        pix_vals = []
        
        # Read tif file
        with rasterio.open(tif) as src:
        
            # Iterate over each point
            for pnt in gdf.geometry:
                
                # Get row and column indices
                row, col = src.index(pnt.x, pnt.y)
                
                # Extract pixel value at point location
                pix_val = src.read(1)[row, col]
                
                # Append pixel value to list
                pix_vals.append(pix_val)
            
        # Add new column to geodataframe
        gdf[name] = pix_vals
        
    return gdf

# Define function to save gdf as csv
def write_csv(gdf, out_dir, outfilename):
    
    # Convert the geometry to WKT format
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.wkt)
    
    # Define output path
    outfilepath = os.path.join(out_dir, f"{outfilename}.csv")

    # Save the GeoDataFrame as a CSV file
    gdf.to_csv(outfilepath, index=True)
    
    # Print statement
    print(f"File saved to {outfilepath}")

# Define list of rasters
tiflist = [gfc_lossyear_path, tmf_defordegra_path, sensitive_early_path]

# Define names of rasters
tifnames = ['gfc', 'tmf', 'se']

# Extract raster values
valpoints = extract_val(points_60, tiflist, tifnames)

# Write points to file
write_csv(valpoints, val_dir, "validation_points_buffer")


# %%
############################################################################


# CHECK: HOW MANY POINTS ARE CURRENTLY IN BUFFER?


############################################################################
import pandas as pd

# Read validation data
val_data = pd.read_csv("data/validation/validation_points_labelled.csv", 
                       delimiter=",", index_col=0)

# Convert csv geometry to WKT
val_data['geometry'] = gpd.GeoSeries.from_wkt(val_data['geometry'])

# Convert dataframe to geodataframe
val_data = gpd.GeoDataFrame(val_data, geometry='geometry', crs="EPSG:32629") 

# Define strata file path
strat_path = os.path.join(out_dir, "stratification_layer_buffered.tif")

# Extract buffered strata
val_data_strat = extract_val(val_data, [strat_path], ["buff_strat"])

# Extract counts from buffered strata column
counts = val_data_strat['buff_strat'].value_counts()


# %%
############################################################################


# CHECK: ARE THERE DUPLICATEPOINTS?


############################################################################
# Subset the original gdf to only keep strata 23
val_data_23 = val_data[val_data['strata'] == 23]

# Subset new gdf to only keep strata 23, 24
valpoints_2324 = valpoints[valpoints['strata'].isin([23, 24])]

# Combine the datasets
valdata_comb = pd.concat([val_data_23, valpoints_2324], ignore_index=True)
    
# Check for any duplicated geometries
duplicate_points = valdata_comb[valdata_comb['geometry'].duplicated()]

# Print the duplicate points
print(f"Number of duplicate geometries: {len(duplicate_points)}")
print(duplicate_points)


# %%
############################################################################


# CREATE MISSING POINTS SHAPEFILE / CSV


############################################################################
# Subset missing point counts from buffer version
missing = 60 - counts

# Take missing points from strata 23
strat23 = valpoints[valpoints['strata'] == 23][:missing[23]]

# Take missing points from strata 24
strat24 = valpoints[valpoints['strata'] == 24][:missing[24]]

# Combine missing points
missing_gdf = pd.concat([strat23, strat24], ignore_index=True)

# Convert csv geometry to WKT
missing_gdf['geometry'] = gpd.GeoSeries.from_wkt(missing_gdf['geometry'])

# Convert dataframe to geodataframe
missing_gdf = gpd.GeoDataFrame(missing_gdf, geometry='geometry', crs="EPSG:32629") 

# Write to shapefile
shp_write(missing_gdf, "validation_points_missing.shp", val_dir)

# Write to csv
write_csv(missing_gdf, val_dir, "validation_points_missing")

