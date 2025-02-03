# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:50:37 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import geometry_mask
from shapely.geometry import Point
from rasterio.mask import mask
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd



############################################################################


# SET UP DIRECTORY AND NODATA


############################################################################
# Define function to create new folders (if necessary)
def newfolder(folderlist, parfolder):
    
    # Iterate over each folder name
    for folder in folderlist:
        
        # Define folder path
        path = os.path.join(parfolder, folder)
        
        # Check if folder does not exist
        if not os.path.exists(path):
            
            # Create folder
            os.makedirs(path)
            
            # Print statement
            print(f"{path} created")
          
        # If folder already exists
        else:
            
            # Print statement
            print(f"{path} already exists")
            
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())

# Define output directory (validation data)
val_dir = os.path.join("data", "validation")

# Create new folders (if necessary)
newfolder(["stratification_maps", "validation_datasets"], val_dir)

# Set nodata value
nodata_val = 255

# Define output directory (stratification maps)
strat_dir = os.path.join("data", "validation", "stratification_maps")

# Redefine output directory (validation data)
val_dir = os.path.join("data", "validation", "validation_datasets")

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

# Read villages data
villages = gpd.read_file("data/village polygons/village_polygons.shp")

# Simplify villages dataframe into only REDD+ and non-REDD+ groups
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k').reset_index()

# Create redd+ polygon
redd_union = gpd.GeoSeries(villages.loc[1].geometry).unary_union

# Create nonredd+ polygon
nonredd_union = gpd.GeoSeries(villages.loc[0].geometry).unary_union


# %%
############################################################################


# RECLASSIFY AGREEMENT ARRAYS 


############################################################################
"""
Sampling strata are based on forest disagreement and agreement area for each 
year between 2013-2023. Therefore, the classes "only GFC deforestation" (pixel 
value 6) and "only TMF deforestation" (pixel value 7) are combined to create
a general "disagreement" class
"""
# Define function to check unique values in a given array
def valcheck(array, dataname):
    uniquevals = np.unique(array)
    print(f"Unique values in the {dataname} are {uniquevals}")

# Define function to reclassify list of arrays
def arr_reclass(arrlist, oldclass, newclass):
    
    # Create empty list to hold reclassified arrays
    reclassed = []
    
    # Iterate over each array
    for arr in arrlist:
        
        # Copy array
        mod_arr = arr.copy()
        
        # Replace old class with new class
        mod_arr[mod_arr == oldclass] = newclass
        
        # Append modified array to list
        reclassed.append(mod_arr)
        
    return reclassed

# Define function to check for overlaps in disagreement/agreement over time
def class_overlaps(arrlist):
    
    # Initialize counter for class overlaps
    conflicts = 0
    
    # Iterate over number of arrays
    for i in range(len(arrlist)):
        
        # Iterate over pairs of arrays
        for j in range(i+1, len(arrlist)):
            
            # Create conflict mask where pixel values differ between arrays
            conflict_mask = (arrlist[i] == 6) & (arrlist[j] == 8) | \
                (arrlist[i] == 8) & (arrlist[j] == 6)
                
            # Count number of conflicts in mask (True values)
            conflicts += np.sum(conflict_mask)
            
    # Print results
    print("Total pixels with deforestation disagreement and agreement at "
          f"different years but the same location: {conflicts}")

# Check values of agreement array
valcheck(agreement_arrs[0], "agreement 2013")

# Merge disagreement classes (7 and 6)
combdis_arrays = arr_reclass(agreement_arrs, 7, 6)

# Check unique values in combined disagreement array
valcheck(combdis_arrays[0], "combined disagreement array 2013")

# Check for temporal overlaps between deforestation classes
class_overlaps(combdis_arrays)


# %%
############################################################################


# CREATE STRATIFICATION ARRAY


############################################################################
"""
Agreement arrays per year are merged into a single stratification array using 
the reclassification key defined below. A total of 23, non-overlapping strata 
should cover the new array. 

Note: execution time for this segment is ~15min
"""
def create_reclass_key(class1, class2, num_dicts):
    
    # Create empty list to hold reclassification dictionaries
    reclass_key = []
    
    # Initiate starting reclass value
    reclass_value = 1
    
    # Iterate over the number of dictionaries needed
    for _ in range(num_dicts):
        
        # Input reclass values to dictionary
        reclass_key.append({class1: reclass_value, class2: reclass_value + 1})
        
        # Increment 2 for the next class pair dictionary
        reclass_value += 2
    
    return reclass_key

# Define function to create stratification layer
def strat_layer(arrlist, reclass_key):
    
    # Create array filled with no deforestation class (value = 23)
    strat_arr = np.full(combdis_arrays[0].shape, 23)
    
    # Iterate over each array
    for i in range(len(arrlist)):
        
        # Iterate over each row
        for j in range(arrlist[i].shape[0]):
            
            # Iterate over each column
            for k in range(arrlist[i].shape[1]):
                
                # Extract pixel value at array, row, column
                pixel_value = arrlist[i][j, k]
                
                # Reclassify disagreement areas (value 6 or 8)
                if pixel_value in (6, 8):
                    strat_arr[j, k] = reclass_key[i][pixel_value]
                
                # Keep nodata pixels as nodata
                elif pixel_value == nodata_val: 
                    strat_arr[j, k] = nodata_val
        
        # Print statement for status
        print(f"Array {i} incorporated into stratification layer")
                    
    return strat_arr

# Define function to remove polygon space from array
def rmv_polyspace(array, geometry, transform, nodataval):
    
    # Create mask to remove geometry area
    mask = geometry_mask(geometry, transform = transform, invert=False, 
                         out_shape = array.shape)
    
    # Remove geometry area from array
    arr_masked = np.where(mask, array, nodataval)
    
    return arr_masked

# Define function to write file to drive
def rast_write(array, filename, out_dir, profile):
    
    # Define output filepath
    output_filepath = os.path.join(out_dir, filename)
    
    # Save to file
    with rasterio.open(output_filepath, "w", **profile) as dst:
        dst.write(array, 1)

# Create reclassification key for pixel values per each year (2013 to 2023)
reclass_key = create_reclass_key(6, 8, len(years))

# Create stratification layer for sampling
strat_arr = strat_layer(combdis_arrays, reclass_key)

# Check values in stratified array
valcheck(strat_arr, "stratification array")

# Remove GRNP from sample space
nogrnp_stratarr = rmv_polyspace(strat_arr, grnp.geometry, transform, nodata_val)

# Write stratified layer to file
rast_write(nogrnp_stratarr, "stratification_layer_nogrnp.tif", strat_dir, profile)


# %%
############################################################################


# CREATE BUFFER STRATA


############################################################################
# Create a undisturbed forest mask (strata 23)
forest = (nogrnp_stratarr == 23)

# Define the structure for erosion (1 pixel width)
buffer_structure = np.ones((3, 3))

# Perform binary erosion to shrink clusters
buffer = ndimage.binary_erosion(forest, structure = buffer_structure)

# Copy stratified array to store results
buff_stratarr = np.copy(nogrnp_stratarr)

# Assign buffer forest to straat 23
buff_stratarr[forest & ~buffer] = 23

# Assign undisturbed forest to strata 24
buff_stratarr[buffer] = 24

# Write stratified layer to file
rast_write(buff_stratarr, "stratification_layer_buffered.tif", strat_dir, profile)


# %%
############################################################################


# EXPLORE STRATA SIZE


############################################################################
# Extract values and ocunts of each strata
values, counts = np.unique(buff_stratarr, return_counts = True)

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


# RANDOM SAMPLING PER STRATA (ALL STRATA)


############################################################################
"""
Because there are many more points in strata 23 and 24, more points will be 
sampled from these strata (total 780):
    Strata 1-22: 30 points (660)
    Strata 23-24: 60 points (120)
"""

# Read stratification array to avoid re-running code
with rasterio.open("data/intermediate/stratification_layer_buffered.tif") as rast:
    stratarr = rast.read(1)

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

# Create sample points with 30 points per strata 1-22
points_660 = strat_sample(stratarr, 30, transform, profile, max_strata = 22,
                         random_state = 33)

# Create sample points with 60 points per strata 23-24
points_120 = strat_sample(stratarr, 60, transform, profile, min_strata = 23,
                          random_state = 33)

# Combine points
points_buff = pd.concat([points_660, points_120], ignore_index = True)

# Copy buffered point dataset
points_nobuff = points_buff.copy()

# Reclassify strata 24 as strata 23 (merge strata)
points_nobuff.loc[points_nobuff["strata"] == 24, "strata"] = 23

# Write sample points to file
shp_write(points_nobuff, "validation_points_780.shp", val_dir)


# %%
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

# Extract raster values (buffered dataset)
valpoints_buff = extract_val(points_buff, tiflist, tifnames)

# Write points to file (buffered dataset)
write_csv(valpoints_buff, val_dir, "validation_points_780_buffer_nolabel")

# Extract raster values (no buffer dataset)
valpoints_nobuff = extract_val(points_nobuff, tiflist, tifnames)

# Write points to file (no buffer dataset)
write_csv(valpoints_nobuff, val_dir, "validation_points_780_nobuffer_nolabel")


# %%
############################################################################


# SPLIT STRATIFICATION MAP INTO REDD+ / NONREDD+ AREAS


############################################################################
# Define function to crop to geometry
def geom_crop(strat_path, geometry, lab):
    
    # Read stratification map
    with rasterio.open(strat_path) as rast:
        
        # Mask to geometry
        out_image, out_transform = mask(rast, [geometry], crop = True)
        
        # Extract metadata
        meta = rast.meta.copy()
        
        # Update metadata
        meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
            })
        
        # Define filename
        filename = f"stratification_layer_{lab}.tif"
        
        # Define output path
        output_path = os.path.join(strat_dir, filename)
        
        # Save clipped raster
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(out_image)
            
    return out_image

# Define function to calculate strata size
def incl_prob(stratmap, valdata):
    
    # Calculate number of pixels per strata
    pixvals, pixcounts = np.unique(stratmap, return_counts = True)

    # Create dataframe
    strata_size = pd.DataFrame({'strata': pixvals[:-1],
                                'size': pixcounts[:-1]})
    
    # Calculate number of samples per strata
    sampvals, sampcounts = np.unique(valdata['strata'], return_counts = True)
    
    # Create dataframe
    samples = pd.DataFrame({'strata': sampvals,
                            'samples': sampcounts})
    
    # Calculate inclusion probability
    prob = pd.DataFrame({'strata': sampvals,
                         'incl_prob': (samples['samples'] / strata_size['size'])
                         })
    
    return prob

# Define stratification map path (no buffer)
strat_nobuff = os.path.join(strat_dir, "stratification_layer_nogrnp.tif")

# Define stratification map path (with buffer)
strat_buff = os.path.join(strat_dir, "stratification_layer_buffered.tif")

# Clip to redd geometry
redd_strat = geom_crop(strat_nobuff, redd_union, "redd")

# Clip to nonredd geometry
nonredd_strat = geom_crop(strat_nobuff, nonredd_union, "nonredd")

# Clip to redd geometry
redd_strat_buff = geom_crop(strat_buff, redd_union, "redd_buff")

# Clip to nonredd geometry
nonredd_strat_buff = geom_crop(strat_buff, nonredd_union, "nonredd_buff")

# Filter points within REDD+ multipolygon
points_redd = valpoints_buff[valpoints_buff.geometry.within(redd_union)]

# Filter points within non-REDD+ multipolygon
points_nonredd = valpoints_buff[valpoints_buff.geometry.within(nonredd_union)]

# Calculate inclusion probability for redd
redd_prob = incl_prob(redd_strat, points_redd)

# Calculate inclusion probability for nonredd
nonredd_prob = incl_prob(nonredd_strat, points_nonredd)





