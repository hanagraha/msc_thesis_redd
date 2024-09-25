# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:58:43 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import pandas as pd
import numpy as np
import glob



############################################################################


# SET UP DIRECTORY


############################################################################
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())

# Get file paths
gfc_folder = "data/hansen_raw"
tmf_folder = "data/jrc_raw"

# Get list of all tif files in the folder
gfc_files = [f"{gfc_folder}/{file}" for file in os.listdir(gfc_folder) if file.endswith('.tif')]
tmf_files = [f"{tmf_folder}/{file}" for file in os.listdir(tmf_folder) if file.endswith('.tif')]

# Print the file paths to check
print(gfc_files)
print(tmf_files)



############################################################################


# IMPORT DEFORESTATION DATASETS (LOCAL DRIVE)


############################################################################

### READ DATA
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")
villages = gpd.read_file("data/village polygons/VillagePolygons.geojson")

### CHECK PROJECTIONS 
# Get epsg codes
villages_epsg = villages.crs.to_epsg()
grnp_epsg = grnp.crs.to_epsg()

# Check if epsgs match:
if villages_epsg == grnp_epsg:
    print(f"Both GeoDataFrames have the same EPSG: {villages_epsg}")
else:
    print(f"Different EPSG codes: Villages has {villages_epsg}, GRNP has {grnp_epsg}")
    # Reproject if necessary
    grnp = grnp.to_crs(epsg=villages_epsg)
    print(f"GRNP has been reprojected to EPSG: {villages_epsg}")

# Create string of EPSG code for raster reprojection
epsg_string = f"EPSG:{villages_epsg}"

### CREATE AOI
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()
aoi_geom = aoi.geometry



############################################################################


# REPROJECT RASTERS


############################################################################

### DEFINE FUNCTION TO REPROJECT RASTERS
def reproject_raster(raster_path, epsg, output_folder):
    # open raster
    with rasterio.open(raster_path) as rast:
        transform, width, height = calculate_default_transform(
            rast.crs, epsg, rast.width, rast.height, *rast.bounds)
        
        kwargs = rast.meta.copy()
        kwargs.update({'crs': epsg,
                       'transform': transform,
                       'width': width,
                       'height': height})
        
        output_path = raster_path.replace(".tif", "_AOI.tif")
        folder_name = os.path.basename(os.path.dirname(output_path))
        output_path = output_path.replace(folder_name, output_folder)

        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, rast.count + 1):
                reproject(
                    source=rasterio.band(rast, i),
                    destination=rasterio.band(dst, i),
                    rast_transform=rast.transform,
                    rast_crs=rast.crs,
                    dst_transform=transform,
                    dst_crs=epsg,
                    resampling=Resampling.nearest)
        
        print(f"Reprojection Complete for {output_path}")
        
        return output_path
    

gfc_processed_folder = "hansen_preprocessed"
tmf_processed_folder = "jrc_preprocessed"


# Reproject GFC images
gfc_reproj_files = []

for tif in gfc_files:
    tif_reproj_file = reproject_raster(tif, epsg_string, gfc_processed_folder)
    gfc_reproj_files.append(tif_reproj_file)
    
    
# Reproject TMF images
tmf_reproj_files = []

for tif in tmf_files:
    tif_reproj_file = reproject_raster(tif, epsg_string, tmf_processed_folder)
    tmf_reproj_files.append(tif_reproj_file)
    
    
    
############################################################################


# CLIP RASTERS


############################################################################

### DEFINE FUNCTION TO CLIP RASTERS
def clip_raster(raster_path, aoi_geom, nodata_val):
    
    # open raster
    with rasterio.open(raster_path) as rast:
        
        # Clip raster
        raster_clip, out_transform = mask(rast, aoi_geom, crop=True, nodata=nodata_val)
        
        out_meta = rast.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'count': 1,
            'height': raster_clip.shape[1],
            'width': raster_clip.shape[2],
            'transform': out_transform,
            'nodata': nodata_val})
        
    raster_clip = raster_clip.astype('float32')
    # raster_clip[raster_clip == nodata_val] = np.nan
    
    # outpath = raster_path.replace("_reprojected.tif", "_clipped.tif")
    
    # if os.path.exists(outpath): 
    #     os.remove(outpath)
    
    with rasterio.open(raster_path, 'w', **out_meta) as dest:
        dest.write(raster_clip)
    
    print(f"Clipping Complete for {raster_path}")
    
    return raster_clip, out_meta  
    

# Clip GFC rasters
for file in gfc_reproj_files:
    clip_raster(file, aoi_geom, 255)
    
# Clip TMF rasters
for file in tmf_reproj_files:
    clip_raster(file, aoi_geom, 255)
    
    
    
############################################################################


# RECLASSIFICATION


############################################################################
"""
GFC lossyear will be reclassified to match the year format of TMF degradation year
"""

# Re-do the clip function to get metadata
gfc_lossyear, gfc_lossyear_meta = clip_raster(gfc_reproj_files[0], aoi_geom, 255)

# Get path to save the reclassified file
# gfc_lossyear_path = gfc_reproj_files[0].replace("_reprojected.tif", "_reclassified.tif")

# Add 2000 to non-NA years
gfc_lossyear_new = np.where(gfc_lossyear != 255, gfc_lossyear + 2000, gfc_lossyear)

# Make sure data type can hold 2000 values
gfc_lossyear_meta.update(dtype='int16')

# Write file to drive
with rasterio.open(gfc_reproj_files[0], 'w', **gfc_lossyear_meta) as dest:
    dest.write(gfc_lossyear_new)
print(f"Reclassification Complete for {gfc_reproj_files[0]}")
    
    
    
############################################################################


# RESAMPLING


############################################################################    
"""
GFC has shape (3030, 3654) and TMF has shape (2812, 3389). TMF will be 
resampled to match the shape of GFC to retain spatial accuracy
"""    

### COMPARE GFC and TMF SHAPES
# Read GFC
gfc_lossyear = "data/hansen_preprocessed/gfc_lossyear_AOI.tif"
with rasterio.open(gfc_lossyear) as gfc:
    gfc_lossyear = gfc.read(1)
    gfc_lossyear_profile = gfc.profile
    gfc_lossyear_shape = gfc.shape
    gfc_lossyear_transform = gfc.transform

# Read TMF
tmf_deforyear = "data/jrc_preprocessed/tmf_DeforestationYear_AOI.tif"
with rasterio.open(tmf_deforyear) as tmf:
    tmf_deforyear = tmf.read(1)
    tmf_deforyear_profile = tmf.profile
    tmf_deforyear_shape = tmf.shape
    tmf_deforyear_transform = tmf.transform

# Compare shapes
if gfc_lossyear_shape == tmf_deforyear_shape:
    print("Both rasters have the same shape.")
    print("Shape:", gfc_lossyear_shape)
else:
    print("The shapes of the two rasters are different.")
    print("Shape of GFC Lossyear:", gfc_lossyear_shape)
    print("Shape of TMF Deforestation Year:", tmf_deforyear_shape)


### DOUBLE CHECK THAT GFC FILES ALL HAVE THE SAME SHAPE
# Read GFC treecover
gfc_treecover = "data/hansen_preprocessed/gfc_treecover2000_AOI.tif"
with rasterio.open(gfc_treecover) as gfc:
    gfc_treecover = gfc.read(1)
    gfc_treecover_profile = gfc.profile
    gfc_treecover_shape = gfc.shape
    gfc_treecover_transform = gfc.transform

# Compare shapes
if gfc_treecover_shape == gfc_lossyear_shape:
    print("Both rasters have the same shape.")
    print("Shape:", gfc_treecover_shape)
else:
    print("The shapes of the two rasters are different.")
    print("Shape of GFC Treecover:", gfc_treecover_shape)
    print("Shape of GFC Lossyear:", gfc_lossyear_shape)


### RESAMPLE TMF TO GFC
tmf_folder = "data/jrc_preprocessed"
tmf_files = glob.glob(os.path.join(tmf_folder, "*AOI.tif*"))

# Make sure the list only has tif files
tmf_files = ([file for file in tmf_files if file.endswith('.tif') and not 
              file.endswith(('.xml', '.ovr'))])

# Resample all tmf tif files
for raster_path in tmf_files:
    with rasterio.open(raster_path) as src:
        source_data = src.read(1)  # Read the first band
        source_profile = src.profile
        source_transform = src.transform

    # Create an output array for the resampled data
    resampled_data = np.zeros(gfc_treecover_shape, dtype=source_data.dtype)

    # Perform the resampling
    rasterio.warp.reproject(
        source=source_data,
        destination=resampled_data,
        src_transform=source_transform,
        dst_transform=gfc_treecover_transform,
        src_crs=source_profile['crs'],
        dst_crs=gfc_treecover_profile['crs'],
        resampling=Resampling.nearest
    )

    # Update the profile for the resampled raster
    gfc_treecover_profile.update({
        'height': gfc_treecover_shape[0],
        'width': gfc_treecover_shape[1],
        'transform': gfc_treecover_transform,
        'nodata': source_profile['nodata']  # Keep the same NoData value
    })

    # Save the resampled raster
    output_file = raster_path.replace(".tif", "_resampled.tif")
    with rasterio.open(output_file, 'w', **gfc_treecover_profile) as dst:
        dst.write(resampled_data, 1)

    print(f"Resampled raster saved as {output_file}")
    
    
    
############################################################################


# FOREST MASKS


############################################################################    

### READ RELEVANT FILES
gfc_treecover = "data/hansen_preprocessed/gfc_treecover2000_AOI.tif"
with rasterio.open(gfc_treecover) as gfc:
    gfc_treecover = gfc.read(1)
    gfc_treecover_profile = gfc.profile

tmf_trans_sub = "data/jrc_preprocessed/tmf_TransitionMap_Subtypes_AOI_resampled.tif"
with rasterio.open(tmf_trans_sub) as tmf:
    tmf_trans_sub = tmf.read(1)
    tmf_trans_sub_profile = tmf.profile


### CREATE TMF BASELINE FOREST 
"""
"The initial tropical moist forest domain can be derived from the Transition 
Map - Sub types map by selecting all pixels belonging to classes 10 to 89 
excluding classes 71 and 72."
- taken from https://forobs.jrc.ec.europa.eu/TMF/resources#how_to
"""

# Create baseline forest mask
tmf_baseforest_mask = ((tmf_trans_sub >= 10) & (tmf_trans_sub <= 89) & 
                       (tmf_trans_sub != 71) & (tmf_trans_sub != 72))

# Mask TMF forest
tmf_baseforest = np.where(tmf_baseforest_mask, tmf_trans_sub, 255)

# Update data type and nodata value for saving
tmf_trans_sub_profile.update(dtype=rasterio.float32, nodata=255)  

# Save to drive
tmf_baseforest_file = 'data/jrc_preprocessed/tmf_baselineforest.tif'

with rasterio.open(tmf_baseforest_file, 'w', **tmf_trans_sub_profile) as dst:
    dst.write(tmf_baseforest.astype(rasterio.float32), 1)

print(f"TMF baseline forest saved as {tmf_baseforest_file}")


### CREATE GFC BASELINE FOREST 
"""
Forest mask will use 50% forest cover threshold from Malan et al. (2024)
"""

# Create baseline forest mask
gfc_baseforest_mask = (gfc_treecover >= 50)

# Mask GFC forest
gfc_baseforest = np.where(gfc_baseforest_mask, gfc_treecover, 255)

# Update data type and nodata value for saving
gfc_treecover_profile.update(dtype=rasterio.float32, nodata=255)  

# Save to drive
gfc_baseforest_file = 'data/hansen_preprocessed/gfc_baselineforest.tif'

with rasterio.open(gfc_baseforest_file, 'w', **gfc_treecover_profile) as dst:
    dst.write(gfc_baseforest.astype(rasterio.float32), 1)

print(f"GFC baseline forest saved as {gfc_baseforest_file}")
    
    
### FOREST MASK SPATIAL AGREEMENT
nodata_val = 255

# Convert forest masks to binary layers
binary_gfc_baseforest = np.where(gfc_baseforest == nodata_val, 0, 1)  # 1 where data exists, 0 where NoData
binary_tmf_baseforest = np.where(tmf_baseforest == nodata_val, 0, 2)  # 2 where data exists, 0 where NoData

# Add binary layers to get spatial agreement map
agreement_map = binary_gfc_baseforest + binary_tmf_baseforest

agreement_profile = gfc_treecover_profile
agreement_profile.update(dtype=rasterio.uint8, nodata=0)

# Save the result as a new raster file
agreement_path = "data/intermediate/forestmask_spatial_agreement.tif"

with rasterio.open(agreement_path, 'w', **agreement_profile) as dst:
    dst.write(agreement_map.astype(rasterio.uint8), 1)

print(f"Spatial agreement map saved as {agreement_path}")
    

### UPDATE GFC BASELINE FOREST
"""
Upon review of the spatial agreement map, the GFC baseline forest is best
suited for this thesis. The mask must be updated to represent forest in 2012
"""

# Read lossyear and baseline forest files
gfc_lossyear = "data/hansen_preprocessed/gfc_lossyear_AOI.tif"
with rasterio.open(gfc_lossyear) as gfc:
    gfc_lossyear = gfc.read(1)
    gfc_lossyear_meta = gfc.meta

gfc_baseforest = "data/hansen_preprocessed/gfc_baselineforest.tif"
with rasterio.open(gfc_baseforest) as gfc:
    gfc_baseforest = gfc.read(1)
    gfc_baseforest_meta = gfc.meta

# Create mask that excludes pixels that were deforested between 2001-2012
mask2012 = (gfc_lossyear < 2001) | (gfc_lossyear > 2012)

# Apply mask, setting pixels that were deforested before 2012 as nodata
gfc_baseforest2012 = np.where(mask2012, gfc_baseforest, gfc_baseforest_meta['nodata'])

# Save new baseline forest
gfc_baseforest2012_meta = gfc_baseforest_meta.copy()
gfc_baseforest2012_file = "data/hansen_preprocessed/gfc_baselineforest_2012.tif"

with rasterio.open(gfc_baseforest2012_file, 'w', **gfc_baseforest2012_meta) as dst:
    dst.write(gfc_baseforest2012, 1)
    
    
### APPLY GFC BASELINE FOREST

# Read GFC files
gfc_folder = "data/hansen_preprocessed"
gfc_files = glob.glob(os.path.join(gfc_folder, "*AOI*"))
gfc_files = ([file for file in gfc_files if file.endswith('.tif') and not 
              file.endswith(('.xml', '.ovr'))])

# Read TMF files
tmf_folder = "data/jrc_preprocessed"
tmf_files = glob.glob(os.path.join(tmf_folder, "*_AOI*"))
tmf_files = ([file for file in tmf_files if file.endswith('_resampled.tif') and not 
              file.endswith(('.xml', '.ovr'))])

# Mask GFC files
for file in gfc_files:
    
    with rasterio.open(file) as src:
        
        file_data = src.read(1)
        file_meta = src.meta

        # Apply mask
        masked_data = np.where(gfc_baseforest2012 != gfc_baseforest2012_meta['nodata'], 
                               file_data, file_meta['nodata'])

        # Create output file name
        output_file = file.replace('_AOI.tif', '_fm.tif')

        # Save the masked output
        with rasterio.open(output_file, 'w', **file_meta) as dst:
            dst.write(masked_data, 1)

        print(f"Masked file saved as: {output_file}")

# Mask TMF files
for file in tmf_files:
    
    with rasterio.open(file) as src:
        
        file_data = src.read(1)
        file_meta = src.meta

        # Apply mask
        masked_data = np.where(gfc_baseforest2012 != gfc_baseforest2012_meta['nodata'], 
                               file_data, file_meta['nodata'])

        # Create output file name
        output_file = file.replace('_AOI_resampled.tif', '_fm.tif')

        # Save the masked output
        with rasterio.open(output_file, 'w', **file_meta) as dst:
            dst.write(masked_data, 1)

        print(f"Masked file saved as: {output_file}")












    