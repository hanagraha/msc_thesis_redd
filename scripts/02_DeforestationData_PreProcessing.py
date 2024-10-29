# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:58:43 2024

@author: hanna

This file conducts the following pre-processing steps: reprojection, clipping, 
reclassifying, resampling, masking, separating, and combining. The inputs for
this file takes outputs from the 01_DataDownload.py file: 2 GFC rasters and 15
TMF rasters. It also uses 1 shapefile and 1 geojson that are found in the 
GIT repository.

Expected execution time: ~6min

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
import shutil



############################################################################


# SET UP DIRECTORY AND NODATA


############################################################################
# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory (ADAPT THIS!!!)
os.chdir("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())

# Define gfc input folder
gfc_folder = "data/hansen_raw"

# Define tmf input folder
tmf_folder = "data/jrc_raw"

# Define gfc output folder
gfc_processed_folder = "data/hansen_preprocessed"

# Define tmf output folder
tmf_processed_folder = "data/jrc_preprocessed"

# Define temporary folder
temp_folder = "data/temp"

# If temporary folder doesn't exist
if not os.path.exists(temp_folder):
    
    # Create temporary folder
    os.makedirs(temp_folder)
    
    print(f"{temp_folder} created.")

# If temporary folder does exist
else:
    print(f"{temp_folder}' already exists.")

# Set nodata value
nodata_val = 255

# Set year range
years = range(2013, 2024)



############################################################################


# IMPORT AND READ DATA


############################################################################
# Define file paths for raw gfc data
gfc_files = [f"{gfc_folder}/{file}" for file in os.listdir(gfc_folder) if 
             file.endswith('.tif')]

# Define file paths for raw tmf data
tmf_files = [f"{tmf_folder}/{file}" for file in os.listdir(tmf_folder) if 
             file.endswith('.tif')]

# Read grnp shapefile
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Read village shapefile
villages = gpd.read_file("data/village polygons/VillagePolygons.geojson")

# Combine grnp and villages to create aoi
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()

# Extract aoi geometry
aoi_geom = aoi.geometry



############################################################################


# REPROJECT RASTERS TO EPSG 32629


############################################################################
"""
This segment will reproject all raster data to match crs of the villages and
GRNP crs (EPSG 32629). This EPSG is also most accurate for Sierra Leone. 

Expected runtime ~6min 
"""
# Define function to check projection of geodataframes
def gdf_epsgcheck(gdf1, gdf2):
    
    # Extract first epsg
    epsg1 = gdf1.crs.to_epsg()
    
    # Extract second epsg
    epsg2 = gdf2.crs.to_epsg()
    
    # If epsgs match
    if epsg1 == epsg2:
        print(f"Both GeoDataFrames have the same EPSG: {epsg1}")
    
    # If epsgs don't match
    else:
        print(f"Different EPSG codes: Villages has {epsg1}, GRNP has {epsg2}")
        
        # Reproject if necessary
        gdf2 = gdf2.to_crs(epsg=epsg1)
        print(f"GRNP has been reprojected to EPSG: {epsg1}")
        
    # Create epsg string
    epsg_string = f"EPSG:{epsg1}"
        
    return epsg_string

# Define function to reproject rasters
def reproject_raster(raster_pathlist, epsg, output_folder):
    
    # Create empty list to hold reprojected raster filepaths
    reproj_files = []
    
    # Iterate over each raster
    for path in raster_pathlist:
    
        # Read raster
        with rasterio.open(path) as rast:
            
            # Calculate transform metrics
            transform, width, height = calculate_default_transform(
                rast.crs, epsg, rast.width, rast.height, *rast.bounds)
            
            # Copy metadata
            kwargs = rast.meta.copy()
            
            # Update metadata with transform metrics
            kwargs.update({'crs': epsg,
                           'transform': transform,
                           'width': width,
                           'height': height})
            
            # Define new filepath
            output_path = os.path.join(output_folder, os.path.basename(path) 
                                       .replace(".tif", "_reprojected.tif"))
    
            # Write raster to file
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
            
        # Print statement
        print(f"Reprojection Complete for {output_path}")
    
        # Append filepath to list
        reproj_files.append(output_path)
            
    return reproj_files
    
# Check village and grnp projections
epsg_string = gdf_epsgcheck(villages, grnp)

# Reproject gfc rasters
gfc_reproj_files = reproject_raster(gfc_files, epsg_string, temp_folder)

# Reproject tmf rasters
tmf_reproj_files = reproject_raster(tmf_files, epsg_string, temp_folder)
    
    
    
############################################################################


# CLIP RASTERS TO AOI


############################################################################
"""
This segment will override the previously created files with updated versions
that are clipped to the AOI (combination of villages and GRNP area)

Expected runtime: <1min
"""

# Define function to clip rasters
def clip_raster(raster_pathlist, aoi_geom, nodata_value, output_folder):
    
    # Create empty list to hold clipped raster filepaths
    clipped_files = []
    
    # Create empty list to hold clipped rasters
    clipped_rasts = []
    
    # Create empty list to hold metadata
    clipped_meta = []
    
    # Iterate over each raster
    for path in raster_pathlist:
    
        # Read raster
        with rasterio.open(path) as rast:
            
            # Clip raster
            raster_clip, out_transform = mask(rast, aoi_geom, crop=True, 
                                              nodata=nodata_value)
            
            # Copy metadata
            out_meta = rast.meta.copy()
            
            # Update metadata
            out_meta.update({
                'driver': 'GTiff',
                'dtype': 'int16', # to incorporate values over 255
                'count': 1,
                'height': raster_clip.shape[1],
                'width': raster_clip.shape[2],
                'transform': out_transform,
                'nodata': nodata_value})
            
            # Define new filepath
            output_path = os.path.join(output_folder, os.path.basename(path) 
                                       .replace(".tif", "_clipped.tif"))
            
            # Append output path to list
            clipped_files.append(output_path)
            
            # Append clipped raster to list
            clipped_rasts.append(raster_clip)
            
            # Append clipped metadata to list
            clipped_meta.append(out_meta)
        
            # Write file to hardrive
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(raster_clip)
        
        # Print statement
        print(f"Clipping Complete for {output_path}")
        
    return clipped_files, clipped_rasts, clipped_meta

# Clip gfc rasters
gfc_clipped_files, gfc_clipped_rasts, gfc_clipped_meta = clip_raster(
    gfc_reproj_files, aoi_geom, nodata_val, temp_folder)

# Clip tmf rasters
tmf_clipped_files, tmf_clipped_rasts, tmf_clipped_meta = clip_raster(
    tmf_reproj_files, aoi_geom, nodata_val, temp_folder)
    
    
    
############################################################################


# RECLASSIFY GFC LOSSYEAR TO MATCH YYYY FORMAT


############################################################################
"""
GFC lossyear will be reclassified to match the format of TMF degradation year
such that the years will be in the format 2013-2023 (instead of 13-23) and 
undisturbed forests will have value 0 (instead of 2000)

Expected runtime: <1min
"""
# Define gfc lossyear raster
gfc_lossyear = gfc_clipped_rasts[0]

# Add 2000 to all non-NA years
gfc_lossyear_new = np.where(gfc_lossyear != nodata_val, gfc_lossyear + 2000, 
                            gfc_lossyear)

# Reclassify undisturbed forest from 2000 to 0
gfc_lossyear_new = np.where(gfc_lossyear_new == 2000, 0, gfc_lossyear_new)

# Write file to hardrive (with the same name as input file, overriding)
with rasterio.open(gfc_clipped_files[0], 'w', **gfc_clipped_meta[0]) as dest:
    dest.write(gfc_lossyear_new)
    
print(f"Reclassification Complete for {gfc_clipped_files[0]}")
    
    
    
############################################################################


# RESAMPLE TMF TO GFC PIXEL SIZES


############################################################################    
"""
GFC has shape (3030, 3654) and TMF has shape (2812, 3389). TMF will be 
resampled to match the shape of GFC to retain spatial accuracy. 

Expected runtime: <1min
"""    
# Define function to compare shapes of two rasters
def rast_shpcomp(rastpath1, rastpath2, rastname1, rastname2):
    
    # Read first raster
    with rasterio.open(rastpath1) as gfc:
        rast1_shape = gfc.shape
        
    # Read second raster
    with rasterio.open(rastpath2) as tmf:
        rast2_shape = tmf.shape
        
    # Compare shapes
    if rast1_shape == rast2_shape:
        print("Both rasters have the same shape.")
        print("Shape:", rast1_shape)
    else:
        print("The shapes of the two rasters are different.")
        print(f"Shape of {rastname1}:", rast1_shape)
        print(f"Shape of {rastname2}:", rast2_shape)
    
# Define function to adjust pixel sizes of raster list
def resamp_pix(resamp_pathlist, ref_rastpath):
    
    # Read reference raster
    with rasterio.open(ref_rastpath) as ref:
        
        # Extract reference raster shape
        ref_shape = ref.shape
        
        # Extract reference transform
        ref_transform = ref.transform
        
        # Extract reference profile
        ref_profile = ref.profile
        
    # Create empty list to hold resampled raster paths
    resamp_files = []
    
    # Create empty list tohold resampled rasters
    resamp_rasts = []
    
    # Iterate over each raster to resample
    for path in resamp_pathlist:
        
        # Read raster data
        with rasterio.open(path) as src:
            
            # Extract source data
            source_data = src.read(1)
            
            # Extract source profile
            source_profile = src.profile
            
            # Extract source transform
            source_transform = src.transform
            
        # Copy dimensions of reference raster into empty array
        resamp_data = np.zeros(ref_shape, dtype=source_data.dtype)
        
        # Resample raster data
        rasterio.warp.reproject(
            
            # Define source and destination arrays
            source=source_data,
            destination=resamp_data,
            
            # Define source and destination transforms
            src_transform=source_transform,
            dst_transform=ref_transform,
            
            # Define source and destination profiles
            src_crs=source_profile['crs'],
            dst_crs=ref_profile['crs'],
            
            # Resample with nearest neighbor
            resampling=Resampling.nearest
        )
        
        # Define output filename
        output_file = path.replace(".tif", "_resampled.tif")
        
        # Write resmpled raster to drive
        with rasterio.open(output_file, 'w', **ref_profile) as dst:
            dst.write(resamp_data, 1)
        
        # Print statement
        print(f"Resampled raster saved as {output_file}")
        
        # Add filepath to list
        resamp_files.append(output_file)
        
        # Add rasters to list
        resamp_rasts.append(resamp_data)
    
    return resamp_files, resamp_rasts

# Compare gfc lossyear and tmf deforestation year
rast_shpcomp(gfc_clipped_files[0], tmf_clipped_files[11], "GFC Lossyear", 
             "TMF Deforestation Year")

# Compare gfc lossyear and gfc treecover2000
rast_shpcomp(gfc_clipped_files[0], gfc_clipped_files[1], "GFC Lossyear", 
             "GFC Treecover2000")

# Resample tmf files to match gfc pixel sizes
tmf_resamp_files, tmf_resamp_rasts = resamp_pix(tmf_clipped_files, 
                                                gfc_clipped_files[1])

# Compare gfc lossyear and resampled tmf deforestation year
rast_shpcomp(gfc_clipped_files[0], tmf_resamp_files[11], "GFC Lossyear", 
             "Resampled TMF Deforestation Year")

    
    
############################################################################


# CREATE TMF AND GFC BASELINE FORESTS


############################################################################
"""
TMF baseline forest definition:
"The initial tropical moist forest domain can be derived from the Transition 
Map - Sub types map by selecting all pixels belonging to classes 10 to 89 
excluding classes 71 and 72."
- taken from https://forobs.jrc.ec.europa.eu/TMF/resources#how_to

GFC baseline forest definition:
GFC forest mask will use 50% forest cover threshold from Malan et al. (2024)

Expected runtime: <1min
"""
# Define function to create forest masks
def custom_forbase(rastpath, mask, nodata_val, outdir, outfile):
    
    # Read raster
    with rasterio.open(rastpath) as rast:
        data = rast.read(1)
        profile = rast.profile
        
    # Mask raster
    data_masked = np.where(mask, data, nodata_val)
    
    # Define output filepath
    outpath = os.path.join(outdir, outfile)
    
    # Write baseline fores to drive
    with rasterio.open(outpath, 'w', **profile) as dst:
        dst.write(data_masked, 1)

    print(f"Custom baseline forest saved as {outpath}")
    
    return data_masked, profile

# Define tmf transition map subtypes raster
tmf_trans = tmf_resamp_rasts[14]

# Define tmf forest mask
tmf_mask = ((tmf_trans >= 10) & (tmf_trans <= 89) & 
            (tmf_trans != 71) & (tmf_trans != 72))

# Create tmf baseline forest
tmf_baseforest, tmf_profile = custom_forbase(tmf_resamp_files[14], tmf_mask, 
    nodata_val, temp_folder, "tmf_baselineforest.tif")

# Define gfc treecover2000 raster
gfc_treecover = np.squeeze(gfc_clipped_rasts[1], axis=0)

# Define gfc forest mask
gfc_mask = (gfc_treecover >= 50)

# Create gfc baseline forest
gfc_baseforest, gfc_profile = custom_forbase(gfc_clipped_files[1], gfc_mask, 
    nodata_val, temp_folder, "gfc_baselineforest.tif")


    
############################################################################


# CALCULATE FOREST MASK SPATIAL AGREEMENT


############################################################################ 
# Define function to create attribute table from array values
def att_table(array):
    
    # Define unique values and counts
    unique_values, pixel_counts = np.unique(array, return_counts=True)
    
    # Create a DataFrame with unique values and pixel counts
    attributes = pd.DataFrame({"Class": unique_values, 
                               "Frequency": pixel_counts})
        
    # Switch rows and columns of dataframe
    attributes = attributes.transpose()
    
    print(attributes)
    
    return attributes
  
# Convert gfc forest to binary (1 for forest, 0 for nodata)
gfc_bin_bf = np.where(gfc_baseforest == nodata_val, 0, 1)

# Convert tmf forest to binary (2 for forest, 0 for nodata)
tmf_bin_bf = np.where(tmf_baseforest == nodata_val, 0, 2)

# Add binary layers to create spatial agreement map
bf_agreement = gfc_bin_bf + tmf_bin_bf

# Explore agreement attributes
agreement_atts = att_table(bf_agreement)

# Copy gfc profile 
agreement_profile = gfc_profile.copy()

# Update agreement profile
agreement_profile.update(dtype=rasterio.uint8, nodata=0)

# Define output filepath
agreement_path = "data/temp/forestmask_spatial_agreement.tif"

# Write baseline forest agreement to file
with rasterio.open(agreement_path, 'w', **agreement_profile) as dst:
    dst.write(bf_agreement.astype(rasterio.uint8), 1)

print(f"Spatial agreement map saved as {agreement_path}")
    
"""
Upon review of the spatial agreement map, the GFC baseline forest is best
suited for this thesis. The mask must be updated to represent forest in 2012

The attribute table indicates that approximately 8% of the pixels within the 
AOI are only covered in GFC data and 2% are only covered by TMF data. After 
taking the GFC baseline forest as the overall forest mask, there will be 
nodata values in the TMF dataset. 

An additional mask will be created to analyze pixels only where both datasets
have data
"""



############################################################################


# ADJUST GFC BASELINE FOREST FOR DEFORESTATION OVER 2000-2012


############################################################################
# Define gfc lossyear raster
gfc_lossyear = np.squeeze(gfc_clipped_rasts[0], axis=0)

# Define gfc baseline forest filepath
gfc_baseforest_path = "data/temp/gfc_baselineforest.tif"

# Create mask for pixels that were NOT deforested between 2001-2012
gfc_mask2012 = (gfc_lossyear < 2001) | (gfc_lossyear > 2012)

# Create gfc baseline forest 2012
gfc_baseforest2012, gfc_profile2012 = custom_forbase(gfc_baseforest_path, 
    gfc_mask2012, nodata_val, temp_folder, "gfc_baselineforest2012.tif")

# Create forest mask where both datasets have data
tmf_gfc_mask = (gfc_baseforest2012 != nodata_val) & \
                (tmf_baseforest != nodata_val)

# Create baseline forest where both datasets have data
gfc_tmf_baseforest2012, gfc_tmf_profile2012 = custom_forbase(gfc_baseforest_path, 
    tmf_gfc_mask, nodata_val, temp_folder, "gfc_tmf_baselineforest2012.tif")
    
    

############################################################################


# APPLY GFC 2012 FOREST MASK


############################################################################ 
# Define function to apply forest mask to raster data
def formask_rasts(rastpathlist, baselineforest, nodata_val, outdir, 
                  orig_rastpathlist, suffix):
    
    # Create empty list to store filepaths
    fm_files = []
    
    # Iterate over files
    for path, name in zip(rastpathlist, orig_rastpathlist):
        
        # Read raster
        with rasterio.open(path) as rast:
            
            # Get raster data
            file_data = rast.read(1)
            
            # Get metadata
            file_meta = rast.meta
            
        # Apply forestmask
        masked_data = np.where(baselineforest != nodata_val, file_data,
                               nodata_val)
        
        # Define output filepath
        output_path = os.path.join(outdir, os.path.basename(name) 
                                   .replace(".tif", f"_{suffix}.tif"))
        
        # Write masked file to drive
        with rasterio.open(output_path, 'w', **file_meta) as dst:
            dst.write(masked_data, 1)
            
        # Append path to list
        fm_files.append(output_path)
            
        # Print statement
        print(f"Masked file saved as: {output_path}")
        
    return fm_files

# Mask gfc files to gfc baseline 2012
gfc_fm_files = formask_rasts(gfc_clipped_files, gfc_baseforest2012, nodata_val, 
                             gfc_processed_folder, gfc_files, "fm")

# Mask tmf files to gfc baseline 2012
tmf_fm_files = formask_rasts(tmf_resamp_files, gfc_baseforest2012, nodata_val, 
                             tmf_processed_folder, tmf_files, "fm")



############################################################################


# SPLIT MULTI-YEAR LAYERS INTO SINGLE-YEAR


############################################################################
"""
For RQ1a and RQ2a, single-year layers are needed to process deforestation
information. This segment converst GFC lossyear, TMF deforestation year, and 
TMF degradation year (all layers that represent multiple years) into individual
layers for each year (11 years per layer, 33 new files total)

Expected runtime: <1min
"""
# Define function to split multi-year rasters into single years
def annsplit_rast(rastpath, yearrange, outdir):
    
    # Create empty list to hold output paths
    ann_files = []
    
    # Read raster
    with rasterio.open(rastpath) as rast:
        
        # Extract data
        raster_data = rast.read(1)
        
        # Extract profile
        profile = rast.profile
        
    # Iterate over each year
    for year in years:
        
        # Create mask for pixels with year value
        year_mask = np.where(raster_data == year, raster_data, nodata_val)
        
        # Define output filename
        output_path = os.path.join(outdir, os.path.basename(rastpath) 
                                   .replace(".tif", f"_{year}.tif"))
        
        # Write file to drive
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(year_mask.astype(rasterio.float32), 1)
        
        # Add output path to list
        ann_files.append(output_path)
        
        # Print statement
        print(f"Single-year file created for {output_path}")
    
    return ann_files

# Split gfc lossyear
gfc_ann_lossyear = annsplit_rast(gfc_fm_files[0], years, gfc_processed_folder)

# Split tmf deforestation year
tmf_ann_deforyear = annsplit_rast(tmf_fm_files[11], years, tmf_processed_folder)

# Split tmf degradation year
tmf_ann_degrayear = annsplit_rast(tmf_fm_files[12], years, tmf_processed_folder)
        


############################################################################


# COMBINE TMF DEFORESTATION AND DEGRADATION YEAR


############################################################################
"""
For all RQs, a combination of TMF Deforestation and Degradation Year is needed.
This segment combines multi-year tmf deforestation and degradation data.

Expected runtime: <1min
"""
# Define function to combine two rasters with same structure
def rast_comb(rastpath1, rastpath2, yearrange, nodata_val, outdir, filename):
    
    # Read both rasters
    with rasterio.open(rastpath1) as src1, rasterio.open(rastpath2) as src2:
        
        # Get data from both rasters
        rast1 = src1.read(1)
        rast2 = src2.read(1)
        
        # Get profile of one raster
        profile = src1.profile
    
    # Only use data within yearrange
    rast1 = np.where((rast1 < min(yearrange)) | (rast1 > max(yearrange)), 
                     nodata_val, rast1)
    rast2 = np.where((rast2 < min(yearrange)) | (rast2 > max(yearrange)), 
                     nodata_val, rast2)
    
    # Create nodata masks for each raster
    rast1nodata = rast1 == nodata_val
    rast2nodata = rast2 == nodata_val
    
    # Combine data with conditions
    combined_data = np.where(
        
        # For pixels where both datasets have data
        ~rast1nodata & ~rast2nodata, 
        
        # Take minimum of the two rasters (earlier year)
        np.minimum(rast1, rast2), 
        
        # For leftover pixels where raster 1 has data, use raster 1 data
        np.where(~rast1nodata, rast1, \
                 
                 # Otherwise where raster 2 has data, use raster 2 data
                 np.where(~rast2nodata, rast2, nodata_val))  
    )
    
    # Define filepath
    output_path = os.path.join(outdir, filename)
    
    # Write file to drive
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(combined_data, 1)
        
    print(f"Combined raster saved to {output_path}")
        
    return output_path
    
# Combine tmf deforestation and degradation year
defordegra_file = rast_comb(tmf_fm_files[11], tmf_fm_files[12], years, 
                            nodata_val, tmf_processed_folder, 
                            "tmf_defordegrayear_fm.tif")

# Separate multi-year defordegra year raster into single year rasters
tmf_ann_defordegra = annsplit_rast(defordegra_file, years, tmf_processed_folder)



############################################################################


# CONVERT TMF TRANSITION RASTER TO ANNUAL DATA


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

# Define function to create attribute table
def att_table(arr, expected_classes=None):
    
    # Extract unique values and pixel counts
    unique_values, pixel_counts = np.unique(arr, return_counts=True)
    
    # Create a DataFrame with unique values and pixel counts
    attributes = pd.DataFrame({"Class": unique_values, 
                               "Frequency": pixel_counts})
    
    # If expected_classes is provided, run the following:
    if expected_classes is not None:
        
        # Reindex DataFrame to include all expected_classes
        attributes = attributes.set_index("Class").reindex(expected_classes, 
                                                           fill_value=0)
        
        # Reset index to have Class as a column again
        attributes.reset_index(inplace=True)
        
    # Switch rows and columns of dataframe
    attributes = attributes.transpose()
    
    return attributes

# Define function to create annual data from a single-layer array
def arr2ann(annual_arrlist, ref_arr, yearrange):
    
    # Create empty list to hold reclassified annual arrays
    ann_arrs = []
    
    # Iterate over annual arrays and years
    for arr, year in zip(annual_arrlist, yearrange):
        
        # Copy array
        reclass_arr = np.copy(arr)
        
        # Create mask for defordegra pixels in that year
        mask = reclass_arr == year
        
        # Replace defordegra pixels with transition map values
        reclass_arr[mask] = ref_arr[mask]
        
        # Add reclassified array to list
        ann_arrs.append(reclass_arr)
        
    return ann_arrs

# Define function to write a list of arrays to file
def filestack_write(arraylist, yearrange, profile, dtype, fileprefix, out_dir):
    
    # Create empty list to store output filepaths
    filelist = []
    
    # Save each array to drive
    for var, year in zip(arraylist, yearrange):
        
        # Adapt file datatype
        data = var.astype(dtype)
        
        # Define file name and path
        output_filename = f"{fileprefix}_{year}.tif"
        output_filepath = os.path.join(out_dir, output_filename)
        
        # Update profile with dtype string
        profile['dtype'] = data.dtype.name
        
        # Write array to file
        with rasterio.open(output_filepath, "w", **profile) as dst:
            dst.write(data, 1)
            
        # Append filepath to list
        filelist.append(output_filepath)
        
        print(f"{output_filename} saved to file")
    
    return filelist

# Read defordegra rasters
tmf_defordegra_arrs, profile = read_files(tmf_ann_defordegra)

# Read tmf transition map
with rasterio.open(tmf_fm_files[13]) as tmf:
    tmf_trans = tmf.read(1)
    
# Identify unique values and counts
trans_attributes = att_table(tmf_trans)

"""
NOTE: the value "0" exists in the transition map. This value is present in the 
raw data and is not described in the TMF data manual. Because it has a limited 
coverage, the value "0" pixels will excluded from the following analysis
"""

# Exclude 0 values
trans_attributes = trans_attributes.drop(columns = trans_attributes.columns[
    trans_attributes.loc['Class'] == 0])

# Reclassify transition map to annual data
trans_annual = arr2ann(tmf_defordegra_arrs, tmf_trans, years)
    
# Write reclassified arrays to file
trans_annual_files = filestack_write(trans_annual, years, profile, 
                                     rasterio.uint8, "tmf_transition_main_fm", 
                                     tmf_processed_folder)



############################################################################


# DELETE TEMPORARY FOLDER


############################################################################
# If the temporary folder still exists, delete it
if os.path.exists(temp_folder):
    shutil.rmtree(temp_folder)
    print(f"{temp_folder} has been deleted.")
else:
    print(f"{temp_folder} does not exist.")
    