# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:39:18 2024

@author: hanna
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import requests
import geopandas as gpd
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import pandas as pd
import numpy as np



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



############################################################################


# IMPORT DEFORESTATION DATASETS (WEB DOWNLOAD TO DRIVE)


############################################################################

### DOWNLOAD GFC DATA
gfc_urls = ["https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_treecover2000_10N_020W.tif",
            "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_lossyear_10N_020W.tif"]

# Directory to save the files
gfc_dir = "data/hansen"

# Create the directory if it doesn't exist
os.makedirs(gfc_dir, exist_ok=True)  

# Loop through each URL
for url in gfc_urls:
    
    # Extract filename from the URL
    filename = url.split("/")[-1]
    local_filename = os.path.join(gfc_dir, filename)

    # Send GET request to the URL
    response = requests.get(url, stream=True)

    # Open the file in write-binary mode and write the content
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Download complete for {filename}")


### DOWNLOAD TMF DATA
jrc_urls = ["https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=DegradationYear&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=DeforestationYear&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2013&lat=N10&lon=W20",
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2014&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2015&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2016&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2017&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2018&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2019&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2020&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2021&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2022&lat=N10&lon=W20", 
            "https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_2023&lat=N10&lon=W20"]


# Directory to save the files
jrc_dir = "data/jrc"

# Create the directory if it doesn't exist
os.makedirs(jrc_dir, exist_ok=True)  

# Loop through each URL
for url in jrc_urls:
    
    # Extract filename from the URL
    filename = url.split("dataset=")[1].split("&")[0]
    filename = f"tmf_{filename}.tif"
    local_filename = os.path.join(jrc_dir, filename)

    # Send GET request to the URL
    response = requests.get(url, stream=True)

    # Open the file in write-binary mode and write the content
    with open(local_filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Download complete for {filename}")


############################################################################


# IMPORT DEFORESTATION DATASETS (WEB DOWNLOAD TO DRIVE)


############################################################################

### REPROJECT TMF
with rasterio.open('data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, epsg_string, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': epsg_string,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('data/jrc/tmf_defor_reproj.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=epsg_string,
                resampling=Resampling.nearest)
            

### REPROJECT GFC
with rasterio.open('data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, epsg_string, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': epsg_string,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('data/hansen/gfc_defor_reproj.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=epsg_string,
                resampling=Resampling.nearest)


### READ REPROJECTED FILES
with rasterio.open("data/jrc/tmf_defor_reproj.tif") as defor:
    tmf_defor = defor.read(1)
    
with rasterio.open("data/hansen/gfc_defor_reproj.tif") as gfc:
    gfc_defor = gfc.read(1)

# Check if epsgs match:
if defor.crs == epsg_string:
    print(f"Both GeoDataFrames have the same {epsg_string}")
else:
    print(f"Different EPSG codes: raster dataset has {defor.crs}, destination crs is {epsg_string}")



############################################################################


# CLIP TO AOI


############################################################################

### CREATE AOI
aoi = gpd.GeoDataFrame(pd.concat([villages, grnp], ignore_index=True)).dissolve()
aoi_geom = aoi.geometry


### MASK TMF
with rasterio.open("data/jrc/tmf_defor_reproj.tif") as src:
    # Read the raster's CRS and bounds
    raster_crs = src.crs
    raster_transform = src.transform
    
    # Clip the raster to the polygon shape
    tmf_defor_clipped, out_transform = mask(src, aoi_geom, crop=True, nodata=9999)
    
    out_meta = src.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count': 1,
        'height': tmf_defor_clipped.shape[1],
        'width': tmf_defor_clipped.shape[2],
        'transform': out_transform })
    
tmf_defor_clipped = tmf_defor_clipped.astype('float32')   
tmf_defor_clipped[tmf_defor_clipped == 9999] = np.nan

# Save the clipped raster to a new file
tmf_defor_clipped_path = 'data/jrc/tmf_defor_reproj_clipped.tif'
with rasterio.open(tmf_defor_clipped_path, 'w', **out_meta) as dest:
    dest.write(tmf_defor_clipped)
    

### MASK GFC
with rasterio.open("data/hansen/gfc_defor_reproj.tif") as src:
    # Read the raster's CRS and bounds
    raster_crs = src.crs
    raster_transform = src.transform
    
    # Clip the raster to the polygon shape
    gfc_defor_clipped, out_transform = mask(src, aoi_geom, crop=True, nodata = 255)
    
    out_meta = src.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count': 1,
        'height': gfc_defor_clipped.shape[1],
        'width': gfc_defor_clipped.shape[2],
        'transform': out_transform })
    
# gfc_defor_clipped = gfc_defor_clipped + 2000
gfc_defor_clipped = gfc_defor_clipped.astype('float32')   
gfc_defor_clipped[gfc_defor_clipped == 255] = np.nan
gfc_defor_clipped = gfc_defor_clipped + 2000


# Save the clipped raster to a new file
gfc_defor_clipped_path = 'data/hansen/gfc_defor_reproj_clipped.tif'
with rasterio.open(gfc_defor_clipped_path, 'w', **out_meta) as dest:
    dest.write(gfc_defor_clipped)


############################################################################


# PLOTTING


############################################################################

### FOR RASTERS (ARRAY)
plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
show(gfc_defor_clipped, title='TMF Deforestation Year', cmap='gist_ncar')

# ### FOR VECTORS (GDF)
# aoi.plot()
# plt.show()


# Figure with three subplots, unpack directly
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

# Populate the three subplots with raster data
show(tmf_defor_clipped, ax=ax1, title='TMF Deforestation Year')
show(gfc_defor_clipped, ax=ax2, title='GFC Deforestation Year')

plt.show()




############################################################################


# R SCRIPT WORK


############################################################################

"""

To create line graph on forest loss %

loop for every year
anything with forest loss before 2013 subtract from forest in 2000

forest2013 = forest2012 - loss2013
lossperc2013 = loss2013 / study area (this is what the paper did, 
                                      reasoning that dividing by forest in 
                                      that area changes, not super reliable)



"""



############################################################################


# ARCHIVE


############################################################################

### DEFINE FUNCTION TO REPROJECT AND CLIP
def reproject_and_clip_raster(tif_path, target_crs, clip_boundary_gdf):
    with rasterio.open(tif_path) as src:
        
        # Calculate the transform and dimensions for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        # Create a metadata template for the reprojected raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject the raster in memory
        reprojected_array = rasterio.MemoryFile().open(**kwargs)
        
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(reprojected_array, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
        
        # Clip the reprojected raster with the boundary
        clip_boundary = [feature['geometry'] for feature in clip_boundary_gdf.__geo_interface__['features']]
        clipped_image, clipped_transform = mask(reprojected_array, clip_boundary, crop=True, nodata=255)
        clipped_image = clipped_image.astype('float32')   
        clipped_image[clipped_image == 255] = np.nan
        
        return clipped_image, clipped_transform, kwargs


### REPROJECT AND CLIP GFC
gfc_clipped_dict = {}

for tif in gfc_files:
    clipped_image, clipped_transform, metadata = reproject_and_clip_raster(tif, epsg_string, aoi_geom)
    non_nan_mask = ~np.isnan(clipped_image) 
    clipped_image[non_nan_mask] += 2000
    
    # Save to drive
    output_path = tif.replace(".tif", "_clipped.tif")
    gfc_clipped_dict[output_path] = clipped_image
    print(f"Preprocessing Complete for {output_path}")
    
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(clipped_image)
        
        
### REPROJECT AND CLIP TMF
tmf_clipped_dict = {}

for tif in tmf_files:
    clipped_image, clipped_transform, metadata = reproject_and_clip_raster(tif, epsg_string, aoi_geom)
    
    # Save to drive
    output_path = tif.replace(".tif", "_clipped.tif")
    tmf_clipped_dict[output_path] = clipped_image
    print(f"Preprocessing Complete for {output_path}")
    
    
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(clipped_image)


# Function to reproject and clip a TIFF
def reproject_and_clip_raster(tif_path, target_crs, clip_boundary_gdf):
    with rasterio.open(tif_path) as src:
        # Calculate the transform and dimensions for the target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        # Create a metadata template for the reprojected raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject the raster in memory
        reprojected_array = rasterio.MemoryFile().open(**kwargs)
        
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(reprojected_array, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
        
        # Clip the reprojected raster with the boundary
        clip_boundary = [feature['geometry'] for feature in clip_boundary_gdf.__geo_interface__['features']]
        clipped_image, clipped_transform = mask(reprojected_array, clip_boundary, crop=True)
        
        return clipped_image, clipped_transform, kwargs

# Load your clip boundary (e.g., GeoDataFrame)
clip_boundary_gdf = gpd.read_file("path_to_clip_boundary.shp")
clip_boundary_gdf = clip_boundary_gdf.to_crs("EPSG:4326")  # Ensure the boundary is in the same CRS as the target CRS
clip_boundary_gdf = aoi_geom

# Define the target CRS (e.g., EPSG code or proj string)
target_crs = epsg_string  # Replace with your target CRS

# List of raster datasets (TIFF files) to process
tif_files = ["data/jrc/tmf_DeforestationYear.tif", "data/jrc/tmf_DegradationYear.tif"]

# Loop through datasets, reproject, and clip
for tif in tif_files:
    clipped_image, clipped_transform, metadata = reproject_and_clip_raster(tif, target_crs, clip_boundary_gdf)
    
    # Now you can either save the clipped image or continue processing
    output_path = tif.replace(".tif", "_clipped.tif")
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(clipped_image)



# vil_shape = villages[['geometry']]
# vil_new = vil_shape.explode(ignore_index=True)
# gola_shape = grnp[['geometry']]
# non_na_pixel_count = np.sum(~np.isnan(out_image))
# ref_pix_count = np.sum(out_image)*0.01
# ref_pix_count

### DOWNLOAD DATA

# # URL of the file to download
# url = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# # Local filename to save the file
# local_filename = "data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# # Send GET request to the URL
# response = requests.get(url, stream=True)

# # Open the file in write-binary mode and write the content
# with open(local_filename, 'wb') as file:
#     for chunk in response.iter_content(chunk_size=8192):
#         file.write(chunk)

# print("Download complete.")


### READ DATA
    
# with rasterio.open("data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as defor:
#     tmf_defor = defor.read(1)

# with rasterio.open("data/jrc/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as degra:
#     tmf_degra = degra.read(1)
    
# with rasterio.open("data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif") as gfc:
#     gfc_defor = gfc.read(1)
    
# # Add 2000 to GFC so instead of 22 looks like 2022
# gfc_defor = gfc_defor + 2000

# "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_treecover2000_10N_020W.tif"
# "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2023-v1.11/Hansen_GFC-2023-v1.11_lossyear_10N_020W.tif"




# # URL of the file to download
# url = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# # Local filename to save the file
# local_filename = "data/hansen/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# # Send GET request to the URL
# response = requests.get(url, stream=True)

# # Open the file in write-binary mode and write the content
# with open(local_filename, 'wb') as file:
#     for chunk in response.iter_content(chunk_size=8192):
#         file.write(chunk)

# print("Download complete.")

# def reproject_raster2(input_raster_path, epsg_string):
#     # Open the source raster
#     with rasterio.open(input_raster_path) as src:
#         # Calculate the new transform, width, and height for the target CRS
#         transform, width, height = calculate_default_transform(
#             src.crs, epsg_string, src.width, src.height, *src.bounds)
        
#         # Copy the metadata and update with new projection details
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': epsg_string,
#             'transform': transform,
#             'width': width,
#             'height': height
#         })
        
#         # Create an empty array to hold the reprojected data
#         reprojected_array = np.empty((src.count, height, width), dtype=src.meta['dtype'])
        
#         # Loop through each band and reproject
#         for i in range(1, src.count + 1):
#             reproject(
#                 source=rasterio.band(src, i),
#                 destination=reprojected_array[i - 1],
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=epsg_string,
#                 resampling=Resampling.nearest)
        
#         return reprojected_array

############################################################################


# VISUALIZE LANDSAT


############################################################################


from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep


landsat_bands_data_path = "data\\landsat\\LC09_L2SP_200055_20240207_20240209_02_T1_SR_B*[1-7]*.TIF"
stack_band_paths = glob(landsat_bands_data_path)
stack_band_paths.sort()

# Create image stack and apply nodata value for Landsat
arr_st, meta = es.stack(stack_band_paths, nodata=-9999)

# Create figure with one plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot red, green, and blue bands, respectively
ep.plot_rgb(arr_st, rgb=(3, 2, 1), ax=ax, title="Landsat 9 RGB Image")
plt.show()


# Create figure with one plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot bands with stretched applied
ep.plot_rgb(
    arr_st,
    rgb=(3, 2, 1),
    ax=ax,
    stretch=True,
    str_clip=0.5,
    title="Landsat 8 RGB Image with Stretch Applied",
)
plt.show()



### MORE ARCHIVE
############################################################################


# REPROJECT AND CLIP RASTERS


############################################################################

# ### DEFINE FUNCTION TO REPROJECT RASTERS
# def reproject_raster(raster_path, epsg):
#     # open raster
#     with rasterio.open(raster_path) as rast:
#         transform, width, height = calculate_default_transform(
#             rast.crs, epsg, rast.width, rast.height, *rast.bounds)
        
#         kwargs = rast.meta.copy()
#         kwargs.update({'crs': epsg,
#                        'transform': transform,
#                        'width': width,
#                        'height': height})
        
#         output_path = raster_path.replace(".tif", "_reprojected.tif")
        
#         with rasterio.open(output_path, 'w', **kwargs) as dst:
#             for i in range(1, rast.count + 1):
#                 reproject(
#                     source=rasterio.band(rast, i),
#                     destination=rasterio.band(dst, i),
#                     rast_transform=rast.transform,
#                     rast_crs=rast.crs,
#                     dst_transform=transform,
#                     dst_crs=epsg,
#                     resampling=Resampling.nearest)
        
#         print(f"Reprojection Complete for {output_path}")
        
#         return output_path
    
    
# # Reproject GFC images
# gfc_reproj_files = []

# for tif in gfc_files:
#     tif_reproj_file = reproject_raster(tif, epsg_string)
#     gfc_reproj_files.append(tif_reproj_file)
    
    
# # Reproject TMF images
# tmf_reproj_files = []

# for tif in tmf_files:
#     tif_reproj_file = reproject_raster(tif, epsg_string)
#     tmf_reproj_files.append(tif_reproj_file)



# ### DEFINE FUNCTION TO CLIP RASTERS
# def clip_raster(raster_path, aoi_geom, nodata_val):
#     # open raster
#     with rasterio.open(raster_path) as rast:
        
#         # Clip raster
#         raster_clip, out_transform = mask(rast, aoi_geom, crop=True, nodata=nodata_val)
        
#         out_meta = rast.meta.copy()
#         out_meta.update({
#             'driver': 'GTiff',
#             'count': 1,
#             'height': raster_clip.shape[1],
#             'width': raster_clip.shape[2],
#             'transform': out_transform,
#             'nodata': nodata_val})
        
#     raster_clip = raster_clip.astype('float32')
#     # raster_clip[raster_clip == nodata_val] = np.nan
    
#     outpath = raster_path.replace(".tif", "_clipped.tif")
    
#     # if os.path.exists(outpath): 
#     #     os.remove(outpath)
    
#     with rasterio.open(outpath, 'w', **out_meta) as dest:
#         dest.write(raster_clip)
    
#     print(f"Clipping Complete for {outpath}")
    
#     return raster_clip, out_meta


# # Clip images of interest
# gfc_lossyear, gfc_lossyear_meta = clip_raster(gfc_reproj_files[1], aoi_geom, 255)
# gfc_treeloss2000, gfc_treeloss2000_meta = clip_raster(gfc_reproj_files[0], aoi_geom, 255)
# tmf_deforyear, tmf_deforyear_meta = clip_raster(tmf_reproj_files[1], aoi_geom, 255)
# tmf_degrayear, tmf_degrayear_meta = clip_raster(tmf_reproj_files[0], aoi_geom, 255)



# ############################################################################


# # RECLASSIFICATION


# ############################################################################
# # Reclassify GFC 
# gfc_lossyear_path = gfc_reproj_files[1].replace(".tif", "_clipped_reclassified.tif")
# gfc_lossyear_new = np.where(gfc_lossyear != 255, gfc_lossyear + 2000, gfc_lossyear)
# gfc_lossyear_meta.update(dtype='int16')
# gfc_lossyear_new = gfc_lossyear + 2000

# with rasterio.open(gfc_lossyear_path, 'w', **gfc_lossyear_meta) as dest:
#     dest.write(gfc_lossyear_new)

# ### READ RELEVANT DATA FILES
# with rasterio.open(tmf_reproj_files[1]) as tmf:
#     deforyear = tmf.read(1)



# ############################################################################


# # PLOTTING


# ############################################################################

# # ### FOR RASTERS (ARRAY)
# plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
# show(tmf_deforyear, title='TMF Deforestation Year', cmap='gist_ncar')

# # # ### FOR VECTORS (GDF)
# # # aoi.plot()
# # # plt.show()


# ### FIGURE WITH SUBPLOTS (ARRAY)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

# # Populate the three subplots with raster data
# show(gfc_lossyear, ax=ax1, title='TMF Deforestation Year')
# show(gfc_lossyear_new, ax=ax2, title='GFC Deforestation Year')

# plt.show()

# ############################################################################


# # IMPORT RELEVANT DATASETS (LOCAL DRIVE)


# ############################################################################

# ### READ POLYGON DATA
# grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")
# villages = gpd.read_file("data/village polygons/VillagePolygons.geojson")

# # Annual Change data
# tmf_folder = "data/jrc_preprocessed"
# tmf_annual_files = glob.glob(os.path.join(tmf_folder, "*AnnualChange*"))
# tmf_annual_files = ([file for file in tmf_annual_files if file.endswith('_fm.tif') 
#                      and not file.endswith(('.xml', '.ovr'))])

# tmf_annual_dict = {}

# for file in tmf_annual_files:
#     filename = os.path.basename(file)
#     year = filename.split("AnnualChange_")[1].split('_')[0]
#     var = f"tmf_{year}"
    
#     with rasterio.open(file) as tmf:
#         tmf_annual = tmf.read(1)
#         tmf_annual_dict[var] = tmf_annual

#     print(f"Stored {var}, data shape: {tmf_annual.shape}")
    





