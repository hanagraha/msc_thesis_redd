# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:49:58 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
import pandas as pd
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.mask import mask
from matplotlib.patches import Rectangle
import base64
import io
import numpy as np
import plotly.graph_objs as go



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
years = range(2013, 2025)

# Define landsat data folder
l8_folder = os.path.join("data", "cc_composites", "L8_Annual")

# Define landsat jan data folder
l8_jan_folder = os.path.join("data", "cc_composites", "L8_Jan")

# Define sentinel data folder
s2_folder = os.path.join("data", "cc_composites", "S2_Annual")

# Define sentinel jan data folder
s2_jan_folder = os.path.join("data", "cc_composites", "S2_Jan")

# Define planet data folder
planet_folder = os.path.join("data", "cc_composites", "Planet_Feb")



############################################################################


# READ INPUT DATA


############################################################################
# Read landsat data
l8 = [os.path.join(l8_folder, file) for file in os.listdir(l8_folder) if \
      file.endswith('.tif')]
    
# Read landsat jan data
l8_jan = [os.path.join(l8_jan_folder, file) for file in os.listdir(l8_jan_folder) \
          if file.endswith('.tif')]

# Read sentinel data
s2 = [os.path.join(s2_folder, file) for file in os.listdir(s2_folder) if \
      file.endswith('.tif')]
    
# Read sentinel jan data
s2_jan = [os.path.join(s2_jan_folder, file) for file in os.listdir(s2_jan_folder) \
          if file.endswith('.tif')]

# Read planet data
planet = [os.path.join(planet_folder, file) for file in os.listdir(planet_folder) \
          if file.endswith('.tif')]
    
# Read validation points
valpoints = gpd.read_file("data/validation/validation_points.shp")

# Define default image path (show it is in wdir)
def_img = "/assets/def_ts_small.png"

# Read ndvi data
pnt_ndvi = pd.read_csv("data/validation/jan_ndvi.csv")

# Read ndmi data
pnt_ndmi = pd.read_csv("data/validation/jan_ndmi.csv")


# %%

############################################################################


# FUNCTIONS FOR IMAGE PLOTTING


############################################################################
# Define function to create frames for each point
def point_frame(point_gdf, framesize):
    
    # Create empty list to store frames
    bbox_list = []
    
    # Iterate over each point
    for i in range(len(point_gdf)):
        
        # Extract point geometry
        geom = point_gdf.geometry[i]
        
        # Extract pixel bounds
        minx, miny, maxx, maxy = geom.bounds
        
        # Create a rectangular bounding box
        bbox = box(minx - framesize / 2, miny - framesize / 2, 
                    maxx + framesize / 2, maxy + framesize / 2)
        
        bbox_list.append(bbox)
    
    # Copy validation points gdf
    bbox_gdf = point_gdf.copy()
    
    # Add bbox infomration to gdf
    bbox_gdf['frame'] = bbox_list
    
    # Make sure crs stays the same
    bbox_gdf.crs = point_gdf.crs
    
    return bbox_gdf

# Create frames for each validation point (500m frame)
val_frames500 = point_frame(valpoints, 500)

# Create frames for each validation point (1000m frame)
val_frames1000 = point_frame(valpoints, 1000)

# Define function to clip validation data to each frame
def clip_landsat(l8_raster_pathlist, geom, nodata_value = 1):
    
    # Create empty lists to hold clipped arrays and metadata
    l8_clipped_arrs = []
    l8_metadata = []
    
    # Iterate over rasters
    for l8_file in l8_raster_pathlist:

        # Read raster
        with rasterio.open(l8_file) as l8_rast:
            
            # Only process the first three bands (RGB)
            l8_indices = [4, 3, 2]
            
            # Mask pixels outside aoi with NoData values
            l8_raster_clip, l8_out_transform = mask(l8_rast, geom, crop = True, 
                                                    nodata = nodata_value, 
                                                    indexes = l8_indices)
            
            # Create empty list to hold normalized arrays
            l8_norm_list = []
            
            # Iterate over each band
            for band in l8_raster_clip:
                
                # Exclude nan pixels
                valid_mask = ~np.isnan(band) & (band != nodata_value)
                
                # Extract minimum
                band_min = np.min(band[valid_mask])
                
                # Extract maximum
                band_max = np.max(band[valid_mask])
                
                # Create empty array to hold normalized values
                band_norm = np.full_like(band, nodata_value, dtype=np.float32)
                
                # Normalize valid pixels only
                band_norm[valid_mask] = (band[valid_mask] - band_min) / (band_max - band_min)
                
                # Add normalized array to list
                l8_norm_list.append(band_norm)
            
            # Convert array list to array
            l8_norm_arrays = np.stack(l8_norm_list)
        
            # Copy metadata
            l8_out_meta = l8_rast.meta.copy()
            
            # Update metadata
            l8_out_meta.update({
                'driver': 'GTiff',
                'dtype': 'float32',
                'count': len(l8_indices),
                'height': l8_norm_arrays.shape[1],
                'width': l8_norm_arrays.shape[2],
                'transform': l8_out_transform,
                'nodata': nodata_value})
        
        # Add clipped array to list
        l8_clipped_arrs.append(l8_norm_arrays)
        
        # Add metadata to list
        l8_metadata.append(l8_out_meta)
        
    return l8_clipped_arrs, l8_metadata

# Define function to clip validation data to each frame
def clip_sentinel(s2_raster_pathlist, geom, nodata_value = 0):
    
    # Create empty lists to hold clipped arrays and metadata
    s2_clipped_arrs = []
    s2_metadata = []
    
    # Iterate over rasters
    for s2_file in s2_raster_pathlist:

        # Read raster
        with rasterio.open(s2_file) as s2_rast:
            
            # Only process the first three bands (RGB)
            s2_indices = [16, 17, 18]
            
            # Mask pixels outside aoi with NoData values
            s2_raster_clip, s2_out_transform = mask(s2_rast, geom, crop = True, 
                                                    nodata = nodata_value, 
                                                    indexes = s2_indices)
            
            # Make sure the nan values have the same nodata value
            s2_raster_clip_int = np.where(np.isnan(s2_raster_clip), nodata_value, 
                                          s2_raster_clip).astype(int)
            
            # Copy metadata
            s2_out_meta = s2_rast.meta.copy()
            
            # Update metadata
            s2_out_meta.update({
                'driver': 'GTiff',
                'dtype': 'uint16',
                'count': len(s2_indices),
                'height': s2_raster_clip_int.shape[1],
                'width': s2_raster_clip_int.shape[2],
                'transform': s2_out_transform,
                'nodata': nodata_value})
        
        # Add clipped array to list
        s2_clipped_arrs.append(s2_raster_clip_int)
        
        # Add metadata to list
        s2_metadata.append(s2_out_meta)
        
    return s2_clipped_arrs, s2_metadata

# Define function to clip validation data to each frame
def clip_planet(pl_raster_pathlist, geom, nodata_value):
    
    # Create empty lists to hold clipped arrays and metadata
    pl_clipped_arrs = []
    pl_metadata = []
    
    # Iterate over rasters
    for pl_file in pl_raster_pathlist:

        # Read raster
        with rasterio.open(pl_file) as pl_rast:
            
            # Only process the first three bands (RGB)
            pl_indices = [1,2,3]
            
            # Mask pixels outside aoi with NoData values
            pl_raster_clip, pl_out_transform = mask(pl_rast, geom, crop = True, 
                                                    nodata = nodata_value, 
                                                    indexes = pl_indices)
        
            # Copy metadata
            pl_out_meta = pl_rast.meta.copy()
            
            # Update metadata
            pl_out_meta.update({
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': len(pl_indices),
                'height': pl_raster_clip.shape[1],
                'width': pl_raster_clip.shape[2],
                'transform': pl_out_transform,
                'nodata': nodata_value})
        
        # Add clipped array to list
        pl_clipped_arrs.append(pl_raster_clip)
        
        # Add metadata to list
        pl_metadata.append(pl_out_meta)
        
    return pl_clipped_arrs, pl_metadata


# Define function to plot frames based on point
def landsat_plot(pntindex, l8):
    
    # Extract relevant point
    point = val_frames1000["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames1000["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    l8_clipped_arrs, l8_metas = clip_landsat(l8, frame)
    
    # Transpose clipped array to match format for imshow
    l8_clipped_rgb_arrs = [l8_arr.transpose(1, 2, 0) for l8_arr in l8_clipped_arrs]
    
    # Define labels for subplots
    l8_labels = list(years)
    
    # Create dictionary to hold l8 data
    l8_dic = {2013 + i: (arr, meta, label) for i, (arr, meta, label) in
              enumerate(zip(l8_clipped_rgb_arrs, l8_metas, l8_labels))}

    # Initialize figure with 3x4 subplots
    l8_fig, l8_axs = plt.subplots(2, 6, figsize=(12, 4))
    
    # Flatten axes array
    l8_axs = l8_axs.flatten()
    
    # Iterate over years in full range (2013-2024)
    for i, year in enumerate(list(years)):
        
        # Extract corresponding axis
        ax = l8_axs[i]
        
        # If there is s2 data for that year
        if year in l8_dic:
            
            # Extract data for that year
            l8_rgb_array, l8_meta, l8_label = l8_dic[year]
            
            # Display rgb image
            ax.imshow(l8_rgb_array)
            
            # Extract specific transform for year
            l8_transform = l8_meta['transform']
            
            # Calculate pixel size in pixels for each raster
            pxsize_px = 30 / abs(l8_transform.a)
            
            # Convert xy coordinate to image coordinates
            px, py = ~l8_transform * (point.x, point.y)
             
            # Create pixel rectangle
            rect = Rectangle(
                (px - pxsize_px / 2, py - pxsize_px / 2),
                pxsize_px, pxsize_px, linewidth=1, edgecolor='red', facecolor='none'
            )
             
            # Overlay validation pixel area
            ax.add_patch(rect)
     
            # Set title
            ax.set_title(f'{l8_label}', fontsize=12)
            
        # If there is no s2 data for that year
        else:
            
            # Set title
            ax.set_title(f'{year}', fontsize=12)
         
        # Remove axis labels for all subplots
        ax.axis('off')

    # Create empty BytesIO buffer
    l8_buffer = io.BytesIO()
    
    # Create tight layout
    plt.tight_layout()
    # plt.show
    
    # Save figure to buffer as png
    l8_fig.savefig(l8_buffer, format="png")
    
    # Close figure
    plt.close(l8_fig)
    
    # Read buffer from beginning
    l8_buffer.seek(0)
    
    # Encode image to base64 for Dash
    l8_encoded_image = base64.b64encode(l8_buffer.read()).decode('utf-8')
    
    # Close buffer
    l8_buffer.close()
    
    # Return buffer string for <img> tag in Dash
    return f"data:image/png;base64,{l8_encoded_image}"


# Define function to plot frames based on point
def sentinel_plot(pntindex, s2):
    
    # Extract relevant point
    point = val_frames1000["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames1000["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    s2_clipped_arrs, s2_metas = clip_sentinel(s2, frame)
    
    # Transpose clipped array to match format for imshow
    s2_clipped_rgb_arrs = [s2_arr.transpose(1, 2, 0) for s2_arr in s2_clipped_arrs]
    
    # Define start year
    start_year = 2018 if len(s2) == 7 else 2019
    
    # Define labels for subplots
    # s2_labels = list(years)[5:]
    s2_labels = [str(start_year + i) for i in range(len(s2_clipped_rgb_arrs))]
    
    # Initialize figure
    s2_fig, s2_axs = plt.subplots(2, 6, figsize=(12, 4))
    
    # Flatten axes array for easy iteration
    s2_axs = s2_axs.flatten()
    
    # Create a dictionary mapping year to its data for easier alignment
    s2_dic = {start_year + i: (arr, meta, label) for i, (arr, meta, label) in
              enumerate(zip(s2_clipped_rgb_arrs, s2_metas, s2_labels))}
     
    # Iterate over years in full range (2013-2024)
    for i, year in enumerate(list(years)):
        
        # Extract corresponding axis
        ax = s2_axs[i]
        
        # If there is s2 data for that year
        if year in s2_dic:
            
            # Extract data for that year
            s2_rgb_array, s2_meta, s2_label = s2_dic[year]
            
            # Display rgb image
            ax.imshow(s2_rgb_array)
            
            # Extract specific transform for year
            s2_transform = s2_meta['transform']
            
            # Calculate pixel size in pixels for each raster
            pxsize_px = 30 / abs(s2_transform.a)
            
            # Convert xy coordinate to image coordinates
            px, py = ~s2_transform * (point.x, point.y)
             
            # Create pixel rectangle
            rect = Rectangle(
                (px - pxsize_px / 2, py - pxsize_px / 2),
                pxsize_px, pxsize_px, linewidth=1, edgecolor='red', facecolor='none'
            )
             
            # Overlay validation pixel area
            ax.add_patch(rect)
     
            # Set title
            ax.set_title(f'{s2_label}', fontsize=12)
            
        # If there is no s2 data for that year
        else:
            
            # Set title
            ax.set_title(f'{year}', fontsize=12)
         
        # Remove axis labels for all subplots
        ax.axis('off')
    
    # Create empty BytesIO buffer
    s2_buffer = io.BytesIO()
    
    # Create tight layout
    plt.tight_layout()
    
    # Save figure to buffer as png
    s2_fig.savefig(s2_buffer, format="png")
    
    # Close figure
    plt.close(s2_fig)
    
    # Read buffer from beginning
    s2_buffer.seek(0)
    
    # Encode image to base64 for Dash
    s2_encoded_image = base64.b64encode(s2_buffer.read()).decode('utf-8')
    
    # Close buffer
    s2_buffer.close()
    
    # Return buffer string for <img> tag in Dash
    return f"data:image/png;base64,{s2_encoded_image}"


# Define function to plot frames based on point
def planet_plot(pntindex):
    
    # Extract relevant point
    point = val_frames500["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames500["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    pl_clipped_arrs, pl_metas = clip_planet(planet, frame, nodata_val)
    
    # Transpose clipped array to match format for imshow
    pl_clipped_rgb_arrs = [pl_arr.transpose(1, 2, 0) for pl_arr in pl_clipped_arrs]
    
    # Define labels for subplots
    pl_labels = ["2013 Jan/Feb", "2013 Dec", "2015 Annual", "2016 Annual", 
                 "2017 Feb", "2018 Feb", "2019 Feb", "2020 Feb", 
                 "2021 Feb", "2022 Feb", "2023 Feb", "2024 Feb"]
    
    # Initialize figure with 3x4 subplots
    pl_fig, pl_axs = plt.subplots(2, 6, figsize=(12, 4))
    
    # Flatten axes array
    pl_axs = pl_axs.flatten()
    
    # Iterate over axis, arrays, and metadata
    for pl_i, (pl_rgb_array, pl_meta) in enumerate(zip(pl_clipped_rgb_arrs, pl_metas)):
        
        # Display rgb image
        pl_axs[pl_i].imshow(pl_rgb_array)
        
        # Remove axis labels
        pl_axs[pl_i].axis('off')
        
        # Set subplot titles
        pl_axs[pl_i].set_title(f'{pl_labels[pl_i]}')
        
        # Extract specific transform for each year
        pl_transform = pl_meta['transform']
        
        # Calculate pixel size in pixels for each raster
        pxsize_px = 30 / abs(pl_transform.a)
        
        # Convert xy coordinate to image coordinates
        px, py = ~pl_transform * (point.x, point.y)
        
        # Create pixel rectangle
        rect = Rectangle(
            (px - pxsize_px / 2, py - pxsize_px / 2),
            pxsize_px, pxsize_px, linewidth=1, edgecolor='red', facecolor='none'
        )
        
        # Overlay validation pixel area
        pl_axs[pl_i].add_patch(rect)

    # Create empty BytesIO buffer
    pl_buffer = io.BytesIO()
    
    # Create tight layout
    plt.tight_layout()
    
    # Save figure to buffer as png
    pl_fig.savefig(pl_buffer, format="png")
    
    # Close figure
    plt.close(pl_fig)
    
    # Read buffer from beginning
    pl_buffer.seek(0)
    
    # Encode image to base64 for Dash
    pl_encoded_image = base64.b64encode(pl_buffer.read()).decode('utf-8')
    
    # Close buffer
    pl_buffer.close()
    
    # Return buffer string for <img> tag in Dash
    return f"data:image/png;base64,{pl_encoded_image}"


#%%
############################################################################


# MY DASHBOARD (WITH BUTTONS, ONE PLOT AT A TIME)


############################################################################
# Initiate dashboard
app = Dash()

# Define dashboard layout
app.layout = html.Div([

    # Dashboard heading
    html.H1("Sample-Based Deforestation Validation Dashboard", style={
            "font-family": "Arial", "font-size": "36px", "color": "darkslategrey",
            "text-align": "center"}),

    # Input box for validation point ID
    html.Div([

        # Input box label
        html.Label("Enter Validation Point ID (0-1379): ", style={
            "font-size": "18px", "font-family": "Arial"}),

        # Input box
        dcc.Input(
            id="input-id",
            type="number",
            min=0,
            max=1379,
            step=1,
            value=None,
            placeholder="Enter ID...",
            style={"margin-right": "10px"}
        ),

    ], style={"text-align": "center", "margin-top": "20px"}),
    
    # Visual divider
    html.Br(),
    
    # Radio buttons for indice selection
    html.Div([
        
        html.Div([
            
            # Create description label before buttons
            html.Label("Select Indice for Time Series:", style={"font-size": "24px", 
                        "font-family": "Arial", "color": "slategrey", 
                        "margin-right": "20px", "font-weight": "bold"}),
            
            # Create radio buttons
            dcc.RadioItems(
            
                id="indice-selector",
                
                # Button options
                options=[
                    {"label": "NDMI", "value": "ndmi"},
                    {"label": "NDVI", "value": "ndvi"}], 
                
                # Set default selection
                value="ndmi", 
                
                # Set buttons side by side
                inline=True, 
                
                # Button text formatting
                labelStyle = {"font-family": "Arial", "margin-right": "20px", 
                              "font-size": "18px"}, 
                
                # Add spacing between buttons
                style={"margin-left": "25px", "margin-right": "25px"})], 
        
        # Positioning of button section
        style={"display": "flex", "align-items": "center", "justify-content": 
                  "center", "margin-top": "10px"})]),

    # Graph output
    dcc.Graph(id="indice-plot", style={"margin-top": "0px", "margin": "0 auto",
                                       "width": "80%"}),

    # Visual divider
    html.Br(),
    
    # Radio buttons for plot selection
    html.Div([
        
        html.Div([
            
            # Create description label before buttons
            html.Label("Select Imagery Source:", style={"font-size": "24px", 
                        "font-family": "Arial", "color": "slategrey", 
                        "margin-right": "20px", "font-weight": "bold"}),
            
            # Create radio buttons
            dcc.RadioItems(
            
                id="plot-selector",
                
                # Button options
                options=[
                    {"label": "Landsat-8 (Annual)", "value": "landsat"},
                    {"label": "Landsat-8 (January)", "value": "landsat_jan"},
                    {"label": "Sentinel-2 (Annual)", "value": "sentinel"},
                    {"label": "Sentinel-2 (January)", "value": "sentinel_jan"},
                    {"label": "RapidEye + PlanetScope", "value": "planet"}], 
                
                # Set default selection
                value="landsat", 
                
                # Set buttons side by side
                inline=True, 
                
                # Button text formatting
                labelStyle = {"font-family": "Arial", "margin-right": "20px", 
                              "font-size": "18px"}, 
                
                # Add spacing between buttons
                style={"margin-left": "5px", "margin-right": "5px"})], 
        
        # Positioning of button section
        style={"display": "flex", "align-items": "center", "justify-content": 
                  "center", "margin-top": "10px"})]),
    
    # Dynamic image display
    html.Div([html.Img(id = "time-series-plot", src = def_img, style = 
                       {"width": "80%", "height": "100%", "margin-top": "10px",
                        "display": "block", "margin": "0 auto"})]),
    
    # Visual divider
    html.Br()
    
])

# Define callback for validation point details
@app.callback(
    Output("indice-plot", "figure"),
    Input("input-id", "value"),
    Input("indice-selector", "value")
)

# Define function to display point details
def indice_plotting(point_id, indice):
    
    # If no valid point id provided
    if point_id is None:
        
        # Return empty figure
        return go.Figure()
    
    # If NDVI is selected
    if indice == 'ndvi':
        indice_row = pnt_ndvi.iloc[point_id]
        labs = pnt_ndvi.columns.tolist()[2:]
        ylab = "NDVI"
        
    # If NDMI is selected
    elif indice == 'ndmi':
        indice_row = pnt_ndmi.iloc[point_id]
        labs = pnt_ndmi.columns.tolist()[2:]
        ylab = "NDMI"
    
    # Extract ndvi values for selected point
    indice_vals = [indice_row[col] for col in labs]

    # Create line plot
    fig = go.Figure(
        
        # Define input data
        data=[
            
            # Initialize scatter plotting
            go.Scatter(
                
                # Set x data to years
                x = list(years),
                
                # Set y data to ndvi value
                y = indice_vals,
                
                # Connect with lines
                mode = "lines",
                
                # Make line green
                line = dict(color="darkslategrey"),
                
                # Add data label
                name = ylab
            )
        ],
        
        # Adjust plot layout
        layout=go.Layout(
 
            # Set x axis label
            xaxis = dict(title="Year", tickmode="linear", tick0=2013, dtick=1),
            
            # Set y axis label
            yaxis = dict(title=ylab),
            
            # Use template
            template = "plotly_white",
            
            # Remove unnecessary margin or padding
            margin=dict(t=0)
        )
    )

    return fig

# Define callback for time-series plot selection
@app.callback(
    Output("time-series-plot", "src"),
    [Input("input-id", "value"), Input("plot-selector", "value")]
)

# Define function to display selected time series plot
def ts_plotting(point_id, plot_type):
    
    # If no point is given
    if point_id is None:
        
        # Print default image
        return def_img
    
    # If landsat is selected
    if plot_type == "landsat":
        
        # Plot landsat time series
        return landsat_plot(point_id, l8)
    
    # If landsat (jan) is selected
    if plot_type == "landsat_jan":
        
        # Plot landsat time series
        return landsat_plot(point_id, l8_jan)
    
    # If sentinel is selected
    elif plot_type == "sentinel":
        
        # Plot sentinel time series
        return sentinel_plot(point_id, s2)
    
    # If sentinel (jan) is selected
    if plot_type == "sentinel_jan":
        
        # Plot landsat time series
        return sentinel_plot(point_id, s2_jan)
    
    # If planet is selected
    elif plot_type == "planet":
        
        # Plot planet time series
        return planet_plot(point_id)
    
    return ""

# Run/update dashboard
if __name__ == '__main__':
    app.run_server(debug=True)

    
