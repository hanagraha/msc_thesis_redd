# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:49:58 2024

@author: hanna
"""

############################################################################


# IMPORT PACKAGES


############################################################################

import os
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

# Define sentinel data folder
s2_folder = os.path.join("data", "cc_composites", "S2_Annual")

# Define planet data folder
planet_folder = os.path.join("data", "cc_composites", "Planet_Feb")



############################################################################


# READ INPUT DATA


############################################################################
# Read landsat data
l8 = [os.path.join(l8_folder, file) for file in os.listdir(l8_folder) if \
      file.endswith('.tif')]

# Read sentinel data
s2 = [os.path.join(s2_folder, file) for file in os.listdir(s2_folder) if \
      file.endswith('.tif')]

# Read planet data
planet = [os.path.join(planet_folder, file) for file in os.listdir(planet_folder) \
          if file.endswith('.tif')]
    
# Read validation points
valpoints = gpd.read_file("data/validation/validation_points_geometry.shp")

# Define default image path (show it is in wdir)
def_img = "/assets/def_ts_small.png"



# %%

############################################################################


# CALCULATE INDICES


############################################################################
# Iterate over each landsat path
for path in l8:
    
    # Create empty lists to hold indice arrays
    ndvi_arrs = []
    ndmi_arrs = []
    
    # Read raster
    with rasterio.open(path) as rast:
        
        # Extract nir data
        nir = rast.read(5)
        
        # Extract red data
        red = rast.read(4)
        
        # Extract swir1 data
        swir1 = rast.read(6)
        
        # Calculate ndvi
        ndvi = (nir - red) / (nir + red)
        
        # Calculated ndmi
        ndmi = (nir - swir1) / (nir + swir1)
        
        # Add ndvi to list
        ndvi_arrs.append(ndvi)
        
        # Add ndmi to list
        ndmi_arrs.append(ndmi)




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

# Create frames for each validation point
val_frames = point_frame(valpoints, 500)

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
def landsat_plot(pntindex):
    
    # Extract relevant point
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    l8_clipped_arrs, l8_metas = clip_landsat(l8, frame)
    
    # Transpose clipped array to match format for imshow
    l8_clipped_rgb_arrs = [l8_arr.transpose(1, 2, 0) for l8_arr in l8_clipped_arrs]
    
    # Define labels for subplots
    l8_labels = list(years)
    
    # Initialize figure with 3x4 subplots
    l8_fig, l8_axs = plt.subplots(2, 6, figsize=(12, 4))
    
    # Flatten axes array
    l8_axs = l8_axs.flatten()
    
    # Iterate over axis, arrays, and metadata
    for l8_i, (l8_rgb_array, l8_meta) in enumerate(zip(l8_clipped_rgb_arrs, l8_metas)):
        
        # Display rgb image
        l8_axs[l8_i].imshow(l8_rgb_array)
        
        # Remove axis labels
        l8_axs[l8_i].axis('off')
        
        # Set subplot titles
        l8_axs[l8_i].set_title(f'{l8_labels[l8_i]}')
        
        # Extract specific transform for each year
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
        l8_axs[l8_i].add_patch(rect)

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
def sentinel_plot(pntindex):
    
    # Extract relevant point
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    s2_clipped_arrs, s2_metas = clip_sentinel(s2, frame)
    
    # Transpose clipped array to match format for imshow
    s2_clipped_rgb_arrs = [s2_arr.transpose(1, 2, 0) for s2_arr in s2_clipped_arrs]
    
    # Define labels for subplots
    s2_labels = list(years)[5:]
    
    # Initialize figure
    s2_fig, s2_axs = plt.subplots(2, 6, figsize=(12, 4))
    
    # Flatten axes array for easy iteration
    s2_axs = s2_axs.flatten()
    
    # Create a dictionary mapping year to its data for easier alignment
    s2_dic = {2018 + i: (arr, meta, label) for i, (arr, meta, label) in
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
            ax.set_title(f'{s2_label}')
            
        # If there is no s2 data for that year
        else:
            
            # Set title
            ax.set_title(f'{year}', fontsize=10)
         
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
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    pl_clipped_arrs, pl_metas = clip_planet(planet, frame, nodata_val)
    
    # Transpose clipped array to match format for imshow
    pl_clipped_rgb_arrs = [pl_arr.transpose(1, 2, 0) for pl_arr in pl_clipped_arrs]
    
    # Define labels for subplots
    pl_labels = list(years)
    
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




# %%
############################################################################


# MY DASHBOARD (NOO BUTTONS, PLOTS VISIBLE SIMULTANEOUSLY)


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
        html.Label("Enter Validation Point ID (0-505): ", style={
            "font-size": "18px", "font-family": "Arial"}),
        
        # Input format requirements
        dcc.Input(id="input-id", type="number", min=0, max=505, step=1, 
                  value=None, placeholder="Enter ID...", style={"margin-right": 
                                                                "10px"})
        
        # Style for input box
        ], style={"text-align": "center", "margin-top": "20px"}),

    # Output for displaying validation point info
    html.Div(id="output-div", style={"text-align": "center", "margin-top": "20px"}),
    
    # Visual divider
    html.Br(), 
    
    # Heading for landsat plotting
    html.H3("Landsat-8 Time Series", style={"font-family": "Arial", 
            "font-size": "24px", "color": "slategrey", "text-align": "center"}),
    
    # Timeseries plot for landsat
    html.Div([
        html.Img(
            id="landsat-plot",
            src="",  # Will be set by the callback
            style={"width": "100%", "height": "100%"}
        )
    ]),
    
    # Heading for sentinel plotting
    html.H3("Sentinel-2 Time Series", style={"font-family": "Arial", 
            "font-size": "24px", "color": "slategrey", "text-align": "center"}),
    
    # Timeseries plot for sentinel
    html.Div([
        html.Img(
            id="sentinel-plot",
            src="",  # Will be set by the callback
            style={"width": "100%", "height": "100%"}
        )
    ]),
    
    # Heading for planet plotting
    html.H3("RapidEye (2013-2016) + PlanetScope (2016-2023) Time Series", 
            style={"font-family": "Arial", "font-size": "24px", "color": 
                   "slategrey", "text-align": "center"}),
    
    # Timeseries plot for planet
    html.Div([
        html.Img(
            id="planet-plot",
            src="",  # Will be set by the callback
            style={"width": "100%", "height": "100%"}
        )
    ])
    
])

# Define callback for validation details
@app.callback(
    Output("output-div", "children"),
    Input("input-id", "value")
)

# Define function to extract validation point info
def display_validation_point(point_id):
    
    # If no input provided
    if point_id is None:
        return "Enter a point ID to begin validation."
    
    # Check if point_id exists in valpoints
    if point_id in valpoints.index:
        
        # Extract column data for identified valpoint
        point_data = valpoints.loc[point_id]
        
        # Extract point data
        x_coord = point_data.geometry.x
        y_coord = point_data.geometry.y
        
        return html.Div([
            html.P(f"Validation Point ID: {point_id}", style={"font-weight": "bold"}),
            html.P(f"Coordinates: ({x_coord}, {y_coord}), Strata: {point_data.strata}")
        ])
    
    else:
        return f"Validation Point ID {point_id} does not exist."

# Callback for landsat plotting
@app.callback(
    Output("landsat-plot", "src"),
    Input("input-id", "value")
)

# Define function to add landsat plots
def update_landsat(landsat_id):
    
    # Set default ID if no input
    if landsat_id is None:
        return def_img

    # Update plot
    l8html = landsat_plot(landsat_id)
    
    return l8html

# Callback for sentinel plotting
@app.callback(
    Output("sentinel-plot", "src"),
    Input("input-id", "value")
)

# Define function to add sentinel plots
def update_sentinel(sentinel_id):
    
    # Set default ID if no input
    if sentinel_id is None:
        return def_img
    
    # Create output html string
    s2html = sentinel_plot(sentinel_id)
    
    return s2html

# Callback for planet plotting
@app.callback(
    Output("planet-plot", "src"),
    Input("input-id", "value")
)

# Define function to add planet plots
def update_planet(planet_id):
    
    # Set default ID if no input
    if planet_id is None:
        return def_img

    # Create output html string
    plhtml = planet_plot(planet_id)
    
    return plhtml

# Run/update dashboard
if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    

# %%
############################################################################


# MY DASHBOARD (WITH BUTTONS, ONE PLOT AT A TIME)


############################################################################
# Initiate dashboard
app = Dash()

# Define dashboard layout
app.layout = html.Div([
    
    # Dashboard heading text
    html.H1("Sample-Based Deforestation Validation Dashboard", 
            
            # Heading formatting
            style={"font-family": "Arial", "font-size": "36px", 
                   "color": "darkslategrey", "text-align": "center"}),

    # Input box for validation point ID
    html.Div([
        
        # Input box label
        html.Label("Enter Validation Point ID (0-505): ", style={
            "font-size": "24px", "font-family": "Arial", "color": "slategrey",
            "font-weight": "bold"}),
        
        # Input format requirements
        dcc.Input(id="input-id", type="number", min=0, max=505, step=1, 
                  value=None, placeholder="Enter ID...", style={"font-size": "18px",
                      "font-family": "Arial", "margin-right": "10px"})
        
    ], style={"text-align": "center", "margin-top": "20px"}),

    # Output for displaying validation point info
    html.Div(id="output-div", style={"text-align": "center", "font-family": "Arial",
                                     "margin-top": "20px"}),

    # Visual divider
    html.Br(),
    
    # Radio buttons for plot selection
    html.Div([
        
        html.Div([
            
            # Create description label before buttons
            html.Label("Select Time Series Plot:", style={"font-size": "24px", 
                        "font-family": "Arial", "color": "slategrey", 
                        "margin-right": "20px", "font-weight": "bold"}),
            
            # Create radio buttons
            dcc.RadioItems(
            
                id="plot-selector",
                
                # Button options
                options=[
                    {"label": "Landsat-8 Time Series", "value": "landsat"},
                    {"label": "Sentinel-2 Time Series", "value": "sentinel"},
                    {"label": "RapidEye + PlanetScope Time Series", 
                     "value": "planet"}], 
                
                # Set default selection
                value="landsat", 
                
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
    
    # Dynamic image display
    html.Div([html.Img(id = "time-series-plot", src = def_img, style = 
                       {"width": "100%", "height": "100%"})])
])

# Define callback for validation point details
@app.callback(
    Output("output-div", "children"),
    Input("input-id", "value")
)

# Define function to display point details
def valpoint_details(point_id):
    if point_id is None:
        return "Enter a point ID to begin validation."

    if point_id in valpoints.index:
        point_data = valpoints.loc[point_id]
        x_coord = point_data.geometry.x
        y_coord = point_data.geometry.y
        return html.Div([
            html.P(f"Validation Point ID: {point_id}", style={"font-weight": "bold"}),
            html.P(f"Coordinates: ({x_coord}, {y_coord}), Strata: {point_data.strata}")
        ])
    else:
        return f"Validation Point ID {point_id} does not exist."

# Define callback for time-series plot selection
@app.callback(
    Output("time-series-plot", "src"),
    [Input("input-id", "value"), Input("plot-selector", "value")]
)

# Define function to display selected time series plot
def update_plot(point_id, plot_type):
    
    # If no point is given
    if point_id is None:
        
        # Print default image
        return def_img
    
    # If landsat is selected
    if plot_type == "landsat":
        
        # Plot landsat time series
        return landsat_plot(point_id)
    
    # If sentinel is selected
    elif plot_type == "sentinel":
        
        # Plot sentinel time series
        return sentinel_plot(point_id)
    
    # If planet is selected
    elif plot_type == "planet":
        
        # Plot planet time series
        return planet_plot(point_id)
    
    return ""

# Run/update dashboard
if __name__ == '__main__':
    app.run_server(debug=True)

    
