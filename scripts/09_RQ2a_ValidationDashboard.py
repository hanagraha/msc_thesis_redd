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
import plotly.express as px
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import box
from rasterio.mask import mask
from matplotlib.patches import Rectangle
import base64
import io
import dash



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



############################################################################


# SAMPLE DASH SETUP A


############################################################################
# visit: http://127.0.0.1:8050/
app = Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for your data.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run(debug=True)



############################################################################


# SAMPLE DASH SETUP B


############################################################################
app = Dash(__name__)

df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Dash_Introduction/intro_bees.csv")

df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
df.reset_index(inplace=True)
print(df[:5])

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2015", "value": 2015},
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018}],
                 multi=False,
                 value=2015,
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd)

    dff = df.copy()
    dff = dff[dff["Year"] == option_slctd]
    dff = dff[dff["Affected by"] == "Varroa_mites"]

    # Plotly Express
    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_dark'
    )

    return container, fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)


# %%
############################################################################


# MY DASHBOARD (CLEAN)


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
    
    # Visual divider
    html.Br(), 
    
    # Heading for sentinel plotting
    html.H3("Sentinel-2 Time Series", style={"font-family": "Arial", 
            "font-size": "24px", "color": "slategrey", "text-align": "center"}),
    
    # Visual divider
    html.Br(), 
    
    # Heading for planet plotting
    html.H3("RapidEye (2013-2016) + PlanetScope (2016-2023) Time Series", 
            style={"font-family": "Arial", "font-size": "24px", "color": 
                   "slategrey", "text-align": "center"})
    
])

# Define callback to handle input and display validation point info
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

# Run/update dashboard
if __name__ == '__main__':
    app.run_server(debug=True)
    
    

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

# Define function to clip validation data to each frame
def clip_raster(raster_pathlist, geom, nodata_value):
    
    # Create empty lists to hold clipped arrays and metadata
    clipped_arrs = []
    metadata = []
    
    # Iterate over rasters
    for file in raster_pathlist:

        # Read raster
        with rasterio.open(file) as rast:
            
            # Only process the first three bands (RGB)
            indices = [1,2,3]
            
            # Mask pixels outside aoi with NoData values
            raster_clip, out_transform = mask(rast, geom, crop = True, 
                                              nodata = nodata_value, 
                                              indexes = indices)
        
            # Copy metadata
            out_meta = rast.meta.copy()
            
            # Update metadata
            out_meta.update({
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': len(indices),
                'height': raster_clip.shape[1],
                'width': raster_clip.shape[2],
                'transform': out_transform,
                'nodata': nodata_value})
        
        # Add clipped array to list
        clipped_arrs.append(raster_clip)
        
        # Add metadata to list
        metadata.append(out_meta)
        
    return clipped_arrs, metadata

# Define function to plot frames based on point
def planet_plot(pntindex):
    
    # Extract relevant point
    point = val_frames["geometry"][pntindex]
    
    # Extract relevant frame
    frame = [val_frames["frame"][pntindex]]
    
    # Clip validation data to frame and retrieve list of metadata
    clipped_arrs, metas = clip_raster(planet, frame, nodata_val)
    
    # Transpose clipped array to match format for imshow
    clipped_rgb_arrs = [arr.transpose(1, 2, 0) for arr in clipped_arrs]
    
    # Define labels for subplots
    labels = list(years)
    
    # Initialize figure with 3x4 subplots
    fig, axs = plt.subplots(1, 12, figsize=(12, 2))
    
    # Flatten axes array
    axs = axs.flatten()
    
    # Iterate over axis, arrays, and metadata
    for i, (rgb_array, meta) in enumerate(zip(clipped_rgb_arrs, metas)):
        
        # Display rgb image
        axs[i].imshow(rgb_array)
        
        # Remove axis labels
        axs[i].axis('off')
        
        # Set subplot titles
        axs[i].set_title(f'{labels[i]}')
        
        # Extract specific transform for each year
        transform = meta['transform']
        
        # Calculate pixel size in pixels for each raster
        pxsize_px = 30 / abs(transform.a)
        
        # Convert xy coordinate to image coordinates
        px, py = ~transform * (point.x, point.y)
        
        # Create pixel rectangle
        rect = Rectangle(
            (px - pxsize_px / 2, py - pxsize_px / 2),
            pxsize_px, pxsize_px, linewidth=1, edgecolor='red', facecolor='none'
        )
        
        # Overlay validation pixel area
        axs[i].add_patch(rect)
        
        # Remove empty subplot axes
        for j in range(len(clipped_rgb_arrs), len(axs)):
            axs[j].axis('off')

    # Show plot
    # plt.tight_layout()
    # plt.show()
    
    return fig

# Create frames for each validation point
val_frames = point_frame(valpoints, 500)



# %%
############################################################################


# MY DASHBOARD (TESTING)


############################################################################
# Initialize the dashboard
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

    ], style={"text-align": "center", "margin-top": "20px"}),

    # Output for displaying validation point info
    html.Div(id="output-div", style={"text-align": "center", "margin-top": "20px"}),

    # Visual divider
    html.Br(), 
    
    # Heading for planet plotting
    html.H3("RapidEye (2013-2016) + PlanetScope (2016-2023) Time Series", 
            style={"font-family": "Arial", "font-size": "24px", "color": 
                   "slategrey", "text-align": "center"}),

    # Embed the Matplotlib plot
    html.Img(
        src="data:image/png;base64," + update_plot(),
        style={"width": "60%", "height": "60%"}
    )

])
    
# Callback to dynamically handle user input and update the plot
@app.callback(
    Output("planet-plot", "src"),  # Update the image source dynamically
    Input("input-id", "value")
)

def update_plot(point_id):
    
    planet_plot(point_id)
    
    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    
    # Convert to base64 string
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    return encoded_image
    
# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)

# %%

# Initialize the Dash app
app = dash.Dash(__name__)


# Function to create and encode a matplotlib plot
def create_plot():

    # Create plots for planet    
    planet_plot(7)
    
    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    
    # Convert to base64 string
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    return encoded_image


# Layout of the Dash dashboard
app.layout = html.Div([
    html.H1("Matplotlib Plot in Dash Example"),
    
    # Embed the Matplotlib plot
    html.Img(
        src="data:image/png;base64," + create_plot(),
        style={"width": "60%", "height": "60%"}
    )
])


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)





# %%

# Initialize the Dash app
app = dash.Dash(__name__)


# Function to create and encode a matplotlib plot
def create_plot(val_id):

    # Create plots for planet    
    planet_plot(7)
    
    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    
    # Convert to base64 string
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    return encoded_image


# Layout of the Dash dashboard
app.layout = html.Div([
    html.H1("Dynamic Matplotlib Plot with Dash Example"),

    # Input box for validation point ID
    html.Div([
        html.Label(
            "Enter Validation Point ID (0-505): ",
            style={"font-size": "18px", "font-family": "Arial"}
        ),
        dcc.Input(
            id="input-id",
            type="number",
            min=0,
            max=505,
            step=1,
            value=1,  # Default value for initialization
            placeholder="Enter ID...",
            style={"margin-right": "10px"}
        )
    ], style={"text-align": "center", "margin-top": "20px"}),

    # Embed the dynamically updated Matplotlib plot
    html.Div([
        html.Img(
            id="dynamic-plot",
            src="",  # Will be set by the callback
            style={"width": "100%", "height": "100%"}
        )
    ])
])


# Define the callback logic to dynamically update the image
@app.callback(
    Output("dynamic-plot", "src"),
    Input("input-id", "value")
)

def update_plot(validation_id):
    """
    Callback to dynamically render the plot based on user input.
    """
    if validation_id is None:
        validation_id = 1  # Set a default value if input is None
    # Generate the plot with the current user input
    return f"data:image/png;base64,{create_plot(validation_id)}"


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)

    
    
    
