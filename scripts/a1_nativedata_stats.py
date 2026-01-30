# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 15:32:49 2026

@author: hanna

This file plots runs extra analysis to compare GFC and TMF datasets before resampling. Attempting analysis on 
annual deforestation vs degradation year area, validation of transition map and annual change maps.

Expected runtime XX min
"""

# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import rasterio
import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterstats import zonal_stats

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"Z:\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# DEFINE CONSTANTS
# -------------------------------------------------------------------------
# Define annual change classes
tmf_annualchange_dict = {
    1: 'Undisturbed TMF',
    2: 'Degraded TMF',
    3: 'Deforested TMF',
    4: 'TMF Regrowth',
    5: 'Water',
    6: 'Other',
    0: 'No data'
}

# Define main transition map classes
tmf_maintrans_dict = {
    10: 'Undisturbed TMF',
    20: 'Degraded TMF', 
    30: 'TMF Regrowth',
    41: 'Deforested - Tree Plantations',
    42: 'Deforested - Water',  
    43: 'Deforested - Other',
    50: 'Ongoing Deforestation/Degradation',
    60: 'Water',
    70: 'Other'
}

# Define years
years = list(range(2013, 2024))

# Define color palatte 
bluecols = ['brown', "#1E2A5E", "#83B4FF"]


# -------------------------------------------------------------------------
# GFC DATA
# -------------------------------------------------------------------------
# Read gfc tree cover 2000
with rasterio.open('temp/gfc_treecover2000_reprojected_clipped.tif') as rast:     
    gfc_tc2000 = rast.read(1)

# Read gfc lossyear
with rasterio.open('temp/gfc_lossyear_reprojected_clipped.tif') as rast:     
    gfc_lossyear = rast.read(1)
    gfc_profile = rast.profile
    gfc_res = rast.res

# Calculate pixel area (ha)
gfc_pixarea = gfc_res[0] * gfc_res[1] / 10000


# -------------------------------------------------------------------------
# TMF DATA
# -------------------------------------------------------------------------
# Read tmf deforestation year
with rasterio.open('temp/tmf_DeforestationYear_reprojected_clipped.tif') as rast:     
    tmf_deforyear = rast.read(1)
    tmf_profile = rast.profile
    tmf_res = rast.res

# Calculate pixel area (ha)
tmf_pixarea = tmf_res[0] * tmf_res[1] / 10000

# Read tmf degradation year
with rasterio.open('temp/tmf_DegradationYear_reprojected_clipped.tif') as rast:     
    tmf_degrayear = rast.read(1)

# Read tmf transition map
with rasterio.open('temp/tmf_TransitionMap_MainClasses_reprojected_clipped.tif') as rast:     
    tmf_transition = rast.read(1)

# Define annual change paths
tmf_acpaths = sorted(f"temp/{file}" for file in os.listdir('temp')
    if file.startswith("tmf_AnnualChange") and file.endswith("_clipped.tif"))

# Initialize empty list to hold annual change data
tmf_annualchange = []

# Iterate over each path and read raster data
for path in tmf_acpaths:
    with rasterio.open(path) as rast:
        tmf_annualchange.append(rast.read(1))


# -------------------------------------------------------------------------
# EXTRACT SUMMARY COUNTS (ANNUAL)
# -------------------------------------------------------------------------
# Define function to get count summaries
def count_summary(raster, valuename = 'year'):

    # Filter out nodata values
    raster = raster[raster != 255]

    # Get unique values and counts
    vals, counts = np.unique(raster, return_counts=True)

    # Create dataframe 
    df = pd.DataFrame({valuename: vals, 'counts': counts})

    return df

# Extract disturbance counts
gfc_lossyear_counts = count_summary(gfc_lossyear)
tmf_deforyear_counts = count_summary(tmf_deforyear)
tmf_degrayear_counts = count_summary(tmf_degrayear)

# Extract annual change counts
tmf_annualchange_counts = []
for changeraster in tmf_annualchange:
    annual_summary = count_summary(changeraster, valuename = 'class')
    annual_summary.index = annual_summary['class'].map(tmf_annualchange_dict)
    tmf_annualchange_counts.append(annual_summary)

# Extract transition map counts
tmf_transition_counts = count_summary(tmf_transition, valuename = 'class')


# -------------------------------------------------------------------------
# COMPARE TOTAL AREAS
# -------------------------------------------------------------------------
# Create dataframe of total areas
area_comparison = pd.DataFrame({
    'dataset': [
        'GFC lossyear',
        'TMF deforestation year',
        'TMF degradation year',
        'TMF transition',
        'TMF annual change'
    ],
    'pixels': [
        gfc_lossyear_counts['counts'].sum(),
        tmf_deforyear_counts['counts'].sum(),
        tmf_degrayear_counts['counts'].sum(),
        tmf_transition_counts['counts'].sum(),
        tmf_transition_counts['counts'].sum()
    ],
    'area_ha': [
        gfc_lossyear_counts['counts'].sum() * gfc_pixarea,
        tmf_deforyear_counts['counts'].sum() * tmf_pixarea,
        tmf_degrayear_counts['counts'].sum() * tmf_pixarea,
        tmf_transition_counts['counts'].sum() * tmf_pixarea,
        tmf_transition_counts['counts'].sum() * tmf_pixarea
    ]   
})


# -------------------------------------------------------------------------
# CALCULATE PROPORTIONAL DISTURBANCE AREAS (AOI)
# -------------------------------------------------------------------------
# Calculate proportaional deforestation area
gfc_loss_aoi = gfc_lossyear_counts.copy()
gfc_loss_aoi['area_ha'] = gfc_loss_aoi['counts'] * gfc_pixarea
gfc_loss_aoi['prop_dist'] = gfc_loss_aoi['area_ha'] / gfc_loss_aoi['area_ha'].sum()

tmf_defor_aoi = tmf_deforyear_counts.copy()
tmf_defor_aoi['area_ha'] = tmf_defor_aoi['counts'] * tmf_pixarea
tmf_defor_aoi['prop_dist'] = tmf_defor_aoi['area_ha'] / tmf_defor_aoi['area_ha'].sum()

tmf_degra_aoi = tmf_degrayear_counts.copy()
tmf_degra_aoi['area_ha'] = tmf_degra_aoi['counts'] * tmf_pixarea
tmf_degra_aoi['prop_dist'] = tmf_degra_aoi['area_ha'] / tmf_degra_aoi['area_ha'].sum()

# Filter for years 2013-2023
gfc_loss_aoi = gfc_loss_aoi[gfc_loss_aoi['year'].between(2013, 2023)].reset_index(drop=True)
tmf_defor_aoi = tmf_defor_aoi[tmf_defor_aoi['year'].between(2013, 2023)].reset_index(drop=True)
tmf_degra_aoi = tmf_degra_aoi[tmf_degra_aoi['year'].between(2013, 2023)].reset_index(drop=True)

# Add tmf disturbances
tmf_defordegra_aoi = pd.DataFrame({
    'year': years,
    'prop_dist': tmf_defor_aoi['prop_dist'] + tmf_degra_aoi['prop_dist']
})

# Calculate total annual change map area
ac_totalpix = tmf_annualchange_counts[0][1:]['counts'].sum()

# Create empty list to hold annual data
annual_data = []

# Iterate over each annual change dataframe and year
for df, year in zip(tmf_annualchange_counts, range(2012, 2024)):

    # Calculate proportional change areas
    annual_data.append({
        'year': year,
        'prop_defor': df.loc['Deforested TMF', 'counts'] / ac_totalpix,
        'prop_degr': df.loc['Degraded TMF', 'counts'] / ac_totalpix,
        'prop_dist': (df.loc['Deforested TMF', 'counts'] + df.loc['Degraded TMF', 'counts']) / ac_totalpix
    })

# Convert to dataframe
tmf_annualchange_propdist = pd.DataFrame(annual_data)


# -------------------------------------------------------------------------
# CALCULATE ANNUAL CHANGE TRANSITIONS
# -------------------------------------------------------------------------
# Extract raster shape
shape = tmf_annualchange[0].shape

# Create empty arrays to hold change year
first_changeyear  = np.zeros(shape, dtype=np.int16)
second_changeyear = np.zeros(shape, dtype=np.int16)
third_changeyear  = np.zeros(shape, dtype=np.int16)

# Create empty arrays to hold transitions
first_trans  = np.zeros(shape, dtype=np.int16)
second_trans = np.zeros(shape, dtype=np.int16)
third_trans  = np.zeros(shape, dtype=np.int16)

# Create empty array to hold change counts
change_count = np.zeros(shape, dtype=np.uint8)

# Define years for annual change analysis
annyears = list(range(2012, 2024))

# Iterate over each year
for i in range(len(tmf_annualchange) - 1):

    # Extract array for prev. and current year
    arr_t  = tmf_annualchange[i]
    arr_t1 = tmf_annualchange[i + 1]

    # Define year
    year = annyears[i + 1]

    # Filter no data values
    valid = (arr_t != 255) & (arr_t1 != 255)

    # Create change mask
    changed = (arr_t != arr_t1) & valid

    # Track change year
    first = changed & (change_count == 0)
    second = changed & (change_count == 1)
    third = changed & (change_count == 2)

    # Add change year to pixel
    first_changeyear[first] = year
    second_changeyear[second] = year
    third_changeyear[third] = year

    # Track transition code
    first_trans[first]   = arr_t[first]*10 + arr_t1[first]
    second_trans[second] = arr_t[second]*10 + arr_t1[second]
    third_trans[third]   = arr_t[third]*10 + arr_t1[third]

    # Update change counter
    change_count[changed] += 1

# Filter first change year by transition
first_changeyear_defor = np.where(np.isin(first_trans, [13, 23, 43]),
    first_changeyear, 255).copy()
first_changeyear_degra = np.where(first_trans == 12, first_changeyear, 255).copy()

# Filter second change year by transition
second_changeyear_defor = np.where(np.isin(second_trans, [13, 23, 43]),
    second_changeyear, 255).copy()
second_changeyear_degra = np.where(second_trans == 12, second_changeyear, 255).copy()


# -------------------------------------------------------------------------
# SUMMARIZE TRANSITION COUNTS
# -------------------------------------------------------------------------
# Define function to create annual transition dataframes
def transition_dataframe(changeyear, transition):

    # Create valid mask
    valid = (changeyear != 0) & (transition != 0) & (changeyear != 255) & (transition != 255)

    # Create raveled dataframe
    annual_transition = pd.DataFrame({
        'year': changeyear[valid].ravel(),
        'transition': transition[valid].ravel()   
    }) 

    # Group by year and transition
    annual_transition_summary = annual_transition.groupby(['year', 'transition']) \
        .size().unstack(fill_value=0).reset_index()
    
    # Remove column name
    annual_transition_summary.columns.name = None
    annual_transition_summary.set_index('year', inplace=True)

    # Create copy 
    annual_transition_prop = annual_transition_summary.copy()

    # Calculate proportional areas
    for col in annual_transition_prop.columns:
        annual_transition_prop[col] = annual_transition_prop[col] / ac_totalpix

    return annual_transition_summary, annual_transition_prop

# Create transition dataframes
ac_first_transition, ac_first_transition_prop = transition_dataframe(first_changeyear, first_trans)
ac_second_transition, ac_second_transition_prop = transition_dataframe(second_changeyear, second_trans)
ac_third_transition, ac_third_transition_prop = transition_dataframe(third_changeyear, third_trans)

# Add all transitions (disturbances will cumulate! double counting)
ac_all_transitions_prop = ac_first_transition_prop.add(ac_second_transition_prop, fill_value=0) \
    .add(ac_third_transition_prop, fill_value=0)


# -------------------------------------------------------------------------
# PLOT DISTURBANCE (LOSSYEAR + DEFORDEGRA YEAR)
# -------------------------------------------------------------------------
# Initialize figure
plt.figure(figsize=(10, 6))

# Add gfc deforestation line
plt.plot(years, gfc_loss_aoi['prop_dist']*100, color=bluecols[0], linewidth = 2,
         label='GFC Lossyear')

# Add tmf deforestation line
plt.plot(years, tmf_defor_aoi['prop_dist']*100, color=bluecols[1], linewidth = 2,
         label='TMF Deforestation Year')

# Add tmf deforestation + degradation line
plt.plot(years, tmf_defordegra_aoi['prop_dist']*100, color=bluecols[2], linewidth = 2,
         label='TMF Deforestation + Degradation Year')

# Add labels and title
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Proportional Disturbance Area (%)', fontsize = 17)

# Add x tickmarks
plt.xticks(years, fontsize = 16)

# Edit y tickmark fontsize
plt.yticks(fontsize = 16)

# Add legend
plt.legend(fontsize = 16, loc="upper right")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Tight layout
plt.tight_layout()

# Save plot
plt.savefig("figs/gfc_tmf_native_comparison", dpi=300, bbox_inches='tight', transparent=False)

# Show plot
plt.show()


# -------------------------------------------------------------------------
# PLOT DISTURBANCE (ANNUAL CHANGE FIRST TRANSITION)
# -------------------------------------------------------------------------
# Transition classes to plot (first transition)
a13 = (ac_first_transition_prop[13] + ac_first_transition_prop[43]) * 100 #deforestation
a12 = ac_first_transition_prop[12] * 100 #degradation
a23 = (ac_first_transition_prop[23]) * 100 #degradation to deforestation

# Transition classes to plot (all transitions)
c13 = (ac_all_transitions_prop[13] + ac_all_transitions_prop[43]) * 100 #deforestation
c12 = ac_all_transitions_prop[12] * 100 #degradation
c23 = (ac_all_transitions_prop[23]) * 100 #degradation to deforestation

# Cumulative stacks (first transition)
y1 = a13
y2 = a13 + a23
y3 = a13 + a23 + a12

# Cumulative stacks (all transitions)
y1_all = c13
y2_all = c13 + c23
y3_all = c13 + c23 + c12

# Initialize figure
plt.figure(figsize=(10, 6))

# Bottom layer
plt.fill_between(years, 0, y1, facecolor='#8e1014', alpha=0.4, hatch = '//', edgecolor = '#8e1014', linewidth=0)
plt.plot(years, y1, color = '#8e1014', label='Direct Deforestation', linewidth=2)

# Middle layer
plt.fill_between(years, y1, y2, facecolor='#e67469', alpha=0.4, hatch = '//', edgecolor = '#e67469', linewidth=0)
plt.plot(years, y2, color = '#cc000a', label='Total Deforestation (incl. after Degradation)', linewidth=2)

# Top layer
plt.fill_between(years, y2, y3, facecolor='#e2c94b', alpha=0.2, hatch = '//', edgecolor = '#e2c94b', linewidth=0)
plt.plot(years, y3, color = '#e2c94b', label='Degraded TMF', linewidth=2)

# Add all transitions lines
plt.plot(years, y1_all, color = '#8e1014', linestyle='--', linewidth=1.5)
plt.plot(years, y2_all, color = '#cc000a', linestyle='--', linewidth=1.5)
plt.plot(years, y3_all, color = '#e2c94b', linestyle='--', linewidth=1.5)

# Add labels and title
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Proportional Disturbance Area (%)', fontsize = 17)

# Add x tickmarks
plt.xticks(years, fontsize = 16)

# Edit y tickmark fontsize
plt.yticks(fontsize = 16)

# Create custom legend
legend_elements = [
    Line2D([0], [0], color='#8e1014', lw=2, label='Direct Deforestation'),
    Line2D([0], [0], color='#cc000a', lw=2, label='Total Deforestation'),
    Line2D([0], [0], color='#e2c94b', lw=2, label='Degraded TMF'),
    Line2D([0], [0], color='#25343F', lw=2, label='First Transition (solid)'),
    Line2D([0], [0], color='#25343F', lw=2, linestyle='--', label='All Transitions (dashed)')
]

# Add legend
plt.legend(handles=legend_elements, fontsize=16, loc='upper right')

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Tight layout
plt.tight_layout()

# Save plot
plt.savefig("figs/tmf_annualchange", dpi=300, bbox_inches='tight', transparent=False)

# Show plot
plt.show()


# -------------------------------------------------------------------------
# EXPORT NATIVE DATASETS
# -------------------------------------------------------------------------
# Define function to export raster
def export_raster(data, profile, filename):

    # Define filepath
    filepath = f"native_validation/{filename}.tif"

    # Write the data to a new raster file
    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(data, 1)

    # Print confirmation message
    print(f"Export complete for {filepath}")

# GFC predictions (lossyear)
export_raster(gfc_lossyear, gfc_profile, "gfc_lossyear_native")

# TMF predictions
export_raster(tmf_deforyear, tmf_profile, "tmf_deforyear_native")
export_raster(tmf_degrayear, tmf_profile, "tmf_degrayear_native")
export_raster(tmf_transition, tmf_profile, "tmf_transition_native")
export_raster(first_changeyear_defor, tmf_profile, "tmfac_firstdeforyear_native")
export_raster(second_changeyear_defor, tmf_profile, "tmfac_seconddeforyear_native")
export_raster(first_changeyear_degra, tmf_profile, "tmfac_firstdegrayear_native")
export_raster(second_changeyear_degra, tmf_profile, "tmfac_seconddegrayear_native")