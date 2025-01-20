# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:34:35 2024

@author: hanna
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from scipy.stats import chi2_contingency
from rasterio.mask import mask
from statsmodels.stats.contingency_tables import mcnemar 
from sklearn.metrics import cohen_kappa_score



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

# Set output directory
out_dir = os.path.join(os.getcwd(), 'data', 'intermediate')

# Define study range years
years = range(2013, 2024)

# Set default plotting colors
defaultblue = "#4682B4"
reddcol = "brown"
nonreddcol = "dodgerblue"
grnpcol = "darkgreen"

reddcol = "#820300"  # Darker Red
grnpcol = "#4682B4"  # Darker Blue - lighter

############################################################################


# IMPORT AND READ DATA


############################################################################
# Define function to read list of files
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

# Define spatial agreement paths
spatagree_paths = [f"data/intermediate/agreement_gfc_combtmf_{year}.tif" for 
                   year in years]

# Define gfc lossyear paths
gfc_paths = [f"data/hansen_preprocessed/gfc_lossyear_fm_{year}.tif" for year 
             in years]

# Define tmf defordegra paths
tmf_paths = [f"data/jrc_preprocessed/tmf_defordegrayear_fm_{year}.tif" for 
             year in years]

# Define simple binary gfc paths
gfc_simp_paths = [f"data/intermediate/gfc_simple_binary_{year}.tif" for 
                  year in years]

# Define simple binary gfc paths
tmf_simp_paths = [f"data/intermediate/tmf_simple_binary_{year}.tif" for 
                  year in years]

# Read spatial agreement rasters
spatagree_arrs, spatagree_profile = read_files(spatagree_paths)

# Read gfc paths
gfc_arrs, gfc_profile = read_files(gfc_paths)

# Read tmf paths
tmf_arrs, tmf_profile = read_files(tmf_paths)

# Read gfc simple binary
gfc_simp_arrs, gfc_simp_profile = read_files(gfc_simp_paths)

# Read tmf simple binary
tmf_simp_arrs, tmf_simp_profile = read_files(tmf_simp_paths)

# Read vector data
villages = gpd.read_file("data/village polygons/village_polygons.shp")
grnp = gpd.read_file("data/gola gazetted polygon/Gola_Gazetted_Polygon.shp")

# Create REDD+ and non-REDD+ polygons
villages = villages[['grnp_4k', 'geometry']].dissolve(by='grnp_4k')

# Create REDD+ and non-REDD+ geometries
redd_geom = [villages.loc[1, 'geometry']]
nonredd_geom = [villages.loc[0, 'geometry']]

# Create GRNP geometry
grnp_geom = grnp.geometry



############################################################################


# SPLIT DATA INTO REDD+, NON-REDD+, GRNP


############################################################################
# Define function for clipping stack of agreement rasters
def regions_clip(pathlist):
    
    # Create empty list to hold arrays
    redd_arrs = []
    nonredd_arrs = []
    grnp_arrs = []
    
    # Iterate over each filepath
    for path in pathlist:
        
        # Read raster data
        with rasterio.open(path) as rast:
            
            # Mask + crop data to redd area
            redd_arr, transform = mask(rast, redd_geom, crop = True, 
                                       nodata = nodata_val)
            
            # Mask + crop data to nonredd area
            nonredd_arr, transform = mask(rast, nonredd_geom, crop = True, 
                                          nodata = nodata_val)
            
            # Mask + crop data to redd area
            grnp_arr, transform = mask(rast, grnp_geom, crop = True, 
                                       nodata = nodata_val)
            
        # Add array to list
        redd_arrs.append(redd_arr)
        nonredd_arrs.append(nonredd_arr)
        grnp_arrs.append(grnp_arr)

    return redd_arrs, nonredd_arrs, grnp_arrs

# Clip agreement to redd+, nonredd+, and grnp area
ag_redd, ag_nonredd, ag_grnp = regions_clip(spatagree_paths)

# Clip gfc arrays to redd+, nonredd+, and grnp area
gfc_redd, gfc_nonredd, gfc_grnp = regions_clip(gfc_paths)

# Clip tmf arrays to redd+, nonredd+, and grnp area
tmf_redd, tmf_nonredd, tmf_grnp = regions_clip(tmf_paths)

# Clip gfc simple arrays to redd+, nonredd+, and grnp area
gfc_simp_redd, gfc_simp_nonredd, gfc_simp_grnp = regions_clip(gfc_simp_paths)

# Clip gfc simple arrays to redd+, nonredd+, and grnp area
tmf_simp_redd, tmf_simp_nonredd, tmf_simp_grnp = regions_clip(tmf_simp_paths)



############################################################################


# PLOTTING OPTIONS


############################################################################
# Define function to plot line graph
def lineplot(ydata, title, ylab, yaxis, lim_low, lim_up):
    
    # Initialize figure
    plt.figure(figsize = (10,6))
    
    # Plot line data
    plt.plot(years, ydata, label = ylab, color = defaultblue)
    
    # Add title
    plt.title(title)
    
    # Add axes labels
    plt.xlabel("Year")
    plt.ylabel(yaxis)
    
    # Add gridlines
    plt.grid(True, linestyle = "--")
    
    # Add legend
    plt.legend(loc='best')
    
    # Add x tickmarks
    plt.xticks(years, rotation=45)
    
    # Adjust yaxis limits
    plt.ylim(lim_low, lim_up)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Define function to plot line graph with redd, nonredd, and grnp data
def tripleplot(redd, nonredd, grnp, title, yaxis, lim_low, lim_up):
    
    # Initialize figure
    plt.figure(figsize = (6.9, 4.5))
    
    # Plot redd+ data
    plt.plot(years, redd, label = "REDD+", color = reddcol)
    
    # Plot nonredd data
    plt.plot(years, nonredd, label = "Non-REDD+", color = reddcol, 
             linestyle = "--")
    
    # Plot grnp data
    plt.plot(years, grnp, label = "GRNP", color = grnpcol)
    
    # Add title
    plt.title(title)
    
    # Add axes labels
    plt.xlabel("Year")
    plt.ylabel(yaxis)
    
    # Add gridlines
    plt.grid(True, linestyle = "--")
    
    # Add legend
    plt.legend(loc='best')
    
    # Add x tickmarks
    plt.xticks(years)
    
    # Adjust yaxis limits
    plt.ylim(lim_low, lim_up)
    
    # Show the plot
    plt.tight_layout()
    plt.show()



############################################################################


# OVERALL AGREEMENT


############################################################################
# Define function to calculate overall agreement
def ov_agree(spatagree_arrs):
    
    # Create empty list to hold statistics
    prop_agree = []
    
    # Iterate over each array
    for arr in spatagree_arrs:
        
        # Calculate total agreement
        ag = (np.sum(arr == 5)) + (np.sum(arr == 8))
        
        # Calculate total disagreement
        disag = (np.sum(arr == 6)) + (np.sum(arr == 7))
        
        # Calculate overall agreement
        ov_ag = (ag / (ag + disag))*100
        
        # Add overall agreement to list
        prop_agree.append(ov_ag)
        
    return prop_agree
    
# Calculate overall agreement
ov_ag_aoi = ov_agree(spatagree_arrs)

# Calculate overall agreement for redd
ov_ag_redd = ov_agree(ag_redd)

# Calculate overall agreement for nonredd area
ov_ag_nonredd = ov_agree(ag_nonredd)

# Calculate overall agreement for grnp area
ov_ag_grnp = ov_agree(ag_grnp)

# Plot overall agreement for aoi
lineplot(ov_ag_aoi, "Overall Spatial Agreement between GFC and TMF Datasets", 
         "AOI", "Overall Agreement (%)", 95, 98)

# Plot overall agreement for redd, nonredd, and grnp area
tripleplot(ov_ag_redd, ov_ag_nonredd, ov_ag_grnp, 
           "Overall Spatial Agreement between GFC and TMF Datasets", 
           "Overall Agreement (%)", 92, 100)



############################################################################


# UNDISTURBED AGREEMENT


############################################################################
# Define function to calculate overall agreement
def for_agree(spatagree_arrs):
    
    # Create empty list to hold statistics
    prop_agree = []
    
    # Iterate over each array
    for arr in spatagree_arrs:
        
        # Calculate total agreement
        ag = (np.sum(arr == 5))
        
        # Calculate total disagreement
        disag = (np.sum(arr == 6)) + (np.sum(arr == 7))
        
        # Calculate overall agreement
        ov_ag = (ag / (ag + disag))*100
        
        # Add overall agreement to list
        prop_agree.append(ov_ag)
        
    return prop_agree
    
# Calculate overall agreement
for_ag_aoi = for_agree(spatagree_arrs)

# Calculate overall agreement for redd
for_ag_redd = for_agree(ag_redd)

# Calculate overall agreement for nonredd area
for_ag_nonredd = for_agree(ag_nonredd)

# Calculate overall agreement for grnp area
for_ag_grnp = for_agree(ag_grnp)

# Plot overall agreement for aoi
lineplot(for_ag_aoi, "Undisturbed Spatial Agreement between GFC and TMF Datasets", 
         "AOI", "Undisturbed Agreement (%)", 95, 98)

# Plot overall agreement for redd, nonredd, and grnp area
tripleplot(for_ag_redd, for_ag_nonredd, for_ag_grnp, 
           "Spatial Agreement Relative to Undisturbed Area", 
           "Undisturbed Agreement (%)", 92, 100)



############################################################################


# DEFORESTATION AGREEMENT


############################################################################
# Define function to calculate deforestation agreement
def defor_agree(spatagree_arrs):
    
    # Create empty list to hold statistics
    prop_agree = []
    
    # Iterate over each array
    for arr in spatagree_arrs:
        
        # Calculate total agreement
        ag = np.sum(arr == 8)
        
        # Calculate total disagreement
        disag = (np.sum(arr == 6)) + (np.sum(arr == 7))
        
        # Calculate overall agreement
        ov_ag = (ag / (ag + disag))*100
        
        # Add overall agreement to list
        prop_agree.append(ov_ag)
        
    return prop_agree

# Calculate deforestation agreement for aoi
defor_ag_aoi = defor_agree(spatagree_arrs)

# Calculate deforestation agreement for redd
defor_ag_redd = defor_agree(ag_redd)

# Calculate deforestation agreement for nonredd area
defor_ag_nonredd = defor_agree(ag_nonredd)

# Calculate deforestation agreement for grnp area
defor_ag_grnp = defor_agree(ag_grnp)

# Plot deforestation agreement for aoi
lineplot(defor_ag_aoi, "Deforestation Spatial Agreement between GFC and TMF Datasets", 
         "AOI", "Deforestation Agreement (%)", 10, 30)

# Plot deforestation agreement for redd, nonredd, and grnp area
tripleplot(defor_ag_redd, defor_ag_nonredd, defor_ag_grnp, 
           "Spatial Agreement Relative to Deforestation Area", 
           "Deforestation Agreement (%)", 0, 35)



############################################################################


# PLOT SIDE BY SIDE: OVERALL AND DEFORESTATION AGREEMENT


############################################################################
# %%
# Initialize figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Plot 1: overall spatial agreement
axes[0].plot(years, ov_ag_redd, color=reddcol, linewidth=2,
             label='REDD+')
axes[0].plot(years, ov_ag_nonredd, color=reddcol, linewidth=2,
             label='non-REDD', linestyle = "--")
axes[0].plot(years, ov_ag_grnp, color=grnpcol, linewidth=2, 
             label='GRNP')

# Add x axis label
axes[0].set_xlabel('Year', fontsize=12)

# Add y axis label
axes[0].set_ylabel('Overall Agreement (%)', fontsize=12)

# Add tickmarks
axes[0].set_xticks(years)
axes[0].tick_params(axis='both', labelsize=11)

# Add legend
axes[0].legend(fontsize=11)

# Add gridlines
axes[0].grid(linestyle="--", alpha=0.6)

# Adjust yaxis limits
axes[0].set_ylim(92, 100)

# Plot 2: deforestation spatial agreement
axes[1].plot(years, defor_ag_redd, label = "REDD+", color = reddcol)
axes[1].plot(years, defor_ag_nonredd, label = "Non-REDD+", color = reddcol, 
             linestyle = "--")
axes[1].plot(years, defor_ag_grnp, label = "GRNP", color = grnpcol)

# Add tickmarks
axes[1].set_xticks(years)
axes[1].tick_params(axis='both', labelsize=11)

# Add x axis label
axes[1].set_xlabel('Year', fontsize=12)

# Add y axis label
axes[1].set_ylabel('Deforestation Agreement (%)', fontsize=12)

# Add gridlines
axes[1].grid(True, linestyle = "--")

# Add legend
axes[1].legend(fontsize=11)

# Adjust yaxis limits
axes[1].set_ylim(0, 35)

# Show plot
plt.tight_layout()
plt.show()


# %%
############################################################################


# MATTHEW'S COEFFICIENT


############################################################################
# Define function to calculate matthews coefficient
def matt_coef(arrlist1, arrlist2):
    
    # Create empty list to hold coefficients
    coefs = []
    
    # Iterate over each array
    for arr1, arr2 in zip (arrlist1, arrlist2):
        
        # Calculate matthews coefficient
        matcoef = matthews_corrcoef(arr1.flatten(), arr2.flatten())
        
        # Add coefficient to list
        coefs.append(matcoef)
        
    return coefs

# Calculate matthews coefficient for aoi
matcoef_aoi = matt_coef(gfc_arrs, tmf_arrs)

# Calculate matthews coefficient for redd area
mattcoef_redd = matt_coef(gfc_redd, tmf_redd)

# Calculate matthews coefficient for nonredd area
mattcoef_nonredd = matt_coef(gfc_nonredd, tmf_nonredd)

# Calculate matthews coefficient for grnp area
mattcoef_grnp = matt_coef(gfc_grnp, tmf_grnp)
        
# Plot matthews coefficient for aoi
lineplot(matcoef_aoi, "Matthews Coefficient for GFC and TMF Datasets", 
         "AOI", "Matthews Coefficient", 0.25, 0.45)

# Plot matthews coefficient for redd, nonredd, and grnp area
tripleplot(mattcoef_redd, mattcoef_nonredd, mattcoef_grnp, 
           "Matthews Coefficient for GFC and TMF Datasets", 
           "Matthews Coefficient", 0, 0.5)



############################################################################


# CHI SQUARED


############################################################################
"""
Method: https://www.geeksforgeeks.org/python-pearsons-chi-square-test/
"""
# Define function to create contingency table
def contingency_table(arrlist):
    
    # Create empty list to hold tables
    tabs = []
    
    # Iterate over each array
    for arr in arrlist:
    
        # Count pixels
        count_5 = np.sum(arr == 5) 
        count_6 = np.sum(arr == 6) 
        count_7 = np.sum(arr == 7) 
        count_8 = np.sum(arr == 8) 
        
        # Create contingency table
        matrix = [[count_8, count_6], 
                  [count_7, count_5]]
        
        # Add matrix to list
        tabs.append(matrix)

    return tabs

# Define function to calculate chi squared 
def chi2(tablist):
    
    # Create empty list to hold statistics
    chi = []
    p = []
    
    # Iterate over each array
    for tab in tablist:
        
        # Calculate chi squared
        stat, pval, dof, expected = chi2_contingency(tab)
        
        # Add chisquared to chi list
        chi.append(stat)
        
        # Add p value to p list
        p.append(pval)
        
    return chi, p

# Create contingency tables for aoi
tabs_aoi = contingency_table(spatagree_arrs)

# Create contingency tables for redd area
tabs_redd = contingency_table(ag_redd)

# Create contingency tables for nonredd area
tabs_nonredd = contingency_table(ag_nonredd)

# Create contingency tables for grnp area
tabs_grnp = contingency_table(ag_grnp)

# Calculate chisquared for aoi
chi2_aoi, chi2p_aoi = chi2(tabs_aoi)

# Calculate chisquared for redd area
chi2_redd, chi2p_redd = chi2(tabs_redd)

# Calculate chisquared for nonredd area
chi2_nonredd, chi2p_nonredd = chi2(tabs_nonredd)

# Calculate chisquared for grnp area
chi2_grnp, chi2p_grnp = chi2(tabs_grnp)

# Plot chisquared value for aoi
lineplot(chi2_aoi, "Chi2 Value for GFC and TMF Datasets", 
         "AOI", "Chi2 (Pearson's Coefficient)", 250000, 750000)

# Plot chisquared for redd, nonredd, and grnp area
tripleplot(chi2_redd, chi2_nonredd, chi2_grnp, 
           "Chi2 Value for GFC and TMF Datasets", 
           "Chi2 (Pearson's Coefficient)", 0, 350000)



############################################################################


# MCNEMAR TEST


############################################################################
# Define function to calculate McNemar's statistic
def calcmcnemar(tablist):
    
    # Create empty list to store McNemar statistics
    mcnemar_stats = []
    pval = []
    
    # Iterate over each contingency table
    for tab in tablist:
        
        # Calculate mcnemar statistic
        result = mcnemar(tab)

        # Add mcnemar statistic to list
        mcnemar_stats.append(result.statistic)
        
        # Add pvalue to list
        pval.append(result.pvalue)

    return mcnemar_stats, pval
    
# Calculate mcnemar statistic for aoi
mcnemar_aoi, mcnemarp_aoi = calcmcnemar(tabs_aoi)

# Calculate mcnemar statistic for redd area
mcnemar_redd, mcnemarp_redd = calcmcnemar(tabs_redd)

# Calculate mcnemar statistic for nonredd area
mcnemar_nonredd, mcnemarp_nonredd = calcmcnemar(tabs_nonredd)

# Calculate mcnemar statistic for grnp area
mcnemar_grnp, mcnemarp_grnp = calcmcnemar(tabs_grnp)

# Plot mcnemar statistic for aoi
lineplot(mcnemar_aoi, "McNemar Statistic for GFC and TMF Datasets", 
         "AOI", "McNemar Statistic", 13000, 81000)

# Plot chisquared for redd, nonredd, and grnp area
tripleplot(mcnemar_redd, mcnemar_nonredd, mcnemar_grnp, 
           "McNemar Statistic for GFC and TMF Datasets", 
           "McNemar Statistic", 0, 70000)



############################################################################


# COHEN'S KAPPA


############################################################################
"""
Based on example from documentation: 
https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.cohen_kappa_score.html
"""
# Define function to calculate cohen's kappa
def calckappa(arrlist1, arrlist2):
    
    # Create empty list to hold statistics
    kappas = []
    
    # Iterate over each array
    for arr1, arr2 in zip(arrlist1, arrlist2):
        
        # Convert arrays to float
        arr1 = arr1.astype(float)
        arr2 = arr2.astype(float)

        # Create mask to exclude nodata pixels
        mask = (arr1 != nodata_val) & (arr2 != nodata_val)
        
        # Filter arrays to exclude nodata pixels
        filtered_arr1 = arr1[mask]
        filtered_arr2 = arr2[mask]
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(filtered_arr1, filtered_arr2)
        
        # Store results in list
        kappas.append(kappa)
    
    return kappas

# Calculate cohens kappa for aoi
kappa_aoi = calckappa(gfc_simp_arrs, tmf_simp_arrs)

# Calculate cohens kappa for redd areas
kappa_redd = calckappa(gfc_simp_redd, tmf_simp_redd)

# Calculate cohens kappa for nonredd areas
kappa_nonredd = calckappa(gfc_simp_nonredd, tmf_simp_nonredd)

# Calculate cohens kappa for grnp area
kappa_grnp = calckappa(gfc_simp_grnp, tmf_simp_grnp)

# Plot cohens kappa for aoi
lineplot(kappa_aoi, "Cohen's Kappa for GFC and TMF Datasets", 
         "AOI", "McNemar Statistic", 0.2, 0.45)

# Plot cohens kappa for redd, nonredd, and grnp area
tripleplot(kappa_redd, kappa_nonredd, kappa_grnp, 
           "Cohen's Kappa for GFC and TMF Datasets", 
           "Cohen's Kappa", 0, 0.5)

















