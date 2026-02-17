# -------------------------------------------------------------------------
# IMPORT PACKAGES AND CHECK DIRECTORY
# -------------------------------------------------------------------------
# Import packages
import os
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Check current working directory
print("Current Working Directory:", os.getcwd())

# Change to a new directory 
os.chdir(r"Z:\person\graham\projectdata\redd-sierraleone")

# Verify the working directory has been changed
print("New Working Directory:", os.getcwd())


# -------------------------------------------------------------------------
# DEFINE CONSTANTS
# -------------------------------------------------------------------------
# Define year strings
years = range(2013, 2024)
yearlabs = [0] + list(years) + ['sum']
year_strings = [str(y) for y in yearlabs]

# Define color palette
bluecols = ["#1E2A5E", "#83B4FF"]

# Define data columns
datacols = ["year", "ua", "pa", "area", "se_ua", "se_pa", "se_a"]


# -------------------------------------------------------------------------
# READ INPUT DATA
# -------------------------------------------------------------------------
# GFC predictions (lossyear)
with rasterio.open('native_validation/gfc_lossyear_native.tif') as rast:     
    gfc_lossyear = rast.read(1)
    gfc_totalpix = np.sum(gfc_lossyear != 255)

# TMF predictions (various)
with rasterio.open('native_validation/tmf_deforyear_native.tif') as rast:     
    tmf_deforyear = rast.read(1)
    tmf_totalpix = np.sum(tmf_deforyear != 255)

with rasterio.open('native_validation/tmf_degrayear_native.tif') as rast:     
    tmf_degrayear = rast.read(1)

with rasterio.open('native_validation/tmfac_firstdeforyear_native.tif') as rast:     
    tmfac_firstdeforyear = rast.read(1)

with rasterio.open('native_validation/tmfac_seconddeforyear_native.tif') as rast:     
    tmfac_seconddeforyear = rast.read(1)

with rasterio.open('native_validation/tmfac_firstdegrayear_native.tif') as rast:     
    tmfac_firstdegrayear = rast.read(1)

with rasterio.open('native_validation/tmfac_seconddegrayear_native.tif') as rast:     
    tmfac_seconddegrayear = rast.read(1)


# -------------------------------------------------------------------------
# READ STATISTICS
# -------------------------------------------------------------------------
# Define function to list files in subfolder
def listfiles(folder, suffixes=('_cm', '_stats')):

    # Define folder path
    folderpath = os.path.join('native_validation', folder)
    
    # Create empty list to store files
    paths = []

    # Iterate over items in folder
    for file in os.listdir(folderpath):
        
        # Remove extension before checking
        name, ext = os.path.splitext(file)

        if name.endswith(suffixes):
            filepath = os.path.join(folderpath, file)
            paths.append(filepath)

    return paths

# Define function to read files from list
def list_read(pathlist, suffix, subset=False):
    
    # Create empty dictionary to store outputs
    files = {}
    
    # Iterate over each file in list
    for path in pathlist:
        
        # Read file
        data = pd.read_csv(path)

        # Subset for rows
        if subset: 
            data = data.loc[data["year"].between(2013, 2023),
                            datacols]
        
        # Extract file name
        filename = os.path.basename(path)
        
        # Remove suffix from filename
        var = filename.replace(suffix, "")
        
        # Add data to dictionary
        files[var] = data
    
    return files

# Read time insensitive data
timeinsensitive_stats = list_read(listfiles("timeinsensitive", suffixes=('_stats')), suffix="_stats.csv")

# Read time sensitive data
anyyear_stats = list_read(listfiles("anyyear", suffixes=('_stats')), suffix="_stats.csv", subset=True)
anyyear_cm = list_read(listfiles("anyyear", suffixes=('_cm')), suffix="_cm.csv")
firstyear_stats = list_read(listfiles("firstyear", suffixes=('_stats')), suffix="_stats.csv", subset=True)
firstyear_cm = list_read(listfiles("firstyear", suffixes=('_stats')), suffix="_cm.csv")

# Read area estimate data
dist1_area = pd.read_csv("native_validation/area_estimation/gfc_lossyear_stats.csv")[2:-1]
dist2_area = pd.read_csv("native_validation/area_estimation/gfc_lossyear2_stats.csv")[2:-1]
dist3_area = pd.read_csv("native_validation/area_estimation/gfc_lossyear3_stats.csv")[2:-1]


# -------------------------------------------------------------------------
# DEFINE CONFUSION MATRIX PLOTTING FUNCTIONS
# -------------------------------------------------------------------------
# Define function to formate stehman confusion matrix
def steh_cm(stehman_cm, deci):
    
    # Exclude the sum row and 0 year
    cm = stehman_cm.iloc[1:-1, 1:-1]
    
    # Fill na values with 0
    cm = cm.fillna(0)
    
    # Convert dataframe to array
    cm = cm.to_numpy()
    
    # Multiply by 100 to convert to %
    cm = cm * 100
    
    # Round to (deci) number of decimals
    cm = np.round(cm, deci)
    
    return cm

# Define function to plot three confusion matrices side by side
def matrix_plt(matrices, names, fmt):
    
    # Initiate figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Iterate over each confusion matrix
    for i in range(0, len(matrices)):
        
        # Create heatmap
        sns.heatmap(matrices[i], annot=True, fmt=fmt, cmap='Blues', ax=axes[i],
                    xticklabels=years, yticklabels=years, vmax = 6)
        
        # Add title
        axes[i].set_xlabel('Validation Labels', fontsize = 16)
        axes[i].set_ylabel(f'{names[i]} Predicted Labels', fontsize = 16)
    
        # Adjust tick labels font size
        axes[i].tick_params(axis='both', labelsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# DEFINE STEHMAN STATISTICS PLOTTING FUNCTIONS
# -------------------------------------------------------------------------
# Define function to plot user's and producer's accuracies side by side
def steh_dual_lineplt(datalist, filename, datanames=["GFC", "TMF"]):
    
    # Initialize figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Subplot for User's Accuracy
    axes[0].set_title("User's Accuracy", fontsize = 16)
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add data and error bars
        axes[0].errorbar(data['year'], data['ua'], yerr=data['se_ua'], fmt='-o', 
                         color=col, capsize=5, elinewidth=1, ecolor=col, 
                         label=name, linewidth = 2)
    
    # Add gridlines
    axes[0].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[0].set_xticks(years)
    
    # Adjust tickmark font size
    axes[0].tick_params(axis='both', labelsize = 14)
    
    # Add axes labels
    axes[0].set_xlabel('Year', fontsize = 14)
    axes[0].set_ylabel("Accuracy", fontsize = 14)
    
    # Add legend
    axes[0].legend(fontsize = 16)

    # Subplot for Producer's Accuracy
    axes[1].set_title("Producer's Accuracy", fontsize = 16)
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add data and error bars
        axes[1].errorbar(data['year'], data['pa'], yerr=data['se_pa'], fmt='-o', 
                         color=col, capsize=5, elinewidth=1, ecolor=col, 
                         label=name, linewidth = 2)
    
    # Add gridlines
    axes[1].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[1].set_xticks(years)
    
    # Adjust tickmark font size
    axes[1].tick_params(axis='both', labelsize = 14)
    
    # Add x labels
    axes[1].set_xlabel('Year', fontsize = 14)
    
    # Add legend
    axes[1].legend(fontsize = 16)

    # Define output filepath
    filepath = f"figs/native_validation/{filename}.png"

    # Save plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)

    # Print save confirmation
    print(f"Plot saved as: {filepath}")
    
# Define function to calculate commission and ommission error
def comom_err(dataset):
    
    # Extract producers accuracy
    pa = dataset['pa']
    
    # Extract users accuracy
    ua = dataset['ua']
    
    # Calculate ommission error
    oe = 1 - pa
    
    # Calculate commission error
    ce = 1 - ua
    
    # Combine ommission and commission error
    errors = pd.DataFrame({
        'year': dataset['year'],
        'oe': oe,
        'ce': ce,
        'oe_se': dataset['se_pa'],
        'ce_se': dataset['se_ua']})
    
    return errors

# Define function to plot user's and producer's accuracies side by side
def errors_lineplt(datalist, filename, datanames=["GFC", "TMF"]):
    
    # Initialize figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Subplot for User's Accuracy
    axes[0].set_title("Commission Error", fontsize = 16)
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add ommission error data
        axes[0].errorbar(data['year'], data['ce'], yerr=data['ce_se'], fmt='-o', 
                         color = col, capsize = 5, elinewidth = 1, label=name, 
                         ecolor = col, linewidth=2)
    
    # Add gridlines
    axes[0].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[0].set_xticks(data['year'])
    
    # Adjust tickmark font size
    axes[0].tick_params(axis='both', labelsize = 14)
    
    # Add axes labels
    axes[0].set_xlabel('Year', fontsize = 14)
    axes[0].set_ylabel("Error", fontsize = 14)
    
    # Add legend
    axes[0].legend(fontsize = 16)

    # Subplot for Producer's Accuracy
    axes[1].set_title("Omission Error", fontsize = 16)
    
    # Iterate over datasets
    for data, col, name in zip(datalist, bluecols, datanames):
        
        # Add commission error data
        axes[1].errorbar(data['year'], data['oe'], yerr=data['oe_se'], fmt='-o', 
                         color = col, capsize = 5, elinewidth = 1, label=name, 
                         ecolor = col, linewidth=2)
    
    # Add gridlines
    axes[1].grid(True, linestyle="--")
    
    # Add x ticks for every year
    axes[1].set_xticks(data['year'])
    
    # Adjust tickmark font size
    axes[1].tick_params(axis='both', labelsize = 14)
    
    # Add x labels
    axes[1].set_xlabel('Year', fontsize = 14)
    
    # Add legend
    axes[1].legend(fontsize = 16)

    # Tight layout
    plt.tight_layout()

    # Define output filepath
    filepath = f"figs/native_validation/{filename}.png"

    # Save plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)

    # Print save confirmation
    print(f"Plot saved as: {filepath}")

    # Show plot
    plt.show()


# -------------------------------------------------------------------------
# PLOT OVERALL ACCURACIES
# -------------------------------------------------------------------------
# Plot accuracies
steh_dual_lineplt([anyyear_stats["gfc_lossyear_anyyear"], 
                   anyyear_stats["tmf_dist_anyyear"]], 
                   "anyyear_tmfdist_accuracies")

steh_dual_lineplt([firstyear_stats["gfc_lossyear_firstyear"], 
                   firstyear_stats["tmf_dist_firstyear"]], 
                   "firstyear_tmfdist_accuracies")


# -------------------------------------------------------------------------
# PLOT COMMISSION AND OMISSION ERRORS
# -------------------------------------------------------------------------
# Plot errors
errors_lineplt([comom_err(anyyear_stats["gfc_lossyear_anyyear"]),
                comom_err(anyyear_stats["tmf_dist_anyyear"])],
                "anyyear_tmfdist_errors")

errors_lineplt([comom_err(firstyear_stats["gfc_lossyear_firstyear"]),
                comom_err(firstyear_stats["tmf_dist_firstyear"])],
                "firstyear_tmfdist_errors")


# -------------------------------------------------------------------------
# PREPROCESS AREA DATA
# -------------------------------------------------------------------------
# Define function to get count summaries
def count_summary(raster, totalpix):

    # Filter out nodata values
    raster = raster[raster != 255]

    # Get unique values and counts
    vals, counts = np.unique(raster, return_counts=True)

    # Create dataframe 
    df = pd.DataFrame({'year': vals, 'counts': counts})

    # Calculate total pixels
    total_pix = df['counts'].sum()

    # Calculate proportions
    df['prop'] = (df['counts'] / totalpix) * 100

    # Filter for just 2013-2023
    df = df[(df['year'] >= 2013) & (df['year'] <= 2023)]

    return df.reset_index(drop=True)

# Extract disturbance counts
gfc_lossyear_area = count_summary(gfc_lossyear, gfc_totalpix)
tmf_deforyear_area = count_summary(tmf_deforyear, tmf_totalpix)
tmf_degrayear_area = count_summary(tmf_degrayear, tmf_totalpix)
tmfac_firstdeforyear_area = count_summary(tmfac_firstdeforyear, tmf_totalpix)
tmfac_seconddeforyear_area = count_summary(tmfac_seconddeforyear, tmf_totalpix)
tmfac_firstdegrayear_area = count_summary(tmfac_firstdegrayear, tmf_totalpix)
tmfac_seconddegrayear_area = count_summary(tmfac_seconddegrayear, tmf_totalpix)

# Add deforestation and degradation area
tmf_defordegra_area = tmf_deforyear_area.copy()
tmf_defordegra_area['prop'] = tmf_deforyear_area['prop'] + tmf_degrayear_area['prop']

# Define function to preprocess area estimation data
def area_prop(df):

    # Filter area estimation data
    area_df = df[['year', 'area', 'se_a']].copy()

    # Convert values to percent
    area_df['area'] = area_df['area'] * 100
    area_df['se_a'] = area_df['se_a'] * 100

    # Calculate 95% confidence interval
    area_df['ci95'] = 1.96 * area_df['se_a']

    # Create upper and lower limits
    area_df['lower'] = area_df['area'] - area_df['ci95']
    area_df['upper'] = area_df['area'] + area_df['ci95']

    return area_df

# Preprocess disturbance area
dist1_prop = area_prop(dist1_area)
dist2_prop = area_prop(dist2_area)
dist3_prop = area_prop(dist3_area)

# Combine to one dataframe
dist_all = dist1_prop.merge(dist2_prop, on='year', suffixes=("_1", "_2"), how="left")
dist_all = dist_all.merge(dist3_prop, on="year", how="left")
dist_all = dist_all.rename(columns={"area": "area_3", "se_a": "se_3"})

# Add disturbances
dist_all["area_total"] = (
    dist_all["area_1"] +
    dist_all["area_2"].fillna(0) +
    dist_all["area_3"].fillna(0)
)

# Calculate combined standard error
dist_all["se_total"] = np.sqrt(
    dist_all["se_a_1"]**2 +
    dist_all["se_a_2"].fillna(0)**2 +
    dist_all["se_3"].fillna(0)**2
)

# Calculated combined confidence intervals
dist_all["ci95_total"] = 1.96 * dist_all["se_total"]
dist_all["lower_total"] = dist_all["area_total"] - dist_all["ci95_total"]
dist_all["upper_total"] = dist_all["area_total"] + dist_all["ci95_total"]

# Filter for final information
combined_dist = dist_all[['year', 'area_total', 'se_total', 'ci95_total', 
                          'lower_total', 'upper_total']].copy()


# -------------------------------------------------------------------------
# PLOT STEHMAN AREA ESTIMATES
# -------------------------------------------------------------------------
# Initialize figure
plt.figure(figsize=(10, 6))

# Add area estimation
plt.plot(dist1_prop['year'], dist1_prop['area'], color="#1E2A5E", linewidth = 1.5,
         label='Area estimation (first disturbance)')

# Add confidence intervals
plt.fill_between(dist1_prop['year'], dist1_prop['lower'], dist1_prop['upper'], color='#83B4FF',
                 alpha=0.4)

# Add area estimation
plt.plot(combined_dist['year'], combined_dist['area_total'], color="#1E2A5E", linewidth = 1.3,
         linestyle = '--', label='Area estimation (all disturbances)')

# Add confidence intervals
plt.fill_between(combined_dist['year'], combined_dist['lower_total'], combined_dist['upper_total'], color='#83B4FF',
                 alpha=0.15)

# Add gfc predictions
plt.plot(gfc_lossyear_area['year'], gfc_lossyear_area['prop'], color='#8e1014', 
         linewidth = 1.5, label='GFC lossyear')

# Add tmf deforestation predictions
plt.plot(tmf_deforyear_area['year'], tmf_deforyear_area['prop'], color='#cc000a', 
         linewidth = 1.5, label='TMF deforestation')

# Add tmf deforestation + degradation predictions
plt.plot(tmf_defordegra_area['year'], tmf_defordegra_area['prop'], color='#cc000a', 
         linewidth = 1.3, linestyle = '--', label='TMF deforestation + degradation')

# Add labels and title
plt.xlabel('Year', fontsize = 17)
plt.ylabel('Proportional Deforestation Area (%)', fontsize = 17)

# Add x tickmarks
plt.xticks(years, fontsize = 14)

# Edit y tickmark fontsize
plt.yticks(fontsize = 14)

# Add legend
plt.legend(fontsize = 15, loc="upper right")

# Add gridlines
plt.grid(linestyle = "--", alpha = 0.6)

# Tight layout
plt.tight_layout()

# Save plot
plt.savefig('figs/native_validation/area_estimation.png', dpi=300, 
            bbox_inches='tight', transparent=True)

# Show plot
plt.show()




# Define function to calculate proportional areas
def steh_area(valdata, stratmap, deforlist, pix, ha):
    
    # Calculate number of pixels per strata
    pixvals, pixcounts = np.unique(stratmap, return_counts = True)

    # Create dataframe
    strata_size = pd.DataFrame({'strata': pixvals[:-1],
                                'size': pixcounts[:-1]})

    # Create empty list to hold deforestation area 
    year_defor = pd.DataFrame(index=strata_size['strata'])

    # Iterate over each year
    for year in years:
        
        # Create empty list to hold year deforestation area
        strata_defor = []
        
        # Iterate over each strata
        for idx, row in strata_size.iterrows():
            
            # Extract strata number
            strata = row['strata']
            
            # Extract strata size
            size = row['size']
            
            # Subset validation data for that strata
            data = valdata[valdata['strata'] == strata]
            
            # Count sum of deforestation in that year
            defor = data[deforlist].eq(year).sum().sum()
            
            # Calculate class proportion in that strata
            cp = defor / len(data)
            
            # Multiply by strata size
            area = (cp * size) / pix
            
            # Add deforestation area to list
            strata_defor.append(area)
            
        # Add the list as a column in the DataFrame
        year_defor[year] = strata_defor

    # Take the sum per year
    total_defor = year_defor.sum(axis=0) * ha
    
    return total_defor

# Estimate deforestation for the first year
first_defor = steh_area(valdata, stratmap, ['defor1'], total_pix, total_ha)

# Estimate deforestation for second year
second_defor = steh_area(valdata, stratmap, ['defor2'], total_pix, total_ha)

# Estimate deforestation for third year
third_defor = steh_area(valdata, stratmap, ['defor3'], total_pix, total_ha)

# Estimate deforestation for all years
all_defor = steh_area(valdata, stratmap, ['defor1', 'defor2', 'defor3'], 
                      total_pix, total_ha)

# Extract area calculated by stehman
stehman_area = protc_stats['protc_gfc']['area'] * total_ha
stehman_area.index = pd.Index(years)

# Calculate missed deforestation
miss_defor = all_defor - first_defor



