# -------------------------------------------------------------------------
# IMPORT PACKAGES
# -------------------------------------------------------------------------
# Install packages
install.packages("terra")
install.packages("mapaccuracy")

# Load packages
library(terra)
library(mapaccuracy)


# -------------------------------------------------------------------------
# SET UP DIRECTORY AND STATISTICS
# -------------------------------------------------------------------------
# Set working directory
setwd("Z:/person/graham/projectdata/redd-sierraleone")

# Read stratification map
stratmap <- rast("validation/stratification_maps/stratification_layer_nogrnp.tif")


# -------------------------------------------------------------------------
# DEFINE KEY FUNCTIONS
# -------------------------------------------------------------------------
# Define function to calculate time insensitive statistics
timeinsensitive <- function(valdata, stratmap, filename){
  
  # Calculate number of pixels per strata
  pixel_counts <- freq(stratmap, digits = 0)
  
  # Extract only pixel counts
  strata_counts <- pixel_counts$count
  
  # Assign strata values as name
  names(strata_counts) <- pixel_counts$value
  
  # Calculate statistics
  stats <- stehman2014(valdata$strata, valdata$ref, valdata$map, strata_counts)
  
  # Create dataframe
  stats_df <- data.frame(
    year = as.numeric(names(stats$area)),
    ua = as.numeric(stats$UA),
    pa = as.numeric(stats$PA),
    area = as.numeric(stats$area),
    se_ua = as.numeric(stats$SEua),
    se_pa = as.numeric(stats$SEpa),
    se_a = as.numeric(stats$SEa),
    oa = as.numeric(stats$OA),
    se_oa = as.numeric(stats$SEoa)
  )
  
  # Export annual data
  write.csv(stats_df, 
            file = sprintf("native_validation/timeinsensitive/%s_stats.csv", 
                           filename), row.names = FALSE)
  
  # Export confusion matrix
  write.csv(stats$matrix, 
            file = sprintf("native_validation/timeinsensitive/%s_cm.csv", 
                           filename))
  
  # Print statement
  print("Stehman statistics calculated and saved to file")
  
  return(stats)
}



# -------------------------------------------------------------------------
# TIME INSENSITIVE STATISTICS
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_optA <- read.csv("native_validation/timeinsensitive/gfc_timeinsensitive.csv")
tmf_optA <- read.csv("native_validation/timeinsensitive/tmf_timeinsensitive.csv")
gfc_optA_buff <- read.csv("native_validation/timeinsensitive/gfc_timeinsensitive_buffered.csv")
tmf_optA_dist_buff <- read.csv("native_validation/timeinsensitive/tmf_timeinsensitive_dist_buffered.csv")
tmf_optA_defor_buff <- read.csv("native_validation/timeinsensitive/tmf_timeinsensitive_dist_buffered.csv")

# Calculate statistics
gfc_optA_stats <- timeinsensitive(gfc_optA, stratmap, "gfc_timeinsensitive")
tmf_optA_stats <- timeinsensitive(tmf_optA, stratmap, "tmf_timeinsensitive")
gfc_optA_buff_stats <- timeinsensitive(gfc_optA_buff, stratmap, "gfc_timeinsensitive_buffered")
tmf_optA_distbuff_stats <- timeinsensitive(tmf_optA_dist_buff, stratmap, "tmf_timeinsensitive_dist_buffered")
tmf_optA_deforbuff_stats <- timeinsensitive(tmf_optA_defor_buff, stratmap, "tmf_timeinsensitive_defor_buffered")






