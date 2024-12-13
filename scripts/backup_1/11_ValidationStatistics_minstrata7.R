############################################################################


# IMPORT PACKAGES


###########################################################################
# Install packages
install.packages("terra")
install.packages("mapaccuracy")

# Load packages
library(terra)
library(mapaccuracy)



############################################################################


# SET UP DIRECTORY AND READ DATA


############################################################################
# Set working directory
setwd("C:/Users/hanna/Documents/WUR MSc/MSc Thesis/redd-thesis")

# Read validation dataset
valdata <- read.csv("data/validation/valdata_minstrata7_proc.csv", sep = ",")

# Read stratification map
stratdata <- rast("data/intermediate/stratification_layer_nogrnp.tif")

# Extract strata data
strata <- valdata$strata

# Calculate number of pixels per strata
pixel_counts <- freq(stratdata, digits = 0)

# Extract only pixel counts
strata_counts <- pixel_counts$count

# Assign strata values as name
names(strata_counts) <- pixel_counts$value



############################################################################


# CALCULATE VALIDATION STATISTICS


############################################################################
# Define function to calculate and save statistics
valstats <- function(pred_data, val_data, varname){
  
  # Calculate statistics
  stats <- stehman2014(strata, val_data, pred_data, strata_counts)
  
  # Create dataframe
  stats_df <- data.frame(
    year = as.numeric(names(stats$area)),
    ua = as.numeric(stats$UA),
    pa = as.numeric(stats$PA),
    area = as.numeric(stats$area),
    se_ua = as.numeric(stats$SEua),
    se_pa = as.numeric(stats$SEpa),
    se_a = as.numeric(stats$SEa)
  )
  
  # Export annual data
  write.csv(stats_df, file = sprintf("data/validation/%s_stehman_minstrata7.csv", 
                                     varname), row.names = FALSE)
  
  # Export confusion matrix
  write.csv(stats$matrix, file = sprintf("data/validation/%s_cf_minstrata7.csv", 
                                         varname))
  
  # Print statement
  print("Stehman statistics calculated and saved to file")
  
  return(stats)
}

# Calculate statistics for gfc
gfc_stats <- valstats(valdata$gfc, valdata$gfc_val, "gfc")

# Calculate statistics for tme
tmf_stats <- valstats(valdata$tmf, valdata$tmf_val, "tmf")

# Calculate statistics for se
se_stats <- valstats(valdata$se, valdata$se_val, "se")


