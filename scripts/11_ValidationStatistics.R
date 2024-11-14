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
valdata <- read.csv("data/validation/validation_points_labelled.csv", sep = ";")

# Read stratification map
stratdata <- rast("data/intermediate/stratification_layer_nogrnp.tif")

# Extract strata data
strata <- valdata$strata

# Extract gfc predictions
gfc <- valdata$gfc

# Extract tmf predictions
tmf <- valdata$tmf

# Extract se predictions
se <- valdata$se

# Extract validation labels
val1 <- valdata$defor1

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
    se_pa = as.numeric(stats$SEpa)
  )
  
  # Export dataframe
  write.csv(stats_df, file = sprintf("data/validation/%s_stehmanstats.csv", 
                                     varname), row.names = FALSE)
  
  # Print statement
  print("Stehman statistics calculated and saved to file")
  
  return(stats)
}

# Calculate statistics for gfc
gfc_stats <- valstats(gfc, val1, "gfc")

# Calculate statistics for tme
tmf_stats <- valstats(tmf, val1, "tmf")

# Calculate statistics for se
se_stats <- valstats(se, val1, "se")



############################################################################


# CALCULATE STATISTICS WITH DATA MANIPULATION


############################################################################
# Subtract 1 from all non-0 validation points
val1_sub1 <- ifelse(val1 == 2013, 0, ifelse(val1 != 0, val1 - 1, 0))

# Calculate statistics for gfc
gfc_substats <- valstats(gfc, val1_sub1, "sub_gfc")

# Calculate statistics for tmf
tmf_substats <- valstats(tmf, val1_sub1, "sub_tmf")

# Calculate statistics for se
se_substats <- valstats(se, val1_sub1, "sub_se")











