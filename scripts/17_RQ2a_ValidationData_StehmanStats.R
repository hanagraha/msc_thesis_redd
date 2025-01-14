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

# Read validation dataset (pre-processed)
valdata <- read.csv("data/validation/validation_points_preprocessed.csv")

# Read validation dataset (pre-processed)
# valdata_proc <- read.csv("data/validation/validation_points_preprocessed2.csv")
valdata_proc <- read.csv("data/validation/validation_points_1200_preprocessed.csv")

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

# Extract validation labels (for unprocessed)
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
    se_pa = as.numeric(stats$SEpa),
    se_a = as.numeric(stats$SEa)
  )
  
  # Export annual data
  write.csv(stats_df, file = sprintf("data/validation/%s_stehmanstats.csv", 
                                     varname), row.names = FALSE)
  
  # Export confusion matrix
  write.csv(stats$matrix, file = sprintf("data/validation/%s_confmatrix.csv", 
                                         varname))
  
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
# Extract gfc processed validation data
gfc_val <- valdata_proc$gfc_val

# Extract tmf processed validation data
tmf_val <- valdata_proc$tmf_val

# Extract se processed validation data
se_val <- valdata_proc$se_val

# Extract strata data
strata <- valdata_proc$strata

# Calculate number of pixels per strata
pixel_counts <- freq(stratdata, digits = 0)

# Extract only pixel counts
strata_counts <- pixel_counts$count

# Assign strata values as name
names(strata_counts) <- pixel_counts$value

# Calculate statistics for gfc
gfc_procstats <- valstats(gfc, gfc_val, "proc2_gfc")
gfc_procstats <- valstats(gfc, gfc_val, "proc2_gfc")

# Calculate statistics for tmf
tmf_procstats <- valstats(tmf, tmf_val, "proc2_tmf")

# Calculate statistics for se
se_procstats <- valstats(se, se_val, "proc2_se")




