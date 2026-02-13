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
# Define function to calculate stehman statistics
valstats <- function(valdata, stratmap, folder, filename){
  
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
            file = sprintf("native_validation/%s/%s_stats.csv", 
                           folder, filename), row.names = FALSE)
  
  # Export confusion matrix
  write.csv(stats$matrix, 
            file = sprintf("native_validation/%s/%s_cm.csv", 
                           folder, filename))
  
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
gfc_optA_stats <- timeinsensitive(gfc_optA, stratmap, "timeinsensitive", "gfc_timeinsensitive")
tmf_optA_stats <- timeinsensitive(tmf_optA, stratmap, "timeinsensitive", "tmf_timeinsensitive")
gfc_optA_buff_stats <- timeinsensitive(gfc_optA_buff, stratmap, "timeinsensitive", "gfc_timeinsensitive_buffered")
tmf_optA_distbuff_stats <- timeinsensitive(tmf_optA_dist_buff, stratmap, "timeinsensitive","tmf_timeinsensitive_dist_buffered")
tmf_optA_deforbuff_stats <- timeinsensitive(tmf_optA_defor_buff, stratmap, "timeinsensitive","tmf_timeinsensitive_defor_buffered")


# -------------------------------------------------------------------------
# TIME SENSITIVE PIXEL STATISTICS (ANY YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_anyyear <- read.csv("native_validation/anyyear/gfc_lossyear_anyyear.csv")
tmfdefor_anyyear <- read.csv("native_validation/anyyear/tmf_defor_anyyear.csv")
tmfdist_anyyear <- read.csv("native_validation/anyyear/tmf_dist_anyyear.csv")

# Calculate statistics
gfc_anyyear_stats <- valstats(gfc_anyyear, stratmap, "anyyear", "gfc_lossyear_anyyear")
tmfdefor_anyyear_stats <- valstats(tmfdefor_anyyear, stratmap, "anyyear", "tmf_defor_anyyear")
tmfdist_anyyear_stats <- valstats(tmfdist_anyyear, stratmap, "anyyear", "tmf_dist_anyyear")


# -------------------------------------------------------------------------
# TIME SENSITIVE PIXEL STATISTICS (FIRST YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_firstyear <- read.csv("native_validation/firstyear/gfc_lossyear_firstyear.csv")
tmfdefor_firstyear <- read.csv("native_validation/firstyear/tmf_defor_firstyear.csv")
tmfdist_firstyear <- read.csv("native_validation/firstyear/tmf_dist_firstyear.csv")

# Calculate statistics
gfc_firstyear_stats <- valstats(gfc_firstyear, stratmap, "firstyear", "gfc_lossyear_firstyear")
tmfdefor_firstyear_stats <- valstats(tmfdefor_firstyear, stratmap, "firstyear", "tmf_defor_firstyear")
tmfdist_firstyear_stats <- valstats(tmfdist_firstyear, stratmap, "firstyear", "tmf_dist_firstyear")


# -------------------------------------------------------------------------
# TIME SENSITIVE BUFFER STATISTICS (ANY YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_anyyear_buff <- read.csv("native_validation/anyyear/gfc_lossyear_buff_anyyear.csv")
tmfdefor_anyyear_buff <- read.csv("native_validation/anyyear/tmf_defor_buff_anyyear.csv")
tmfdist_anyyear_buff <- read.csv("native_validation/anyyear/tmf_dist_buff_anyyear.csv")

# Calculate statistics
gfc_anyyear_buff_stats <- valstats(gfc_anyyear_buff, stratmap, "anyyear", "gfc_lossyear_anyyear")
tmfdefor_anyyear_buff_stats <- valstats(tmfdefor_anyyear_buff, stratmap, "anyyear", "tmf_defor_anyyear")
tmfdist_anyyear_buff_stats <- valstats(tmfdist_anyyear_buff, stratmap, "anyyear", "tmf_dist_anyyear")


# -------------------------------------------------------------------------
# TIME SENSITIVE BUFFER STATISTICS (FIRST YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_firstyear_buff <- read.csv("native_validation/firstyear/gfc_lossyear_buff_firstyear.csv")
tmfdefor_firstyear_buff <- read.csv("native_validation/firstyear/tmf_defor_buff_firstyear.csv")
tmfdist_firstyear_buff <- read.csv("native_validation/firstyear/tmf_dist_buff_firstyear.csv")

# Calculate statistics
gfc_firstyear_buff_stats <- valstats(gfc_firstyear_buff, stratmap, "firstyear", "gfc_lossyear_buff_firstyear")
tmfdefor_firstyear_buff_stats <- valstats(tmfdefor_firstyear_buff, stratmap, "firstyear", "tmf_defor_buff_firstyear")
tmfdist_firstyear_buff_stats <- valstats(tmfdist_firstyear_buff, stratmap, "firstyear", "tmf_dist_buff_firstyear")



