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
# TIME INSENSITIVE PIXEL STATISTICS
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_timeinsensitive <- read.csv("native_validation/timeinsensitive/gfc_lossyear_timeinsensitive.csv")
gfc50cc_timeinsensitive <- read.csv("native_validation/timeinsensitive/gfc_lossyear50cc_timeinsensitive.csv")
tmfdefor_timeinsensitive <- read.csv("native_validation/timeinsensitive/tmf_defor_timeinsensitive.csv")
tmfdist_timeinsensitive <- read.csv("native_validation/timeinsensitive/tmf_dist_timeinsensitive.csv")

# Calculate statistics
gfc_timeinsensitive_stats <- valstats(gfc_timeinsensitive, stratmap, "timeinsensitive", "gfc_lossyear_timeinsensitive")
gfc50cc_timeinsensitive_stats <- valstats(gfc50cc_timeinsensitive, stratmap, "timeinsensitive", "gfc_lossyear50cc_timeinsensitive")
tmfdefor_timeinsensitive_stats <- valstats(tmfdefor_timeinsensitive, stratmap, "timeinsensitive", "tmf_defor_timeinsensitive")
tmfdist_timeinsensitive_stats <- valstats(tmfdist_timeinsensitive, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive")


# -------------------------------------------------------------------------
# TIME INSENSITIVE BUFFER STATISTICS
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_timeinsensitive_buff <- read.csv("native_validation/timeinsensitive/gfc_lossyear_buff_timeinsensitive.csv")
gfc50cc_timeinsensitive_buff <- read.csv("native_validation/timeinsensitive/gfc_lossyear50cc_buff_timeinsensitive.csv")
tmfdefor_timeinsensitive_buff <- read.csv("native_validation/timeinsensitive/tmf_defor_buff_timeinsensitive.csv")
tmfdisttimeinsensitive_buff <- read.csv("native_validation/timeinsensitive/tmf_dist_buff_timeinsensitive.csv")

# Calculate statistics
gfc_timeinsensitive_buff_stats <- valstats(gfc_timeinsensitive_buff, stratmap, "timeinsensitive", "gfc_lossyear_buff_timeinsensitive")
gfc50cc_timeinsensitive_buff_stats <- valstats(gfc50cc_timeinsensitive_buff, stratmap, "timeinsensitive", "gfc_lossyear50cc_buff_timeinsensitive")
tmfdefor_timeinsensitive_buff_stats <- valstats(tmfdefor_timeinsensitive_buff, stratmap, "timeinsensitive", "tmf_defor_buff_timeinsensitive")
tmfdist_timeinsensitive_buff_stats <- valstats(tmfdisttimeinsensitive_buff, stratmap, "timeinsensitive", "tmf_dist_buff_timeinsensitive")


# -------------------------------------------------------------------------
# TIME SENSITIVE PIXEL STATISTICS (ANY YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_anyyear <- read.csv("native_validation/anyyear/gfc_loss_anyyear.csv")
gfc50cc_anyyear <- read.csv("native_validation/anyyear/gfc_loss50cc_anyyear.csv")
tmfdefor_anyyear <- read.csv("native_validation/anyyear/tmf_defor_anyyear.csv")
tmfdist_anyyear <- read.csv("native_validation/anyyear/tmf_dist_anyyear.csv")

# Calculate statistics
gfc_anyyear_stats <- valstats(gfc_anyyear, stratmap, "anyyear", "gfc_loss_anyyear")
gfc50cc_anyyear_stats <- valstats(gfc50cc_anyyear, stratmap, "anyyear", "gfc_loss50cc_anyyear")
tmfdefor_anyyear_stats <- valstats(tmfdefor_anyyear, stratmap, "anyyear", "tmf_defor_anyyear")
tmfdist_anyyear_stats <- valstats(tmfdist_anyyear, stratmap, "anyyear", "tmf_dist_anyyear")


# -------------------------------------------------------------------------
# TIME SENSITIVE BUFFER STATISTICS (ANY YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_anyyear_buff <- read.csv("native_validation/anyyear/gfc_lossyear_buff_anyyear.csv")
gfc50cc_anyyear_buff <- read.csv("native_validation/anyyear/gfc_lossyear_50cc_buff_anyyear.csv")
tmfdefor_anyyear_buff <- read.csv("native_validation/anyyear/tmf_defor_buff_anyyear.csv")
tmfdist_anyyear_buff <- read.csv("native_validation/anyyear/tmf_dist_buff_anyyear.csv")

# Calculate statistics
gfc_anyyear_buff_stats <- valstats(gfc_anyyear_buff, stratmap, "anyyear", "gfc_lossyear_buff_anyyear")
gfc50cc_anyyear_buff_stats <- valstats(gfc50cc_anyyear_buff, stratmap, "anyyear", "gfc_lossyear_50cc_buff_anyyear")
tmfdefor_anyyear_buff_stats <- valstats(tmfdefor_anyyear_buff, stratmap, "anyyear", "tmf_defor_buff_anyyear")
tmfdist_anyyear_buff_stats <- valstats(tmfdist_anyyear_buff, stratmap, "anyyear", "tmf_dist_buff_anyyear")


# -------------------------------------------------------------------------
# TIME SENSITIVE PIXEL STATISTICS (FIRST YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_firstyear <- read.csv("native_validation/firstyear/gfc_loss_firstyear.csv")
gfc50cc_firstyear <- read.csv("native_validation/firstyear/gfc_loss50cc_firstyear.csv")
tmfdefor_firstyear <- read.csv("native_validation/firstyear/tmf_defor_firstyear.csv")
tmfdist_firstyear <- read.csv("native_validation/firstyear/tmf_dist_firstyear.csv")

# Calculate statistics
gfc_firstyear_stats <- valstats(gfc_firstyear, stratmap, "firstyear", "gfc_lossyear_firstyear")
gfc50cc_firstyear_stats <- valstats(gfc50cc_firstyear, stratmap, "firstyear", "gfc_loss50cc_firstyear")
tmfdefor_firstyear_stats <- valstats(tmfdefor_firstyear, stratmap, "firstyear", "tmf_defor_firstyear")
tmfdist_firstyear_stats <- valstats(tmfdist_firstyear, stratmap, "firstyear", "tmf_dist_firstyear")


# -------------------------------------------------------------------------
# TIME SENSITIVE BUFFER STATISTICS (FIRST YEAR)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_firstyear_buff <- read.csv("native_validation/firstyear/gfc_lossyear_buff_firstyear.csv")
gfc50cc_firstyear_buff <- read.csv("native_validation/firstyear/gfc_lossyear_50cc_buff_firstyear.csv")
tmfdefor_firstyear_buff <- read.csv("native_validation/firstyear/tmf_defor_buff_firstyear.csv")
tmfdist_firstyear_buff <- read.csv("native_validation/firstyear/tmf_dist_buff_firstyear.csv")

# Calculate statistics
gfc_firstyear_buff_stats <- valstats(gfc_firstyear_buff, stratmap, "firstyear", "gfc_lossyear_buff_firstyear")
gfc50cc_firstyear_buff_stats <- valstats(gfc50cc_firstyear_buff, stratmap, "firstyear", "gfc_lossyear_50cc_buff_firstyear")
tmfdefor_firstyear_buff_stats <- valstats(tmfdefor_firstyear_buff, stratmap, "firstyear", "tmf_defor_buff_firstyear")
tmfdist_firstyear_buff_stats <- valstats(tmfdist_firstyear_buff, stratmap, "firstyear", "tmf_dist_buff_firstyear")


# -------------------------------------------------------------------------
# STATISTICS FOR AREA ESTIMATION
# -------------------------------------------------------------------------
# Read map and reference labels
native_valdata <- read.csv("native_validation/validation_mapdata.csv")

# Filter data for each disturbance
area_dist1 <- native_valdata[, c("strata", "defor1", "gfc_lossyear")]
area_dist2 <- native_valdata[, c("strata", "defor2", "gfc_lossyear")]
area_dist3 <- native_valdata[, c("strata", "defor3", "gfc_lossyear")]

# Add column names
colnames(area_dist1) <- c("strata", "ref", "map")
colnames(area_dist2) <- c("strata", "ref", "map")
colnames(area_dist3) <- c("strata", "ref", "map")

# Calculate statistics
dist1_stats <- valstats(area_dist1, stratmap, "areaestimation", "area_dist1")
dist2_stats <- valstats(area_dist2, stratmap, "areaestimation", "area_dist2")
dist3_stats <- valstats(area_dist3, stratmap, "areaestimation", "area_dist3")



# -------------------------------------------------------------------------
# ARCHIVE
# -------------------------------------------------------------------------

# Try again (defor)
tmfdefor_anyyear2 <- read.csv("native_validation/anyyear/tmf_defor_anyyear_manual.csv")
tmfdefor_anyyear2_stats <- valstats(tmfdefor_anyyear2, stratmap, "anyyear", "tmf_defor_anyyear_manual")

# Try again (dist)
tmfdist_anyyear2 <- read.csv("native_validation/anyyear/tmf_dist2_anyyear.csv")
tmfdist_anyyear2_stats <- valstats(tmfdist_anyyear2, stratmap, "anyyear", "tmf_dist2_anyyear")

# Try again
tmfdefor2_anyyear_buff <- read.csv("native_validation/anyyear/tmf_defor_buff_anyyear2.csv")


tmfdefor_timeinsensitive2 <- read.csv("native_validation/timeinsensitive/tmf_defor_timeinsensitive2.csv")
tmfdefor_timeinsensitive2_stats <- valstats(tmfdefor_timeinsensitive2, stratmap, "timeinsensitive", "tmf_defor_timeinsensitive2")

tmfdist_timeinsensitive2 <- read.csv("native_validation/timeinsensitive/tmf_dist_timeinsensitive2.csv")
tmfdist_timeinsensitive2_stats <- valstats(tmfdist_timeinsensitive2, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive2")



