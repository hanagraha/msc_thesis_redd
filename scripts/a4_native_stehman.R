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
tmfdefor_timeinsensitive <- read.csv("native_validation/timeinsensitive/tmf_defor_timeinsensitive.csv")
tmfdist_timeinsensitive <- read.csv("native_validation/timeinsensitive/tmf_dist_timeinsensitive.csv")

# Calculate statistics
gfc_timeinsensitive_stats <- valstats(gfc_timeinsensitive, stratmap, "timeinsensitive", "gfc_lossyear_timeinsensitive")
tmfdefor_timeinsensitive_stats <- valstats(tmfdefor_timeinsensitive, stratmap, "timeinsensitive", "tmf_defor_timeinsensitive")
tmfdist_timeinsensitive_stats <- valstats(tmfdist_timeinsensitive, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive")

tmfdefor_timeinsensitive2 <- read.csv("native_validation/timeinsensitive/tmf_defor_timeinsensitive2.csv")
tmfdefor_timeinsensitive2_stats <- valstats(tmfdefor_timeinsensitive2, stratmap, "timeinsensitive", "tmf_defor_timeinsensitive2")

tmfdist_timeinsensitive2 <- read.csv("native_validation/timeinsensitive/tmf_dist_timeinsensitive2.csv")
tmfdist_timeinsensitive2_stats <- valstats(tmfdist_timeinsensitive2, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive2")

# -------------------------------------------------------------------------
# TIME INSENSITIVE BUFFER STATISTICS
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_timeinsensitive_buff <- read.csv("native_validation/timeinsensitive/gfc_lossyear_buff_timeinsensitive.csv")
tmfdefor_timeinsensitive_buff <- read.csv("native_validation/timeinsensitive/tmf_defor_buff_timeinsensitive.csv")
tmfdisttimeinsensitive_buff <- read.csv("native_validation/timeinsensitive/tmf_dist_buff_timeinsensitive.csv")

# Calculate statistics
gfc_anyyear_buff_stats <- valstats(gfc_timeinsensitive_buff, stratmap, "timeinsensitive", "gfc_lossyear_buff_timeinsensitive")
tmfdefor_anyyear_buff_stats <- valstats(tmfdefor_timeinsensitive_buff, stratmap, "timeinsensitive", "tmf_defor_buff_timeinsensitive")
tmfdist_anyyear_buff_stats <- valstats(tmfdisttimeinsensitive_buff, stratmap, "timeinsensitive", "tmf_dist_buff_timeinsensitive")


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
gfc_anyyear_buff_stats <- valstats(gfc_anyyear_buff, stratmap, "anyyear", "gfc_lossyear_buff_anyyear")
tmfdefor_anyyear_buff_stats <- valstats(tmfdefor_anyyear_buff, stratmap, "anyyear", "tmf_defor_buff_anyyear")
tmfdist_anyyear_buff_stats <- valstats(tmfdist_anyyear_buff, stratmap, "anyyear", "tmf_dist_buff_anyyear")


tmfdefor2_anyyear_buff <- read.csv("native_validation/anyyear/tmf_defor_buff_anyyear2.csv")


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


# -------------------------------------------------------------------------
# STATISTICS FOR AREA ESTIMATION
# -------------------------------------------------------------------------
# Read map and reference labels
native_valdata <- read.csv("native_validation/validation_mapdata.csv")

# Filter data for gfc
gfc_area <- native_valdata[, c("strata", "defor1", "gfc_lossyear")]
colnames(gfc_area) <- c("strata", "ref", "map")

# Filter data for tmf defor year
tmfdefor_area <- native_valdata[, c("strata", "defor1", "tmf_deforyear")]
colnames(tmfdefor_area) <- c("strata", "ref", "map")

# Calculate statistics
native_gfc_stats <- valstats(gfc_area, stratmap, "area_estimation", "gfc_lossyear")
native_tmfdefor_stats <- valstats(tmfdefor_area, stratmap, "area_estimation", "tmf_deforyear")

# Filter data for second disturbance
gfc_area2 <- native_valdata[, c("strata", "defor2", "gfc_lossyear")]
colnames(gfc_area2) <- c("strata", "ref", "map")

# Filter data for third disturbance
gfc_area3 <- native_valdata[, c("strata", "defor3", "gfc_lossyear")]
colnames(gfc_area3) <- c("strata", "ref", "map")

# Calculate statistics
native_gfc2_stats <- valstats(gfc_area2, stratmap, "area_estimation", "gfc_lossyear2")
native_gfc3_stats <- valstats(gfc_area3, stratmap, "area_estimation", "gfc_lossyear3")







