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
  write.csv(stats_df, file=sprintf('ee_processed/%s/%s_stats.csv', folder,
                                   filename), row.names=FALSE)
  
  # Export confusion matrix
  write.csv(stats$matrix, file=sprintf('ee_processed/%s/%s_cm.csv', folder,
                                       filename))
  
  # Print statement
  print("Stehman statistics calculated and saved to file")
  
  # return(stats)
}


# -------------------------------------------------------------------------
# TIME INSENSITIVE (PIXEL)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_timeinsensitive <- read.csv("ee_processed/timeinsensitive/gfc_timeinsensitive.csv")
gfc_timeinsensitive_fm <- read.csv("ee_processed/timeinsensitive/gfc_timeinsensitive_fm.csv")
gfc_timeinsensitive_tm <- read.csv("ee_processed/timeinsensitive/gfc_timeinsensitive_tm.csv")
tmf_dist_timeinsensitive <- read.csv("ee_processed/timeinsensitive/tmf_dist_timeinsensitive.csv")
tmf_dist_timeinsensitive_fm <- read.csv("ee_processed/timeinsensitive/tmf_dist_timeinsensitive_fm.csv")
tmf_dist_timeinsensitive_tm <- read.csv("ee_processed/timeinsensitive/tmf_dist_timeinsensitive_tm.csv")

# Calculate statistics
valstats(gfc_timeinsensitive, stratmap, "timeinsensitive", "gfc_timeinsensitive")
valstats(gfc_timeinsensitive_fm, stratmap, "timeinsensitive", "gfc_timeinsensitive_fm")
valstats(gfc_timeinsensitive_tm, stratmap, "timeinsensitive", "gfc_timeinsensitive_tm")
valstats(tmf_dist_timeinsensitive, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive")
valstats(tmf_dist_timeinsensitive_fm, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive_fm")
valstats(tmf_dist_timeinsensitive_tm, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive_tm")


# -------------------------------------------------------------------------
# TIME INSENSITIVE (WINDOW)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_timeinsensitive_window <- read.csv("ee_processed/timeinsensitive/gfc_timeinsensitive_window.csv")
gfc_timeinsensitive_fm_window <- read.csv("ee_processed/timeinsensitive/gfc_timeinsensitive_fm_window.csv")
gfc_timeinsensitive_tm_window <- read.csv("ee_processed/timeinsensitive/gfc_timeinsensitive_tm_window.csv")
tmf_dist_timeinsensitive_window <- read.csv("ee_processed/timeinsensitive/tmf_dist_timeinsensitive_window.csv")
tmf_dist_timeinsensitive_fm_window <- read.csv("ee_processed/timeinsensitive/tmf_dist_timeinsensitive_fm_window.csv")
tmf_dist_timeinsensitive_tm_window <- read.csv("ee_processed/timeinsensitive/tmf_dist_timeinsensitive_tm_window.csv")

# Calculate statistics
valstats(gfc_timeinsensitive_window, stratmap, "timeinsensitive", "gfc_timeinsensitive_window")
valstats(gfc_timeinsensitive_fm_window, stratmap, "timeinsensitive", "gfc_timeinsensitive_fm_window")
valstats(gfc_timeinsensitive_tm_window, stratmap, "timeinsensitive", "gfc_timeinsensitive_tm_window")
valstats(tmf_dist_timeinsensitive_window, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive_window")
valstats(tmf_dist_timeinsensitive_fm_window, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive_fm_window")
valstats(tmf_dist_timeinsensitive_tm_window, stratmap, "timeinsensitive", "tmf_dist_timeinsensitive_tm_window")


# -------------------------------------------------------------------------
# FIRST YEAR (PIXEL)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_firstyear <- read.csv("ee_processed/firstyear/gfc_firstyear.csv")
gfc_firstyear_fm <- read.csv("ee_processed/firstyear/gfc_firstyear_fm.csv")
gfc_firstyear_tm <- read.csv("ee_processed/firstyear/gfc_firstyear_tm.csv")
tmf_dist_firstyear<- read.csv("ee_processed/firstyear/tmf_dist_firstyear.csv")
tmf_dist_firstyear_fm <- read.csv("ee_processed/firstyear/tmf_dist_firstyear_fm.csv")
tmf_dist_firstyear_tm <- read.csv("ee_processed/firstyear/tmf_dist_firstyear_tm.csv")

# Calculate statistics
valstats(gfc_firstyear, stratmap, "firstyear", "gfc_firstyear")
valstats(gfc_firstyear_fm, stratmap, "firstyear", "gfc_fistyear_fm")
valstats(gfc_firstyear_tm, stratmap, "firstyear", "gfc_firstyear_tm")
valstats(tmf_dist_firstyear, stratmap, "firstyear", "tmf_dist_firstyear")
valstats(tmf_dist_firstyear_fm, stratmap, "firstyear", "tmf_dist_firstyear_fm")
valstats(tmf_dist_firstyear_tm, stratmap, "firstyear", "tmf_dist_firstyear_tm")


# -------------------------------------------------------------------------
# FIRST YEAR (WINDOW)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_firstyear_window <- read.csv("ee_processed/firstyear/gfc_firstyear_window.csv")
gfc_firstyear_fm_window <- read.csv("ee_processed/firstyear/gfc_firstyear_fm_window.csv")
gfc_firstyear_tm_window <- read.csv("ee_processed/firstyear/gfc_firstyear_tm_window.csv")
tmf_dist_firstyear_window <- read.csv("ee_processed/firstyear/tmf_dist_firstyear_window.csv")
tmf_dist_firstyear_fm_window <- read.csv("ee_processed/firstyear/tmf_dist_firstyear_fm_window.csv")
tmf_dist_firstyear_tm_window <- read.csv("ee_processed/firstyear/tmf_dist_firstyear_tm_window.csv")

# Calculate statistics
valstats(gfc_firstyear_window, stratmap, "firstyear", "gfc_firstyear_window")
valstats(gfc_firstyear_fm_window, stratmap, "firstyear", "gfc_firstyear_fm_window")
valstats(gfc_firstyear_tm_window, stratmap, "firstyear", "gfc_firstyear_tm_window")
valstats(tmf_dist_firstyear_window, stratmap, "firstyear", "tmf_dist_firstyear_window")
valstats(tmf_dist_firstyear_fm_window, stratmap, "firstyear", "tmf_dist_firstyear_fm_window")
valstats(tmf_dist_firstyear_tm_window, stratmap, "firstyear", "tmf_dist_firstyear_tm_window")


# -------------------------------------------------------------------------
# ANY YEAR (PIXEL)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_anyyear <- read.csv("ee_processed/anyyear/gfc_anyyear.csv")
gfc_anyyear_fm <- read.csv("ee_processed/anyyear/gfc_anyyear_fm.csv")
gfc_anyyear_tm <- read.csv("ee_processed/anyyear/gfc_anyyear_tm.csv")
tmf_dist_anyyear<- read.csv("ee_processed/anyyear/tmf_dist_anyyear.csv")
tmf_dist_anyyear_fm <- read.csv("ee_processed/anyyear/tmf_dist_anyyear_fm.csv")
tmf_dist_anyyear_tm <- read.csv("ee_processed/anyyear/tmf_dist_anyyear_tm.csv")

# Calculate statistics
valstats(gfc_anyyear, stratmap, "anyyear", "gfc_anyyear")
valstats(gfc_anyyear_fm, stratmap, "anyyear", "gfc_anyyear_fm")
valstats(gfc_anyyear_tm, stratmap, "anyyear", "gfc_anyyear_tm")
valstats(tmf_dist_anyyear, stratmap, "anyyear", "tmf_dist_anyyear")
valstats(tmf_dist_anyyear_fm, stratmap, "anyyear", "tmf_dist_anyyear_fm")
valstats(tmf_dist_anyyear_tm, stratmap, "anyyear", "tmf_dist_anyyear_tm")


# -------------------------------------------------------------------------
# ANY YEAR (WINDOW)
# -------------------------------------------------------------------------
# Read map and reference labels
gfc_anyyear_window <- read.csv("ee_processed/anyyear/gfc_anyyear_window.csv")
gfc_anyyear_fm_window <- read.csv("ee_processed/anyyear/gfc_anyyear_fm_window.csv")
gfc_anyyear_tm_window <- read.csv("ee_processed/anyyear/gfc_anyyear_tm_window.csv")
tmf_dist_anyyear_window <- read.csv("ee_processed/anyyear/tmf_dist_anyyear_window.csv")
tmf_dist_anyyear_fm_window <- read.csv("ee_processed/anyyear/tmf_dist_anyyear_fm_window.csv")
tmf_dist_anyyear_tm_window <- read.csv("ee_processed/anyyear/tmf_dist_anyyear_tm_window.csv")

# Calculate statistics
valstats(gfc_anyyear_window, stratmap, "anyyear", "gfc_anyyear_window")
valstats(gfc_anyyear_fm_window, stratmap, "anyyear", "gfc_anyyear_fm_window")
valstats(gfc_anyyear_tm_window, stratmap, "anyyear", "gfc_anyyear_tm_window")
valstats(tmf_dist_anyyear_window, stratmap, "anyyear", "tmf_dist_anyyear_window")
valstats(tmf_dist_anyyear_fm_window, stratmap, "anyyear", "tmf_dist_anyyear_fm_window")
valstats(tmf_dist_anyyear_tm_window, stratmap, "anyyear", "tmf_dist_anyyear_tm_window")



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



