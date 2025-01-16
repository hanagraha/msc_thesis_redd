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


# SET UP DIRECTORY AND STATISTICS


############################################################################
# Set working directory
setwd("C:/Users/hanna/Documents/WUR MSc/MSc Thesis/redd-thesis")

# Read stratification map
stratmap <- rast("data/intermediate/stratification_layer_nogrnp.tif")

# Define function to calculate and save statistics
valstats <- function(valdata, stratmap, varname, protname, ext = NULL){
  
  # Extract strata
  strata <- valdata$strata
  
  # Extract validation column name
  val_col <- sub("(.)$", "_\\1", protname)
    
  # Extract validation deforestation
  val_defor <- valdata[[val_col]]
  
  # Extract predicted deforestation
  pred_defor <- valdata[[varname]]
  
  # Calculate number of pixels per strata
  pixel_counts <- freq(stratmap, digits = 0)
  
  # Extract only pixel counts
  strata_counts <- pixel_counts$count
  
  # Assign strata values as name
  names(strata_counts) <- pixel_counts$value
  
  # Calculate statistics
  stats <- stehman2014(strata, val_defor, pred_defor, strata_counts)
  
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
  
  # Define subfolder
  subfolder <- sprintf("val_%s", protname)
  
  # Define suffix
  suffix <- if (!is.null(ext)) paste0("_", ext) else ""
  
  # Export annual data
  write.csv(stats_df, 
            file = sprintf("data/validation/%s/%s_%s%s_stehmanstats.csv", 
                           subfolder, protname, varname, suffix), 
            row.names = FALSE)
  
  # Export confusion matrix
  write.csv(stats$matrix, 
            file = sprintf("data/validation/%s/%s_%s%s_confmatrix.csv", 
                           subfolder, protname, varname, suffix))
  
  # Print statement
  print("Stehman statistics calculated and saved to file")
  
  return(stats)
}

############################################################################


# STATISTICS FOR PROTOCOL A


############################################################################
# Read gfc validation datasets
gfc_prota <- read.csv("data/validation/val_prota/prota_gfc.csv")
gfc_prota_redd <- read.csv("data/validation/val_prota/prota_gfc_redd.csv")
gfc_prota_nonredd <- read.csv("data/validation/val_prota/prota_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_prota <- read.csv("data/validation/val_prota/prota_tmf.csv")
tmf_prota_redd <- read.csv("data/validation/val_prota/prota_tmf_redd.csv")
tmf_prota_nonredd <- read.csv("data/validation/val_prota/prota_tmf_nonredd.csv")

# Read se validation datasets
se_prota <- read.csv("data/validation/val_prota/prota_se.csv")
se_prota_redd <- read.csv("data/validation/val_prota/prota_se_redd.csv")
se_prota_nonredd <- read.csv("data/validation/val_prota/prota_se_nonredd.csv")

# Calculate statistics for gfc
gfc_prota_stats <- valstats(gfc_prota$prot_a, stratmap, gfc_prota$gfc, "prota_gfc")
gfc_prota_redd_stats <- valstats(gfc_prota_redd$prot_a, stratmap, gfc_prota_redd$gfc, "prota_gfc_redd")


# Calculate statistics for tme
tmf_stats <- valstats(tmf, valdata$tmf_val, "tmf_1380")

# Calculate statistics for se
se_stats <- valstats(se, valdata$se_val, "se_1380")



############################################################################


# STATISTICS FOR PROTOCOL B


############################################################################
# Read gfc validation datasets
gfc_protb <- read.csv("data/validation/val_protb/protb_gfc.csv")
gfc_protb_redd <- read.csv("data/validation/val_protb/protb_gfc_redd.csv")
gfc_protb_nonredd <- read.csv("data/validation/val_protb/protb_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_protb <- read.csv("data/validation/val_protb/protb_tmf.csv")
tmf_protb_redd <- read.csv("data/validation/val_protb/protb_tmf_redd.csv")
tmf_protb_nonredd <- read.csv("data/validation/val_protb/protb_tmf_nonredd.csv")

# Read se validation datasets
se_protb <- read.csv("data/validation/val_protb/protb_se.csv")
se_protb_redd <- read.csv("data/validation/val_protb/protb_se_redd.csv")
se_protb_nonredd <- read.csv("data/validation/val_protb/protb_se_nonredd.csv")

# Calculate statistics for gfc
gfc_protb_stats <- valstats(gfc_protb, stratmap, "gfc", "protb")
gfc_protb_redd_stats <- valstats(gfc_protb_redd, stratmap, "gfc", "protb", "redd")
gfc_protb_nonredd_stats <- valstats(gfc_protb_nonredd, stratmap, "gfc", "protb", "nonredd")

# Calculate statistics for tmf
tmf_protb_stats <- valstats(tmf_protb, stratmap, "tmf", "protb")
tmf_protb_redd_stats <- valstats(tmf_protb_redd, stratmap, "tmf", "protb", "redd")
tmf_protb_nonredd_stats <- valstats(tmf_protb_nonredd, stratmap, "tmf", "protb", "nonredd")

# Calculate statistics for se
se_protb_stats <- valstats(se_protb, stratmap, "se", "protb")
se_protb_redd_stats <- valstats(se_protb_redd, stratmap, "se", "protb", "redd")
se_protb_nonredd_stats <- valstats(se_protb_nonredd, stratmap, "se", "protb", "nonredd")



############################################################################


# STATISTICS FOR PROTOCOL C


############################################################################
# Read gfc validation datasets
gfc_protc <- read.csv("data/validation/val_protc/protc_gfc.csv")
gfc_protc_redd <- read.csv("data/validation/val_protc/protc_gfc_redd.csv")
gfc_protc_nonredd <- read.csv("data/validation/val_protc/protc_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_protc <- read.csv("data/validation/val_protc/protc_tmf.csv")
tmf_protc_redd <- read.csv("data/validation/val_protc/protc_tmf_redd.csv")
tmf_protc_nonredd <- read.csv("data/validation/val_protc/protc_tmf_nonredd.csv")

# Read se validation datasets
se_protc <- read.csv("data/validation/val_protc/protc_se.csv")
se_protc_redd <- read.csv("data/validation/val_protc/protc_se_redd.csv")
se_protc_nonredd <- read.csv("data/validation/val_protc/protc_se_nonredd.csv")

# Calculate statistics for gfc
gfc_protc_stats <- valstats(gfc_protc, stratmap, "gfc", "protc")
gfc_protc_redd_stats <- valstats(gfc_protc_redd, stratmap, "gfc", "protc", "redd")
gfc_protc_nonredd_stats <- valstats(gfc_protc_nonredd, stratmap, "gfc", "protc", "nonredd")

# Calculate statistics for tmf
tmf_protc_stats <- valstats(tmf_protc, stratmap, "tmf", "protc")
tmf_protc_redd_stats <- valstats(tmf_protc_redd, stratmap, "tmf", "protc", "redd")
tmf_protc_nonredd_stats <- valstats(tmf_protc_nonredd, stratmap, "tmf", "protc", "nonredd")

# Calculate statistics for se
se_protc_stats <- valstats(se_protc, stratmap, "se", "protc")
se_protc_redd_stats <- valstats(se_protc_redd, stratmap, "se", "protc", "redd")
se_protc_nonredd_stats <- valstats(se_protc_nonredd, stratmap, "se", "protc", "nonredd")

############################################################################


# STATISTICS FOR PROTOCOL D


############################################################################
# Read gfc validation datasets
gfc_protd <- read.csv("data/validation/val_protd/protd_gfc.csv")
gfc_protd_redd <- read.csv("data/validation/val_protd/protd_gfc_redd.csv")
gfc_protd_nonredd <- read.csv("data/validation/val_protd/protd_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_protd <- read.csv("data/validation/val_protd/protd_tmf.csv")
tmf_protd_redd <- read.csv("data/validation/val_protd/protd_tmf_redd.csv")
tmf_protd_nonredd <- read.csv("data/validation/val_protd/protd_tmf_nonredd.csv")

# Read se validation datasets
se_protd <- read.csv("data/validation/val_protd/protd_se.csv")
se_protd_redd <- read.csv("data/validation/val_protd/protd_se_redd.csv")
se_protd_nonredd <- read.csv("data/validation/val_protd/protd_se_nonredd.csv")

# Calculate statistics for gfc
gfc_protd_stats <- valstats(gfc_protd, stratmap, "gfc", "protd")
gfc_protd_redd_stats <- valstats(gfc_protd_redd, stratmap, "gfc", "protd", "redd")
gfc_protd_nonredd_stats <- valstats(gfc_protd_nonredd, stratmap, "gfc", "protd", "nonredd")

# Calculate statistics for tmf
tmf_protd_stats <- valstats(tmf_protd, stratmap, "tmf", "protd")
tmf_protd_redd_stats <- valstats(tmf_protd_redd, stratmap, "tmf", "protd", "redd")
tmf_protd_nonredd_stats <- valstats(tmf_protd_nonredd, stratmap, "tmf", "protd", "nonredd")

# Calculate statistics for se
se_protd_stats <- valstats(se_protd, stratmap, "se", "protd")
se_protd_redd_stats <- valstats(se_protd_redd, stratmap, "se", "protd", "redd")
se_protd_nonredd_stats <- valstats(se_protd_nonredd, stratmap, "se", "protd", "nonredd")
