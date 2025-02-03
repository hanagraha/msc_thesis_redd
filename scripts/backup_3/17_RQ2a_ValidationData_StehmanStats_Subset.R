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
stratmap <- rast("data/intermediate/stratification_layer_buffered.tif")

# Define function to calculate and save statistics (specifically for prota)
valstats_a <- function(valdata, stratmap, varname, protname, ext = NULL){
  
  # Extract strata
  strata <- valdata$strata
  
  # Extract validation deforestation
  val_defor <- valdata['prot_a_val']
  
  # Extract predicted deforestation
  pred_defor <- valdata['prot_a_pred']
  
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
  subfolder <- sprintf("val_%s_sub", protname)
  
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
  subfolder <- sprintf("val_%s_sub", protname)
  
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

# Define function to extract overall accuracies
ov_acc <- function(stats_list) {
  
  # Initialize an empty data frame
  results <- data.frame(
    Variable = character(),
    OA = numeric(),
    SEoa = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop through the list and extract OA and SEoa
  for (var_name in names(stats_list)) {
    stats <- stats_list[[var_name]]
    results <- rbind(
      results,
      data.frame(
        Variable = var_name,
        OA = stats$OA,
        SEoa = stats$SEoa,
        stringsAsFactors = FALSE
      )
    )
  }
  
  return(results)
}

############################################################################


# STATISTICS FOR PROTOCOL A


############################################################################
# Read gfc validation datasets
gfc_prota <- read.csv("data/validation/val_prota_sub/prota_gfc.csv")
gfc_prota_redd <- read.csv("data/validation/val_prota_sub/prota_gfc_redd.csv")
gfc_prota_nonredd <- read.csv("data/validation/val_prota_sub/prota_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_prota <- read.csv("data/validation/val_prota_sub/prota_tmf.csv")
tmf_prota_redd <- read.csv("data/validation/val_prota_sub/prota_tmf_redd.csv")
tmf_prota_nonredd <- read.csv("data/validation/val_prota_sub/prota_tmf_nonredd.csv")

# Read se validation datasets
se_prota <- read.csv("data/validation/val_prota_sub/prota_se.csv")
se_prota_redd <- read.csv("data/validation/val_prota_sub/prota_se_redd.csv")
se_prota_nonredd <- read.csv("data/validation/val_prota_sub/prota_se_nonredd.csv")

# Calculate statistics for gfc
gfc_prota_stats <- valstats_a(gfc_prota, stratmap, "gfc", "prota")
gfc_prota_redd_stats <- valstats_a(gfc_prota_redd, stratmap, "gfc", "prota", "redd")
gfc_prota_nonredd_stats <- valstats_a(gfc_prota_nonredd, stratmap, "gfc", "prota", "nonredd")

# Calculate statistics for tmf
tmf_prota_stats <- valstats_a(tmf_prota, stratmap, "tmf", "prota")
tmf_prota_redd_stats <- valstats_a(tmf_prota_redd, stratmap, "tmf", "prota", "redd")
tmf_prota_nonredd_stats <- valstats_a(tmf_prota_nonredd, stratmap, "tmf", "prota", "nonredd")

# Calculate statistics for se
se_prota_stats <- valstats_a(se_prota, stratmap, "se", "prota")
se_prota_redd_stats <- valstats_a(se_prota_redd, stratmap, "se", "prota", "redd")
se_prota_nonredd_stats <- valstats_a(se_prota_nonredd, stratmap, "se", "prota", "nonredd")

# Store variables in list
prota_stats <- list(
  gfc_prota_stats = gfc_prota_stats, 
  gfc_prota_redd_stats = gfc_prota_redd_stats, 
  gfc_prota_nonredd_stats = gfc_prota_nonredd_stats, 
  tmf_prota_stats = tmf_prota_stats, 
  tmf_prota_redd_stats = tmf_prota_redd_stats, 
  tmf_prota_nonredd_stats = tmf_prota_nonredd_stats, 
  se_prota_stats = se_prota_stats, 
  se_prota_redd_stats = se_prota_redd_stats, 
  se_prota_nonredd_stats = se_prota_nonredd_stats
)

# Extract overall accuracies
prota_acc <- ov_acc(prota_stats)

# Write to file
write.csv(prota_acc, "data/validation/val_prota_sub/prota_overall_accuracy.csv", 
          row.names = FALSE)



############################################################################


# STATISTICS FOR PROTOCOL B


############################################################################
# Read gfc validation datasets
gfc_protb <- read.csv("data/validation/val_protb_sub/protb_gfc.csv")
gfc_protb_redd <- read.csv("data/validation/val_protb_sub/protb_gfc_redd.csv")
gfc_protb_nonredd <- read.csv("data/validation/val_protb_sub/protb_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_protb <- read.csv("data/validation/val_protb_sub/protb_tmf.csv")
tmf_protb_redd <- read.csv("data/validation/val_protb_sub/protb_tmf_redd.csv")
tmf_protb_nonredd <- read.csv("data/validation/val_protb_sub/protb_tmf_nonredd.csv")

# Read se validation datasets
se_protb <- read.csv("data/validation/val_protb_sub/protb_se.csv")
se_protb_redd <- read.csv("data/validation/val_protb_sub/protb_se_redd.csv")
se_protb_nonredd <- read.csv("data/validation/val_protb_sub/protb_se_nonredd.csv")

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

# Store variables in list
protb_stats <- list(
  gfc_protb_stats = gfc_protb_stats, 
  gfc_protb_redd_stats = gfc_protb_redd_stats, 
  gfc_protb_nonredd_stats = gfc_protb_nonredd_stats, 
  tmf_protb_stats = tmf_protb_stats, 
  tmf_protb_redd_stats = tmf_protb_redd_stats, 
  tmf_protb_nonredd_stats = tmf_protb_nonredd_stats, 
  se_protb_stats = se_protb_stats, 
  se_protb_redd_stats = se_protb_redd_stats, 
  se_protb_nonredd_stats = se_protb_nonredd_stats
)

# Extract overall accuracies
protb_acc <- ov_acc(protb_stats)

# Write to file
write.csv(protb_acc, "data/validation/val_protb_sub/protb_overall_accuracy.csv", 
          row.names = FALSE)



############################################################################


# STATISTICS FOR PROTOCOL C


############################################################################
# Read gfc validation datasets
gfc_protc <- read.csv("data/validation/val_protc_sub/protc_gfc.csv")
gfc_protc_redd <- read.csv("data/validation/val_protc_sub/protc_gfc_redd.csv")
gfc_protc_nonredd <- read.csv("data/validation/val_protc_sub/protc_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_protc <- read.csv("data/validation/val_protc_sub/protc_tmf.csv")
tmf_protc_redd <- read.csv("data/validation/val_protc_sub/protc_tmf_redd.csv")
tmf_protc_nonredd <- read.csv("data/validation/val_protc_sub/protc_tmf_nonredd.csv")

# Read se validation datasets
se_protc <- read.csv("data/validation/val_protc_sub/protc_se.csv")
se_protc_redd <- read.csv("data/validation/val_protc_sub/protc_se_redd.csv")
se_protc_nonredd <- read.csv("data/validation/val_protc_sub/protc_se_nonredd.csv")

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

# Store variables in a list
protc_stats <- list(
  gfc_protc_stats = gfc_protc_stats, 
  gfc_protc_redd_stats = gfc_protc_redd_stats, 
  gfc_protc_nonredd_stats = gfc_protc_nonredd_stats, 
  tmf_protc_stats = tmf_protc_stats, 
  tmf_protc_redd_stats = tmf_protc_redd_stats, 
  tmf_protc_nonredd_stats = tmf_protc_nonredd_stats, 
  se_protc_stats = se_protc_stats, 
  se_protc_redd_stats = se_protc_redd_stats, 
  se_protc_nonredd_stats = se_protc_nonredd_stats
)

# Extract overall accuracies
protc_acc <- ov_acc(protc_stats)

# Write to file
write.csv(protc_acc, "data/validation/val_protc_sub/protc_overall_accuracy.csv", 
          row.names = FALSE)



############################################################################


# STATISTICS FOR PROTOCOL D


############################################################################
# Read gfc validation datasets
gfc_protd <- read.csv("data/validation/val_protd_sub/protd_gfc.csv")
gfc_protd_redd <- read.csv("data/validation/val_protd_sub/protd_gfc_redd.csv")
gfc_protd_nonredd <- read.csv("data/validation/val_protd_sub/protd_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_protd <- read.csv("data/validation/val_protd_sub/protd_tmf.csv")
tmf_protd_redd <- read.csv("data/validation/val_protd_sub/protd_tmf_redd.csv")
tmf_protd_nonredd <- read.csv("data/validation/val_protd_sub/protd_tmf_nonredd.csv")

# Read se validation datasets
se_protd <- read.csv("data/validation/val_protd_sub/protd_se.csv")
se_protd_redd <- read.csv("data/validation/val_protd_sub/protd_se_redd.csv")
se_protd_nonredd <- read.csv("data/validation/val_protd_sub/protd_se_nonredd.csv")

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

# Store variables in a list
protd_stats <- list(
  gfc_protd_stats = gfc_protd_stats, 
  gfc_protd_redd_stats = gfc_protd_redd_stats, 
  gfc_protd_nonredd_stats = gfc_protd_nonredd_stats, 
  tmf_protd_stats = tmf_protd_stats, 
  tmf_protd_redd_stats = tmf_protd_redd_stats, 
  tmf_protd_nonredd_stats = tmf_protd_nonredd_stats, 
  se_protd_stats = se_protd_stats, 
  se_protd_redd_stats = se_protd_redd_stats, 
  se_protd_nonredd_stats = se_protd_nonredd_stats
)

# Extract overall accuracies
protd_acc <- ov_acc(protd_stats)

# Write to file
write.csv(protd_acc, "data/validation/val_protd_sub/protd_overall_accuracy.csv", 
          row.names = FALSE)



############################################################################


# STATISTICS FOR PROTOCOL E


############################################################################
# Read gfc validation datasets
gfc_prote <- read.csv("data/validation/val_prote_sub/prote_gfc.csv")
gfc_prote_redd <- read.csv("data/validation/val_prote_sub/prote_gfc_redd.csv")
gfc_prote_nonredd <- read.csv("data/validation/val_prote_sub/prote_gfc_nonredd.csv")

# Read tmf validation datasets
tmf_prote <- read.csv("data/validation/val_prote_sub/prote_tmf.csv")
tmf_prote_redd <- read.csv("data/validation/val_prote_sub/prote_tmf_redd.csv")
tmf_prote_nonredd <- read.csv("data/validation/val_prote_sub/prote_tmf_nonredd.csv")

# Read se validation datasets
se_prote <- read.csv("data/validation/val_prote_sub/prote_se.csv")
se_prote_redd <- read.csv("data/validation/val_prote_sub/prote_se_redd.csv")
se_prote_nonredd <- read.csv("data/validation/val_prote_sub/prote_se_nonredd.csv")

# Calculate statistics for gfc
gfc_prote_stats <- valstats(gfc_prote, stratmap, "gfc", "prote")
gfc_prote_redd_stats <- valstats(gfc_prote_redd, stratmap, "gfc", "prote", "redd")
gfc_prote_nonredd_stats <- valstats(gfc_prote_nonredd, stratmap, "gfc", "prote", "nonredd")

# Calculate statistics for tmf
tmf_prote_stats <- valstats(tmf_prote, stratmap, "tmf", "prote")
tmf_prote_redd_stats <- valstats(tmf_prote_redd, stratmap, "tmf", "prote", "redd")
tmf_prote_nonredd_stats <- valstats(tmf_prote_nonredd, stratmap, "tmf", "prote", "nonredd")

# Calculate statistics for se
se_prote_stats <- valstats(se_prote, stratmap, "se", "prote")
se_prote_redd_stats <- valstats(se_prote_redd, stratmap, "se", "prote", "redd")
se_prote_nonredd_stats <- valstats(se_prote_nonredd, stratmap, "se", "prote", "nonredd")

# Store variables in a list
prote_stats <- list(
  gfc_prote_stats = gfc_prote_stats, 
  gfc_prote_redd_stats = gfc_prote_redd_stats, 
  gfc_prote_nonredd_stats = gfc_prote_nonredd_stats, 
  tmf_prote_stats = tmf_prote_stats, 
  tmf_prote_redd_stats = tmf_prote_redd_stats, 
  tmf_prote_nonredd_stats = tmf_prote_nonredd_stats, 
  se_prote_stats = se_prote_stats, 
  se_prote_redd_stats = se_prote_redd_stats, 
  se_prote_nonredd_stats = se_prote_nonredd_stats
)

# Extract overall accuracies
prote_acc <- ov_acc(prote_stats)

# Write to file
write.csv(prote_acc, "data/validation/val_prote_sub/prote_overall_accuracy.csv", 
          row.names = FALSE)




