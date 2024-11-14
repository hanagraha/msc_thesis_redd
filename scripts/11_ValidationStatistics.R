
# Set working directory
setwd("C:/Users/hanna/Documents/WUR MSc/MSc Thesis/redd-thesis")

# Install packages
install.packages("terra")
install.packages("mapaccuracy")

# Load packages
library(terra)
library(mapaccuracy)

# Read data
valdata <- read.csv("data/validation/validation_points_labelled.csv", sep = ";")
stratdata <- rast("data/intermediate/stratification_layer_nogrnp.tif")

# Extract data of interest
strata <- valdata$strata
gfc <- valdata$gfc
tmf <- valdata$tmf
se <- valdata$se
val1 <- valdata$defor1

# Calculate number of pixels per strata
pixel_counts <- freq(stratdata, digits = 0)
strata_counts <- pixel_counts$count
names(strata_counts) <- pixel_counts$value

# Calculate statistics
stehman2014(strata, val1, gfc, strata_counts)

























