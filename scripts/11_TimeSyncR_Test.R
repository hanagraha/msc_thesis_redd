library(devtools)
install_github("bendv/timeSyncR")
library(timeSyncR)

setwd("C:\\Users\\hanna\\Documents\\WUR MSc\\MSc Thesis\\redd-thesis\\data")
dir.create("test")
fl <- system.file("extdata", "MDD.tar.gz", package = "timeSyncR")
newfl <- "test/MDD.tar.gz"
file.copy(fl, newfl)

untar(newfl, exdir = "test")

r <- brick("test/MDD_red.grd")
g <- brick("test/MDD_green.grd")
b <- brick("test/MDD_blue.grd")

nlayers(r)
plot(g, 1:9)

# layer names correspond to Landsat ID's
names(b)
# see ?getSceneinfo to get more info out of these
s <- getSceneinfo(names(b))
print(s)

# each brick has a z-dimension containing date info
getZ(g)
# also found in the date column of the getSceneinfo() data.frame
s$date
all(getZ(g) == s$date)

xy <- c(472000, -1293620)
# save original (default) plotting parameters to workspace
op <- par()

tsChipsRGB(xr = r, xg = g, xb = b, loc = xy, start = "2000-01-01")
# reset plotting parameters (plotRGB() changes parameters internally)
par(op)

