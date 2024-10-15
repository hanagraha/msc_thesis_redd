# REDD+ Thesis

## Description
This project is part of Hannah Graham's MSc thesis on comparing Global Forest Change and Tropical Moist Forest datasets for their suitability in REDD+ evaluation. 

## Usage
To reproduce results, first create the environment using the thesisenv.yaml file. All scripts are found in the scripts folder and should be run sequentially. This repository does not include village and grnp polygon data, which contains sensitive information. The scripts assume these datasets exist as shapefiles in the filepaths: "data/gola gazetted polygon/ Gola_Gazetted_Polygon.shp" and "data/village polygons/village_polygons.shp" respectively. In case of errors, the package rasterstats may need to be updated in the command line using code: pip install rasterstats --upgrade. 

**00_SetUpDirectory.py** no inputs. outputs subfolders in "data" folder, it previously nonexistent

**01_DataDownload.py** no inputs. downloads raster data from web. outputs raw hansen and jrc tif files from 2013-2023. 

**02_PreProcessing.py** inputs all raw hansen and jrc tif files, village and grnp polygons. 

**03_RQ1a_DeforestationRateComparison.py**

**04_RQ1a_ForestPatchAnalysis.py**

**04_RQ1b_SpatialAgreement.py**

**05_RQ2a_DeforestationValidation.py** inputs spatial agreement tif files and grnp polygon. outputs sample point coordinates shapefile. 

## Authors and acknowledgment
All scripts are created by Hannah Graham, with help from supervisors Dr. Nandika Tsendbazar and Dr. Maarten Voors. 

## License
This project is licensed under the MIT open source license. It is intended for academic use. 

## Project status
This project began in August and is expected to be finalized in February. 
