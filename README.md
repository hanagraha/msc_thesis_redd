# REDD+ Thesis

## Description
This project is part of Hannah Graham's MSc thesis on comparing Global Forest Change and Tropical Moist Forest datasets for their suitability in REDD+ evaluation. 

## Usage
To reproduce results, first create the environment using the thesisenv.yaml file. All scripts are found in the scripts folder and should be run sequentially. This repository does not include village and grnp polygon data, which contains sensitive information. The scripts assume these datasets exist as shapefiles in the filepaths: "data/gola gazetted polygon/ Gola_Gazetted_Polygon.shp" and "data/village polygons/village_polygons.shp" respectively. In case of errors, the package rasterstats may need to be updated in the command line using code: pip install rasterstats --upgrade. 

**00_SetUpDirectory.py** no inputs. outputs subfolders in "data" folder, it previously nonexistent

**01_DataDownload.py** no inputs. downloads raster data from web. outputs raw hansen and jrc tif files from 2013-2023. 

**02_DeforestationData_PreProcessing.py** input: all raw hansen and jrc tif files, village and grnp polygons. output: preprocessed hansen and jrc, singleyear lossyear, singleyear deforestation, singleyear degradation, multiyear defordegra, singleyear defordegra

**03_PlanetData_PreProcessing.py** input: planet tifs, villages, grnp, output: planet tifs

**04_RQ1a_DeforestationRateComparison.py** input: singleyear lossyear, singleyear deforestation, singleyear degradation, annual change, transition map main classes, villages, grnp

**05_RQ1a_TransitionMapConversion.py** input: singleyear defordegra, annual chance, singleyear lossyear, transition map main classes

**06_RQ1b_SpatialAgreement_DeforestationPixels.py** input: multiyear lossyear, gfc_tmf_combyear (?? - should be defordegra?), villages, grnp, output: singleyear agreement_gfc_combtmf, singleyear gfc_simple_binary, singleyear tmf_simple_binary, multiyear gfc_tmf_sensitive_early

**07_RQ1b_SpatialAgreement_DeforestationPatches.py** input: singleyear lossyear, singleyear defordegra, singleyear agreement, villages, grnp, output: singleyear gfc forclust, singleyear tmf forclust, singleyear agreement clusters, singleyear disagreement clusters, 

**08_RQ2a_ValidationSampling.py** input: singleyear agreement, grnp, villages. output: stratification tif, validation points shp, 

**09_RQ2a_ValidationLabelling.py** input: planet tifs, valpoints shp, 

## Authors and acknowledgment
All scripts are created by Hannah Graham, with help from supervisors Dr. Nandika Tsendbazar and Dr. Maarten Voors. 

## License
This project is licensed under the MIT open source license. It is intended for academic use. 

## Project status
This project began in August and is expected to be finalized in February. 
