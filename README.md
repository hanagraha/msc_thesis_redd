# REDD+ Thesis

## Description
This project is part of Hannah Graham's MSc thesis on comparing Global Forest Change and Tropical Moist Forest datasets for their suitability in REDD+ evaluation. 

## Usage
To reproduce results, first create the environment using the thesisenv.yaml file. All scripts are found in the scripts folder and should be run sequentially. This repository does not include village and grnp polygon data, which contains sensitive information. The scripts assume these datasets exist as shapefiles in the filepaths: "data/gola gazetted polygon/ Gola_Gazetted_Polygon.shp" and "data/village polygons/village_polygons.shp" respectively. In case of errors, the package rasterstats may need to be updated in the command line using code: pip install rasterstats --upgrade. 

## Authors and acknowledgment
All scripts are created by Hannah Graham, with help from supervisors Dr. Nandika Tsendbazar and Dr. Maarten Voors. 

## License
This project is licensed under the MIT open source license. It is intended for academic use. 

## Project status
This project began in August 2024 and was successfully defended in March 2025. 
