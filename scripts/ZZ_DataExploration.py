# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:39:18 2024

@author: hanna
"""


############################################################################


# IMPORT PACKAGES


############################################################################

import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt



############################################################################


# IMPORT DATASETS


############################################################################


import requests

# URL of the file to download
url = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# Local filename to save the file
local_filename = "Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif"

# Send GET request to the URL
response = requests.get(url, stream=True)

# Open the file in write-binary mode and write the content
with open(local_filename, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print("Download complete.")




    
with rasterio.open("data/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as defor:
    deforestation_year = defor.read(1)

with rasterio.open("data/JRC_TMF_DeforestationYear_INT_1982_2023_v1_AFR_ID51_N10_W20.tif") as degra:
    degradation_year = degra.read(1)
    
with rasterio.open("data/Hansen_GFC-2019-v1.7_lossyear_10N_020W.tif") as gfc:
    gfc_year = gfc.read(1)
    
with rasterio.open("Hansen_GFC-2022-v1.10_lossyear_10N_020W.tif") as gfc:
    gfc_year = gfc.read(1)
    
defor_min = deforestation_year.min()
defor_max = deforestation_year.max()
degra_min = degradation_year.min()
degra_max = degradation_year.max()



############################################################################


# PLOTTING


############################################################################


# plt.figure(figsize=(5, 5), dpi=300)  # adjust size and resolution
# show(dat, title='Chech Area', cmap='gist_ncar')