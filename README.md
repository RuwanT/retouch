# retouch

Code for [RETOUCH challenge](https://retouch.grand-challenge.org)
 
 
 
## Installation

`>> pip install SimpleITK`

## File description

* `prepare_data.py` : Read all OCT images and segmentations and do the following:
    1) if oct segmentation -> write each slice as tiff file
    2) if oct image and vendor spectralasis -> write eac slice
    3) if oct image and vendor Cirrius -> match histogram , do median filtering and write to tiff
    
    also writes a slice information csv
    
* `custom_networks.py` : Keras model definitions

* `image_preprocessing.py` : deprecated 