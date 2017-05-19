"""
Train the slice classification network fro RETOUCH data

"""

from nutsflow import *
from nutsml import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def train_network():
    """
    
    :return: 
    """
    ImagePatchesByMask()



if __name__ == "__main__":
    import utils.mhd as mhd
    image_data, a, b = mhd.load_itk('/home/truwan/DATA/retouch/Spectralis/4a8a81b1c06072385738775dccdc7942/oct.mhd')
    print image_data.shape, a, b

    train_network()