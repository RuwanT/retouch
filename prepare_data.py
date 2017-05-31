from utils import mhd
from utils.slice_op import hist_match
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFilter
import platform
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle
from skimage.filters.rank import median

if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/'
else:
    DATA_ROOT = '/Users/ruwant/DATA/retouch/'

IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3

# Prepare reference image for histogram matching (randomly selected from Spectralis dataset)
filepath_r = '/home/truwan/DATA/retouch/Spectralis/7501081e3e7577af524c6f7703d8d538/oct.mhd'
oct_r, _, _ = mhd.load_oct_image(filepath_r)

image_names = list()
for subdir, dirs, files in os.walk(DATA_ROOT):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith("reference.mhd"):
            image_name = filepath.split('/')[-2]
            vendor = filepath.split('/')[-3]
            img, _, _ = mhd.load_oct_seg(filepath)
            num_slices = img.shape[0]
            for slice_num in range(0, num_slices):
                im_slice = img[slice_num, :, :]
                image_names.append([image_name, vendor, subdir, slice_num, int(np.any(im_slice == IRF_CODE)),
                                    int(np.any(im_slice == SRF_CODE)), int(np.any(im_slice == PED_CODE))])
                im_slice = Image.fromarray(im_slice, mode='L')
                save_name = DATA_ROOT + 'pre_processed/oct_masks/' + vendor + '_' + image_name + '_' + str(slice_num).zfill(
                    3) + '.tiff'
                im_slice.save(save_name)

        elif filepath.endswith("oct.mhd"):
            image_name = filepath.split('/')[-2]
            vendor = filepath.split('/')[-3]
            img, _, _ = mhd.load_oct_image(filepath)
            if 'Cirrus' in vendor:
                img = hist_match(img, oct_r)
            num_slices = img.shape[0]
            for slice_num in range(0, num_slices):
                if 'Cirrus' in vendor and (slice_num > 0) and (slice_num < num_slices-1):
                    im_slice = np.median(img[slice_num-1:slice_num+2, :, :].astype(np.int32), axis=0).astype(np.int32)
                else:
                    im_slice = img[slice_num, :, :].astype(np.int32)
                im_slice = Image.fromarray(im_slice, mode='I')
                # TODO : check if this second filtering is useful
                if 'Cirrus' in vendor:
                    im_slice = im_slice.filter(ImageFilter.MedianFilter(size=3))
                save_name = DATA_ROOT + 'pre_processed/oct_imgs/' + vendor + '_' + image_name + '_' + str(slice_num).zfill(3) + '.tiff'
                im_slice.save(save_name)


col_names = ['image_name', 'vendor', 'root', 'slice', 'is_IRF', 'is_SRF', 'is_PED']
df = pd.DataFrame(image_names, columns=col_names)
df.to_csv(DATA_ROOT + 'pre_processed/slice_gt.csv', index=False)