import os
from utils import mhd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc, scipy.signal
import skimage.filters
import skimage
import skimage.transform
from skimage.morphology import disk
from skimage.filters.rank import entropy

IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3

DATA_ROOT = '/home/truwan/DATA/retouch/Cirrus/'
# DATA_ROOT = '/home/truwan/DATA/retouch/'
DATA_ROOT = '/Users/ruwant/DATA/retouch/'

from collections import defaultdict
count =0
image_names = list()
for subdir, dirs, files in os.walk(DATA_ROOT):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith("oct.mhd"):
            # img, _, _ = mhd.load_oct_image(filepath)
            # num_slices = img.shape[0]
            # info = defaultdict(int)
            # for slice_num in range(0, num_slices):
            #     im_slice = img[slice_num, :, :].astype(int)
            #
            #     if IRF_CODE in im_slice:
            #         info[IRF_CODE] += 1
            #     else:
            #         info[IRF_CODE] += 0
            #
            #     if SRF_CODE in im_slice:
            #         info[SRF_CODE] += 1
            #     else:
            #         info[SRF_CODE] += 0
            #
            #     if PED_CODE in im_slice:
            #         info[PED_CODE] += 1
            #     else:
            #         info[PED_CODE] += 0

            img, _, _ = mhd.load_oct_image(filepath)
            num_slices = img.shape[0]
            for slice_num in range(0, num_slices):
                im_slice = img[slice_num, :, :].astype(np.float32)
                # print np.max(im_slice), np.min(im_slice), np.median(im_slice)
                # im_slice = scipy.misc.imresize(im_slice, (512, 512), mode='F')
                # # im_slice = scipy.signal.medfilt(im_slice, 3)
                # im_slice = frangi(im_slice)
                # plt.imshow(im_slice)
                # plt.show()
                # plt.pause(1)
                im_slice = skimage.img_as_float(im_slice)
                im_slice = skimage.transform.resize(im_slice, (512, 512))
                # im_slice = frangi(im_slice) * 128. + 128.
                # im_slice = skimage.filters.median(im_slice)
                im_slice = entropy(im_slice, disk(10)) * 128. + 128.
                dist = np.mean(im_slice,axis=1)

                print np.max(dist), np.min(dist), np.median(dist)

                # plt.plot(dist)
                # plt.show()
                # plt.pause(1)








            #count = count + 1
            #print str(count) + ', ' + filepath.split('/')[-2] + ', ' +str(info[1]) + ', ' + str(info[2]) + ', ' + str(info[3])





