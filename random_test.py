import os
from utils import mhd, msse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc, scipy.signal
import skimage.filters
import skimage
import skimage.transform
from skimage.morphology import disk, rectangle
from skimage.filters.rank import entropy
from skimage.filters.rank import median
import platform

#
# IRF_CODE = 1
# SRF_CODE = 2
# PED_CODE = 3
#
# if platform.system() == 'Linux':
#     DATA_ROOT = '/home/truwan/DATA/retouch/'
# else:
#     DATA_ROOT = '/Users/ruwant/DATA/retouch/'


# # Plot average histograms for vendor
# n_bins = 255
#
# p_s = np.zeros(n_bins,dtype=np.float)
# x_ = np.zeros(n_bins,dtype=np.float)
# p_c = np.zeros(n_bins,dtype=np.float)
#
# for subdir, dirs, files in os.walk(DATA_ROOT):
#     for file in files:
#         filepath = subdir + os.sep + file
#         if filepath.endswith("oct.mhd"):
#             img, _, _ = mhd.load_oct_image(filepath)
#             num_slices = img.shape[0]
#             for slice_num in range(0, num_slices):
#                 im_slice = img[slice_num, :, :].astype(np.float32)
#                 im_slice = skimage.img_as_float(im_slice)
#                 im_slice = skimage.transform.resize(im_slice, (512, 512))
#                 # plt.imshow(im_slice)
#                 # plt.pause(0.1)
#                 if 'Spectralis' in filepath:
#                     p, x = np.histogram(im_slice, bins=n_bins, range=(-1., 1.), normed=True)
#                     p_s = p_s + p/num_slices
#                 elif 'Cirrus' in filepath:
#                     p, x = np.histogram(im_slice, bins=n_bins, range=(-1., 1.), normed=True)
#                     p_c = p_c + p/num_slices
#                     x_ = x
#
# plt.plot(x_[:-1], p_c/24., color='blue')
# plt.plot(x_[:-1], p_s/24., color='red')
# plt.show()

# # get the y (height) range of a slice using entropy
# for subdir, dirs, files in os.walk(DATA_ROOT):
#     for file in files:
#         filepath = subdir + os.sep + file
#         if filepath.endswith("oct.mhd") and 'Cirrus' in filepath:
#             img, _, _ = mhd.load_oct_image(filepath)
#             num_slices = img.shape[0]
#             for slice_num in range(0, num_slices,10):
#                 im_slice = img[slice_num, :, :]
#                 im_slice = skimage.img_as_float(im_slice)
#                 #im_slice = skimage.transform.resize(im_slice, (512, 512))
#                 im_slice_ = entropy(im_slice, disk(15))
#                 p_ = np.mean(im_slice_, axis=1)
#                 p_ = p_ / (np.max(p_) + 1e-16)
#                 p_ = np.asarray(p_ > 1e-10, dtype=np.int)
#                 inx = np.where(p_ == 1)
#                 p_ = np.zeros(p_.shape, dtype=np.float32)
#                 if len(inx[0]) > 0:
#                     p_[inx[0][0]:inx[0][-1]] = 512.
#                 plt.imshow(im_slice)
#
#                 plt.plot(p_, range(0,len(p_)), color='red' )
#                 plt.pause(0.02)
#                 plt.clf()


# # doing histogram matching
# from utils.slice_op import hist_match
# filepath_s = '/home/truwan/DATA/retouch/Spectralis/fe02f982b78218ab05c755c01c7876b1/oct.mhd'
# filepath_c = '/home/truwan/DATA/retouch/Cirrus/1e0e71d2acdc57f10ab6712ab87b2ef7/oct.mhd'
#
# img_s, _, _ = mhd.load_oct_image(filepath_s)
# img_c, _, _ = mhd.load_oct_image(filepath_c)
#
# img_c_t = hist_match(img_c, img_s)
# num_slices = img_c_t.shape[0]
#
# # p_s, x = np.histogram(img_s, bins=255, range=(0, 255), density=True)
# # p_c, x = np.histogram(img_c, bins=255, range=(0, 255), density=True)
# # p_t, x = np.histogram(img_c_t, bins=255, range=(0, 255), density=True)
#
# # plt.plot(x[:-1],p_s, color='blue')
# # plt.plot(x[:-1],p_c, color='red')
# # plt.plot(x[:-1],p_t, color='green')
# # plt.show()
#
#
#
# for slice_num in range(0, num_slices,1):
#     plt.subplot(1,2,1)
#     plt.imshow(img_c[slice_num, :, :])
#     plt.subplot(1,2,2)
#     plt.imshow(img_c_t[slice_num, :, :])
#     plt.pause(0.1)


# # test image write
# DATA_ROOT = '/home/truwan/DATA/retouch/pre_processed/oct_imgs/'
#
# for subdir, dirs, files in os.walk(DATA_ROOT):
#     for file in files:
#         filepath = subdir + os.sep + file
#
#         if filepath.endswith(".tiff"):
#             img = Image.open(filepath)
#             img = img.resize((512,512), Image.LANCZOS)
#             img = np.asarray(img)
#             print img.shape
#             plt.imshow(img)
#             plt.pause(1)

# from custom_nuts import sample_retouch_patches, calculate_oct_y_range
# DATA_ROOT = '/home/truwan/DATA/retouch/pre_processed/oct_imgs/'
# for subdir, dirs, files in os.walk(DATA_ROOT):
#     for file in files:
#         filepath = subdir + os.sep + file
#         if filepath.endswith(".tiff"):
#             img = Image.open(filepath)
#             img = np.asarray(img)
#             calculate_oct_y_range(img, tresh=1e-10)


# DATA_ROOT = '/home/truwan/DATA/retouch/Spectralis/'
# sshape = set()
# sdim = set()
# for subdir, dirs, files in os.walk(DATA_ROOT):
#     for file in files:
#         filepath = subdir + os.sep + file
#         if filepath.endswith(".mhd"):
#             img, _, s = mhd.load_oct_image(filepath)
#             print img.shape, s
#             sshape.add(img.shape)
#             sdim.add(tuple(s))
#
# print sshape
# print sdim


from nutsflow import *
from nutsml import *
import numpy as np
a = np.zeros((100, 100), dtype=np.int8)
a[40:60, 40:60] = 4

augment_2 = (AugmentImage((1,)).by('rotate', 1.0, [10, 50]))
img_viewer = ViewImage(imgcols=(0, 1), layout=(1, 2), pause=10)
[(a, a),] >> augment_2 >> img_viewer >> Consume()
