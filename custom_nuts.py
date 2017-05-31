import numpy as np
import os.path as path
import random as rnd
import nutsml.datautil as ut
import nutsml.imageutil as ni
from skimage.morphology import disk, rectangle
from skimage.filters.rank import entropy
import matplotlib.pyplot as plt
from nutsflow import Nut, NutFunction, nut_processor, as_tuple, as_set
import skimage

IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3


def calculate_oct_y_range(img, tresh=1e-10):
    """
    Calculate the interesting y region of the image using entropy
    :param img: 
    :param tresh: entropy cutoff threshold
    :return: min and maximum y index containing the interesting region
    """
    im_slice = skimage.img_as_float(img.astype(np.float32) / 128. - 1.)
    im_slice_ = entropy(im_slice, disk(15))
    p_ = np.mean(im_slice_, axis=1)
    p_ = p_ / (np.max(p_) + 1e-16)
    p_ = np.asarray(p_ > tresh, dtype=np.int)
    inx = np.where(p_ == 1)

    # p_ = np.zeros(p_.shape, dtype=np.float32)
    # if len(inx[0]) > 0:
    #     p_[inx[0][0]:inx[0][-1]] = 512.
    # plt.imshow(img)
    # print img.shape
    # plt.plot(p_, range(0, len(p_)), color='red')
    # plt.pause(1)
    # plt.clf()

    if len(inx[0]) > 0:
        return np.min(inx[0]), np.max(inx[0])
    else:
        return 0, img.shape[0]


def sample_retouch_patches(img, mask=None, pshape=(224,224), npos=10, nneg=1, pos=255, neg=0, patch_border=12):
    """
    Generate patches from the interesting region of the OCT slice
    :param img: oct image slice
    :param mask: oct segmentation GT
    :param pshape: patch shape
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from non interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negative
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: 
    """
    y_min, y_max = calculate_oct_y_range(img)
    roi_mask = np.zeros(mask.shape, dtype=np.int8)
    # print y_min, y_max
    roi_mask[y_min:y_max,:] = pos
    roi_mask[y_min:y_min+32, :] = 0
    roi_mask[y_max-32:y_max, :] = 0
    # plt.imshow(img)
    # print np.max(roi_mask), np.min(roi_mask)

    it = ni.sample_patch_centers(roi_mask, pshape=pshape, npos=npos, nneg=nneg, pos=pos, neg=neg)
    for r, c, label in it:
        img_patch = ni.extract_patch(img, pshape, r, c)
        mask_patch = ni.extract_patch(mask, pshape, r, c)
        label_IRF = int(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == IRF_CODE))
        label_SRF = int(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == SRF_CODE))
        label_PED = int(np.any(mask_patch[patch_border:-patch_border, patch_border:-patch_border] == PED_CODE))
        yield img_patch, mask_patch, label_IRF, label_SRF, label_PED
        # plt.plot(c, r, 'ro')

    # plt.pause(.1)
    # plt.clf()



@nut_processor
def ImagePatchesByMaskRetouch(iterable, imagecol, maskcol, IRFcol, SRFcol, PEDcol, pshape, npos,
                       nneg, pos=255, neg=0, patch_border=12):
    """
    :param iterable: iterable: Samples with images
    :param imagecol: Index of sample column that contain image
    :param maskcol: Index of sample column that contain mask
    :param IRFcol: Index of sample column that contain IRF label
    :param SRFcol: Index of sample column that contain SRF label
    :param PEDcol: Index of sample column that contain PED label
    :param pshape: Shape of patch
    :param npos: Number of patches to sample from interesting region
    :param nneg: Number of patches to sample from outside interesting region
    :param pos: Mask value indicating positives
    :param neg: Mask value indicating negativr
    :param patch_border: boder to ignore when creating IRF,SRF,PED labels for patches (ignore border pixels for predicting labels)
    :return: Iterator over samples where images and masks are replaced by image and mask patches
        and labels are replaced by labels [0,1] for patches
    """

    for sample in iterable:
        image, mask = sample[imagecol], sample[maskcol]
        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask size don''t match!')

        it = sample_retouch_patches(image, mask, pshape=pshape, npos=npos, nneg=nneg, pos=pos, neg=neg, patch_border=patch_border)

        for img_patch, mask_patch, label_IRF, label_SRF, label_PED in it:
            outsample = list(sample)[:]
            outsample[imagecol] = img_patch
            outsample[maskcol] = mask_patch
            outsample[IRFcol] = label_IRF
            outsample[SRFcol] = label_SRF
            outsample[PEDcol] = label_PED

            yield tuple(outsample)