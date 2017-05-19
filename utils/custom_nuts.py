import numpy as np
import os.path as path
import random as rnd
import nutsml.datautil as ut
import nutsml.imageutil as ni

from nutsflow import Nut, NutFunction, nut_processor, as_tuple, as_set

@nut_processor
def ImagePatchesByMaskRETOUCH(iterable, imagecol, maskcol, pshape, npos,
                       nneg=lambda npos: npos, pos=255, neg=0, retlabel=True):
    """
    samples >> ImagePatchesByMask(imagecol, maskcol, pshape, npos,
                                 nneg=lambda npos: npos,
                                 pos=255, neg=0, retlabel=True)


    Randomly sample positive/negative patches from image based on mask.

    A patch is positive if its center point has the value 'pos' in the
    mask (corresponding to the input image) and is negative for value 'neg'
    The mask must be of same size as image.

    >>> import numpy as np
    >>> np.random.seed(0)    # just to ensure stable doctest
    >>> img = np.reshape(np.arange(25), (5, 5))
    >>> mask = np.eye(5, dtype='uint8') * 255
    >>> samples = [(img, mask)]

    >>> getpatches = ImagePatchesByMask(0, 1, (3, 3), 2, 1)
    >>> for (p, l) in samples >> getpatches:
    ...     print p, l
    [[10 11 12]
     [15 16 17]
     [20 21 22]] 0
    [[12 13 14]
     [17 18 19]
     [22 23 24]] 1
    [[ 6  7  8]
     [11 12 13]
     [16 17 18]] 1

    >>> np.random.seed(0)    # just to ensure stable doctest
    >>> patches = ImagePatchesByMask(0, 1, (3, 3), 1, 1, retlabel=False)
    >>> for (p, m) in samples >> getpatches:
    ...     print p
    ...     print m
    [[10 11 12]
     [15 16 17]
     [20 21 22]]
    0
    [[12 13 14]
     [17 18 19]
     [22 23 24]]
    1
    [[ 6  7  8]
     [11 12 13]
     [16 17 18]]
    1

    :param iterable iterable: Samples with images
    :param int imagecol: Index of sample column that contain image
    :param int maskcol: Index of sample column that contain mask
    :param tuple pshape: Shape of patch
    :param int npos: Number of positive patches to sample
    :param int|function nneg: Number of negative patches to sample or
        a function hat returns the number of negatives
        based on number of positives.
    :param int pos: Mask value indicating positives
    :param int neg: Mask value indicating negatives
    :param bool retlabel: True return label, False return mask patch
    :return: Iterator over samples where images are replaced by image patches
        and masks are replaced by labels [0,1] or mask patches
    :rtype: generator
    """
    for sample in iterable:
        image, mask = sample[imagecol], sample[maskcol]
        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask size don''t match!')

        it = ni.sample_pn_patches(image, mask, pshape, npos, nneg, pos, neg)
        for img_patch, mask_patch, label in it:
            outsample = list(sample)[:]
            outsample[imagecol] = img_patch
            outsample[maskcol] = label if retlabel else mask_patch
            yield tuple(outsample)