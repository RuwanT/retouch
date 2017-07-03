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
import matplotlib.pyplot as plt
from nutsml import *
from nutsflow import *

if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/'
else:
    DATA_ROOT = '/Users/ruwant/DATA/retouch/'

IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3

CHULL = False


def preprocess_oct_images():
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
                    save_name = DATA_ROOT + 'pre_processed/oct_masks/' + vendor + '_' + image_name + '_' + str(
                        slice_num).zfill(
                        3) + '.tiff'
                    im_slice.save(save_name)

            elif filepath.endswith("oct.mhd"):
                image_name = filepath.split('/')[-2]
                vendor = filepath.split('/')[-3]
                img, _, _ = mhd.load_oct_image(filepath)
                if 'Cirrus' in vendor:
                    img = hist_match(img, oct_r)
                elif 'Topcon' in vendor:
                    img = hist_match(img, oct_r)
                num_slices = img.shape[0]
                for slice_num in range(0, num_slices):
                    if 'Cirrus' in vendor and (slice_num > 0) and (slice_num < num_slices - 1):
                        im_slice = np.median(img[slice_num - 1:slice_num + 2, :, :].astype(np.int32), axis=0).astype(
                            np.int32)
                    if 'Topcon' in vendor and (slice_num > 0) and (slice_num < num_slices - 1):
                        im_slice = np.median(img[slice_num - 1:slice_num + 2, :, :].astype(np.int32), axis=0).astype(
                            np.int32)
                    else:
                        im_slice = img[slice_num, :, :].astype(np.int32)
                    im_slice = Image.fromarray(im_slice, mode='I')
                    # TODO : check if this second filtering is useful
                    if 'Cirrus' in vendor:
                        im_slice = im_slice.filter(ImageFilter.MedianFilter(size=3))
                    elif 'Topcon' in vendor:
                        im_slice = im_slice.filter(ImageFilter.MedianFilter(size=3))
                    save_name = DATA_ROOT + 'pre_processed/oct_imgs/' + vendor + '_' + image_name + '_' + str(
                        slice_num).zfill(3) + '.tiff'
                    im_slice.save(save_name)

    col_names = ['image_name', 'vendor', 'root', 'slice', 'is_IRF', 'is_SRF', 'is_PED']
    df = pd.DataFrame(image_names, columns=col_names)
    df.to_csv(DATA_ROOT + 'pre_processed/slice_gt.csv', index=False)


def create_test_train_set():

    print 'generating new test train SPLIT'
    # reading training data
    train_file = DATA_ROOT + '/pre_processed/slice_gt.csv'
    data = ReadPandas(train_file, dropnan=True)
    data = data >> Shuffle(7000) >> Collect()

    case_names = data >> GetCols(0, 1) >> Collect(set)
    # case_names >> Print() >> Consume()

    # is_topcon = lambda s: s[1] == 'Topcon'
    # is_cirrus = lambda s: s[1] == 'Cirrus'
    # is_spectralis = lambda s: s[1] == 'Spectralis'

    train_cases, test_cases = case_names >> Shuffle(70) >> SplitRandom(ratio=0.75)

    print train_cases >> GetCols(1) >> CountValues()
    print test_cases >> GetCols(1) >> CountValues()

    train_cases = train_cases >> GetCols(0) >> Collect()
    test_cases = test_cases >> GetCols(0) >> Collect()

    is_in_train = lambda sample: (sample[0],) in list(train_cases)

    writer = WriteCSV('./outputs/train_data.csv')
    data >> Filter(is_in_train) >> writer
    writer = WriteCSV('./outputs/test_data.csv')
    data >> FilterFalse(is_in_train) >> writer

    case_names = data >> Filter(is_in_train) >> GetCols(0, 1) >> Collect(set)
    print case_names >> GetCols(1) >> CountValues()
    case_names = data >> FilterFalse(is_in_train) >> GetCols(0, 1) >> Collect(set)
    print case_names >> GetCols(1) >> CountValues()


def create_roi_masks(tresh=1e-2):
    import skimage.io as sio
    import skimage
    from skimage.morphology import disk, rectangle, closing, opening, binary_closing, convex_hull_image
    from skimage.filters.rank import entropy

    MASK_PATH = '/home/truwan/DATA/retouch/pre_processed/oct_masks/'

    image_names = list()
    for subdir, dirs, files in os.walk(DATA_ROOT + 'pre_processed/oct_imgs/'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".tiff"):
                image_name = filepath.split('/')[-1]
                image_names.append([filepath, image_name])

    for filepath, image_name in image_names:
        # print filepath, image_name
        img = sio.imread(filepath)
        im_mask = sio.imread(MASK_PATH + image_name)
        im_mask = im_mask.astype(np.int8)
        im_slice = skimage.img_as_float(img.astype(np.float32) / 128. - 1.)

        im_slice_ = entropy(im_slice, disk(15))
        im_slice_ = im_slice_ / (np.max(im_slice_) + 1e-16)
        im_slice_ = np.asarray(im_slice_ > tresh, dtype=np.int8)
        im_slice_ = np.bitwise_or(im_slice_,im_mask)
        selem = disk(55)
        im_slice_ = binary_closing(im_slice_, selem=selem)

        h, w = im_slice_.shape
        rnge = list()
        for x in range(0, w):
            col = im_slice_[:, x]
            col = np.nonzero(col)[0]
            # print col, col.shape
            if len(col) > 0:
                y_min = np.min(col)
                y_max = np.max(col)
                rnge.append(int((float(y_max) - y_min)/h*100.))
                im_slice_[y_min:y_max, x] = 1
        if len(rnge) > 0:
            print image_name, np.max(rnge)
        else:
            print image_name, "**************"

        if CHULL:
            im_slice_ = convex_hull_image(im_slice_)

        # plt.imshow(im_slice, cmap='gray')
        # plt.imshow(im_slice_, cmap='jet', alpha=0.5)
        # plt.pause(.1)

        im_slice_ = Image.fromarray(im_slice_, mode='L')
        save_name = DATA_ROOT + 'pre_processed/roi_masks_chull/' + image_name
        im_slice_.save(save_name)


if __name__ == "__main__":
    # create_roi_masks()
    create_test_train_set()

