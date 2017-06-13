from custom_networks import retouch_dual_net, retouch_vgg_net, retouch_unet
from custom_nuts import ImagePatchesByMaskRetouch, ReadOCT
from nutsflow import *
from nutsml import *
import platform
import numpy as np
from custom_networks import retouch_dual_net
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/pre_processed/'
else:
    DATA_ROOT = '/Users/ruwant/DATA/retouch/pre_processed/'

BATCH_SIZE = 16
EPOCH = 10


def visualize_images():
    train_file = DATA_ROOT + 'slice_gt.csv'
    data = ReadPandas(train_file, dropnan=True)
    data = data >> Shuffle(4000) >> Collect()

    is_topcon = lambda v: v[1] == 'Cirrus'

    def rearange_cols(sample):
        """
        Re-arrange the incoming data stream to desired outputs
        :param sample: 
        :return: 
        """
        img = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        mask = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        IRF_label = sample[4]
        SRF_label = sample[5]
        PED_label = sample[6]

        return (img, mask, IRF_label, SRF_label, PED_label)

    # setting up image ad mask readers
    imagepath = DATA_ROOT + 'oct_imgs/*'
    maskpath = DATA_ROOT + 'oct_masks/*'
    img_reader = ReadOCT(0, imagepath)
    mask_reader = ReadImage(1, maskpath)

    viewer = ViewImage(imgcols=(0, 1), layout=(1, 2), pause=.1)
    slice_oct = lambda x: x[:,:,1]

    # randomly sample image patches from the interesting region (based on entropy)
    image_patcher = ImagePatchesByMaskRetouch(imagecol=0, maskcol=1, IRFcol=2, SRFcol=3, PEDcol=4, pshape=(224, 224),
                                              npos=20, nneg=2, pos=1)

    data >> NOP(Filter(is_topcon)) >> Map(
        rearange_cols) >> img_reader >> mask_reader >> MapCol(0, slice_oct) >> image_patcher >> Consume()

def train_model():
    # reading training data
    train_file = DATA_ROOT + 'slice_gt.csv'
    data = ReadPandas(train_file, dropnan=True)
    data = data >> Shuffle(7000) >> Collect()

    # Split the data set into train and test sets with all the slices from the same volume remaining in one split
    same_image = lambda s: s[0]
    train_data, val_data = data >> SplitRandom(ratio=0.75, constraint=same_image)

    val_images = val_data >> GetCols(0, 1) >> Collect(container=set)
    print 'printing validation set: '
    val_images >> Print() >> Consume()
    print '...'

    def rearange_cols(sample):
        """
        Re-arrange the incoming data stream to desired outputs
        :param sample: 
        :return: 
        """
        img = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        mask = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        IRF_label = sample[4]
        SRF_label = sample[5]
        PED_label = sample[6]

        return (img, mask, IRF_label, SRF_label, PED_label)

    # training image augementation (flip-lr and rotate)
    # TODO : test adding a contrast enhancement to image
    augment_1 = (AugmentImage((0, 1))
                 .by('identical', 1.0)
                 .by('fliplr', 0.5))

    augment_2 = (AugmentImage((0, 1))
                 .by('identical', 1.0)
                 .by('rotate', 0.5, [0, 10]))

    # augment_3 = (AugmentImage((0))
    #             .by('contrast', 1.0, [0.7, 1.3]))


    # setting up image ad mask readers
    imagepath = DATA_ROOT + 'oct_imgs/*'
    maskpath = DATA_ROOT + 'oct_masks/*'
    img_reader = ReadOCT(0, imagepath)
    mask_reader = ReadImage(1, maskpath)

    # randomly sample image patches from the interesting region (based on entropy)
    image_patcher = ImagePatchesByMaskRetouch(imagecol=0, maskcol=1, IRFcol=2, SRFcol=3, PEDcol=4, pshape=(224, 224),
                                              npos=20, nneg=2, pos=1)

    # viewer = ViewImage(imgcols=(0, 1), layout=(1, 2), pause=1)

    # building image batches
    build_batch_train = (BuildBatch(BATCH_SIZE, prefetch=0)
                         .by(0, 'image', 'float32', channelfirst=False)
                         .by(1, 'one_hot', 'uint8', 4)
                         .by(2, 'one_hot', 'uint8', 2)
                         .by(3, 'one_hot', 'uint8', 2)
                         .by(4, 'one_hot', 'uint8', 2))

    is_cirrus = lambda v: v[1] == 'Cirrus'

    # TODO : Should I drop non-pathelogical slices
    # Filter to drop all non-pathology patches
    no_pathology = lambda s: (s[2] == 0) and (s[3] == 0) and (s[4] == 0)

    def drop_patch(sample, drop_prob=0.9):
        """
        Randomly drop a patch from iterator if there is no pathology
        :param sample: 
        :param drop_prob: 
        :return: 
        """
        if (int(sample[2]) == 0) and (int(sample[3]) == 0) and (int(sample[4]) == 0):
            return float(np.random.random_sample(1)) < drop_prob
        else:
            return False

    # define the model
    # model = retouch_vgg_net(input_shape=(224, 224, 3))
    model = retouch_unet(input_shape=(224, 224, 3))

    def train_batch(sample):
        # outp = model.train_on_batch(sample[0], [sample[2], sample[3], sample[4], sample[1]])
        outp = model.train_on_batch(sample[0], [sample[1],])
        return (outp,)

    def test_batch(sample):
        # outp = model.test_on_batch(sample[0], [sample[2], sample[3], sample[4], sample[1]])
        outp = model.test_on_batch(sample[0], [sample[1],])
        return (outp,)

    log_cols_train = LogCols('./outputs/train_log.csv', cols=None, colnames=model.metrics_names)
    log_cols_test = LogCols('./outputs/test_log.csv', cols=None, colnames=model.metrics_names)

    filter_batch_shape = lambda s: s[0].shape[0] == BATCH_SIZE

    patch_mean = 116.
    print patch_mean
    remove_mean = lambda s: s - patch_mean

    print 'Starting network training'
    for e in range(0, EPOCH):
        print "Training Epoch", str(e)
        train_data >> NOP(Filter(is_cirrus)) >> Map(
            rearange_cols) >> img_reader >> mask_reader >> augment_1 >> augment_2 >> Shuffle(
            100) >> image_patcher >> MapCol(0, remove_mean) >> Shuffle(1000) >> FilterFalse(drop_patch) >> NOP(
            viewer) >> build_batch_train >> Filter(filter_batch_shape) >> Map(
            train_batch) >> log_cols_train >> Consume()

        print "Testing Epoch", str(e)
        val_data >> NOP(Filter(is_cirrus)) >> Map(
            rearange_cols) >> img_reader >> mask_reader >> image_patcher >> MapCol(0, remove_mean) >> build_batch_train >> Filter(
            filter_batch_shape) >> Map(test_batch) >> log_cols_test >> Consume()

        # save weights
        model.save_weights('./outputs/weights.h5')


if __name__ == "__main__":
    train_model()
    # visualize_images()
