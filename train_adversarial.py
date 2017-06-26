from custom_networks import retouch_unet, multiclass_balanced_cross_entropy_loss_unet, retouch_discriminator
from custom_nuts import ImagePatchesByMaskRetouch, ReadOCT, ImagePatchesByMaskRetouch_resampled
from nutsflow import *
from nutsml import *
import platform
import numpy as np
from custom_networks import retouch_dual_net
import os
from hyper_parameters import *
import skimage.transform as skt
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model
from keras.layers import Cropping2D

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/pre_processed/'
else:
    DATA_ROOT = '/Users/ruwant/DATA/retouch/pre_processed/'

weight_file = './outputs/weights.h5'
save_weight_file = './outputs/gan_weights.h5'


def set_trainability(model, trainable=True):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def print_trainability(model):
    print 'trainble: ',
    for l in model.layers:
        if l.trainable:
            print l.name,
    print " "

    print 'non trainble: ',
    for l in model.layers:
        if not l.trainable:
            print l.name,

    print " "


def train_model():
    if not os.path.isfile('./outputs/train_data_.csv'):
        print 'generating new test train SPLIT'
        # reading training data
        train_file = DATA_ROOT + 'slice_gt.csv'
        data = ReadPandas(train_file, dropnan=True)
        data = data >> Shuffle(7000) >> Collect()

        # Split the data set into train and test sets with all the slices from the same volume remaining in one split
        same_image = lambda s: s[0]
        train_data, val_data = data >> SplitRandom(ratio=0.75, constraint=same_image)

        writer = WriteCSV('./outputs/train_data_.csv')
        train_data >> writer

        writer = WriteCSV('./outputs/test_data_.csv')
        val_data >> writer
    else:
        print 'Using existing test train SPLIT'
        train_file = './outputs/train_data_.csv'
        data = ReadPandas(train_file, dropnan=True)
        train_data = data >> Collect()

        train_file = './outputs/test_data_.csv'
        data = ReadPandas(train_file, dropnan=True)
        val_data = data >> Collect()

    def rearange_cols(sample):
        """
        Re-arrange the incoming data stream to desired outputs
        :param sample: (image_name, vendor, root, slice, is_IRF, is_SRF, is_PED)
        :return: 
        """
        img = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        mask = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        IRF_label = sample[4]
        SRF_label = sample[5]
        PED_label = sample[6]
        roi_m = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'

        return (img, mask, IRF_label, SRF_label, PED_label, roi_m)

    # training image augementation (flip-lr and rotate)
    # TODO : test adding a contrast enhancement to image
    def myrotate(image, angle):
        return skt.rotate(image, angle, preserve_range=True, order=0).astype('uint8')

    TransformImage.register('myrotate', myrotate)

    augment_1 = (AugmentImage((0, 1, 5))
                 .by('identical', 1.0)
                 .by('fliplr', 0.5))

    augment_2 = (AugmentImage((0, 1, 5))
                 .by('identical', 1.0)
                 .by('myrotate', 0.5, [0, 10]))

    # augment_3 = (AugmentImage((0))
    #             .by('contrast', 1.0, [0.7, 1.3]))


    # setting up image ad mask readers
    imagepath = DATA_ROOT + 'oct_imgs/*'
    maskpath = DATA_ROOT + 'oct_masks/*'
    roipath = DATA_ROOT + 'roi_masks/*'
    img_reader = ReadOCT(0, imagepath)
    mask_reader = ReadImage(1, maskpath)
    roi_reader = ReadImage(5, roipath)

    # randomly sample image patches from the interesting region (based on entropy)
    image_patcher = ImagePatchesByMaskRetouch_resampled(imagecol=0, maskcol=1, IRFcol=2, SRFcol=3, PEDcol=4, roicol=5,
                                                        pshape=(PATCH_SIZE_H, PATCH_SIZE_W),
                                                        npos=7, nneg=2, pos=1, use_entropy=True, patch_border=42)

    # img_viewer = ViewImage(imgcols=(0, 1), layout=(1, 2), pause=1)

    # building image batches
    build_batch_train = (BuildBatch(BATCH_SIZE, prefetch=0)
                         .by(0, 'image', 'float32', channelfirst=False)
                         .by(1, 'one_hot', 'uint8', 4)
                         .by(2, 'one_hot', 'uint8', 2)
                         .by(3, 'one_hot', 'uint8', 2)
                         .by(4, 'one_hot', 'uint8', 2))

    is_cirrus = lambda v: v[1] == 'Cirrus'
    is_topcon = lambda v: v[1] == 'Topcon'
    is_spectralis = lambda v: v[1] == 'Spectralis'

    # TODO : Should I drop non-pathelogical slices
    # Filter to drop some non-pathology patches
    def drop_patch(sample, drop_prob=0.9):
        """
        Randomly drop a patch from iterator if there is no pathology
        :param sample: 
        :param drop_prob: 
        :return: 
        """
        if (int(sample[2]) == 0) and (int(sample[3]) == 0) and (int(sample[4]) == 0):
            return float(np.random.random_sample()) < drop_prob
        else:
            return False

    # define the model
    # TODO : change optimizer
    model_D = retouch_discriminator(input_shape=(PATCH_SIZE_H-BORDER_WIDTH*2, PATCH_SIZE_W-BORDER_WIDTH*2, 3))
    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False, clipvalue=1.)
    set_trainability(model_D, trainable=True)
    model_D.compile(optimizer=sgd, loss='binary_crossentropy')
    # model_D.summary()
    # print "Disc"
    # print_trainability(model_D)

    model_G = retouch_unet(input_shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False, clipvalue=1.)
    model_G.compile(optimizer=sgd, loss=multiclass_balanced_cross_entropy_loss_unet)
    # print "Gen"
    # print_trainability(model_G)

    set_trainability(model_D,trainable=False)
    im_input = Input(shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    im2_input = Cropping2D(cropping=((46, 46), (46, 46)), data_format='channels_last')(im_input)
    # im2_input = Input(shape=(PATCH_SIZE_H-BORDER_WIDTH*2, PATCH_SIZE_W-BORDER_WIDTH*2, 3))
    G_out = model_G(im_input)
    D_out = model_D([im2_input, G_out])
    model_D_G = Model([im_input], [D_out, G_out])
    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False, clipvalue=1.)
    model_D_G.compile(optimizer=sgd, loss=['binary_crossentropy', multiclass_balanced_cross_entropy_loss_unet])

    if LOAD_WEIGTHS:
        assert os.path.isfile(weight_file)
        model_G.load_weights(weight_file)

    def train_batch(sample):
        # train the discriminator
        set_trainability(model_D, trainable=True)
        generated_images = model_G.predict(sample[0], batch_size=BATCH_SIZE)
        # print generated_images.shape, sample[1][:, BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH, :].shape
        images = np.copy(sample[0][:, BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH, :])
        X_img = np.concatenate((sample[0][:, BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH, :], images), axis=0)
        X_mask = np.concatenate((sample[1][:, BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH, :], generated_images), axis=0)
        y = np.asarray([1]*BATCH_SIZE + [0]*BATCH_SIZE, dtype=np.int8)
        D_error = model_D.train_on_batch([X_img, X_mask], y)

        # train the Generator
        set_trainability(model_D, trainable=False)
        y = np.asarray([1] * BATCH_SIZE, dtype=np.int8)
        G_error = model_D_G.train_on_batch(sample[0], [y, sample[1]])

        return (D_error, G_error)

    def test_batch(sample):
        # outp = model.test_on_batch(sample[0], [sample[2], sample[3], sample[4], sample[1]])
        outp = model_G.test_on_batch(sample[0], sample[1])
        return (outp,)

    log_cols_train = LogCols('./outputs/train_log.csv', cols=None, colnames=('D error', 'G error'))
    log_cols_test = LogCols('./outputs/test_log.csv', cols=None, colnames=('D error', 'G error'))

    filter_batch_shape = lambda s: s[0].shape[0] == BATCH_SIZE

    patch_mean = 128.
    patch_sd = 128.
    remove_mean = lambda s: (s.astype(np.float32) - patch_mean) / patch_sd

    best_error = float("inf")
    print 'Starting network training'
    for e in range(0, EPOCH):
        print "Training Epoch", str(e)
        train_data >> Shuffle(1000) >> Map(
            rearange_cols) >> img_reader >> mask_reader >> roi_reader >> augment_1 >> augment_2 >> Shuffle(
            100) >> image_patcher >> MapCol(0, remove_mean) >> Shuffle(1000) >> FilterFalse(
            drop_patch) >> build_batch_train >> Filter(filter_batch_shape) >> Map(
            train_batch) >> log_cols_train >> Consume()

        print "Testing Epoch", str(e)
        val_error = val_data >> Map(
            rearange_cols) >> img_reader >> mask_reader >> roi_reader >> image_patcher >> MapCol(0,
                                                                                                 remove_mean) >> FilterFalse(
            drop_patch) >> build_batch_train >> Filter(
            filter_batch_shape) >> Map(test_batch) >> log_cols_test >> Collect()

        val_error = np.mean([v[0] for v in val_error])
        print 'validation error : ', e, val_error
        if val_error < best_error:
            # save weights
            print 'saving weights at epoch: ', e, val_error
            model_G.save_weights(save_weight_file)
            best_error = val_error


if __name__ == "__main__":
    train_model()
