import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from custom_networks import retouch_dual_net, retouch_vgg_net, retouch_unet, multiclass_balanced_cross_entropy_loss_unet
from custom_nuts import ImagePatchesByMaskRetouch, ReadOCT, ImagePatchesByMaskRetouch_resampled
from nutsflow import *
from nutsml import *
import platform
import numpy as np
from custom_networks import retouch_dual_net
import os
from hyper_parameters import *
import skimage.transform as skt
from keras.optimizers import Adam


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/pre_processed/'
else:
    DATA_ROOT = '/Users/ruwant/DATA/retouch/pre_processed/'

weight_file = './outputs/weights.h5'
load_weight_file = '/home/truwan/projects/retouch/outputs/weights.h5'

def train_model():
    assert os.path.isfile('./outputs/train_data.csv')
    assert os.path.isfile('./outputs/test_data.csv')
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
        :return: (image_name, GT_mask_name, is_IRF, is_SRF, is_PED, ROI_mask_name)
        """
        img = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        mask = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        IRF_label = sample[4]
        SRF_label = sample[5]
        PED_label = sample[6]
        roi_m = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'

        return (img, mask, IRF_label, SRF_label, PED_label, roi_m)

    # training image augementation (flip-lr and rotate)
    # normal rotate would interpolate pixel values
    def myrotate(image, angle):
        return skt.rotate(image, angle, preserve_range=True, order=0).astype('uint8')

    TransformImage.register('myrotate', myrotate)

    augment_1 = (AugmentImage((0, 1, 5))
                 .by('identical', 1.0)
                 .by('fliplr', 0.5))

    augment_2 = (AugmentImage((0, 1, 5))
                 .by('identical', 1.0)
                 .by('myrotate', 0.5, [0, 10]))

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
                                                        npos=12, nneg=2, pos=1, use_entropy=True, patch_border=BORDER_WIDTH)

    # img_viewer = ViewImage(imgcols=(0, 1), layout=(1, 2), pause=1)

    # building image batches
    build_batch_train = (BuildBatch(BATCH_SIZE, prefetch=0)
                         .by(0, 'image', 'float32', channelfirst=False)
                         .by(1, 'one_hot', 'uint8', 4)
                         .by(2, 'number', 'uint8')
                         .by(3, 'number', 'uint8')
                         .by(4, 'number', 'uint8'))

    is_cirrus = lambda v: v[1] == 'Cirrus'
    is_topcon = lambda v: v[1] == 'Topcon'
    is_spectralis = lambda v: v[1] == 'Spectralis'

    # Filter to drop some non-pathology patches
    def drop_patch(sample, drop_prob=0.75):
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
    model = retouch_unet(input_shape=(PATCH_SIZE_H, PATCH_SIZE_W, 3))
    model.compile(optimizer=Adam(lr=ADAM_LR, beta_1=ADAM_BETA_1, clipvalue=1.), loss=multiclass_balanced_cross_entropy_loss_unet)

    if LOAD_WEIGTHS:
        assert os.path.isfile(load_weight_file)
        model.load_weights(load_weight_file)

    def train_batch(sample):
        if not TRAIN_CLASSES:
            outp = model.train_on_batch(sample[0], sample[1])
            outp = (outp,)
        else:
            outp = model.train_on_batch(sample[0], [sample[1], sample[2], sample[3], sample[4]])

        return outp

    def test_batch(sample):
        if not TRAIN_CLASSES:
            outp = model.test_on_batch(sample[0], sample[1])
            outp = (outp,)
        else:
            outp = model.test_on_batch(sample[0], [sample[1], sample[2], sample[3], sample[4]])

        return outp

    log_cols_train = LogCols('./outputs/train_log.csv', cols=None, colnames=model.metrics_names)
    log_cols_test = LogCols('./outputs/test_log.csv', cols=None, colnames=model.metrics_names)

    filter_batch_shape = lambda s: s[0].shape[0] == BATCH_SIZE

    remove_mean = lambda s: (s.astype(np.float32) - SLICE_MEAN) / SLICE_SD

    best_error = float("inf")
    print 'Starting network training'
    error_hold = list()
    for e in range(0, EPOCH):
        # print "Training Epoch", str(e)
        train_error = train_data >> Stratify(1, mode='up') >> Shuffle(1000) >> Map(
            rearange_cols) >> img_reader >> mask_reader >> roi_reader >> augment_1 >> augment_2 >> Shuffle(
            100) >> image_patcher >> MapCol(0, remove_mean) >> Shuffle(1000) >> FilterFalse(
            drop_patch) >> build_batch_train >> Filter(filter_batch_shape) >> Map(
            train_batch) >> log_cols_train >> Collect()

        # print "Testing Epoch", str(e)
        val_error = val_data >> Map(rearange_cols) >> Shuffle(
            1000) >> img_reader >> mask_reader >> roi_reader >> image_patcher >> Shuffle(1000) >> MapCol(0,
                                                                                                         remove_mean) >> FilterFalse(
            drop_patch) >> build_batch_train >> Filter(filter_batch_shape) >> Map(
            test_batch) >> log_cols_test >> Collect()

        error_hold.append(
            [e, np.mean([v[0] for v in train_error]), np.mean([v[0] for v in val_error]),
             np.std([v[0] for v in train_error]), np.std([v[0] for v in val_error])])
        val_error = np.mean([v[0] for v in val_error])
        train_error = np.mean([v[0] for v in train_error])
        print 'epoch ', e, 'train_error = ', train_error, 'val_error = ', val_error,
        if val_error < best_error:
            # save weights
            print '... saving weights ...'
            model.save_weights(weight_file)
            best_error = val_error
        else:
            print "..."
        if e >= EPOCH - 1:
            model.save_weights('./outputs/final_weights.h5')

        if DRAW_ERRORS_EPOCH:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            error_hold_np = np.asarray(error_hold)
            ax.errorbar(error_hold_np[:, 0], error_hold_np[:, 1], error_hold_np[:, 3], label='train error')
            ax.errorbar(error_hold_np[:, 0], error_hold_np[:, 2], error_hold_np[:, 4], label='test error')
            ax.legend()
            fig.savefig('./outputs/res_plot.png')

    error_writer = WriteCSV('./outputs/epoch_errors.csv')
    error_hold >> error_writer


if __name__ == "__main__":
    # TODO : changed regularize weigths
    # TODO : added dice loss
    train_model()
