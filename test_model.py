from custom_networks import retouch_dual_net, retouch_vgg_net, retouch_unet
from custom_nuts import ImagePatchesByMaskRetouch, ReadOCT
from nutsflow import *
from nutsml import *
import platform
import numpy as np
from custom_networks import retouch_dual_net
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/pre_processed/'
else:
    DATA_ROOT = '/Users/ruwant/DATA/retouch/pre_processed/'

BATCH_SIZE = 1
EPOCH = 10


def test_model():
    # reading training data
    train_file = './outputs/train_data_.csv'
    data = ReadPandas(train_file, dropnan=True)
    train_data = data >> Collect()

    train_file = './outputs/test_data_.csv'
    data = ReadPandas(train_file, dropnan=True)
    val_data = data >> Collect()

    print 'data reading done'

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

    # randomly sample image patches from the interesting region (based on entropy)
    image_patcher = ImagePatchesByMaskRetouch(imagecol=0, maskcol=1, IRFcol=2, SRFcol=3, PEDcol=4, pshape=(224, 224),
                                              npos=20, nneg=2, pos=1)

    # building image batches
    build_batch_train = (BuildBatch(BATCH_SIZE, prefetch=0)
                         .by(0, 'image', 'float32', channelfirst=False)
                         .by(1, 'one_hot', 'uint8', 4)
                         .by(2, 'one_hot', 'uint8', 2)
                         .by(3, 'one_hot', 'uint8', 2)
                         .by(4, 'one_hot', 'uint8', 2))

    model = retouch_unet(input_shape=(224, 224, 3))
    model.load_weights('/home/truwan/projects/retouch/outputs/weights.h5')

    def predict_batch(sample):
        outp = model.predict(sample[0])

        return (sample[0][0,:,:,1], np.argmax(sample[1][0,:],axis=-1 ) , np.argmax(outp[0,:],axis=-1 ))

    patch_mean = 128.
    patch_sd = 128.
    print patch_mean
    remove_mean = lambda s: (s - patch_mean) / patch_sd
    add_mean = lambda s: (s * patch_sd) - patch_mean

    debatch = lambda s: s[0,:]

    is_cirrus = lambda v: v[1] == 'Spectralis'

    def plot_res(sample):
        plt.imshow(sample[0], cmap='gray')
        plt.imshow(sample[1], cmap='jet', alpha=0.5)
        plt.pause(.1)

    # viewer = ViewImage(imgcols=(0, 1, 2), layout=(1, 3), pause=1)
    train_data >> Filter(is_cirrus) >> Map(rearange_cols) >> img_reader >> mask_reader >> image_patcher >> MapCol(0,
                                                                                             remove_mean) >> build_batch_train >> Map(
        predict_batch) >> MapCol(0, add_mean) >> PrintColType() >> Map(plot_res) >> Consume()

MapMulti
if __name__ == '__main__':
    test_model()