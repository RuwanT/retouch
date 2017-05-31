from custom_networks import retouch_dual_net
from custom_nuts import ImagePatchesByMaskRetouch
from nutsflow import *
from nutsml import *
import platform


if platform.system() == 'Linux':
    DATA_ROOT = '/home/truwan/DATA/retouch/pre_processed/'
else:
    DATA_ROOT = '/Users/ruwant/DATA/retouch/pre_processed/'

def train_model(model):
    # create something to generate parches
    print "nothing"

if __name__ == "__main__":
    #model = retouch_dual_net(input_shape=(224,224,3))
    #model.compile(optimizer='sgd')
    #train_model(model)

    train_file = DATA_ROOT + 'slice_gt.csv'
    data = ReadPandas(train_file, dropnan=True)

    def rearange_cols(sample):
        img = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        mask = sample[1] + '_' + sample[0] + '_' + str(sample[3]).zfill(3) + '.tiff'
        IRF_label = sample[4]
        SRF_label = sample[5]
        PED_label = sample[6]

        return (img, mask, IRF_label, SRF_label, PED_label)


    imagepath = DATA_ROOT + 'oct_imgs/*'
    maskpath = DATA_ROOT + 'oct_masks/*'
    img_reader = ReadImage(0, imagepath)
    mask_reader = ReadImage(1, maskpath)

    image_patcher = ImagePatchesByMaskRetouch(imagecol=0, maskcol=1, IRFcol=2, SRFcol=3, PEDcol=4, pshape=(224,224), npos=20, nneg=2, pos=1)

    viewer = ViewImage(imgcols=(0,1), layout=(1,2), pause=.1)

    data >> Shuffle(1000) >> Map(rearange_cols) >> img_reader >> mask_reader >> image_patcher >> NOP(viewer) >> NOP(PrintColType()) >> Consume()









