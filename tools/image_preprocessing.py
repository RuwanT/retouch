import utils.mhd as mhd
import utils.slice_op
import os
import numpy as np
import pandas as pd

DATA_ROOT = '/home/truwan/DATA/retouch/'
IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3

image_names = list()
for subdir, dirs, files in os.walk(DATA_ROOT):
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith("reference.mhd"):
            image_name = filepath.split('/')[-2]
            vendor = filepath.split('/')[-3]
            img, _, _ = mhd.load_oct_seg(filepath)
            num_slices = img.shape[0]
            for slice_num in range(0, num_slices):
                im_slice = img[slice_num, :, :].astype(int)

                image_names.append([image_name, vendor, subdir, slice_num, int(np.any(im_slice == IRF_CODE)),
                                    int(np.any(im_slice == SRF_CODE)), int(np.any(im_slice == PED_CODE))])

        if filepath.endswith("oct.mhd"):
            image_name = filepath.split('/')[-2]
            vendor = filepath.split('/')[-3][0]
            img, _, _ = mhd.load_oct_image(filepath)
            num_slices = img.shape[0]
            for slice_num in range(0, num_slices):
                im_slice = img[slice_num, :, :].astype(np.float32)  # TODO: check float type in nuts-flow
                im_slice = utils.slice_op.pre_process_slice(im_slice)  # TODO: pre_process_slice
                save_name = DATA_ROOT + 'oct_slices/' + vendor + '_' + image_name + '_' + str(slice_num).zfill(
                    3) + '.npy'
                np.save(save_name, im_slice)

col_names = ['image_name', 'vendor', 'root', 'slice', 'is_IRF', 'is_SRF', 'is_PED']
df = pd.DataFrame(image_names, columns=col_names)
df.to_csv(DATA_ROOT + '/slice_gt.csv', index=False)
