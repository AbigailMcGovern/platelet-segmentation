import numpy as np
import os
import torch
import train

# Directory for training data and network output 
data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
out_dir = os.path.join(data_dir, '210309_training')
suffix = 'first-attempt'

# if training data has not yet been produced
image_paths = [os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_t58_zyx-coords.zarr')] #, 
              # os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_t74_zyx-coords.zarr')]
labels_paths = [os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_labels_58_GT_zyx-coords.zarr')]#, 
              #  os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_labels_t74_GT_zyx-coords.zarr')]
unet = train.train_unet(out_dir, suffix, image_paths=image_paths, labels_paths=labels_paths, train_data='get')

# if training data already exists and is present in the output directory
unet = train.train_unet(out_dir, suffix)