import numpy as np
import os
import torch
import train

METHOD = 'load'

# Directory for training data and network output 
data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
out_dir = os.path.join(data_dir, '210311_training')
suffix = 'dice-loss'

if METHOD == 'get':
    # if training data has not yet been produced
    image_paths = [os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_t58_zyx-coords.zarr')] #, 
                  # os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_t74_zyx-coords.zarr')]
    labels_paths = [os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_labels_58_GT_zyx-coords.zarr')]#, 
                  #  os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_labels_t74_GT_zyx-coords.zarr')]
    unet = train.train_unet(out_dir, suffix, image_paths=image_paths, labels_paths=labels_paths, train_data='get')

if METHOD == 'load':
    # if training data already exists and is present in the output directory
    data_dir = os.path.join(data_dir, '210309_training')
    unet = train.train_unet(out_dir, suffix, data_dir=data_dir)

if METHOD == 'load weights':
    train_dir = os.path.join(data_dir, '210309_training')
    wp = os.path.join(out_dir, '211003_213141_unet_sigmoid.pt')
    suffix = 'sigmoid'
    weights = torch.load(wp)
    unet = train.train_unet(out_dir, suffix, data_dir=train_dir, weights=weights)