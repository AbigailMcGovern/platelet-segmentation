import napari
import numpy as np
import os
import torch
import train


# ------------------
# Stable Local Paths
# ------------------
# Directory for training data and network output 
data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
# Path for original image volumes for which GT was generated
image_paths = [os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_t58_zyx-coords.zarr')] #, 
                  # os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_t74_zyx-coords.zarr')]
# Path for GT labels volumes
labels_paths = [os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_labels_58_GT_zyx-coords.zarr')]#, 
                  #  os.path.join(data_dir, '191113_IVMTR26_Inj3_cang_exp3_labels_t74_GT_zyx-coords.zarr')]


# -----------------------
# CHANGE THESE EACH TIME:
# ---------------------------------------------------------------------------------------
METHOD = 'get'
out_dir = os.path.join(data_dir, '210329_training_0')
suffix = 'z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl_cg'
# ---------------------------------------------------------------------------------------


if METHOD == 'get':
    # --------------------
    # CAN CHANGE THESE TOO
    # --------------------
    n_each = 100
    channels = ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log', 'centroid-gauss')
    validation_prop = 0.2
    scale = (4, 1, 1)
    epochs = 4
    lr = .01
    loss_function = 'BCELoss'
    chan_weights = (1., 2., 2.) # only used for weighted BCE 
    weights = None # can load and continue training
    update_every = 20 # how many batches before printing loss
    # -----------
    # Train U-net
    # ----------- 
    # with newly generated label chunks
    unet = train.train_unet_get_labels(
                                       out_dir, 
                                       suffix, 
                                       image_paths, 
                                       labels_paths, 
                                       n_each=n_each, 
                                       channels=channels, 
                                       validation_prop=validation_prop, 
                                       scale=scale, 
                                       epochs=epochs,
                                       lr=lr,
                                       loss_function=loss_function,
                                       chan_weights=chan_weights,
                                       weights=weights,
                                       update_every=update_every
                                       )

if METHOD == 'load':
    # to be continued ...
    pass
