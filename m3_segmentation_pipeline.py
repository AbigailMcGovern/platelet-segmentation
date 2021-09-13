import logging
from numpy.lib import npyio
from datetime import datetime
from scipy.ndimage.filters import rank_filter

now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")
#LOG_NAME = f'/projects/rl54/results/{dt}_segmentation-log.log'
LOG_NAME = f'/Users/amcg0011/Data/pia-tracking/debugging/{dt}_segmentation-log.log'
log_basic = False
if log_basic:
    logging.basicConfig(filename=LOG_NAME, encoding='utf-8', level=logging.DEBUG)
else:
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.DEBUG) 
    handler = logging.FileHandler(LOG_NAME, 'w', 'utf-8') 
    root_logger.addHandler(handler)
logging.debug('Starting script...')


try:
    import argparse
    from nd2_dask.nd2_reader import nd2_reader
    from nd2reader import ND2Reader
    import numpy as np
    import os
    import pandas as pd
    from skimage.exposure import rescale_intensity
    from skimage.measure import regionprops_table
    from unet import UNet
    import torch
    from pathlib import Path
    from plateseg.predict import predict_output_chunks, make_chunks
    from segmentation import segment_output_image
    import dask.array as da
    import json
    from time import time
    import zarr
    from zarpaint._zarpaint import create_ts_meta
    from train_io import normalise_data
    logging.debug('Finished imports.')
except:
    logging.exception('Got an exception during imports')
    raise

# ---------------
# Parse Arguments
# ---------------
try:
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--info', help='JSON file containing info for segmentation')
    args = p.parse_args()
    args.info = '/Users/amcg0011/GitRepos/platelet-segmentation/untracked/local-seg.json'
    logging.debug('Parsed arguments.')
    info_path = args.info
    # load the required variables
    with open(info_path, 'r') as f:
        info = json.load(f)
    logging.debug('Loaded JSON info.')
    out_dir = info['out_dir']
    image_path = info['image_path']
    scratch_dir = info['scratch_dir']
    unet_path = info['unet_path']
    batch_name = info['batch_name']
    logging.debug('Loaded json info for segmentation')
except:
    logging.exception('Got an exception whilst parsing info')
    raise


# ---------
# Functions
# ---------

def read_image(image_path):
    t = time()
    p = Path(image_path)
    name = p.stem
    cohort = p.parents[1].stem
    treatment = p.parents[0].stem
    logging.debug(f'Processing {image_path}...')
    # get the nd2 metadata for rois
    logging.debug('Obtaining ND2 metadata using nd2reader')
    data = ND2Reader(image_path)
    rois = data.metadata['rois'][0]
    axes = data.axes
    px_microns = data.metadata['pixel_microns']
    frame_rate = float(data.frame_rate)
    del data
    # read in with nd2-dask (nd2reader can have issues)
    logging.debug('Reading image using nd2-dask...')
    # keep only the Alxa 647 channel
    layerlist = nd2_reader(image_path)
    images = []
    image, data = None, None
    channels = []
    channels_dict = {}
    for i, layertuple in enumerate(layerlist):
        channel = layertuple[1]['name']
        if '647' in channel:
            image = layertuple[0]
            data = layertuple[1]
        images.append(layertuple[0])
        channels.append(channel)
        channels_dict[channel] = i
    scale = [abs(scl) for scl in data['scale']]
    translate = data['translate']
    image = image / np.max(image, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))

    #pixel_size = data.scale
    try:
        logging.debug('Obtaining metadata...')
        metadata_dict = get_metadata(name, axes, scale, translate, channels, rois, 
                                    px_microns, frame_rate, cohort, treatment)
    except:
        logging.exception('Got exeption whilst attemping to obtain metadata')
        raise
    t = time() - t
    logging.debug(f'Read in image and obtained metadata in {t} sec')
    return images, image, metadata_dict, channels_dict


def get_channels_dict(meta, channels=('Alxa 647', 'GaAsP Alexa 488', 'GaAsP Alexa 568')):
    channels_dict = {}
    for st in channels:
        logging.debug(f'Attempting to obtain channel {st}')
        channels = meta['name']
        st_channel = next(
            i
            for i, name in enumerate(channels)
            if st in name
        )
        channels_dict[st] = st_channel
    logging.debug(f'Found the following channels: {channels}')
    return channels_dict


def get_channel_names(channel_names, file_name):
    n = ': ' + file_name
    chans = []
    for c in channel_names:
        i = c.find(n)
        if i == -1:
            chans.append(c)
        else:
            chans.append(c[:i])
    return chans


def get_metadata(name, axes, scale, translate, channels, rois, px_microns, frame_rate, cohort, treatment):
    '''
    Generate metadata to be saved for each image (including about rois)
    '''
    logging.debug('Initiating metadata dict...')
    metadata_dict = {
        'file' : name,
        'px_microns' : px_microns, 
        'axes' : axes,
        'cohort': cohort, 
        'treatment' : treatment, 
        'scale' : scale,
        'translate' : translate,
    }
    axes = ['t', 'z', 'y', 'x']
    metadata_dict.update(**{ m:scale[i] for i, m in enumerate(axes)})
    metadata_dict.update(**{ 'channel_'+ str(i):c for i, c in enumerate(channels)})
    metadata_dict.update(frame_rate = frame_rate)
    metadata_dict.update(roi_t = float(np.mean(rois['timepoints'])))
    metadata_dict.update(roi_x = rois['positions'][0][1])
    metadata_dict.update(roi_y = rois['positions'][0][0])
    metadata_dict.update(roi_size = float(rois['sizes'][0][0]))
    logging.debug('Obtained initial metadata dict.')
    return metadata_dict


def get_labels_info(labels, meta, t_max, out_dir, images, channels_dict, dt):
    labs_df = []
    a647 = channels_dict['Alxa 647'] 
    a488 = channels_dict['GaAsP Alexa 488']
    a568 = channels_dict['GaAsP Alexa 568']
    logging.debug(f'Channels dict: {channels_dict}')
    logging.debug(f'Image: {str(image)}')
    logging.debug(f'labels shape: {labels.shape}')
    for t in range(t_max):
        try:
            l_max = np.max(labels[t])
            logging.debug(f'labels max at {t}: {l_max}')
            im = images[a647][t, ...].compute()
            im = np.array(im)
            logging.debug(f'A647 image frame: {im.shape}, {type(im)}')
            df = regionprops_table(labels[t], 
                                   intensity_image=im, 
                                   properties=('label', 'centroid', 'inertia_tensor_eigvals',
                                   'area', 'mean_intensity', 'max_intensity'))
            df['t'] = [t,] * len(df['area']) 
            df = pd.DataFrame(df)
            df = df.set_index('label')
            df = df.rename(columns={
                'mean_intensity' : 'Alxa 647: mean_intensity',
                'max_intensity' : 'Alxa 647: max_intensity',
                })
        except:
            logging.debug(f't = {t}, a647 = {a647}')
            logging.exception('Got exeption whist geting A647 intensity info')
            raise
        try:
            im = images[a488][t, ...].compute()
            im = np.array(im)
            logging.debug(f'A488 image frame: {im.shape}, {type(im)}')
            df_a488 = regionprops_table(labels[t], 
                                        intensity_image=im, 
                                        properties=('label','mean_intensity', 'max_intensity'))
            df_a488 = pd.DataFrame(df_a488).rename(columns={
                                                            'mean_intensity' : 'GaAsP Alexa 488: mean_intensity',
                                                            'max_intensity' : 'GaAsP Alexa 488: max_intensity'
                                                            })
            df_a488 = df_a488.set_index('label')
        except:
            logging.debug(f't = {t}, a488 = {a488}')
            logging.exception('Got exeption whist geting A488 intensity info')
            raise
        try:
            im = images[a568][t, ...].compute()
            im = np.array(im)
            logging.debug(f'A568 image frame: {im.shape}, {type(im)}')
            df_a568 = regionprops_table(labels[t], 
                                        intensity_image=im, 
                                        properties=('label','mean_intensity', 'max_intensity'))
            df_a568 = pd.DataFrame(df_a568).rename(columns={
                                                            'mean_intensity' : 'GaAsP Alexa 568: mean_intensity',
                                                            'max_intensity' : 'GaAsP Alexa 568: max_intensity'
                                                            })
            df_a568 = df_a568.set_index('label')
            del im
        except:
            logging.debug(f't = {t}, a568 = {a568}')
            logging.exception('Got exeption whist geting A568 intensity info')
            raise
        df = pd.concat([df, df_a488, df_a568], axis=1)
        labs_df.append(df)
    labs_df = pd.concat(labs_df)
    cols = df.columns.values
    # rename the voxel coordinate columns
    cols = [c for c in cols if c.find('centroid') != -1]
    ax = ['z_pixels', 'y_pixels', 'x_pixels'] # this should be true after np.transpose in read_image()
    rename = {cols[i] : ax[i] for i in range(len(cols))}
    labs_df = labs_df.rename(columns=rename)
    # add coloumn with coordinates in microns
    microns = ['zs', 'ys', 'xs']
    for m, a in zip(microns, ax):
        labs_df[m] = labs_df[a] * meta[a[0]]
    # add volume column (in microns)
    one_voxel = meta['x'] * meta['y'] * meta['z']
    labs_df['volume'] = labs_df['area'] * one_voxel
    # get flatness (or lineness) scores
    labs_df['elongation'] = np.sqrt(1 - labs_df['inertia_tensor_eigvals-2'] / labs_df['inertia_tensor_eigvals-0'])
    labs_df['flatness'] = np.sqrt(1 - labs_df['inertia_tensor_eigvals-2'] / labs_df['inertia_tensor_eigvals-1'])
    # ensure unique labels
    labs_df['pid'] = range(len(labs_df)) # ensure unique labels
    # add file info
    labs_df['file'] = meta['file']
    labs_df['cohort'] = meta['cohort']
    labs_df['treatment'] = meta['treatment']
    # save the results into a file structure that reflects that in which the 
    #   original data is saved
    name = meta['file'] +'_' + dt + '_platelet-coords.csv'
    os.makedirs(os.path.join(out_dir, meta['cohort']), exist_ok=True)
    info_out_dir = os.path.join(out_dir, meta['cohort'], meta['treatment'])
    os.makedirs(info_out_dir, exist_ok=True)
    path = os.path.join(info_out_dir, name)
    labs_df.to_csv(path)
    return labs_df, path


# -------
# Compute
# -------
try:
    local = True
    # Get U-net with correct number of channels (doesn't work for forked)
    if local:
        sd = torch.load(unet_path, map_location=torch.device('cpu'))
    else:
        sd = torch.load(unet_path)
    out_channels = sd['c8_0.batch1.weight'].shape[0]
    unet = UNet(out_channels=out_channels)
    logging.debug('Loaded unet.')
except:
    logging.exception('Got an exception whilst preparing to process files')
    raise

# /Users/amcg0011/Data/pia-tracking/191016_IVMTR12_Inj4_cang_exp3.nd2   
try:
    logging.debug(f'Attempting to process file: {image_path}')
    images, image, meta, channels_dict = read_image(image_path)
    image = image[73:75]
    meta['batch_name'] = batch_name
    logging.debug(str(meta))
    # get some debuggin info
    frame = tuple(image[0, ...].shape)
    t_max = image.shape[0]
    im_max = image[0, ...].max().compute()
    im_type = image[0, ...].dtype
    logging.debug(f't max: {t_max}, image max val: {im_max}, image data type: {im_type}')
    # get prediciton output volume to which to write frame unet predictions
    prediction_output = np.zeros((out_channels,) + frame, dtype=np.float32)
    logging.debug('Prediction output shape: ' + str(prediction_output[0].shape))
    # get the labels output volume to which to write frame labels
    labels = [zarr.zeros(frame, dtype=np.uint32, chunk_size=(1,) + frame) for _ in range(t_max)]
    #logging.debug('Labels shape: ' + str(labels.shape))
    # closure to connect to threadworker signal
    def segment(prediction, t):
        yield from segment_output_image(prediction, affinities_channels=(0, 1, 2), 
                                        centroids_channel=4, thresholding_channel=3, 
                                        out=labels[t], use_logging=LOG_NAME)
    size = (10, 256, 256)
except:
    logging.exception('Got an exception whilst preparing to segment')
    raise
for t in range(t_max):
    try:
        logging.debug('Predicting output chunks...')
        x_input = normalise_data(image[t].compute().astype(np.float32))
        predict_output_chunks(unet, x_input, size, prediction_output, margin=(1, 64, 64))
        pred_val = np.array(prediction_output[0, 16]).mean()
        logging.debug(f'Prediction output 16th z-slice average: {pred_val}')
        logging.debug('Running affinities watershed...')
        for i in segment(prediction_output, t):
            pass
    except:
        logging.exception(f'Got an exception whist segmenting frame {t}')
        raise
try:
    # make cohort and treatment directories in scratch if they don't exist
    os.makedirs(os.path.join(scratch_dir, meta['cohort']), exist_ok=True)
    scratch_out_dir = os.path.join(scratch_dir, meta['cohort'], meta['treatment'])
    os.makedirs(scratch_out_dir, exist_ok=True)
    # save the labels as a zarr (to scratch)
    labels_name = dt + '_' + meta['file'] + '_labels.zarr'
    labels_path = os.path.join(scratch_out_dir, labels_name) 
    pred_name = dt + '_' + meta['file'] + '_pred_frame.zarr' 
    pred_path = os.path.join(scratch_out_dir, pred_name) 
    zarr.save(pred_path, prediction_output)
    # covert list of zarrs to single
    full_labels = zarr.zeros(image.shape, dtype=np.uint32, chunk_size=(1,) + frame)
    for i, za in enumerate(labels):
        full_labels[i] = za
    labels = full_labels
    logging.debug(f'Saving labels zarr with shape {labels.shape} and chunks of shape {labels.chunks}...')
    zarr.save(labels_path, labels)
    create_ts_meta(labels_path, {'scale': meta['scale'], 'translate': meta['translate']})
    meta['labels_path'] = labels_path
    lab_max = labels[0].max()
    logging.debug(f'Labels max val: {lab_max}')
except:
    logging.exception('Got an exeption whilst saving labels')
    raise
try:
    labs_df, df_path = get_labels_info(labels, meta, t_max, out_dir, images, channels_dict, dt)
    meta['platelets_info_path'] = df_path
except:
    logging.exception('Got an exeption whilst attempting to extract segmentation info')
    raise
try:
    meta['datetime'] = dt
    metadata_name = 'segmentation-metadata.csv'
    metadata_path = os.path.join(out_dir, metadata_name)
    if os.path.exists(metadata_path):
        md = pd.read_csv(metadata_path)
        new = pd.DataFrame([meta,])
        md = pd.concat([md, new])
    else:
        md = pd.DataFrame([meta,])
    md.to_csv(metadata_path)
    logging.debug(f'Saved metadata at {metadata_path}')
except:
    logging.exception('Got an exeption whist saving metadata')
    raise

