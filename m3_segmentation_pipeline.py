import argparse
from datetime import datetime
from re import S
from napari_bioformats import read_bioformats
from nd2reader import ND2Reader
import numpy as np
import os
import pandas as pd
#from pims.bioformats import BioformatsReader
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table
from unet import UNet
import torch
from pathlib import Path
from plateseg.predict import predict_output_chunks, make_chunks
from segmentation import segment_output_image
import dask.array as da


# ---------------
# Parse Arguments
# ---------------
p = argparse.ArgumentParser()
p.add_argument('-i', '--images', nargs='*', help='the image paths to process')
p.add_argument('-s', '--scratchdir', help='directory into which temporary output will be saved')
p.add_argument('-o', '--outdir', help='directory into which output will be saved')
p.add_argument('-u', '--unet', help='path to the unet state_dict')
args = p.parse_args()
image_paths = args.images
scratch_dir = args.scratchdir
out_dir = args.outdir
unet_path = args.unet


# ---------
# Functions
# ---------


def read_image(image_path):
    name = Path(image_path).stem
    # get the nd2 metadata for rois
    data = ND2Reader(image_path)
    rois = data.metadata['rois'][0]
    axes = data.axes
    px_microns = data.metadata['pixel_microns']
    frame_rate = float(data.frame_rate)
    # read in with napari_bioformats (nd2reader can have issues)
    image, data = read_bioformats(image_path)[0]
    scale = data['scale']
    print(scale)
    channels = get_channel_names(data['name'], name)
    #pixel_size = data.scale
    metadata_dict = get_metadata(name, axes, scale, channels, 
                                 rois, px_microns, frame_rate)
    a488 = da.from_array(get_channel(image, data, st='GaAsP Alexa 488')[:10])
    print(a488)
    a568 = da.from_array(get_channel(image, data, st='GaAsP Alexa 568')[:10])
    print(a568)
    other_chans = {
        'GaAsP Alexa 488' : a488,
        'GaAsP Alexa 568' : a568
    }
    print(other_chans)
    image = get_channel(image, data)[:10] # for now only use a small portion
    return image, metadata_dict, other_chans


def get_channel(image, meta, st='Alxa 647'):
    channels = meta['name']
    alxa647_channel = next(
        i
        for i, name in enumerate(channels)
        if st in name
    )
    print(alxa647_channel)
    channel_axis = meta['channel_axis']
    channel = np.asarray(np.take(image, alxa647_channel, axis=channel_axis))
    vol2predict = rescale_intensity(channel, out_range=np.float32)
    print(vol2predict.shape)
    return vol2predict


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


def get_metadata(name, axes, scale, channels, rois, px_microns, frame_rate):
    '''
    Generate metadata to be saved for each image (including about rois)
    '''
    metadata_dict = {
        'file' : name,
        'px_microns' : px_microns, 
        'axes' : axes
    }
    axes = ['t', 'z', 'y', 'x']
    metadata_dict.update(**{ m:scale[i] for i, m in enumerate(axes)})
    metadata_dict.update(**{ 'channel_'+ str(i):c for i, c in enumerate(channels)})
    metadata_dict.update(frame_rate = frame_rate)
    metadata_dict.update(roi_t = float(np.mean(rois['timepoints'])))
    metadata_dict.update(roi_x = rois['positions'][0][1])
    metadata_dict.update(roi_y = rois['positions'][0][0])
    metadata_dict.update(roi_size = float(rois['sizes'][0][0]))
    return metadata_dict


def get_labels_info(labels, meta, t_max, out_dir, image, other_chans):
    labs_df = []
    for t in range(t_max):
        df = regionprops_table(labels[t], 
                               intensity_image=image[t], 
                               properties=('label', 'centroid', 'area', 'intensity_mean'))
        df['t'] = [t,] * len(df['area'])
        df = pd.DataFrame(df)
        df.rename(columns={'intensity_mean' : 'Alxa 647: intensity_mean'})
        df = df.set_index('label')
        df_a488 = regionprops_table(labels[t], 
                                    intensity_image=other_chans['GaAsP Alexa 488'][t].compute(), 
                                    properties=('intensity_mean'))
        df_a488 = pd.DataFrame(df_a488).rename(columns={'intensity_mean' : 'GaAsP Alexa 488: intensity_mean'})
        df_a488 = df_a488.set_index('label')
        df_a568 = regionprops_table(labels[t], 
                                    intensity_image=other_chans['GaAsP Alexa 568'][t].compute(), 
                                    properties=('intensity_mean'))
        df_a568 = pd.DataFrame(df_a568).rename(columns={'intensity_mean' : 'GaAsP Alexa 568: intensity_mean'})
        df_a568 = df_a568.set_index('label')
        df = pd.concat([df, df_a488, df_a568], axis=1)
        labs_df.append(df)
    labs_df = pd.concat(labs_df)
    cols = df.columns.values
    cols = [c for c in cols if c.find('centroid') != -1]
    ax = ['z', 'y', 'x'] # this should be true after np.transpose in read_image()
    rename = {cols[i] : ax[i] for i in range(len(cols))}
    labs_df.rename(columns=rename)
    labs_df['labels'] = range(len(labs_df)) # ensure unique labels
    name = meta['file'] + '_platelet-coords.csv'
    path = os.path.join(out_dir, name)
    labs_df.to_csv(path)
    return labs_df, path


# -------
# Compute
# -------

# Get U-net with correct number of channels (doesn't work for forked)
sd = torch.load(unet_path)
out_channels = sd['c8_0.batch1.weight'].shape[0]
unet = UNet(out_channels=out_channels)
now = datetime.now()
metadata_name = now.strftime("%y%m%d_%H%M%S") + '_segmentation-metadata.csv'
metadata_to_save = []
for path in image_paths: # /Users/amcg0011/Data/pia-tracking/191016_IVMTR12_Inj4_cang_exp3.nd2
    image, meta, other_chans = read_image(path)
    print(meta)
    frame = tuple(image[0].shape)
    t_max = image.shape[0]
    print(t_max)
    prediction_output = [np.zeros((out_channels,) + frame, dtype=np.float32)
                            for _ in range(t_max)]
    print(prediction_output[0].shape)
    labels = [np.zeros(prediction_output[0].shape[1:], dtype=np.uint32) for _ in range(t_max)]
    print(labels[0].shape)
    # closure to connect to threadworker signal
    def segment(prediction):
        for t in range(t_max):
            print('Frame: ', t)
            yield from segment_output_image(prediction[t], affinities_channels=(0, 1, 2), 
                                            centroids_channel=4, thresholding_channel=3, 
                                            out=labels[t])
    size = (10, 256, 256)
    predict_output_chunks(unet, image, size, prediction_output, margin=(1, 64, 64))
    for i in segment(prediction_output):
        pass
    prediction_output = np.stack(prediction_output)
    print(prediction_output.shape)
    prediction_name = meta['file'] + '_prediction.zarr'
    prediciton_path = os.path.join(scratch_dir, prediction_name)
    print(prediciton_path)
    chunk_size = (1, 1, ) + frame
    print(chunk_size)
    prediction_output = da.from_array(prediction_output, chunks=chunk_size)
    prediction_output.to_zarr(prediciton_path)
    labels = np.stack(labels)
    labels_name = meta['file'] + '_labels.zarr'
    labels_path = os.path.join(out_dir, labels_name)
    labels = da.from_array(labels, chunks=chunk_size[1:])
    labels.to_zarr(labels_path)
    meta['labels_path'] = labels_path
    labs_df, df_path = get_labels_info(labels, meta, t_max, out_dir)
    meta['platelets_info_path'] = df_path
    metadata_to_save.append(meta)
metadata_to_save = pd.DataFrame(metadata_to_save)
metadata_path = os.path.join(out_dir, metadata_name)
metadata_to_save.to_csv(metadata_path)

