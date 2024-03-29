from tracking import get_platelets_paths
from scipy.ndimage.measurements import label
from nd2_dask.nd2_reader import nd2_reader
from nd2reader import ND2Reader
import numpy as np
import os
import pandas as pd
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops_table
from toolz.itertoolz import count
from unet import UNet
import torch
from pathlib import Path
from plateseg.predict import predict_output_chunks, make_chunks
from plateseg.watershed import segment_output_image
import dask.array as da
import json
from time import time
import zarr
from zarpaint._zarpaint import create_ts_meta
import logging
from datetime import datetime
from tifffile import TiffWriter
from tqdm import tqdm


# Set up logging file in untracked local dir within repo
now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")
THIS_PATH = Path(__file__).parent.resolve()
DIR_PATH = THIS_PATH.parents[0]
LOG_DIR = DIR_PATH / 'untracked' 
os.makedirs(LOG_DIR, exist_ok=True)
LOG_NAME = LOG_DIR / f'{dt}_segmentation-log.log'
logging.basicConfig(filename=LOG_NAME, encoding='utf-8', level=logging.DEBUG)


# ------------------
# Read in Timeseries
# ------------------

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
    scale = [scl for scl in data['scale']]
    translate = data['translate']
    image = image / np.max(image, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
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


def get_metadata(
        name, 
        axes, 
        scale, 
        translate, 
        channels, 
        rois, 
        px_microns, 
        frame_rate, 
        cohort, 
        treatment):
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
    metadata_dict.update(**{ m:abs(scale[i]) for i, m in enumerate(axes)})
    metadata_dict.update(**{ 'channel_'+ str(i):c for i, c in enumerate(channels)})
    metadata_dict.update(frame_rate = frame_rate)
    metadata_dict.update(roi_t = float(np.mean(rois['timepoints'])))
    metadata_dict.update(roi_x = rois['positions'][0][1])
    metadata_dict.update(roi_y = rois['positions'][0][0])
    metadata_dict.update(roi_size = float(rois['sizes'][0][0]))
    logging.debug('Obtained initial metadata dict.')
    return metadata_dict


# --------------
# Segment Volume
# --------------

def segment_volume(
        image, 
        unet, 
        labels,
        t=None,
        affinities_channels=(0, 1, 2),
        centroids_channel=4,
        thresholding_channel=3,
        size=(10, 256, 256),
        save_pred=None,
        ):
    prediction_output = np.zeros((5, ) + image.shape[1:], dtype=np.float32)
    if t is None:
        labels = [labels,]
        image = [image, ]
        t = 0
    # normalise the entire frame @ t -  as was done prior to training
    x_input = rescale_intensity(image[t].compute()).astype(np.float32)
    # predictions by unet
    logging.debug('Predicting output chunks...')
    for _ in predict_output_chunks(
        unet, x_input, size, prediction_output, margin=(1, 64, 64)
    ):
        pass
    pred_val = np.array(prediction_output[:, 16]).mean()
    logging.debug(f'Prediction output 16th z-slice average: {pred_val}')
    if save_pred is not None:
        with TiffWriter(save_pred) as tiff:
            tiff.save(prediction_output)
    # segment with affinities watershed
    logging.debug('Running affinities watershed...')
    labels[t, :, :, :] = segment_output_image(prediction_output, affinities_channels=affinities_channels,
                                  centroids_channel=centroids_channel,
                                  thresholding_channel=thresholding_channel)[0]


# ------------------
# Segment Timeseries
# ------------------

def segment_timeseries(image, 
        unet, 
        meta, 
        out_dir, 
        batch_name,
        create_ts_md=True,
        save_pred=None,
    ):
    frame = image.shape[1:]
    t_max = image.shape[0]
    #padded_frame = [a + 2 for a in frame]
    #padded_frame = tuple(padded_frame)
    # get the labels output volume to which to write frame labels
    labels = zarr.zeros(image.shape, dtype=np.uint32, chunk_size=(1,) + frame)
    for t in tqdm(range(t_max)):
        segment_volume(image, unet, labels, t=t, save_pred=save_pred)
    # make cohort and treatment directories in scratch if they don't exist
    os.makedirs(os.path.join(out_dir, meta['cohort']), exist_ok=True)
    nested_out_dir = os.path.join(out_dir, meta['cohort'], meta['treatment'])
    os.makedirs(nested_out_dir, exist_ok=True)
    # get the path to which to save 
    labels_name = dt + '_' + meta['file'] + '_labels.zarr'
    labels_path = get_labels_path(meta, out_dir, batch_name)
    # save the labels
    logging.debug(f'Saving labels zarr with shape {labels.shape} and chunks of shape {labels.chunks}...')
    zarr.save(labels_path, labels)
    if create_ts_md:
        create_ts_meta(labels_path, {'scale': meta['scale'], 'translate': meta['translate']})
    # add labels path to metadata 
    meta['labels_path'] = labels_path
    # debugging
    lab_max = labels[0].max()
    logging.debug(f'Labels max val: {lab_max}')
    return labels


def get_labels_path(meta, out_dir, batch_name):
    labels_name = batch_name + '_' + meta['file'] + '_labels.zarr'
    nested_out_dir = os.path.join(out_dir, meta['cohort'], meta['treatment'])
    labels_path = os.path.join(nested_out_dir, labels_name) 
    return labels_path

# ---------------------
# Obtain Platelets Info
# ---------------------

def get_labels_info(labels, images, channels_dict, meta, out_dir, batch_name, dt):
    '''
    Parameters
    ----------
    labels: array like
        Int array containing labeled platelets at each point in time (t, z, y, x)
    images: list of dask.array.Array
        List of image timeseries for different channels captured by the imaging modality 
        [<dask array with (t, z, y, x)>, ...]
    channels_dict: dict
        Dict with string keys representing the channels as described in the metadata. Values
        are integers representing the index at which the channel timeseries can be found in 
        the images list. {'Alxa 647' : 2}
    meta: dict
        Dict with metadata. Must contain the following fields:
            - x : micron per pixel x
            - y : micron per pixel y
            - z : micron per pixel z
            - file : name of the file
            - cohort : name of the cohort
            - treatment : name of the treatment
    out_dir: str
        directory into which to save the output csv
    id_str: str
        ID string that will be used in saving output. This is usually a date time id that 
        idetifies the time at which the segmentation was run. 

    Returns
    -------
    labs_df: pandas.DataFrame
        Contains information about each platelet in the timeseries. 
        The columns are outlined below:
        Coordinate information:
            - label : original label from the image (separate sets of labels for each frame)
            - pid : platelet ID (separate IDs for each detected object in timeseries)
            - t : frame number
            - z_pixels : z coordinates in pixels
            - y_pixels : y coordinates in pixels
            - x_pixels : x coordinates in pixels
            - zs : z coordinates in microns
            - ys : y coordinates in microns
            - xs : x coordinates in microns
        Shape information:
            - volume : platelet volume in cubic microns
            - elongation : measure of the degree to which objects are elongated.
                           smallest inertia tensor eigenvalue (of 3) / largest
            - flatness : measure of the degree to which objects are elongated.
                         medium inertia tensor eigenvalue (of 3) / largest
        Channel information:
            - <channel name> : mean intensity : mean intensity for platelet
            - <channel name> : max intensity : max intensity for platelet
    path:
        path at which the data was saved as a csv
    '''
    t_max = labels.shape[0]
    labs_df = []
    started = False
    logging.debug(f'Images: {str(images)}')
    logging.debug(f'labels shape: {labels.shape}')
    logging.debug(f'Channels dict: {channels_dict}')
    # go through each subsequent frame in the labels file
    counter = 0
    for t in tqdm(range(t_max)):
        try:
            l_max = np.max(labels[t])
            logging.debug(f'labels max at {t}: {l_max}')
            chans_dfs = []
            chans_started = False
            for key in channels_dict.keys():
                chan = channels_dict[key]
                logging.debug(f'Segmenting channel {key}')
                im = images[channels_dict[key]][t, ...].compute()
                im = np.array(im)
                logging.debug(f'{key} frame: {im.shape}, {type(im)}')
                if not chans_started:
                    props = ('label', 'centroid', 'inertia_tensor_eigvals',
                                   'area', 'mean_intensity', 'max_intensity')
                    chans_started = True
                else:
                    props = ('label', 'mean_intensity', 'max_intensity')
                df = regionprops_table(labels[t], 
                                   intensity_image=im, 
                                   properties=props)
                df['t'] = [t,] * len(df['label']) 
                df = pd.DataFrame(df)
                df_labs = df['label'].values
                df = df.set_index('label')
                df = df.rename(columns={
                'mean_intensity' : f'{key}: mean_intensity',
                'max_intensity' : f'{key}: max_intensity',
                })
                chans_dfs.append(df)
        except:
            logging.debug(f't = {t}, {key} = {channels_dict[key]}')
            logging.exception('Got exeption whist geting A647 intensity info')
            raise
        df = pd.concat(chans_dfs, axis=1)
        df['label'] = df_labs
        df_pids = [i for i in range(counter, len(df['label']) + counter)]
        counter += len(df['label'])
        df['pid'] = df_pids
        df = df.set_index('pid')
        bbb = len(df)
        df = df.loc[~df.index.duplicated(keep='first')]
        aaa = len(df)
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
    factors = []
    for m, a in zip(microns, ax):
        mic_per_pix = meta[a[0]]
        factors.append(mic_per_pix)
        labs_df[m] = labs_df[a] * mic_per_pix
    # add volume column (in microns)
    one_voxel = meta['x'] * meta['y'] * meta['z']
    labs_df['volume'] = labs_df['area'] * one_voxel
    # get flatness (or lineness) scores
    labs_df = add_elongation(labs_df)
    labs_df = add_flatness(labs_df)
    # add file info
    labs_df['file'] = meta['file']
    labs_df['cohort'] = meta['cohort']
    labs_df['treatment'] = meta['treatment']
    # save the results into a file structure that reflects that in which the 
    #   original data is saved
    os.makedirs(os.path.join(out_dir, meta['cohort']), exist_ok=True)
    info_out_dir = os.path.join(out_dir, meta['cohort'], meta['treatment'])
    os.makedirs(info_out_dir, exist_ok=True)
    path = get_info_path(meta, out_dir, batch_name)
    labs_df.to_csv(path)
    meta['datetime'] = dt
    meta['platelets_info_path'] = path
    return labs_df, path


def add_elongation(df):
    df['elongation'] = np.sqrt(1 - df['inertia_tensor_eigvals-2'] / df['inertia_tensor_eigvals-0'])
    return df


def add_flatness(df):
    df['flatness'] = np.sqrt(1 - df['inertia_tensor_eigvals-2'] / df['inertia_tensor_eigvals-1'])
    return df


def get_info_path(meta, out_dir, batch_name):
    name = batch_name + '_' + meta['file'] +'_platelet-coords.csv'
    info_out_dir = os.path.join(out_dir, meta['cohort'], meta['treatment'])
    path = os.path.join(info_out_dir, name)
    return path

# -------
# Helpers
# -------

def load_unet(unet_path, cpu=False):
    if cpu:
        sd = torch.load(unet_path, map_location=torch.device('cpu'))
    else:
        sd = torch.load(unet_path, map_location=torch.device('cuda'))
    out_channels = sd['c8_0.batch1.weight'].shape[0]
    unet = UNet(in_channels=1, out_channels=out_channels)
    unet.load_state_dict(sd)
    if not cpu:
        unet.cuda()
    return unet


def load_from_json(info_path):
    with open(info_path, 'r') as f:
        info = json.load(f)
    logging.debug('Loaded JSON info.')
    out_dir = info['out_dir']
    image_path = info['image_path']
    scratch_dir = info['scratch_dir']
    unet_path = info['unet_path']
    unet = load_unet(unet_path)
    batch_name = info['batch_name']
    logging.debug('Loaded json info for segmentation')
    return out_dir, image_path, scratch_dir, unet, batch_name


def save_metadata(meta, out_dir, batch_name):
    name = meta['file']
    metadata_name = batch_name + '_' + name + '_seg-md.csv'
    metadata_path = os.path.join(out_dir, metadata_name)
    md = pd.DataFrame([meta,])
    md.to_csv(metadata_path)
    logging.debug(f'Saved metadata at {metadata_path}')


# -------
# Compute
# -------
if __name__ == '__main__':
    #import argparse
    #p = argparse.ArgumentParser()
    #p.add_argument('-i', '--info', help='JSON file containing info for segmentation')
    info_path = '/home/abigail/GitRepos/platelet-segmentation/segmentation-jsons/inj4-dmso_worms.json'
    #args = p.parse_args()
    #info_path = args.info
    out_dir, image_path, scratch_dir, unet, batch_name = load_from_json(info_path)
    images, image, meta, channels_dict = read_image(image_path)
    pred_name = os.path.join(out_dir, 'pred_volume.tif')
    #labels = segment_timeseries(image, unet, meta, scratch_dir, batch_name, save_pred=pred_name)
    import zarr
    labels = zarr.open('/home/abigail/data/plateseg-training/timeseries_seg/plateseg-training/timeseries_seg/210914_195928_191016_IVMTR12_Inj4_cang_exp3_labels.zarr')
    df, _ = get_labels_info(labels, images, channels_dict, meta, out_dir, batch_name, dt)
    save_metadata(meta, out_dir, batch_name)
    import napari
    v = napari.view_image(image, scale=(1, 4, 1, 1), blending='additive')
    v.add_labels(labels, scale=(1, 4, 1, 1), blending='additive')
    napari.run()

    
