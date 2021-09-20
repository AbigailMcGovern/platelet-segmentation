import logging
from datetime import datetime

now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")
LOG_NAME = f'/projects/rl54/results/logging/{dt}_segmentation-log.log'
#LOG_NAME = f'/home/abigail/GitRepos/platelet-segmentation/untracked/{dt}_segmentation-log-local.log'
log_basic = True
if log_basic:
    logging.basicConfig(filename=LOG_NAME, encoding='utf-8', level=logging.DEBUG)
else:
    pass
logging.debug('Starting script...')


try:
    from segmentation_pipeline import load_from_json, read_image, \
        segment_timeseries, get_labels_info, save_metadata, get_labels_path, \
        get_info_path
    import argparse
    from time import time
    import os
    import pandas as pd
    from tracking import track_from_df
    logging.debug('Finished imports.')
    start_time = time()
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
    logging.debug('Parsed arguments.')
    #args.info = '/home/abigail/GitRepos/platelet-segmentation/segmentation-jsons/inj4-dmso_worms.json'
    info_path = args.info
    # load the required variables
    out_dir, image_path, scratch_dir, unet, batch_name = load_from_json(info_path)
except:
    logging.exception('Got an exception whilst parsing info')
    raise

# -------------
# Read ND2 data
# -------------
try:
    images, image, meta, channels_dict = read_image(image_path)
    t = time()
except:
    logging.exception('Got an exception whilst reading info')
    raise

# -------------
# Segment Image
# -------------
try:
    labs_dir = os.path.join(scratch_dir, batch_name + '_segmentations')
    labels_path = get_labels_path(meta, labs_dir, batch_name)
    if os.path.exists(labels_path):
        import zarr
        labels = zarr.open(labels_path)
        logging.debug('Loaded existing labels.')
    else:
        logging.debug('Labels file does not exist, segmenting timeseries')
        labels = segment_timeseries(image, unet, meta, labs_dir, batch_name)
        t = time()
        seg_time = t - start_time
        logging.debug(f'Segmented timeseries in {seg_time} seconds')
except:
    logging.exception('Got an exception whilst segmenting time series')
    raise

# ----------------------
# Extract Platelets Info
# ----------------------
try:
    df_path = get_info_path(meta, labs_dir, batch_name)
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        meta['platelets_info_path'] = df_path
    else:
        df, df_path = get_labels_info(labels, images, channels_dict, meta, labs_dir, batch_name, dt)
        df = pd.read_csv(df_path) # because pandas indexing???
        info_time = time() - t 
        t = time()
        logging.debug(f'Obtained platelet info in {info_time} seconds')
except:
    logging.exception('Got an exception whilst extracting platelet info from labels and images')
    raise

# ----------------------
# Track from Coordinates
# ----------------------
try:
    tracks_dir = os.path.join(out_dir, batch_name + '_platelet-tracks')
    tracks_name = track_from_df(df, meta, tracks_dir)
    meta['platelet_tracks'] = tracks_name
    track_time = time() - t 
    logging.debug(f'Obtained platelet info in {track_time} seconds')
except:
    logging.exception('Got an exception whilst tracking platelets')
    raise

# -------------
# Save Metadata
# -------------
try:
    save_metadata(meta, tracks_dir, batch_name)
except:
    logging.exception('Got an exception whilst tracking platelets')
    raise
