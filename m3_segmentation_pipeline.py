import logging
from datetime import datetime

now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")
LOG_NAME = f'/projects/rl54/results/{dt}_segmentation-log.log'
log_basic = True
logging.basicConfig(filename=LOG_NAME, encoding='utf-8', level=logging.DEBUG)
logging.debug('Starting script...')


try:
    from segmentation_pipeline import load_from_json, read_image, \
        segment_timeseries, get_labels_info, save_metadata
    import argparse
    import os
    from tracking import track_from_df
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
    logging.debug('Parsed arguments.')
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
except:
    logging.exception('Got an exception whilst reading info')
    raise

# -------------
# Segment Image
# -------------
try:
    labels = segment_timeseries(image, unet, meta, scratch_dir)
except:
    logging.exception('Got an exception whilst segmenting time series')
    raise

# ----------------------
# Extract Platelets Info
# ----------------------
try:
    df, _ = get_labels_info(labels, images, channels_dict, meta, out_dir, dt)
except:
    logging.exception('Got an exception whilst extracting platelet info from labels and images')
    raise

# ----------------------
# Track from Coordinates
# ----------------------
try:
    tracks_dir = os.path.join(out_dir, dt + '_platelet-tracks')
    tracks_name = track_from_df(df, meta, tracks_dir)
    meta['platelet_tracks'] = tracks_name
except:
    logging.exception('Got an exception whilst tracking platelets')
    raise

# -------------
# Save Metadata
# -------------
try:
    save_metadata(meta, out_dir, batch_name)
except:
    logging.exception('Got an exception whilst tracking platelets')
    raise
