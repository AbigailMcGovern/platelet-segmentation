import trackpy as tp
import argparse
import os
import pandas as pd
from pathlib import Path
import numpy as np


def track(platelets):
    search_range = 3 
    linked_pc = tp.link_df(platelets, search_range, 
                           pos_columns=['xs', 'ys', 'zs'], 
                           t_column='t', memory=1)
    return linked_pc


def get_platelets_paths(md_path):
    meta = pd.read_csv(md_path)
    platelets_paths = meta['platelets_info_path'].values
    return platelets_paths, meta


def add_track_len(df):
    ids = df['particle'].values
    track_count = np.bincount(ids)
    df['track length'] = track_count[ids]
    return df


def filter_df(df, min_frames=20):
    df_filtered = df.loc[df['track length'] >= min_frames, :]
    return df_filtered


if __name__ == '__main__':
    md_path = '/home/abigail/data/plateseg-training/timeseries_seg/debugging-seg-pipeline_segmentation-metadata.csv'
    out_dir = '/home/abigail/data/plateseg-training/timeseries_seg'
    p_paths, meta = get_platelets_paths(md_path)
    t_paths = []
    for p in p_paths:
        df = pd.read_csv(p)
        df = track(df)
        tracks_name = Path(p).stem + '_tracks.csv'
        tracks_path = os.path.join(out_dir, tracks_name)
        df.to_csv(tracks_path)
        t_paths.append(tracks_path)
    meta['tracks_paths'] = t_paths
    meta.to_csv(md_path)
    