import argparse
import os
import pandas as pd
from pathlib import Path
import trackpy as tp

p = argparse.ArgumentParser()
p.add_argument('-m', '--metadata', nargs='*', help='the csv files containing metadata for processed images')
p.add_argument('-o', '--outdir', help='

#directory into which output will be saved')
args = p.parse_args()
paths = args.metadata
out_dir = args.outdir


def track(platelets):
    search_range = 3 
    linked_pc = tp.link_df(platelets, search_range, 
                           pos_columns=['x', 'y', 'z'], 
                           t_column='t', memory=1)
    return linked_pc


for p in paths:
    # read in the meta data
    df = pd.read_csv(p)
    platelets_paths = df['platelets_info_path'].values
    tracks_df_paths = [] 
    for p_path in platelets_paths:
        platelets = pd.read_csv(p_path)
        # track the platelets
        tracks = track(platelets)
        # save them
        tracks_name = Path(p_path).stem + '_tracks.csv'
        tracks_path = os.path.join(out_dir, tracks_name)
        tracks.to_csv(tracks_path)
        tracks_df_paths.append(tracks_path)
    df['tracks_path'] = tracks_df_paths
    # update the saved data frame with the new 
    df.to_csv(p)





