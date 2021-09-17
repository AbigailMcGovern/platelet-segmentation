# script to remove worms
import numpy as np
from scipy.ndimage.measurements import label
from skimage.measure import regionprops_table
import zarr
import os
from pathlib import Path
from segmentation_pipeline import add_elongation
import pandas as pd


def get_labs_less_than(df, lab_col='label', cond_col='elongation', val=0.9):
    df = df[(df[cond_col] < val)]
    labs = df[lab_col].values
    return labs


def only_include_labels(labels, ids):
    new = np.zeros_like(labels)
    new = new.ravel()
    for i, elem in enumerate(labels.ravel()):
        if elem in labs: # numpy where doesnt like this kind of condition unfortunately
            new[i] = elem
    new = new.reshape(labels.shape)
    return new


def get_files(dirs, ends='.zarr'):
    files = []
    for d in dirs:
        for sub in os.walk(d):
            if ends.endswith('.zarr'):
                if sub[0].endswith(ends):
                    files.append(sub[0])
            for fl in sub[2]:
                f = os.path.join(sub[0], fl)
                if f.endswith(ends):
                    files.append(f)
    return files


dirs = ['/home/abigail/data/plateseg-training/training_gt/Volga', '/home/abigail/data/plateseg-training/training_gt/Pia']
files = get_files(dirs, ends='_labels.zarr')
props = ('label', 'centroid', 'inertia_tensor_eigvals')

for f in files:
    labels = zarr.open(f)
    labels = np.array(labels)
    df = regionprops_table(labels, properties=props)
    df = pd.DataFrame(df)
    df = add_elongation(df)
    labs = get_labs_less_than(df)
    new = only_include_labels(labels, labs)
    p = Path(f)
    new_path = os.path.join(p.parents[0], p.stem + '_no-worms.zarr')
    zarr.save(new_path, new)
