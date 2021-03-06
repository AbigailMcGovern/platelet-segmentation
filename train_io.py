from augment import augment_images
from datetime import datetime
from helpers import get_files, log_dir_or_None, write_log, LINE
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import skimage.filters as filters
from skimage.measure import regionprops
from skimage.morphology._util import _offsets_to_raveled_neighbors
from tifffile import TiffWriter, imread
from time import time
import torch 
from tqdm import tqdm
import zarr


# -------------------
# Generate Train Data
# -------------------
def get_train_data(
                   image_paths, 
                   labels_paths,
                   out_dir, 
                   shape=(10, 256, 256), 
                   n_each=100, 
                   channels=('z-1', 'y-1', 'x-1', 'centreness'),
                   scale=(4, 1, 1), 
                   log=True
                   ):
    """
    Generate training data from whole ground truth volumes.

    Parameters
    ----------
    image_path: str
        path to the image zarr file, which must be in tczyx dim format (ome bioformats)
    labels_path: str
        path to the labels zarr file, which must be in tczyx dim format
    shape: tuple of int
        Shape of the test data to 
    channels: tuple of str
        tuple of channels to be added to the training data. 
            Affinities: 'axis-n' (pattern: r'[xyz]-\d+' e.g., 'z-1')
            Centreness: 'centreness'
    scale: tuple of numeric
        Scale of channels. This is used in calculating centreness score.
    log: bool
        Should print out be recorded in out_dir/log.txt?

    Returns
    -------
    xs: list of torch.Tensor
        List of images for training 
    ys: list of torch.Tensor
        List of affinities for training
    ids: list of str
        ID strings by which each image and label are named.
        Eventually used for correctly labeling network output

    Notes
    -----
    It takes a very long time to obtain training data with sufficient 
    information (as determined by min_affinity param). 
    """
    assert len(image_paths) == len(labels_paths)
    for i in range(len(image_paths)):
        print(LINE)
        s = f'Generating training data from image: {image_paths[i]}, labels: {labels_paths[i]}'
        print(s)
        if log:
            write_log(LINE, out_dir)
            write_log(s, out_dir)
        im = zarr.open_array(image_paths[i])
        l = zarr.open_array(labels_paths[i])
        if i == 0:
            xs, ys, ids = get_random_chunks(im, l, 
                                            out_dir, 
                                            shape=shape, 
                                            n=n_each, 
                                            channels=channels,
                                            scale=scale, 
                                            log=log)
        else:
            xs_n, ys_n, ids_n = get_random_chunks(im, l, 
                                                  out_dir, 
                                                  shape=shape, 
                                                  n=n_each, 
                                                  channels=channels,
                                                  scale=scale, 
                                                  log=log)
            for j in range(len(xs_n)):
                xs.append(xs_n[j])
                ys.append(ys_n[j])
                ids.append(ids_n[j])
    return xs, ys, ids



def get_random_chunks(
                      image, 
                      labels, 
                      out_dir, 
                      shape=(10, 256, 256), 
                      n=25, 
                      min_affinity=300, 
                      channels=('z-1', 'y-1', 'x-1', 'centreness'),
                      scale=(4, 1, 1), 
                      log=True
                      ):
    '''
    Obtain random chunks of data from whole ground truth volumes.

    Parameters
    ----------
    image: array like
        same shape as labels
    labels: array like
        same shape as image
    shape: tuple of int
        shape of chunks to obtain
    n: int
        number of random chunks to obtain
    min_affinity:
        minimum cut off sum of affinities for an image.
        As affinities as belong to {0, 1}, this param
        is the number of voxels that boarder labels.
    
    Returns
    -------
    xs: list of torch.Tensor
        List of images for training 
    ys: list of torch.Tensor
        List of affinities for training
    ids: list of str
        ID strings by which each image and label are named.
        Eventually used for correctly labeling network output
 
    '''
    im = np.array(image)
    l = np.array(labels)
    assert len(im.shape) == len(shape)
    a = get_training_labels(l, channels=channels, scale=scale)
    xs = []
    ys = []
    labs = []
    i = 0
    df = {'z_start' : [],
          'y_start' : [],
          'x_start' : []}
    while i < n:
        dim_randints = []
        for j, dim in enumerate(shape):
            max_ = im.shape[j] - dim - 1
            ri = np.random.randint(0, max_)
            dim_randints.append(ri)
        # Get the network output: affinities 
        s_ = [slice(None, None),] # 
        for j in range(len(shape)):
            s_.append(slice(dim_randints[j], dim_randints[j] + shape[j]))
        s_ = tuple(s_)
        y = a[s_]
        if y.sum() > min_affinity * len(channels): # if there are a sufficient number of boarder voxels
            # add coords to output df
            for j in range(len(shape)):
                _add_to_dataframe(j, dim_randints[j], df)
            # Get the network input: image
            s_ = [slice(dim_randints[j], dim_randints[j] + shape[j]) for j in range(len(shape))]
            s_ = tuple(s_)
            x = im[s_]
            x = normalise_data(x)
            # get the GT labels so that later quatitative comparison can be made with final
            #   segmentation  
            lab = l[s_] # that's right, be confused by my variable names!!
            # data augmentation for better generalisation
            x, y, lab = augment_images(x, y, lab) 
            # add the affinities and image chunk to the training data 
            y = torch.from_numpy(y.copy())
            ys.append(y)
            x = torch.from_numpy(x.copy())
            xs.append(x)
            labs.append(lab)
            # another successful addition, job well done you crazy mofo
            i += 1
    print(LINE)
    s = f'Obtained {n} {shape} chunks of training data'
    print(s)
    if log:
        write_log(LINE, out_dir)
        write_log(s, out_dir)
    log_dir = log_dir_or_None(log, out_dir)
    print_labels_info(channels, out_dir=log_dir)
    ids = save_random_chunks(xs, ys, labs, out_dir)
    now = datetime.now()
    d = now.strftime("%y%m%d_%H%M%S")
    df['data_ids'] = ids
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(out_dir, 'start_coords' + d + '.csv'))
    return xs, ys, ids


def _add_to_dataframe(dim, start, df):
    if dim == 0:
        df['z_start'].append(start)
    if dim == 1:
        df['y_start'].append(start)
    if dim == 2:
        df['x_start'].append(start)


# --------------------------
# Lable Generating Functions
# --------------------------

def get_training_labels(
                        l, 
                        channels=('z-1', 'y-1', 'x-1', 'centreness'),
                        scale=(4, 1, 1)):
    labels = []
    for chan in channels:
        if chan.startswith('z'):
            axis = 0
        elif chan.startswith('y'):
            axis = 1
        elif chan.startswith('x'):
            axis = 2
        n = re.search(r'\d+', chan)
        if n is not None:
            # get the nth affinity
            n = int(n[0]) 
            lab = nth_affinity(l, n, axis)
        elif chan == 'centreness':
            # get the centreness score
            lab = get_centreness(l, scale=scale)
        elif chan == 'centreness-log':
            lab = get_centreness(l, scale=scale, log=True)
        elif chan == 'centroid-gauss':
            lab = get_gauss_centroids(l)
        else:
            m = f'Unrecognised channel type: {chan} \n'
            m = m + 'Please enter str of form axis-n for nth affinity \n'
            m = m + 'or centreness for centreness score.'
            raise ValueError(m)
        if chan.endswith('-smooth'):
            lab = smooth(lab)
        labels.append(lab)
    labels = np.stack(labels, axis=0)
    return labels


def nth_affinity(labels, n, axis):
    affinities = []
    labs_pad = np.pad(labels, n, mode='reflect')
    for i in range(labels.shape[axis]):
        s_0 = [slice(None, None)] * len(labs_pad.shape) 
        s_0[axis] = slice(i, i + 1)
        s_0 = tuple(s_0)
        s_n = [slice(None, None)] * len(labs_pad.shape) 
        s_n[axis] = slice(i + n, i + n + 1)
        s_n = tuple(s_n)
        new = labs_pad[s_0] - labs_pad[s_n]
        new = np.squeeze(new)
        if len(new) > 0:
            affinities.append(new)
    affinities = np.stack(affinities, axis=axis)
    s_ = [slice(n, -n)] * len(labs_pad.shape)
    s_[axis] = slice(None, None)
    s_ = tuple(s_)
    affinities = affinities[s_]
    affinities = np.where(affinities != 0, 1., 0.)
    return affinities


def get_centreness(labels, scale=(4, 1, 1), log=False, power=False):
    """
    Obtains a centreness score for each voxel belonging to a labeled object.
    Values in each object sum to one. Values are inversely proportional
    to euclidian distance from the object centroid.

    Notes
    -----
    Another possible implementation would involve the medioid, as in: 
    Lalit, M., Tomancak, P. and Jug, F., 2021. Embedding-based Instance 
    Segmentation of Microscopy Images. arXiv.

    Unfortunately, skimage doesn't yet have a method for finding the  
    medioid (more dev, *sigh*).
    """
    scale = np.array(scale)
    def dist_score(mask):
        output = np.zeros_like(mask, dtype=np.float32)
        c = np.mean(np.argwhere(mask), axis=0)
        indices, values = inverse_dist_score(
                mask, c, scale, log=log, power=power
                )
        output[indices] = values
        return output
    t = time()
    props = regionprops(labels, extra_properties=(dist_score,))
    new = np.zeros(labels.shape, dtype=np.float32)
    for i, prop in tqdm(enumerate(props), desc='Score centreness'):
        new[prop.slice] += prop.dist_score
    new = np.nan_to_num(new)
    print('------------------------------------------------------------')
    print(f'Obtained centreness scores in {time() - t} seconds')
    return new


def inverse_dist_score(mask, centroid, scale, log, power):
    '''
    Compute euclidian distances of each index from a mask
    representing a single object from the centroid of said object

    Uses scale to account for annisotropy in image
    '''
    indices = np.argwhere(mask > 0)
    distances = []
    centre = centroid
    for i in range(indices.shape[0]):
        ind = indices[i, ...]
        diff = (centre - ind) * scale
        dist = np.linalg.norm(diff)
        if log and abs(dist) > 0:
            m = f'Infinite value with distance of {dist}'
            dist = np.log(dist)
            assert not np.isinf(dist), m
        if power:
            dist = 2 ** dist
        distances.append(dist)
    distances = np.array(distances)
    if log:
        distances = distances + np.abs(distances.min()) # bring min value to 0
    norm_distances = distances / distances.max()
    values = (1 - norm_distances) 
    indices = tuple(indices.T.tolist())
    return indices, values


# not used
def get_gauss_centroids(labels, sigma=1, z=0):
    centroids = [prop['centroid'] for prop in regionprops(labels)]
    centroids = tuple(np.round(np.stack(centroids).T).astype(int))
    centroid_image = np.zeros(labels.shape, dtype=float)
    centroid_image[centroids] = 1.
    gauss_cent = []
    for i in range(labels.shape[z]):
        s_ = [slice(None, None)] * labels.ndim
        s_[z] = slice(i, i+1)
        s_ = tuple(s_)
        plane = np.squeeze(centroid_image[s_])
        gauss_cent.append(filters.gaussian(plane, sigma=sigma))
    out = np.stack(gauss_cent, axis=z)
    out = out - out.min()
    out = out / out.max()
    #print(out.dtype, out.shape, out.max(), out.min())
    return out


def smooth(image, z=0, sigma=1):
    out = []
    for i in range(image.shape[z]):
        s_ = [slice(None, None)] * image.ndim
        s_[z] = slice(i, i+1)
        s_ = tuple(s_)
        plane = np.squeeze(image[s_])
        out.append(filters.gaussian(plane, sigma=sigma))
    out = np.stack(out, axis=z)
    return out


# not currently referenced, uses nth_affinity() for generality
def get_affinities(image):
    """
    Get short-range voxel affinities for a segmentation. Affinities are 
    belonging to {0, 1} where 1 represents a segment boarder voxel in a
    particular direction. Affinities are produced for each dimension of 
    the labels and each dim has its own channel (e.g, (3, z, y, x)). 

    Note
    ----
    Others may represent affinities with {-1, 0}, because technically... 
    My network wasn't designed for this :)
    """
    padded = np.pad(image, 1, mode='reflect')
    affinities = []
    for i in range(len(image.shape)):
        a = np.diff(padded, axis=i)
        a = np.where(a != 0, 1.0, 0.0)
        a = a.astype(np.float32)
        s_ = [slice(1, -1)] * len(image.shape)
        s_[i] = slice(None, -1)
        s_ = tuple(s_)
        affinities.append(a[s_])
    affinities = np.stack(affinities)
    return affinities    


# -------------
# Log and Print
# -------------

def print_labels_info(channels, out_dir=None, log_name='log.txt'):
    print(LINE)
    s = f'Training labels have {len(channels)} output channels: \n'
    print(s)
    if out_dir is not None:
            write_log(LINE, out_dir, log_name)
            write_log(s, out_dir, log_name)
    for i, chan in enumerate(channels):
        affinity_match = re.search(r'[xyz]-\d*', chan)
        if affinity_match is not None:
            n = f'{affinity_match[0]} affinities'
        elif chan == 'centreness':
            n = 'centreness score'
        elif chan == 'centreness-log':
            n = 'log centreness score'
        elif chan == 'centroid-gauss':
            n = 'gaussian centroids'
        else:
            n = 'Unknown channel type'
        s = f'Channel {i}: {n}'
        print(s)
        if out_dir is not None:
            write_log(s, out_dir, log_name)


# -----------
# Save Output
# -----------

def save_random_chunks(xs, ys, labs, out_dir):
    '''
    Save the random chunks as they are sampled
    '''
    os.makedirs(out_dir, exist_ok=True)
    assert len(xs) == len(ys)
    ids = []
    # iterate over the sample
    for i in range(len(xs)):
        # get the datetime to give the samples unique names 
        now = datetime.now()
        d = now.strftime("%y%m%d_%H%M%S") + '_' + str(i)
        ids.append(d)
        # save the image
        i_name = d + '_image.tif'
        i_path = os.path.join(out_dir, i_name)
        with TiffWriter(i_path) as tiff:
            tiff.write(xs[i].numpy())
        # save the labels
        l_name = d + '_labels.tif'
        l_path = os.path.join(out_dir, l_name)
        with TiffWriter(l_path) as tiff:
            tiff.write(ys[i].numpy())
        # save the ground truth segmentation
        s_name = d + '_GT.tif'
        s_path = os.path.join(out_dir, s_name)
        with TiffWriter(s_path) as tiff:
            tiff.write(labs[i]) # this is already ndarray not tensor
    assert len(ids) == len(ys)
    print('------------------------------------------------------------')
    print('Training data saved at:')
    print(out_dir)
    return ids


# ---------------
# Load Train Data
# ---------------

def load_train_data(
                    data_dir, 
                    id_regex=r'\d{6}_\d{6}_\d{1,3}',
                    x_regex=r'\d{6}_\d{6}_\d{1,3}_image.tif', 
                    y_regex=r'\d{6}_\d{6}_\d{1,3}_labels.tif'
                    ):
    '''
    Load train data from a directory according to a naming convention

    Parameters
    ----------
    data_dir: str
        Directory containing data
    id_regex: r string
        regex that will be used to extract IDs that will
        be used to label network output
    x_regex: r string
        regex that represents image file names, complete with
        extension (tiff please)
    y_regex: r string
        regex that represents training label files, complete
        with extension (tiff please)

    Returns
    -------
    xs: list of torch.Tensor
        List of images for training 
    ys: list of torch.Tensor
        List of affinities for training
    ids: list of str
        ID strings by which each image and label are named.
        Eventually used for correctly labeling network output
    
    '''
    # Get file names for images and training labels
    x_paths, y_paths = get_files(
                                 data_dir, 
                                 x_regex=x_regex, 
                                 y_regex=x_regex
                                 )
    # Get IDs 
    id_pattern = re.compile(id_regex)
    ids = []
    x_paths.sort()
    y_paths.sort()
    for i in range(len(x_paths)):
        xn = Path(x_paths[i]).stem # this could have been avoided
        yn = Path(y_paths[i]).stem # why would I bother now though?!
        # assumes there will be a match for each
        xid = id_pattern.search(xn)[0]
        yid = id_pattern.search(yn)[0]
        m = 'There is a mismatch in image and label IDs'
        assert xid == yid, m
        ids.append(xid)
    # Get images and training labels in tensor form 
    xs = []
    ys = []
    for i in range(len(x_paths)):
        xp = x_paths[i]
        yp = y_paths[i]
        x = imread(xp)
        x = normalise_data(x)
        y = imread(yp)
        xs.append(torch.from_numpy(x))
        ys.append(torch.from_numpy(y))
    # returns objects in the same manner as get_train_data()
    print('------------------------------------------------------------')
    print(f'Loaded {len(xs)} sets of training data')
    print_labels_info(ys[0].shape)
    return xs, ys, ids


# ------------------------
# General Helper Functions
# ------------------------

def normalise_data(image):
    '''
    Bring image values to between 0-1

    Parameters
    ----------
    image: np.array
        Image data. Dtype should be float.
    '''
    im = image / image.max()
    return im



if __name__ =="__main__":
    #
    pass
