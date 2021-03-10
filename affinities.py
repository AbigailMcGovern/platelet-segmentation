from datetime import datetime
import numpy as np
import os
from pathlib import Path
import re
from tifffile import TiffWriter, imread
import torch 
import zarr


# -------------------
# Generate Train Data
# -------------------
def get_train_data(
                   image_paths, 
                   labels_paths,
                   out_dir, 
                   shape=(10, 256, 256), 
                   n_each=50, 
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
        print(f'Generating training data from image: {image_paths[i]}, labels: {labels_paths[i]}')
        im = zarr.open_array(image_paths[i])
        l = zarr.open_array(labels_paths[i])
        if i == 0:
            xs, ys, ids = get_random_chunks(im, l, out_dir, shape=shape, n=n_each)
        else:
            xs_n, ys_n, ids_n = get_random_chunks(im, l, out_dir, shape=shape, n=n_each)
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
                      min_affinity=100
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
    a = np.array(labels)
    assert len(im.shape) == len(shape)
    a = get_affinities(a)
    xs = []
    ys = []
    i = 0
    while i < n:
        dim_randints = []
        for j, dim in enumerate(shape):
            max_ = im.shape[j] - dim - 1
            ri = np.random.randint(0, max_)
            dim_randints.append(ri)
        # Get the network output: affinities 
        s_ = [slice(None, None),] # affinities have three channels... we want all of them :)
        for j in range(len(shape)):
            s_.append(slice(dim_randints[j], dim_randints[j] + shape[j]))
        y = a[s_]
        if y.sum() > min_affinity: # if there are a sufficient number of boarder voxels
            # add the affinities and image chunk to the training data 
            y = torch.from_numpy(y)
            ys.append(y)
            # Get the network input: image
            s_ = [slice(dim_randints[j], dim_randints[j] + shape[j]) for j in range(len(shape))]
            s_ = tuple(s_)
            x = im[s_]
            x = torch.from_numpy(x)
            xs.append(x)
            # another successful addition, job well done you crazy mofo
            i += 1
    print(f'Obtained {n} {shape} chunks of training data')
    ids = save_random_chunks(xs, ys, out_dir)
    return xs, ys, ids


def get_affinities(image):
    """
    Get short-range voxel affinities for a segmentation. Affinities are 
    belonging to {0, 1} where 1 represents a segment boarder voxel in a
    particular direction. Affinities are produced for each dimension of 
    the labels and each dim has its own channel (e.g, (3, z, y, x)).

    Note
    ----
    Others may represent affinities with {-1, 0}. My network wasn't designed 
    for this :)
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


def save_random_chunks(xs, ys, out_dir):
    '''
    Save the random chunks as they are sampled
    '''
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
    assert len(ids) == len(ys)
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
    files = os.listdir(data_dir)
    x_paths = []
    y_paths = [] 
    x_pattern = re.compile(x_regex)
    y_pattern = re.compile(y_regex)
    for f in files:
        x_res = x_pattern.search(f)
        if x_res is not None:
            x_paths.append(os.path.join(data_dir, x_res[0]))
        y_res = y_pattern.search(f)
        if y_res is not None:
            y_paths.append(os.path.join(data_dir, y_res[0]))
    m = 'There is a mismatch in the number of images and training labels'
    assert len(x_paths) == len(y_paths), m
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
        y = imread(yp)
        xs.append(torch.from_numpy(x))
        ys.append(torch.from_numpy(y))
    # returns objects in the same manner as get_train_data()
    return xs, ys, ids
    


if __name__ =="__main__":
    #
    pass