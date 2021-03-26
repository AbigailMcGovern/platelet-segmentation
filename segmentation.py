from dask.array.core import Array
from helpers import get_dataset
import numpy as np
from skimage.feature import peak_local_max, blob_dog, blob_log
from skimage.filters import threshold_otsu
from skimage import filters
from watershed import watershed
from time import time

# --------------------
# Segment U-net Output
# --------------------

def segment_output_image(unet_output, affinities_channels, centroids_channel, thresholding_channel):
    '''
    Parameters
    ----------
    unet_output: np.ndarray or dask.array.core.Array
        Output from U-net inclusive of all channels. If there is an extra 
        dim of size 1, this will be squeezed out. Therefore shape may be
        (1, c, z, y, x) or (c, z, y, x).
    affinities_channels: tuple of int
        Ints, in order (z, y, x) describe the channel indicies to which 
        the z, y, and x short-range affinities belong.
    centroids_channel: int
        Describes the channel index for the channel that is used to find 
        centroids.
    thresholding_channel: in
        Describes the channel index for the channel that is used to find
        the mask for watershed.
    '''
    t = time()
    if isinstance(unet_output, Array):
        unet_output = unet_output.compute()
    unet_output = np.squeeze(unet_output)
    # Get the affinities image (a, z, y, x)
    affinties = []
    for c in affinities_channels:
        affinties.append(unet_output[c, ...]/unet_output[c, ...].max())
    affinties = np.stack(affinties)
    affinties = np.pad(affinties, 1, constant_values=0)
    # Get the image for finding centroids
    centroids_img = unet_output[centroids_channel]
    centroids_img = np.pad(centroids_img, 1, constant_values=0)
    # find the centroids
    centroids = _get_centroids(centroids_img)
    # Get the image for finding the mask 
    masking_img = unet_output[thresholding_channel]
    # find the mask for use with watershed
    mask = _get_mask(masking_img)
    mask = np.pad(mask, 1, constant_values=0) # edge voxels must be 0
    # affinity-based watershed
    segmentation = watershed(affinties, centroids, mask, affinities=True)
    segmentation = segmentation[1:-1, 1:-1, 1:-1]
    seeds = centroids - 1
    print(f'Obtained segmentation in {time() - t} seconds')
    return segmentation, seeds


def _get_mask(img, sigma=2):
    thresh = threshold_otsu(filters.gaussian(img, sigma=sigma))
    mask = img > thresh
    return mask


def _get_centroids(cent, gaussian=True):
    if gaussian:
        # won't blur along z, can't afford to do that
        for i in range(cent.shape[0]):
            cent[i, ...] = filters.gaussian(cent[i, ...])
    centroids = peak_local_max(cent, threshold_abs=.04) #* c_scale
    #centroids = blob_log(cent, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    return centroids


if __name__ == '__main__':
    import os
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    train_dir = os.path.join(data_dir, '210324_training_0')
    channels = ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log')
    images, labs, output = get_dataset(train_dir)
    o88 = output[88]
    aff_chans = (0, 2, 5)
    cent_chan = 9
    mask_chan = 8
    seg88, s88 = segment_output_image(o88, aff_chans, cent_chan, mask_chan)
    i88 = images[88]
    l88 = labs[88]
    import napari 
    v = napari.view_image(i88, name='image', scale=(4, 1, 1), blending='additive')
    #v.add_labels(l88, name='labels', scale=(4, 1, 1), visible=False)
    v.add_image(o88[aff_chans[0]], name='z affinities', 
                colormap='bop purple', scale=(4, 1, 1), 
                visible=False, blending='additive')
    v.add_image(o88[aff_chans[1]], name='y affinities', 
                colormap='bop orange', scale=(4, 1, 1), 
                visible=False, blending='additive')
    v.add_image(o88[aff_chans[2]], name='x affinities', 
                colormap='bop blue', scale=(4, 1, 1), 
                visible=False, blending='additive')
    v.add_labels(seg88, name='affinity watershed', 
                 scale=(4, 1, 1), blending='additive')
    v.add_points(s88, name='seeds', scale=(4, 1, 1), size=1)
    napari.run()

