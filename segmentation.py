from dask.array.core import Array
import dask.array as da
from helpers import get_dataset, get_regex_images
import napari
import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage.feature import peak_local_max, blob_dog, blob_log
from skimage.filters import threshold_otsu
from skimage import filters
from skimage.measure import regionprops
from skimage.metrics import variation_of_information
from watershed import watershed
from time import time


# --------------------------------
# Segment and Score from Directory
# --------------------------------

def segment_from_directory(
        directory, 
        suffix,
        affinities_channels, 
        centroids_channel, 
        thresholding_channel, 
        scale = (4, 1, 1),
        w_scale=None, 
        compactness=0.,
        display=True, 
        #
    ):
    images, _, output, GT = get_dataset(directory, GT=True)
    images = da.squeeze(images)
    print(output.shape)
    segmentations = []
    masks = []
    scores = {'GT | Output' : [], 'Output | GT' : []}
    for i in range(output.shape[0]):
        gt = GT[i].compute()
        seg, _, mask = segment_output_image(
                output[i], 
                affinities_channels, 
                centroids_channel, 
                thresholding_channel, 
                scale=w_scale, 
                compactness=0.)
        vi = variation_of_information(gt, seg)
        scores['GT | Output'].append(vi[0])
        scores['Output | GT'].append(vi[1])
        seg = da.from_array(seg)
        segmentations.append(seg)
        masks.append(mask)
    segmentations = da.stack(segmentations)
    masks = da.stack(masks)
    # Save the VI data
    scores = pd.DataFrame(scores)
    s_path = os.path.join(directory, suffix + '_VI.csv')
    scores.to_csv(s_path)
    gt_o = scores['GT | Output'].mean()
    o_gt = scores['Output | GT'].mean()
    print(f'Conditional entropy H(GT|Output): {gt_o}')
    print(f'Conditional entropy H(Output|GT): {o_gt}')
    if display:
        # Now Display
        z_affs = output[:, affinities_channels[0], ...]
        y_affs = output[:, affinities_channels[1], ...]
        x_affs = output[:, affinities_channels[2], ...]
        c = output[:, thresholding_channel, ...]
        cl = output[:, centroids_channel, ...]
        v_scale = [1] * len(images.shape)
        v_scale[-3:] = scale
        print(images.shape, v_scale, z_affs.shape, masks.shape)
        v = napari.Viewer()
        v.add_image(images, name='Input images', blending='additive', visible=True, scale=v_scale)
        v.add_image(c, name='Thresholding channel', blending='additive', visible=False, scale=v_scale)
        v.add_image(cl, name='Centroids channel', blending='additive', visible=False, scale=v_scale)
        v.add_image(z_affs, name='z affinities', blending='additive', visible=False, scale=v_scale, 
                    colormap='bop purple')
        v.add_image(y_affs, name='y affinities', blending='additive', visible=False, scale=v_scale, 
                    colormap='bop orange')
        v.add_image(x_affs, name='x affinities', blending='additive', visible=False, scale=v_scale, 
                    colormap='bop blue')
        v.add_labels(masks, name='Masks', blending='additive', visible=False, scale=v_scale)
        v.add_labels(GT, name='Ground truth', blending='additive', visible=False, scale=v_scale)
        v.add_labels(segmentations, name='Segmentations', blending='additive', visible=True, 
                     scale=v_scale)
        napari.run()


# --------------------
# Segment U-net Output
# --------------------

def segment_output_image(
        unet_output, 
        affinities_channels, 
        centroids_channel, 
        thresholding_channel, 
        scale=None, 
        compactness=0.
    ):
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
    affinties = np.pad(affinties, 
                       ((0, 0), (1, 1), (1, 1), (1, 1)), 
                       constant_values=0)
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
    mask, centroids = _remove_unwanted_objects(mask, centroids, min_area=10, max_area=10000)
    # affinity-based watershed
    segmentation = watershed(affinties, centroids, mask, 
                             affinities=True, scale=scale, 
                             compactness=compactness)
    segmentation = segmentation[1:-1, 1:-1, 1:-1]
    segmentation = segmentation.astype(int)
    seeds = centroids - 1
    print(f'Obtained segmentation in {time() - t} seconds')
    return segmentation, seeds, mask


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


def _remove_unwanted_objects(mask, centroids, min_area=0, max_area=10000):
    labs, _ = label(mask)
    props = regionprops(labs)
    new = np.zeros_like(mask)
    a_s = []
    for prop in props:
        a = prop['area']
        a_s.append(a)
        if a >= min_area and a < max_area:
            l = prop['label']
            new = np.where(labs == l, 1, new)
    new_cent = []
    for c in centroids:
        try:
            if new[c[-3], c[-2], c[-1]] == 1:
                new_cent.append(c)
        except IndexError:
            pass
    #print('min: ', np.min(a_s), ' max: ', np.max(a_s))
    return new, np.array(new_cent)


# ----------------------
# Convert centre offsets
# ----------------------

def convert_axial_offsets(output, chan_axis=1, zyx_chans=(3, 4, 5)):
    # get the slices for each axis
    zs = [slice(None, None)] * output.ndim
    zs[chan_axis] = zyx_chans[0]
    ys = [slice(None, None)] * output.ndim
    ys[chan_axis] = zyx_chans[1]
    xs = [slice(None, None)] * output.ndim
    xs[chan_axis] = zyx_chans[2]
    zs, ys, xs = tuple(zs), tuple(ys), tuple(xs)
    # get the data
    z = output[zs]
    y = output[ys]
    x = output[xs]
    if isinstance(output, Array):
        z = z.compute()
        y = y.compute()
        x = x.compute()
    # get the combined score (l2 norm)
    c = np.sqrt((z**2 + y**2 + x**2))
    # get the new output array
    new_shape = output.shape
    new_shape[chan_axis] = new_shape[chan_axis] - 2
    new = np.zeros(new_shape, dtype=output.dtype)
    # get the slice to take other data from output
    s_ = [slice(None, None)] * output.ndim
    s_[chan_axis] = [i for i in range(output[chan_axis]) if i not in zyx_chans]
    s_ = tuple(s_)
    # get the slice to put other data into new
    ns_ = [slice(None, None)] * len(new_shape)
    ns_[chan_axis] = slice(0, len(s_[chan_axis]))
    ns_ = tuple(ns_)
    # add other channels to new
    new[ns_] = output[s_]
    # get the slice to add the centre scores to new
    ns_ = [slice(None, None)] * len(new_shape)
    ns_[chan_axis] = slice(-1, None)
    ns_ = tuple(ns_)
    # add the centre scores
    new[ns_] = c
    if isinstance(output, Array):
        new = da.array(new)
    return new


if __name__ == '__main__':
    import os
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    train_dir = os.path.join(data_dir, '210416_161026_EWBCE_2F_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl')
    channels = ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log')
    images, labs, output = get_dataset(train_dir)
    #o88 = output[88]
    aff_chans = (0, 2, 5)
    cent_chan = 9
    mask_chan = 8
    #seg88, s88 = segment_output_image(o88, aff_chans, cent_chan, mask_chan) #, scale=(4, 1, 1))
    #seg88s, s88s = segment_output_image(o88, aff_chans, cent_chan, mask_chan, scale=(4, 1, 1))
    #seg88c, s88c = segment_output_image(o88, aff_chans, cent_chan, mask_chan, compactness=0.5) #, scale=(4, 1, 1))
    #i88 = images[88]
    #l88 = labs[88]
    #v = napari.view_image(i88, name='image', scale=(4, 1, 1), blending='additive')
    #v.add_labels(l88, name='labels', scale=(4, 1, 1), visible=False)
    #v.add_image(o88[aff_chans[0]], name='z affinities', 
              #  colormap='bop purple', scale=(4, 1, 1), 
               # visible=False, blending='additive')
    #v.add_image(o88[aff_chans[1]], name='y affinities', 
              #  colormap='bop orange', scale=(4, 1, 1), 
              #  visible=False, blending='additive')
    #v.add_image(o88[aff_chans[2]], name='x affinities', 
              #  colormap='bop blue', scale=(4, 1, 1), 
              #  visible=False, blending='additive')
    #v.add_labels(seg88, name='affinity watershed', 
               #  scale=(4, 1, 1), blending='additive')
    #v.add_labels(seg88s, name='anisotropic affinity watershed', 
               #  scale=(4, 1, 1), blending='additive')
    #v.add_labels(seg88c, name='compact affinity watershed', 
        #         scale=(4, 1, 1), blending='additive')
    #v.add_points(s88, name='seeds', scale=(4, 1, 1), size=1)
    #napari.run()

    segment_from_directory(
        train_dir, 
        'EWBCE_2F_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl',
        aff_chans, 
        cent_chan, 
        mask_chan, 
        scale = (4, 1, 1),
        w_scale=None, 
        compactness=0.,
        display=True)

