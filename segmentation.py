from dask.array.core import Array
import dask.array as da
from helpers import get_dataset, get_regex_images
import napari
import numpy as np
import os
import pandas as pd
from scipy.ndimage import label
from skimage import io
from skimage.feature import peak_local_max, blob_dog, blob_log
from skimage.filters import threshold_otsu
from skimage import filters
from skimage.measure import regionprops
from skimage.metrics import variation_of_information
from watershed import watershed
from time import time
import cv2
from skimage.util import img_as_ubyte
from skimage.segmentation import watershed as skim_watershed
import scipy.ndimage as ndi
import umetrics


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
        validation=False, 
        dog_config=None,
        save=True,
        **kwargs
        #
    ):
    dog_comp = dog_config is not None
    images, _, output, GT, ids = get_dataset(directory, 
                                        GT=True, 
                                        validation=validation, 
                                        return_ID=True)
    images = da.squeeze(images)
    print(output.shape)
    segmentations = []
    masks = []
    scores = {'GT | Output' : [], 'Output | GT' : []}
    IoU_dict = generate_IoU_dict()
    if dog_comp:
        dog_segs = []
        dog_masks = []
        dog_scores = {'GT | Output' : [], 'Output | GT' : []}
        dog_IoU_dict = generate_IoU_dict()
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
        generate_IoU_data(gt, seg, IoU_dict)
        scores['GT | Output'].append(vi[0])
        scores['Output | GT'].append(vi[1])
        if save:
            save_name = ids[i] + '_segmentation.tif'
            save_path = os.path.join(directory, save_name)
            io.imsave(save_path, seg)
        seg = da.from_array(seg)
        segmentations.append(seg)
        masks.append(mask)
        if dog_comp:
            dog_seg, dog_mask = dog_segmentation(images[i], dog_config)
            dog_vi = variation_of_information(gt, dog_seg)
            dog_scores['GT | Output'].append(dog_vi[0])
            dog_scores['Output | GT'].append(dog_vi[1])
            generate_IoU_data(gt, dog_seg, dog_IoU_dict)
            dog_seg = da.from_array(dog_seg)
            if save:
                save_name = ids[i] + '_DoG-segmentation.tif'
                save_path = os.path.join(directory, save_name)
                io.imsave(save_path, dog_seg)
            dog_segs.append(dog_seg)
            dog_masks.append(dog_mask)
    segmentations = da.stack(segmentations)
    masks = da.stack(masks)
    if dog_comp:
        dog_segs = da.stack(dog_segs)
        dog_masks = da.stack(dog_masks)
    # Save the VI data
    scores = pd.DataFrame(scores)
    if dog_comp:
        dog_scores = pd.DataFrame(dog_scores)
    if validation:
        s = 'validation_VI'
        s0 = 'validation_metrics'
        s1 = 'validation_AP'
    else:
        s = '_VI'
        s0 = 'test_metrics'
        s1 = 'test_AP'
    s_VI_path = os.path.join(directory, suffix + s + '.csv')
    scores.to_csv(s_VI_path)
    iou_df = save_data(IoU_dict, suffix, directory, s0)
    ap = generate_ap_scores(iou_df, suffix, directory, s1)
    if dog_comp:
        d_path = os.path.join(directory, suffix + s + '_DOG-seg' + '.csv')
        dog_scores.to_csv(d_path)
        dog_iou_df = save_data(dog_IoU_dict, suffix, directory, s0)
        dog_ap = generate_ap_scores(dog_iou_df, suffix, directory, s1)
    gt_o = scores['GT | Output'].mean()
    o_gt = scores['Output | GT'].mean()
    print(f'Conditional entropy H(GT|Output): {gt_o}')
    print(f'Conditional entropy H(Output|GT): {o_gt}')
    if dog_comp:
        d_gt_o = dog_scores['GT | Output'].mean()
        d_o_gt = dog_scores['Output | GT'].mean()
        print(f'DoG segmentation - Conditional entropy H(GT|Output): {d_gt_o}')
        print(f'DoG segmentation - Conditional entropy H(Output|GT): {d_o_gt}')
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
        if dog_comp:
            v.add_labels(dog_masks, name='DoG Masks', 
                         blending='additive', visible=False, 
                         scale=v_scale)
            v.add_labels(dog_segs, name='DoG Segmentations', 
                         blending='additive', visible=True, 
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
        compactness=0., 
        absolute_thresh=None, 
        out=None,
        use_logging=None,
    ):
    '''
    Parameters
    ----------
    unet_output: np.ndarray or dask.array.core.Array
        Output from U-net inclusive of all channels. If there is an extra 
      r  dim of size 1, this will be squeezed out. Therefore shape may be
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
    if absolute_thresh is None:
        mask = _get_mask(masking_img)
    else:
        mask = masking_img > absolute_thresh
    mask = np.pad(mask, 1, constant_values=0) # edge voxels must be 0
    mask, centroids = _remove_unwanted_objects(mask, centroids, min_area=10, max_area=100000)
    if centroids.shape[0] != 0:
        # affinity-based watershed
        segmentation = watershed(affinties, centroids, mask, 
                             affinities=True, scale=scale, 
                             compactness=compactness)
        segmentation = segmentation[1:-1, 1:-1, 1:-1]
        segmentation = segmentation.astype(int)
        print(f'Obtained segmentation in {time() - t} seconds')
    else:
        segmentation = np.zeros(mask[1:-1, 1:-1, 1:-1].shape, dtype=int)
    seeds = centroids - 1
    if use_logging is not None:
        max_lab = segmentation.max()
        import logging
        logging.basicConfig(filename=use_logging, encoding='utf-8', level=logging.DEBUG)
        logging.debug(f'Internal segmentatation max label: {max_lab}')
    if out is not None:
        out[:] = segmentation[:]
        if use_logging is not None:
            max_lab = np.max(out)
            logging.debug(f'Out segmentation max label: {max_lab}')
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
        z = (z.compute() - 0.5) * 2
        y = (y.compute() - 0.5) * 2
        x = (x.compute() - 0.5) * 2
    # get the combined score (l2 norm)
    c = np.sqrt((z**2 + y**2 + x**2))
    # get the new output array
    new_shape = np.array(output.shape)
    new_shape[chan_axis] = new_shape[chan_axis] - 2
    new = np.zeros(new_shape, dtype=output.dtype)
    # get the slice to take other data from output
    s_ = [slice(None, None)] * output.ndim
    s_[chan_axis] = [i for i in range(output.shape[chan_axis]) if i not in zyx_chans]
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
    new[ns_] = np.expand_dims(c, 1)
    if isinstance(output, Array):
        new = da.array(new)
    return new


# ----------------
# DoG Segmentation
# ----------------

# Directly based on fl.py --- serialised version -- not my creation


def denoise(image):
    res_im = cv2.fastNlMeansDenoising(image, None, 6, 7, 20)
    return res_im


def dog_func(image, conf):
    s1 = conf['dog_sigma1']
    s2 = conf['dog_sigma2']
    image_dog = cv2.GaussianBlur(image.astype('float'),(0,0), s1) - cv2.GaussianBlur(image.astype('float'), (0,0), s2)
    return image_dog


def dog_segmentation(vol, conf):
    # denoise
    vol = img_as_ubyte(vol)
    print(vol.shape)
    vlist = [vol[i, ...] for i in range(vol.shape[0])]
    v_dn = [denoise(im) for im in vlist]
    v_dn = np.stack(v_dn, axis=0)
    v_dn = img_as_ubyte(v_dn)
    # dog volume
    vlist = [v_dn[i, ...] for i in range(v_dn.shape[0])]
    v_dog = [dog_func(im, conf) for im in vlist]
    v_dog = np.stack(v_dog, axis=0)
    # threshold
    v_dog_thr = v_dog > conf['threshold']
    v_dog_thr = img_as_ubyte(v_dog_thr)
    # seeds for watershed
    local_maxi = peak_local_max(v_dog, 
                                indices=False, 
                                min_distance=conf['peak_min_dist'], 
                                labels=v_dog_thr)
    markers, num_objects = ndi.label(local_maxi, structure=np.ones((3,3,3)))
    # watershed
    v_labels = skim_watershed(-v_dog, markers, mask=v_dog_thr,compactness=1)
    return v_labels, v_dog_thr


# --------------------
# Segmentation Metrics
# --------------------

def metrics_for_stack(directory, name, seg, gt):
    assert seg.shape[0] == gt.shape[0]
    IoU_dict = generate_IoU_dict()
    for i in range(seg.shape[0]):
        seg_i = seg[i].compute()
        gt_i = gt[i].compute()
        generate_IoU_data(gt_i, seg_i, IoU_dict)
    df = save_data(IoU_dict, name, directory, 'metrics')
    ap = generate_ap_scores(df, name, directory)
    return df, ap


def calc_ap(result):
        denominator = result.n_true_positives + result.n_false_negatives + result.n_false_positives
        return result.n_true_positives / denominator


def generate_IoU_dict(thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    IoU_dict = {}
    IoU_dict['n_predicted'] = []
    IoU_dict['n_true'] = []
    IoU_dict['n_diff'] = []
    for t in thresholds:
        n = f't{t}_true_positives'
        IoU_dict[n] = []
        n = f't{t}_false_positives'
        IoU_dict[n] = []
        n = f't{t}_false_negatives'
        IoU_dict[n] = []
        n = f't{t}_IoU'
        IoU_dict[n] = []
        n = f't{t}_Jaccard'
        IoU_dict[n] = []
        n = f't{t}_pixel_identity'
        IoU_dict[n] = []
        n = f't{t}_localization_error'
        IoU_dict[n] = []
        n = f't{t}_per_image_average_precision'
        IoU_dict[n] = []
    return IoU_dict


def generate_IoU_data(gt, seg, IoU_dict, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    for t in thresholds:
        result = umetrics.calculate(gt, seg, strict=True, iou_threshold=t)
        n = f't{t}_true_positives'
        IoU_dict[n].append(result.n_true_positives) 
        n = f't{t}_false_positives'
        IoU_dict[n].append(result.n_false_positives) 
        n = f't{t}_false_negatives'
        IoU_dict[n].append(result.n_false_negatives) 
        n = f't{t}_IoU'
        IoU_dict[n].append(result.results.IoU) 
        n = f't{t}_Jaccard'
        IoU_dict[n].append(result.results.Jaccard) 
        n = f't{t}_pixel_identity'
        IoU_dict[n].append(result.results.pixel_identity) 
        n = f't{t}_localization_error'
        IoU_dict[n].append(result.results.localization_error) 
        n = f't{t}_per_image_average_precision'
        IoU_dict[n].append(calc_ap(result))
        if t == thresholds[0]:
            IoU_dict['n_predicted'].append(result.n_pred_labels)
            IoU_dict['n_true'].append(result.n_true_labels)
            IoU_dict['n_diff'].append(result.n_true_labels - result.n_pred_labels)


def save_data(data_dict, name, directory, suffix):
    df = pd.DataFrame(data_dict)
    n = name + '_' + suffix +'.csv'
    p = os.path.join(directory, n)
    df.to_csv(p)
    return df


def generate_ap_scores(df, name, directory, suffix, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)):
    ap_scores = {'average_precision' : [], 
                 'threshold': []}
    for t in thresholds:
        ap_scores['threshold'].append(t)
        n = f't{t}_true_positives'
        true_positives = df[n].sum()
        n = f't{t}_false_positives'
        false_positives = df[n].sum()
        n = f't{t}_false_negatives'
        false_negatives = df[n].sum()
        ap = true_positives / (true_positives + false_negatives + false_positives)
        ap_scores['average_precision'].append(ap)
    print(ap_scores)
    ap_scores = save_data(ap_scores, name, directory, suffix)
    return ap_scores


if __name__ == '__main__':
    import os
    #data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    data_dir = '/home/abigail/data/platelet-segmentation-training'
    train_dir = os.path.join(data_dir, '210505_181203_seed_z-1_y-1_x-1_m_centg')
    channels = ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')
    images, labs, output = get_dataset(train_dir)
    #o88 = output[88]
    aff_chans = (0, 1, 2)
    cent_chan = 4
    mask_chan = 3
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
        'seed_z-1_y-1_x-1_m_centg',
        aff_chans, 
        cent_chan, 
        mask_chan, 
        scale = (4, 1, 1),
        w_scale=None, 
        compactness=0.,
        display=True, 
        validation=True)

