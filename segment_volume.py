from helpers import get_paths
import napari
import numpy as np
import os
from skimage.io import imread
import torch
from unet import UNet


# -----------------------
# Whole Volume Prediction
# -----------------------


def simple_volume_prediciton(image_path, data_dir, chunk_shape=(10, 256, 256), overlap=0):
    image = imread(image_path)
    blocks, slices = get_blocks(image, chunk_shape, overlap)
    unet = load_unet(data_dir)
    input_tensor = prepare_for_torch(blocks)
    output = unet(input_tensor)
    whole_volume = rebuild_volume(output, slices)
    return whole_volume


def get_blocks(image, chunk_shape, overlap=0):
    shape = image.shape
    slices = get_box_coords(shape, chunk_shape, overlap)
    blocks = []
    for s_ in slices:
        blocks.append(image[s_])
    blocks = np.stack(blocks)
    return blocks, slices
    #

def get_box_coords(image_shape, chunk_shape, overlap):
    coords = {}
    # for each dim, get a list of coords
    for i in range(len(image_shape)):
        current = 0
        coords[i] = []
        while current <= image_shape[i]:
            start = current
            stop = current + chunk_shape[i] + overlap
            if stop <= image_shape[i]:
                coords[i].append([start, stop])
            else:
                print(f'Final chunk ends at {current} in axis {i}, which has a length of {image_shape[i]}')
            current = stop 
    print(coords)
    # now we need to extract the actual slices
    out = []
    coords = [coords[key] for key in sorted(coords.keys())]
    out = _recursive_groups(coords, [], [], [], 0)
    print(out)
    slices = []
    for c in out:
        s_ = [slice(*start_stop) for start_stop in c]
        s_ = tuple(s_)
        slices.append(s_)
    return slices
    # 


def _recursive_groups(full_list, used_idx, out_list, final_list, counter):
    for i in range(len(full_list)):
        if i not in used_idx:
            used_idx.append(i)
            for j, item in enumerate(full_list[i]):
                counter += 1
                print('idx: ', i)
                print('item, j: ', item, j)
                print('used idx: ', used_idx)
                print('counter: ', counter)
                new = out_list.copy()
                new.append(item)
                if len(new) == len(full_list) == len(used_idx):
                    final_list.append(new)
                    print('adding: ', new)
                    if j == (len(full_list[i]) - 1):
                        used_idx.pop()
                        print('remove used idx: ', used_idx)
                else:
                    _recursive_groups(full_list, used_idx, new, final_list, counter)
    return final_list


def load_unet(data_dir):
    # get unet input size
    input_files = get_paths(data_dir, r'\d{6}_\d{6}_\d{1,3}_image.tif')
    i0 = imread(input_files[0])
    if i0.ndim == 3:
        in_channels = 1
    else:
        in_channels = i0.shape[-4]
    # get unet output size
    output_files = get_paths(data_dir)
    o0 = imread(output_files[0])
    out_channels = o0.shape[-4]
    # get the unet state dict path
    state_dict_paths = get_paths(data_dir, 
                                 r'\d{6}_\d{6}_unet.*.pt')
    e_state_dict_paths = get_paths(data_dir, 
                                   r'\d{6}_\d{6}_unet.*_epoch-\d.pt')
    state_dict_path = sorted([p for p in state_dict_paths \
                                if p not in e_state_dict_paths])[-1]
    # initialise and load unet
    unet = UNet(in_channels=in_channels, out_channels=out_channels)
    state_dict = torch.load(state_dict_path)
    unet.load_state_dict(state_dict)
    return unet


def prepare_for_torch(blocks):
    blocks = blocks / blocks.max()
    blocks = torch.from_numpy(blocks)
    blocks = torch.unsqueeze(blocks, 1) # I think the first dim is for batches??
    return blocks         


def rebuild_volume(output, slices):
    shape = []
    for i in range(3):
        stop_max = 0
        for s_ in slices:
            stop = s_[i].stop
            if stop > stop_max:
                stop_max = stop
        shape.append(stop_max)
    output = output.numpy()
    new = np.zeros(shape, output.dtype)
    for i, s_ in enumerate(slices):
        new[s_] = output[i]
    return new


# -------------------
# Volume Segmentation
# -------------------




if __name__ == '__main__':
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training/210415_135338_EWBCE_2_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl'
    unet = load_unet(data_dir)
    image_path = '/Users/amcg0011/Data/pia-tracking/cang_training/191113_IVMTR26_I3_E3_t58_cang_training_image.tif'
    vol = imread(image_path)
    #vol = vol / vol.max()
    #tvol = torch.from_numpy(vol)
    #tvol = torch.unsqueeze(tvol, 0)
    #tvol = torch.unsqueeze(tvol, 0)
    #tout = unet(tvol.float())
    #out = tout.numpy()
    #out = np.squeeze(out)
    #v = napari.Viewer()
    #v.add_image(vol, scale=(4, 1, 1))
    #v.add_image(out, scale=(1, 4, 1, 1))
    #napari.run()
    a = get_box_coords((33, 512, 512), (10, 256, 256), 0)
    print(a)
    

