import dask.array as da
from dask import delayed
from helpers import get_dataset, LINE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import napari
import numpy as np
from pathlib import Path
import re
from skimage import io
from skimage.feature import peak_local_max
from skimage import filters

# -------------
# QT Management
# -------------

def qt_session(func):
    '''
    Qt session decorator. When the visualise 
    '''
    def wrapper_gui_qt(*args, **kwargs):
        visualise = kwargs.get('visualise')
        if visualise is not None:
            use = visualise
            doing = 'initiating' if use else 'will not initiate'
            print(LINE)
            print(f'Visualise is {use}, {doing} qt session')
        else:
            use = True
        if use:
            with napari.gui_qt():
                func(*args, **kwargs)
        else:
            func(*args, **kwargs)
    return wrapper_gui_qt


# demo functions with decorator
@qt_session
def init_v(visualise):
    v = napari.Viewer()
    return v


def run_example(): # this also works
    example(visualise=True)

@qt_session
def example(visualise):
    print('an example function') # I don't know why this works and the train_unet one doesn't!!!
    if visualise:
        v = napari.Viewer()
        add_some_image(v)
    else:
        pass

import numpy as np

def add_some_image(v):
    img = np.random.random((100, 100))
    v.add_image(img)


# -----------------
# Initialise Viewer 
# -----------------
# with training images and labels (labels are set to not visible)


def view_output(train_dir, channels, scale, out_dir=None, verbose=True):
    '''
    Parameters
    ----------
    train_dir: str
        Directory containing training data
    channels: tuple of str
        Labels for channels
    scale: tuple of number
        Scale for zxy dims (z, y, x)
    out_dir: str
        Directory containing output. Required only if this is
        different from that containing train data.
    verbose: bool
        Print outs :)
    '''
    assert len(scale) == 3
    images, labs, output = get_dataset(train_dir, out_dir)
    # prepare labels and output data for display in different channels
    # z, y, and x afiniies will be displayed according to BOP colour scheme
    labs_channels_dict = prep_channels(labs, channels, scale=scale)
    out_channels_dict = prep_channels(output, channels, scale=scale, gaussian=True)
    # initialise viewer
    v = napari.Viewer()
    image_scale = [1] * images.ndim
    image_scale[-3:] = scale
    v.add_image(images, name='images', scale=image_scale)
    if verbose:
        print(f'Added input image dataset with shape {images.shape}')
    for chan in out_channels_dict.keys():
        # prepare output data for addition to viewer
        output = out_channels_dict[chan]['data']
        output_scale = [1] * (len(output.shape)) 
        output_scale[-3:] = scale # last three channels should be spatial
        output_offsets = [0.] * len(output.shape)
        output_offsets[-3:] = out_channels_dict[chan]['offsets']
        output_name = out_channels_dict[chan]['name']
        oc = out_channels_dict[chan]['colour']
        # prepare labels data for addition to viewer
        lab = labs_channels_dict[chan]['data']
        lab_scale = [1] * len(lab.shape)
        lab_scale[-3:] = scale
        name = 'Labels: ' + labs_channels_dict[chan]['name']
        lab_offsets = [0.] * len(lab.shape)
        lab_offsets[-3:] = labs_channels_dict[chan]['offsets']
        lc = labs_channels_dict[chan]['colour']
        # add to viewer according to content
        if chan == 'centreness':
            v.add_image(output, name=output_name, scale=output_scale, visible=False, 
                        colormap=oc, translate=output_offsets, blending='additive')
            v.add_image(lab, name=name, scale=lab_scale, visible=False, 
                        colormap=lc, translate=lab_offsets, blending='additive')
        elif chan == 'centroids':
            c_scale = [1] * output.shape[1]
            c_scale[-3:] = scale
            v.add_points(output, name=output_name, scale=c_scale, size=1, visible=False)
            v.add_points(lab, name=name, scale=c_scale, size=1, visible=False)
        else:
            v.add_image(output, name=output_name, scale=output_scale, 
                        colormap=oc, translate=output_offsets, visible=True, 
                        blending='additive')
            v.add_labels(lab, name=name, scale=lab_scale, visible=False, 
                         translate=lab_offsets, blending='additive')
        if verbose:
            print(f'Added empty output data with shape {output.shape} to {output_name}')
            print(f'Added labels data with shape {lab.shape} to {name}')
    napari.run()
    return v



def get_empty_output(ys):
    output_shape = ys[0].shape[-4:]
    l = len(ys)
    shape = [l, ]
    for d in output_shape:
        shape.append(d)
    empty = np.zeros(shape, dtype=float)
    return empty


def prep_channels(dask_stack, channels, scale=(4, 1, 1), gaussian=False, **kwargs):
    channels_dict = _empty_channels_dict(scale=scale)
    aff_pattern = re.compile(r'-\d+')
    if 'centreness-log' in channels:
        use_log = True
    for i, c in enumerate(channels):
        aff = aff_pattern.search(c)
        if c.startswith('z'):
            channels_dict['z_affinities']['data'].append(dask_stack[:, i, ...])
            n = channels_dict['z_affinities']['name']
            channels_dict['z_affinities']['name'] = n + aff[0] + ', '
            channels_dict['z_affinities']['idx'].append(i)
        elif c.startswith('y'):
            channels_dict['y_affinities']['data'].append(dask_stack[:, i, ...])
            n = channels_dict['y_affinities']['name']
            channels_dict['y_affinities']['name'] = n + aff[0] + ', '
            channels_dict['y_affinities']['idx'].append(i)
        elif c.startswith('x'):
            channels_dict['x_affinities']['data'].append(dask_stack[:, i, ...])
            n = channels_dict['x_affinities']['name']
            channels_dict['x_affinities']['name'] = n + aff[0] + ', '
            channels_dict['x_affinities']['idx'].append(i)
        elif c == 'centreness':
            cent = dask_stack[:, i, ...]
            channels_dict['centreness']['data'].append(cent)
            channels_dict['centreness']['idx'].append(i)
            if not use_log:
                centroids = _centroids(cent, gaussian)
                channels_dict['centroids']['data'] = centroids # will only find this channel once
        elif c == 'centreness-log':
            cent = dask_stack[:, i, ...]
            channels_dict['centreness-log']['data'].append(cent)
            channels_dict['centreness-log']['idx'].append(i)
            centroids = _centroids(cent, gaussian)
            channels_dict['centroids']['data'] = centroids
    for chan in channels_dict.keys():
        if chan != 'centroids':
            channels_dict[chan]['data'] = da.stack(channels_dict[chan]['data'], axis=1)
    return channels_dict


def _empty_channels_dict(scale=(4, 1, 1)):
    channels_dict = {
        'z_affinities': {
            'data' : [],
            'name' : 'z affinities: ',
            'idx' : [],
            'offsets': [scale[0] * -0.5, 0, 0], 
            'colour' : 'bop purple'
        },
        'y_affinities': {
            'data' : [],
            'name' : 'y affinities: ',
            'idx' : [],
            'offsets': [0, scale[0] * -0.5, 0], 
            'colour' : 'bop orange'
        },
        'x_affinities': {
            'data' : [],
            'name' : 'x affinities: ',
            'idx' : [],
            'offsets': [0, 0, scale[0] * -0.5], 
            'colour' : 'bop blue'
        }, 
        'centreness': {
            'data' : [],
            'name' : 'centreness',
            'idx' : [],
            'offsets': [0, 0, 0], 
            'colour' : 'viridis'
        },
        'centreness-log': {
            'data' : [],
            'name' : 'log centreness',
            'idx' : [],
            'offsets': [0, 0, 0], 
            'colour' : 'viridis'
        },
        'centroids': {
            'data' : [],
            'name' : 'centroids',
            'idx' : None,
            'offsets': [0, 0, 0], 
            'colour' : None
        }
    }
    return channels_dict

def _centroids(cent, gaussian):
    cent = cent.compute()
    if gaussian:
        for i in range(cent.shape[0]):
            cent[i, ...] = filters.gaussian(cent[i, ...])
    centroids = peak_local_max(cent) #* c_scale
    new_centroids = np.zeros((centroids.shape[0], centroids.shape[1] + 1))
    new_centroids[:, 0] = centroids[:, 0]
    new_centroids[:, -3:] = centroids[:, -3:]
    print('centroids: ', new_centroids.shape)
    return new_centroids


# unused
def fix_id_order(ids, incorrect_ids, stack):
    assert len(ids) == len(incorrect_ids) == stack.shape[0]
    new = stack.copy()
    for i in range(len(ids)):
        ID = ids[i]
        for j in range(len(incorrect_ids)):
            if incorrect_ids[i] == ID:
                idx = j
        new[i, ...] = stack[idx, ...]
    return new


# ----------------
# Napariboard Prep
# ----------------


# ----------
# Loss Plots
# ----------


def initialise_plot():
    with plt.style.context('dark_background'):
        loss_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        loss_axes = loss_canvas.figure.subplots()
        lines = loss_axes.plot([], [])  # make empty plot
        loss_axes.set_xlim(0, NUM_ITER)
        loss_axes.set_xlabel('batch number')
        loss_axes.set_ylabel('loss')
        loss_canvas.figure.tight_layout()
        loss_line = lines[0]
        plot_dict = {
            'canvas' : loss_canvas,
            'axes' : loss_axes,
            'lines' : lines,
            'loss_line' : loss_line
        }
    return plot_dict
    

def update_plot(loss, plot_dict):
    x, y = plot_dict['loss_line'].get_data()
    new_y = np.append(y, loss)
    new_x = np.arange(len(new_y))
    plot_dict['loss_line'].set_data(new_x, new_y)
    plot_dict['axes'].set_ylim(
        np.min(new_y) * (-0.05), np.max(new_y) * (1.05)
    )
    plot_dict['canvas'].draw_idle()


if __name__ == '__main__':
    import os
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    train_dir = os.path.join(data_dir, '210324_training_0')
    channels = ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log')
    scale = (4, 1, 1)
    v = view_output(train_dir, channels, scale)
