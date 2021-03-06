from custom_loss import WeightedBCELoss, DiceLoss, EpochwiseWeightedBCELoss, \
    channel_losses_to_dict
from datetime import datetime
import dask.array as da
from helpers import LINE, write_log
import napari
import numpy as np
import os
import pandas as pd
from plots import save_loss_plot, save_channel_loss_plot
from tifffile import TiffWriter 
import torch 
import torch.nn as nn
import torch.optim as optim
from train_io import get_train_data, load_train_data
from tqdm import tqdm
from unet import UNet, ForkedUNet


# DICE seems to be more common but BCE Loss is also used for 
#   image segmentation (Jadon et al., 2020, arXiv) and seems 
#   to give better results here


def train_unet(
               # training data
               xs, 
               ys, 
               ids, 
               # output information
               out_dir, 
               suffix, 
               channels=None,
               # validation data
               v_xs=None,
               v_ys=None,
               v_ids=None,
               validate=False,
               # training variables
               log=True,
               epochs=3, 
               lr=0.01, 
               loss_function='WeightedBCE', 
               chan_weights=(1., 2., 2.), # for weighted BCE
               weights=None,
               update_every=20, 
               fork_channels=None,
               **kwargs
               ):
    '''
    Train a basic U-Net on affinities data.

    Parameters
    ----------
    xs: list of torch.tensor
        Input images for which the network will be trained to predict
        inputted labels.
    ys: list of torch.tensor
        Input labels that represent target output that the network will
        be trained to predict.
    ids: list of str
        ID strings corresponding to xs and ys (as they are named on disk). 
        Used for saving output.
    out_dir: str
        Directory to which to save network output
    suffix: str
        Suffix used in naming pytorch state dictionary file
    channels: tuple of str or None
        Names of output channels to be used for labeling channelwise
        loss columns in output loss csv. If none, names are generated.
    v_xs: list of torch.Tensor or None
        Validation images
    v_y: list of torch.Tensor or None
        Validation labels
    v_ids: list of str or None
        Validation IDs
    validate: bool
        Will a validation be done at the end of every epoch?
    log: bool
        Will a log.txt file containing all console print outs be saved?
    epochs: int
        How many times should we go through the training data?
    lr: float
        Learning rate for Adam optimiser
    loss_function: str
        Which loss function will be used for training & validation?
        Current options include:
            'BCELoss': Binary cross entropy loss
            'WeightedBCE': Binary cross entropy loss whereby channels are weighted
                according to chan_weights parameter. Quick way to force network to
                favour learning information about a given channel/s.
            'DiceLoss': 1 - DICE coefficient of the output-target pair
    chan_weights: tuple of float
        WEIGHTEDBCE: Weights for BCE loss for each output channel. 
    weights: None or nn.Model().state_dict()
        Prior weights with which to initalise the network.
    update_every: int
        Determines how many batches are processed before printing loss

    Returns
    -------
    unet: UNet (unet.py)

    Notes
    -----
    When data is loaded from a directory, it will be recognised according
    to the following naming convention:

        IDs: YYMMDD_HHMMSS_{digit/s} 
        Images: YYMMDD_HHMMSS_{digit/s}_image.tif
        Affinities: YYMMDD_HHMMSS_{digit/s}_labels.tif
    
    E.g., 210309_152717_7_image.tif, 210309_152717_7_labels.tif

    For each ID, a labels and an image file must be found or else an
    assertion error will be raised.
    '''
    # Device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    # initialise U-net
    if fork_channels is None:
        unet = UNet(out_channels=len(channels)).to(device, dtype=torch.float32)
    else:
        unet = UNet(out_channels=fork_channels).to(device, dtype=torch.float32)
    # load weights if applicable 
    weights_are = _load_weights(weights, unet)
    # define the optimiser
    optimiser = optim.Adam(unet.parameters(), lr=lr)
    # define the loss function
    loss = _get_loss_function(loss_function, chan_weights, device)
    # get the dictionary that will be converted to a csv of losses
    #   contains columns for each channel, as we record channel-wise
    #   BCE loss in addition to the loss used for backprop
    channels = _index_channels_if_none(channels, xs) 
    loss_dict = _get_loss_dict(channels)
    if validate:
        v_loss = _get_loss_function(loss_function, chan_weights, device)
        validation_dict = {'epoch' : [], 
                           'validation_loss' : [], 
                           'data_id' : [], 
                           'batch_id': []
                           }
        no_iter = (epochs * len(xs)) + ((epochs + 1) * len(v_xs))
    else:
        no_iter = epochs * len(xs)
    # print the training into and log if applicable 
    bce_weights = _bce_weights(loss) # gets weights if using WeightedBCE
    _print_train_info(loss_function, bce_weights, epochs, lr, 
                     weights_are, device_name, out_dir, log)
    # loop over training data 
    y_hats, v_y_hats = _train_loop(no_iter, epochs, xs, ys, ids, device, unet, 
                                   out_dir, optimiser, loss, loss_dict,  
                                   validate, v_xs, v_ys, v_ids, validation_dict, 
                                   v_loss, update_every, log, suffix, channels)
    _save_final_results(unet, out_dir, suffix, y_hats, ids, validate, 
                        loss_dict, v_y_hats, v_ids, validation_dict)
    #_plots(out_dir, suffix, loss_function, validate) # 2 leaked semaphore objects... pytorch x mpl??
    return unet



def _plots(out_dir, suffix, loss_function, validate):
    l_path = os.path.join(out_dir, 'loss_' + suffix + '.csv')
    v_path = None
    if validate:
        vl_path = os.path.join(out_dir, 'validation-loss_' + suffix + '.csv')
    save_loss_plot(l_path, loss_function, v_path=vl_path, show=False)
    save_channel_loss_plot(l_path, show=False)



def _get_loss_function(loss_function, chan_weights, device):
    # define the loss function
    if loss_function == 'BCELoss':
        loss = nn.BCELoss()
    elif loss_function == 'DiceLoss':
        loss = DiceLoss()
    elif loss_function == 'WeightedBCE':
        loss = WeightedBCELoss(chan_weights=chan_weights, device=device)
    #elif loss_function == 'BCECentrenessPenalty':
       # loss = BCELossWithCentrenessPenalty()
    elif loss_function == 'EpochWeightedBCE':
        loss = EpochwiseWeightedBCELoss(weights_list=chan_weights, device=device)
    else:
        m = 'Valid loss options are BCELoss, WeightedBCE, and DiceLoss'
        raise ValueError(m)
    return loss


def _load_weights(weights, unet):
    weights_are = 'naive'
    if weights is not None:
        unet.load_state_dict(weights)
        weights_are = 'pretrained'
    return weights_are


def _index_channels_if_none(channels, xs):
    if channels is None:
        new_chans = ['channel_' + str(i) for i in range(xs[0].shape[1])]
        return tuple(new_chans)
    else:
        return channels


def _get_loss_dict(channels):
    loss_dict = {'epoch' : [], 
                 'batch_num' : [], 
                 'loss' : [], 
                 'data_id' : []}
    for c in channels:
        loss_dict[c] = []
    return loss_dict


def _bce_weights(loss):
    bce_weights = None
    try:
        bce_weights = loss.chan_weights.data
    except:
        pass
    return bce_weights


def _print_train_info(loss_function, bce_weights, epochs, lr, 
                     weights_are, device_name, out_dir, log):
    s = LINE + '\n' + f'Loss function: {loss_function} \n'
    if bce_weights is not None:
        s = s + f'Loss function channel weights: {bce_weights} \n'
    s = s + 'Optimiser: Adam \n' + f'Learning rate: {lr} \n'
    s = s + LINE + '\n' 
    s = s + f'Training {weights_are} U-net for {epochs} '
    s = s + 'epochs with batch size 1 \n'
    s = s + f'Device: {device_name} \n' + LINE
    print(s)
    if log:
        write_log(LINE, out_dir)
        write_log(s, out_dir)



def _train_loop(no_iter, epochs, xs, ys, ids, device, unet, out_dir,
                optimiser, loss, loss_dict,  validate, v_xs, v_ys, 
                v_ids, validation_dict, v_loss, update_every, log, 
                suffix, channels):
    v_y_hats = None
    # loop over training data 
    unet = unet.to(device=device, dtype=torch.float32)
    with tqdm(total=no_iter, desc='unet training') as progress:
        for e in range(epochs):
            _set_epoch_if_epoch_weighted(loss, e)
            if validate and e == 0:
                _set_epoch_if_epoch_weighted(v_loss, e, verbose=False)
                if e == 0:
                    # first validation at the start of the first epoch
                    v_y_hats = _validate(v_xs, v_ys, v_ids, 
                                         device, unet, v_loss, 
                                         progress, log, out_dir, 
                                         validation_dict, e, 0)
            running_loss = 0.0
            y_hats = []
            for i in range(len(xs)):
                l = _train_step(i, xs, ys, ids, device, unet, optimiser, 
                                y_hats, loss, loss_dict, e, channels)
                optimiser.step()
                running_loss += l.item()
                progress.update(1)
                if i % update_every == (update_every - 1):
                    s = f'Epoch {e} - running loss: ' 
                    s = s + f'{running_loss / update_every}'
                    print(s)
                    if log:
                        write_log(s, out_dir)
                    running_loss = 0.0
            if validate:
                # validation at the end of the epoch
                batch_no = ((e + 1) * len(xs))
                v_y_hats = _validate(v_xs, v_ys, v_ids, 
                                     device, unet, v_loss, 
                                     progress, log, out_dir, 
                                     validation_dict, e, batch_no)
            _save_checkpoint(unet.state_dict(), out_dir, 
                             f'{suffix}_epoch-{e}')  
    return y_hats, v_y_hats


def _set_epoch_if_epoch_weighted(loss, e, verbose=True):
    if isinstance(loss, EpochwiseWeightedBCELoss):
        loss.current_epoch = e
        if verbose:
            print(f'Channel weights set to {loss.current_weights.data} for epoch {e} ')


def _train_step(i, xs, ys, ids, device, unet, optimiser, 
                y_hats, loss, loss_dict, e, channels):
    x, y = _prep_x_y(xs[i], ys[i], device)
    optimiser.zero_grad()
    y_hat = unet(x.float())
    y_hats.append(y_hat)
    l = loss(y_hat, y)
    l.backward()
    optimiser.step()
    loss_dict['epoch'].append(e) 
    loss_dict['batch_num'].append(i)
    loss_dict['loss'].append(l.item())
    loss_dict['data_id'].append(ids[i])
    channel_losses_to_dict(y_hat, y, channels, loss_dict)
    return l


def _validate(v_xs, v_ys, v_ids, device, unet, v_loss, progress, 
             log, out_dir, validation_dict, e, batch_no):
    validation_loss = 0.0
    with torch.no_grad():
        v_y_hats = []
        for i in range(len(v_xs)):
            v_x, v_y = _prep_x_y(v_xs[i], v_ys[i], device)
            v_y_hat = unet(v_x.float())
            v_y_hats.append(v_y_hat)
            vl = v_loss(v_y_hat, v_y)
            validation_loss += vl.item()
            validation_dict['epoch'].append(e)
            validation_dict['validation_loss'].append(vl.item())
            validation_dict['data_id'].append(v_ids[i])
            validation_dict['batch_id'].append(batch_no)
            progress.update(1)
        score = validation_loss / len(v_xs)
        s = f'Epoch {e} - validation loss: {score}'
        print(s)
        if log:
            write_log(s, out_dir)
    return v_y_hats


def _prep_x_y(x, y, device):
    x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)
    x = torch.unsqueeze(x, 0)
    x, y = x.to(device), y.to(device)
    y = y.type(torch.float32)
    return x, y


def _save_final_results(unet, out_dir, suffix, y_hats, ids, validate,
                        loss_dict, v_y_hats, v_ids, validation_dict):
    _save_checkpoint(unet.state_dict(), out_dir, suffix)
    _save_output(y_hats, ids, out_dir)
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(os.path.join(out_dir, 'loss_' + suffix + '.csv'))
    if validate:
        _save_output(v_y_hats, v_ids, out_dir, suffix='_validation')
        v_loss_df = pd.DataFrame(validation_dict)
        v_loss_df.to_csv(os.path.join(out_dir, 
                         'validation-loss_' + suffix + '.csv'))


def _save_checkpoint(checkpoint, out_dir, suffix):
    now = datetime.now()
    d = now.strftime("%y%d%m_%H%M%S")
    name = d + '_unet_' + suffix + '.pt'
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, path)


def _save_output(y_hats, ids, out_dir, suffix=''):
    assert len(y_hats) == len(ids)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(y_hats)):
        n = ids[i] + suffix +'_output.tif'
        p = os.path.join(out_dir, n)
        with TiffWriter(p) as tiff:
            tiff.write(y_hats[i].detach().cpu().numpy())


#def test_unet(unet, image_paths, labels_paths, out_dir):
 #   xs, ys, ids = load_train_data(image_paths, labels_paths, out_dir, n_each=10)
  #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   # y_hats = []
#    correct = []
 #   for i in range(len(xs)):
  #      x, y = xs[i].to(device), ys[i].to(device)
   #     y_hat = unet(x.float())
    #    y_hats.append(y_hat)
     #   same = y.numpy() == y_hat.numpy()
      #  prop_correct = same.sum() / same.size
       # correct.append(prop_correct)
#        print(f'The proportion of correct pixels for image {ids[i]}: {prop_correct}') 
 #   save_output(y_hats, ids, out_dir)
  #  correct = np.array(correct)
   # print(f'Overall, the model scored {correct.mean() * 100} %')
#    return unet



# -----------------
# Wrapper Functions
# -----------------


def train_unet_from_directory(
                              out_dir, 
                              suffix, 
                              data_dir,
                              validation_dir=None,
                              epochs=4, 
                              lr=0.01, 
                              loss_function='BCELoss', 
                              chan_weights=(1., 2., 2.), # for weighted BCE
                              weights=None,
                              update_every=20, 
                              channels=None, 
                              **kwargs
                              ):
    '''
    Train a basic U-Net on affinities data. Load chunks of training data
    from directory.


    Parameters
    ----------
    out_dir: str
        Directory to which to save network output
    suffix: str
        Suffix used in naming pytorch state dictionary file
    data_dir: None or str 
        LOAD: Only applicable when loading training data. If None
        training data is assumed to be in the output directory.
        Otherwise, data_dir should be the directory in which 
        training data is located
    validation_dir: None or str
        LOAD: If none, no validation is performed. If provided, validation
        data is loaded from the given directory according to the 
        same naming convention as training data. Validation is performed 
        at the end of every epoch. 
        Labels are expected to be in int form (typical segmentation)
    channels: tuple of str
        Types of output channels to be obtained.
            Affinities: 'axis-n' (pattern: r'[xyz]-\d+' e.g., 'z-1')
            Centreness: 'centreness'
    epochs: int
        How many times should we go through the training data?
    lr: float
        Learning rate for Adam optimiser
    loss_function: str
        Which loss function will be used for training & validation?
        Current options include:
            'BCELoss': Binary cross entropy loss
            'WeightedBCE': Binary cross entropy loss whereby channels are weighted
                according to chan_weights parameter. Quick way to force network to
                favour learning information about a given channel/s.
            'DiceLoss': 1 - DICE coefficient of the output-target pair
    chan_weights: tuple of float
        WEIGHTEDBCE: Weights for BCE loss for each output channel. 
    weights: None or nn.Model().state_dict()
        Prior weights with which to initalise the network.
    update_every: int
        Determines how many batches are processed before printing loss

    Returns
    -------
    unet: UNet (unet.py)

    Notes
    -----
    When data is loaded from a directory, it will be recognised according
    to the following naming convention:

        IDs: YYMMDD_HHMMSS_{digit/s} 
        Images: YYMMDD_HHMMSS_{digit/s}_image.tif
        Affinities: YYMMDD_HHMMSS_{digit/s}_labels.tif
    
    E.g., 210309_152717_7_image.tif, 210309_152717_7_labels.tif

    For each ID, a labels and an image file must be found or else an
    assertion error will be raised.
    '''
    log = True
    if data_dir == None:
        d = out_dir
    else:
        d = data_dir
    xs, ys, ids = load_train_data(d)
    # if applicable, load the validation data
    if validation_dir is not None:
        validate = True
        v_xs, v_ys, v_ids = _load_validation(validation_dir, out_dir, log)
    else:
        v_xs, v_ys, v_ids = None, None, None
        validate = False
    unet = train_unet(
                      # training data
                      xs, 
                      ys, 
                      ids, 
                      # output information
                      out_dir, 
                      suffix, 
                      channels,
                      # validation data
                      v_xs=v_xs,
                      v_ys=v_ys,
                      v_ids=v_ids,
                      validate=validate,
                      # training variables
                      log=log,
                      epochs=epochs, 
                      lr=lr, 
                      loss_function=loss_function, 
                      chan_weights=chan_weights, # for weighted BCE
                      weights=weights,
                      update_every=update_every
                      )
    return unet


def _load_validation(validation_dir, out_dir, log):
    v_xs, v_ys, v_ids = load_train_data(validation_dir)
    print(LINE)
    s = f'Loaded {len(v_xs)} sets of validation data'
    print(s)
    if log:
        write_log(LINE, out_dir)
        write_log(s, out_dir)
    return v_xs, v_ys, v_ids


def train_unet_get_labels(
                          out_dir, 
                          image_paths, 
                          labels_paths,
                          suffix='',
                          channels=('z-1', 'y-1', 'x-1', 'centreness'), 
                          n_each=100,
                          validation_prop=None, 
                          scale=(4, 1, 1),
                          epochs=3, 
                          lr=0.01, 
                          loss_function='BCELoss', 
                          chan_weights=(1., 2., 2.), # for weighted BCE
                          weights=None,
                          update_every=20,
                          fork_channels=None,
                          **kwargs
                          ):
    '''
    Train a basic U-Net on affinities data. Generates chunks of training data
    in which case chunks with spatial dimensions of (10, 256, 256).

    Different types of channels can be generated from a segmentation as training
    data. These include z, y, and x affinities of specified degree and scores
    for centreness (i.e., scores segmented voxels according to closeness to centre).

    Parameters
    ----------
    out_dir: str
        Directory to which to save network output
    suffix: str
        Suffix used in naming pytorch state dictionary file
    image_paths: None or list of str
        Only applicable if generating trainig data from volumes.
        Paths to whole voume images.
    labels_paths: None or list of str
        Only applicable if generating trainig data from volumes.
        Paths to whole voume labels. 
        Labels are expected to be in int form (typical segmentation)
    channels: tuple of str
        Types of output channels to be obtained.
            Affinities: 'axis-n' (pattern: r'[xyz]-\d+' e.g., 'z-1')
            Centreness: 'centreness'
    n_each: int
        Number of image-labels pairs to obtain from each image-GT volume
        provided.
    scale: tuple of numeric
        Scale of channels. This is used in calculating centreness score.
    validation_prop: float
        If greater than 0, validation data will be generated and a 
        validation performed at the end of every epoch. The number of 
        pairs generated correspond to the proportion inputted.  
    epochs: int
        How many times should we go through the training data?
    lr: float
        Learning rate for Adam optimiser
    loss_function: str
        Which loss function will be used for training & validation?
        Current options include:
            'BCELoss': Binary cross entropy loss
            'WeightedBCE': Binary cross entropy loss whereby channels are weighted
                according to chan_weights parameter. Quick way to force network to
                favour learning information about a given channel/s.
            'DiceLoss': 1 - DICE coefficient of the output-target pair
    chan_weights: tuple of float
        WEIGHTEDBCE: Weights for BCE loss for each output channel. 
    weights: None or nn.Model().state_dict()
        Prior weights with which to initalise the network.
    update_every: int
        Determines how many batches are processed before printing loss

    Returns
    -------
    unet: UNet (unet.py)
    '''
    log = True
    validate = False
    xs, ys, ids = get_train_data(
                                 image_paths, 
                                 labels_paths, 
                                 out_dir, 
                                 n_each=n_each,
                                 channels=channels,
                                 scale=scale, 
                                 log=log
                                 )
    if validation_prop > 0:
        validate = True
        v_n_each = np.round(validation_prop * n_each)
        v_xs, v_ys, v_ids = get_train_data(
                                 image_paths, 
                                 labels_paths, 
                                 out_dir, 
                                 n_each=v_n_each,
                                 channels=channels,
                                 scale=scale, 
                                 log=log
                                 )
    else:
        v_xs, v_ys, v_ids = None, None, None
    unet = train_unet(
                      # training data
                      xs, 
                      ys, 
                      ids, 
                      # output information
                      out_dir, 
                      suffix, 
                      channels,
                      # validation data
                      v_xs=v_xs,
                      v_ys=v_ys,
                      v_ids=v_ids,
                      validate=validate,
                      # training variables
                      log=log,
                      epochs=epochs, 
                      lr=lr, 
                      loss_function=loss_function, 
                      chan_weights=chan_weights, # for weighted BCE
                      weights=weights,
                      update_every=update_every, 
                      fork_channels=fork_channels
                      )
    return unet


# Another option, but need to build ResBlock :) in unet.py

def train_resunet():
    pass

