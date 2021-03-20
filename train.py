from custom_loss import WeightedBCELoss, DiceLoss
from datetime import datetime
from helpers import LINE, write_log
import numpy as np
import os
import pandas as pd
from tifffile import TiffWriter 
import torch 
import torch.nn as nn
import torch.optim as optim
from train_io import get_train_data, load_train_data
from tqdm import tqdm
from unet import UNet

# Note: not sure if the default loss function is the most approapriate.
#   I know that BCE Loss is used for image segmentation (Jadon et al., 2020, arXiv)
#   so what the hell, give it a go...
def train_unet(
               # output information
               out_dir, 
               suffix, 
               log=True,
               train_data='load', 
               # load train data
               data_dir=None,
               validation_dir=None,
               # Get train data
               image_paths=None, 
               labels_paths=None,
               n_each=100,
               channels=('z-1', 'y-1', 'x-1', 'centreness'),
               validation_prop=None,
               scale=(4, 1, 1), # for centreness
               # training variables
               epochs=3, 
               lr=0.01, 
               loss_function='WeightedBCE', 
               chan_weights=(1., 2., 2.), # for weighted BCE
               weights=None,
               update_every=20
               ):
    '''
    Train a basic U-Net on affinities data. Works with both whole volumes, 
    in which case chunks of (10, 256, 256) training data are generated, and 
    already generated and saved training data. This should probably be two 
    different functions... Oh well *laughs mischievously*

    Parameters
    ----------
    out_dir: str
        Directory to which to save network output
    suffix: str
        Suffix used in naming pytorch state dictionary file
    data_dir: None or str
        Only applicable when loading training data. If None
        training data is assumed to be in the output directory.
        Otherwise, data_dir should be the directory in which 
        training data is located
    validation_dir: None or str
        If none, no validation is performed. If provided, validation
        data is loaded from the given directory according to the 
        same naming convention as training data. Validation is then
        performed at the end of every epoch. 
    image_paths: None or list of str
        Only applicable if generating trainig data from volumes.
        Paths to whole voume images.
    labels_paths: None or list of str
        Only applicable if generating trainig data from volumes.
        Paths to whole voume labels. 
        Labels are expected to be in int form (typical segmentation)
    epochs: int
        How many times should we go through the training data?
    lr: float
        Learning rate for Adam optimiser
    train_data: str 
        'load' or 'get'. Should training data be loaded from 
        a directory or generated from whole volumes.
    weights: None or nn.Model().state_dict()
        Prior weights with which to initalise the network.
    auxillary: None or str
        'centroid' or 'long_range_affs'

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
    # Get training data from lists of whole volumes
    validate = False
    if train_data == 'get':
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

    # Get training data stored in a dictionary according to a naming pattern
    if train_data == 'load':
        if data_dir == None:
            d = out_dir
        else:
            d = data_dir
        xs, ys, ids = load_train_data(d)
        # if applicable, load the validation data
        if validation_dir is not None:
            validate = True
            v_xs, v_ys, v_ids = load_train_data(validation_dir)
            print(LINE)
            s = f'Loaded {len(v_xs)} sets of validation data'
            print(s)
            if log:
                write_log(LINE, out_dir)
                write_log(s, out_dir)
    # initialise U-net
    unet = UNet(out_channels=len(channels))
    # load weights if applicable 
    weights_are = 'naive'
    if weights is not None:
        unet.load_state_dict(weights)
        weights_are = 'pretrained'
    # define the optimiser
    optimiser = optim.Adam(unet.parameters(), lr=lr)
    # define the loss function
    bce_weights = None
    loss = _get_loss_function(loss_function, chan_weights)
    loss_dict = {'epoch' : [], 
                 'batch_num' : [], 
                 'loss' : [], 
                 'data_id' : []}
    bce_weights = None
    try:
        bce_weights = loss.chan_weights
    except:
        pass
    if validate:
        v_loss = _get_loss_function(loss_function, chan_weights)
        validation_dict = {'epoch' : [], 
                           'validation_loss' : []}
    # Device
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    # how many iterations will be done
    if validate:
        no_iter = (epochs * len(xs)) + (epochs * len(v_xs))
    else:
        no_iter = epochs * len(xs)
    # print the training into and log if applicable 
    _print_train_info(loss_function, bce_weights, epochs, lr, 
                     weights_are, device_name, out_dir, log)
    # loop over training data 
    with tqdm(total=no_iter, desc='unet training') as progress:
        for e in range(epochs):
            running_loss = 0.0
            y_hats = []
            for i in range(len(xs)):
                l = _train_step(i, xs, ys, ids, device, unet, optimiser, 
                                y_hats, loss, loss_dict, e)
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
                v_y_hats = _validate(v_xs, v_ys, device, unet, v_loss, 
                                     progress, log, out_dir, validation_dict, e)
            _save_checkpoint(unet.state_dict(), out_dir, 
                             f'{suffix}_epoch-{e}')
    _save_checkpoint(unet.state_dict(), out_dir, suffix)
    _save_output(y_hats, ids, out_dir)
    loss_df = pd.DataFrame(loss_dict)
    loss_df.to_csv(os.path.join(out_dir, 'loss_' + suffix + '.csv'))
    if validate:
        _save_output(v_y_hats, v_ids, out_dir, suffix='_validation')
        v_loss_df = pd.DataFrame(validation_dict)
        v_loss_df.to_csv(os.path.join(out_dir, 
                         'validation-loss_' + suffix + '.csv'))
    return unet


def _get_loss_function(loss_function, chan_weights):
    # define the loss function
    if loss_function == 'BCELoss':
        loss = nn.BCELoss()
    elif loss_function == 'DiceLoss':
        loss = DiceLoss()
    elif loss_function == 'WeightedBCE':
        loss = WeightedBCELoss(chan_weights=chan_weights)
    #elif loss_function == 'BCECentrenessPenalty':
       # loss = BCELossWithCentrenessPenalty()
    else:
        m = 'Valid loss options are BCELoss, WeightedBCE, and DiceLoss'
        raise ValueError(m)
    return loss


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


def _train_step(i, xs, ys, ids, device, unet,
                optimiser, y_hats, loss, loss_dict, e):
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
    return l


def _validate(v_xs, v_ys, device, unet, v_loss, progress, 
             log, out_dir, validation_dict, e):
    validation_loss = 0.0
    with torch.no_grad():
        v_y_hats = []
        for i in range(len(v_xs)):
            v_x, v_y = _prep_x_y(v_xs[i], v_ys[i], device)
            v_y_hat = unet(v_x.float())
            v_y_hats.append(v_y_hat)
            vl = v_loss(v_y_hat, v_y)
            validation_loss += vl.item()
            progress.update(1)
        score = validation_loss / len(v_xs)
        s = f'Epoch {e} - validation loss: {score}'
        print(s)
        if log:
            write_log(s, out_dir)
        validation_dict['epoch'].append(e)
        validation_dict['validation_loss'].append(score)
    return v_y_hats
        

def _prep_x_y(x, y, device):
    x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)
    x = torch.unsqueeze(x, 0)
    x, y = x.to(device), y.to(device)
    y = y.type(torch.float32)
    return x, y


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
            tiff.write(y_hats[i].detach().numpy())


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
                              update_every=20
                              ):
    unet = train_unet(
                      # output information
                      out_dir, 
                      suffix, 
                      log=True,
                      train_data='load', 
                      # load train data
                      data_dir=data_dir,
                      validation_dir=validation_dir,
                      # training variables
                      epochs=epochs, 
                      lr=lr, 
                      loss_function=loss_function, 
                      chan_weights=chan_weights, # for weighted BCE
                      weights=weights,
                      update_every=update_every
                      )
    return unet


def train_unet_get_labels(
                          out_dir, 
                          suffix,
                          image_paths, 
                          labels_paths,
                          n_each=100,
                          channels=('z-1', 'y-1', 'x-1', 'centreness'), 
                          validation_prop=None, 
                          scale=(4, 1, 1),
                          epochs=3, 
                          lr=0.01, 
                          loss_function='BCELoss', 
                          chan_weights=(1., 2., 2.), # for weighted BCE
                          weights=None,
                          update_every=20
                          ):
    unet = train_unet(
                      # output information
                      out_dir, 
                      suffix, 
                      log=True,
                      train_data='get', 
                      # Get train data
                      image_paths=image_paths, 
                      labels_paths=labels_paths,
                      n_each=n_each,
                      channels=channels,
                      validation_prop=validation_prop,
                      scale=scale, # for centreness
                      # training variables
                      epochs=epochs, 
                      lr=lr, 
                      loss_function=loss_function, 
                      chan_weights=chan_weights, # for weighted BCE
                      weights=weights,
                      update_every=update_every
                      )
    return unet


# Another option, but need to build ResBlock :) in unet.py

def train_resunet():
    pass

