from affinities import get_train_data, load_train_data
from datetime import datetime
import numpy as np
import os
from tifffile import TiffWriter 
import torch 
import torch.nn as nn
import torch.optim as optim
from unet import UNet
from tqdm import tqdm

# Note: not sure if the default loss function is the most approapriate.
#   I know that BCE Loss is used for image segmentation (Jadon et al., 2020, arXiv)
#   so what the hell, give it a go...
def train_unet(
               out_dir, 
               suffix, 
               data_dir=None,
               image_paths=None, 
               labels_paths=None, 
               epochs=3, 
               lr=0.01, 
               train_data='load', 
               weights=None
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
        Learning rate for SGD optimiser
    train_data: str 
        'load' or 'get'. Should training data be loaded from 
        a directory or generated from whole volumes.
    weights: None or nn.Model().state_dict()
        Prior weights with which to initalise the network.

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
    if train_data == 'get':
        xs, ys, ids = get_train_data(image_paths, labels_paths, out_dir)
    # Get training data stored in a dictionary according to a naming pattern
    if train_data == 'load':
        if data_dir == None:
            d = out_dir
        else:
            d = data_dir
        xs, ys, ids = load_train_data(d)
        print(len(xs))
    unet = UNet()
    # load weights if applicable 
    if weights is not None:
        unet.load_state_dict(weights)
    optimiser = optim.SGD(unet.parameters(), lr=lr)
    loss = nn.BCELoss()
    # loop over training data 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tqdm(total=epochs*len(xs), desc='unet training') as progress:
        for e in range(epochs):
            running_loss = 0.0
            y_hats = []
            for i in range(len(xs)):
                x, y = torch.unsqueeze(xs[i], 0), torch.unsqueeze(ys[i], 0)
                x = torch.unsqueeze(x, 0)
                # x = x.type(torch.DoubleTensor)
                x, y = x.to(device), y.to(device)
                y = y.type(torch.float32)
                optimiser.zero_grad()
                y_hat = unet(x.float())
                y_hats.append(y_hat)
                l = loss(y_hat, y)
                l.backward()
                optimiser.step()
                running_loss += l.item()
                progress.update(1)
                if i % 20 == 19:
                    print(f'Epoch {e} - running loss: {running_loss / 20}')
                    running_loss = 0.0
            save_checkpoint(unet.state_dict(), out_dir, f'{suffix}_epoch-{e}')
    save_checkpoint(unet.state_dict(), out_dir, suffix)
    save_output(y_hats, ids, out_dir)
    return unet


def test_unet(unet, image_paths, labels_paths, out_dir):
    xs, ys, ids = load_train_data(image_paths, labels_paths, out_dir, n_each=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_hats = []
    correct = []
    for i in range(len(xs)):
        x, y = xs[i].to(device), ys[i].to(device)
        y_hat = unet(x.float())
        y_hats.append(y_hat)
        same = y.numpy() == y_hat.numpy()
        prop_correct = same.sum() / same.size
        correct.append(prop_correct)
        print(f'The proportion of correct pixels for image {ids[i]}: {prop_correct}') 
    save_output(y_hats, ids, out_dir)
    correct = np.array(correct)
    print(f'Overall, the model scored {correct.mean() * 100} %')
    return unet


def save_checkpoint(checkpoint, out_dir, suffix):
    now = datetime.now()
    d = now.strftime("%y%d%m_%H%M%S")
    name = d + '_unet_' + suffix + '.pt'
    path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, path)


def save_output(y_hats, ids, out_dir):
    assert len(y_hats) == len(ids)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(y_hats)):
        n = ids[i] + '_output.tif'
        p = os.path.join(out_dir, n)
        with TiffWriter(p) as tiff:
            tiff.write(y_hats[i].detach().numpy())

