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
               validation_dir=None,
               image_paths=None, 
               labels_paths=None, 
               epochs=3, 
               lr=0.01, 
               train_data='load', 
               weights=None, 
               loss_function='WeightedBCE'
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
        print('------------------------------------------------------------')
        print(f'Loaded {len(xs)} sets of training data')
    # if applicable, load the validation data
    validate = False
    if validation_dir is not None:
        validate = True
        v_xs, v_ys, v_ids = load_train_data(validation_dir)
        print('------------------------------------------------------------')
        print(f'Loaded {len(v_xs)} sets of validation data')
    # initialise U-net
    unet = UNet()
    # load weights if applicable 
    weights_are = 'naive'
    if weights is not None:
        unet.load_state_dict(weights)
        weights_are = 'pretrained'
    # define the optimiser
    optimiser = optim.Adam(unet.parameters(), lr=lr)
    # define the loss function
    if loss_function == 'BCELoss':
        loss = nn.BCELoss()
        if validate:
            v_loss = nn.BCELoss()
    elif loss_function == 'DiceLoss':
        loss = DiceLoss()
        if validate:
            v_loss = DiceLoss()
    elif loss_function == 'WeightedBCE':
        loss = WeightedBCELoss()
        bce_weights = loss.chan_weights
        if validate:
            v_loss = WeightedBCELoss()
    else:
        raise ValueError('Valid loss options are BCELoss, WeightedBCE, and DiceLoss')
    print('------------------------------------------------------------')
    print(f'Loss function: {loss_function}')
    if bce_weights is not None:
        print(f'Loss function channel weights: {bce_weights}')
    print('Optimiser: Adam')
    print(f'Learning rate: {lr}')
    # loop over training data 
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if validate:
        no_iter = (epochs * len(xs)) + (epochs * len(v_xs))
    else:
        no_iter = epochs * len(xs)
    print('------------------------------------------------------------')
    print(f'Training {weights_are} U-net for {epochs} epochs with batch size 1 ')
    print(f'Device: {device_name}')
    print('------------------------------------------------------------')
    with tqdm(total=no_iter, desc='unet training') as progress:
        for e in range(epochs):
            running_loss = 0.0
            if validate:
                validation_loss = 0.0
            y_hats = []
            for i in range(len(xs)):
                x, y = prep_x_y(xs[i], ys[i], device)
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
            if validate:
                with torch.no_grad():
                    v_y_hats = []
                    for i in range(len(v_xs)):
                        v_x, v_y = prep_x_y(v_xs[i], v_ys[i], device)
                        v_y_hat = unet(v_x.float())
                        v_y_hats.append(v_y_hat)
                        vl = v_loss(v_y_hat, v_y)
                        validation_loss += vl.item()
                        progress.update(1)
                    print(f'Epoch {e} - validation loss: {validation_loss / len(v_xs)}')
            save_checkpoint(unet.state_dict(), out_dir, f'{suffix}_epoch-{e}')
    save_checkpoint(unet.state_dict(), out_dir, suffix)
    save_output(y_hats, ids, out_dir)
    if validate:
        save_output(v_y_hats, v_ids, out_dir, suffix='_validation')
    return unet


def prep_x_y(x, y, device):
    x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)
    x = torch.unsqueeze(x, 0)
    x, y = x.to(device), y.to(device)
    y = y.type(torch.float32)
    return x, y


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


def save_output(y_hats, ids, out_dir, suffix=''):
    assert len(y_hats) == len(ids)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(y_hats)):
        n = ids[i] + suffix +'_output.tif'
        p = os.path.join(out_dir, n)
        with TiffWriter(p) as tiff:
            tiff.write(y_hats[i].detach().numpy())


# --------------
# Loss Functions
# --------------


class DiceLoss(nn.Module):
    '''
    DiceLoss: 1 - DICE coefficient 

    Adaptations: weights output channels equally in final loss. 
    This is necessary for anisotropic data.
    '''
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, channel_dim=1, smooth=1):
        '''
        inputs: torch.tensor
            Network predictions. Float
        targets: torch.tensor
            Ground truth labels. Float
        channel_dim: int
            Dimension in which output channels can be found.
            Loss is weighted equally between output channels.
        smooth: int
            Smoothing hyperparameter.
        '''
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs) 
        inputs, targets = flatten_channels(inputs, targets, channel_dim)
        intersection = (inputs * targets).sum(-1) 
        dice = (2.*intersection + smooth)/(inputs.sum(-1) + targets.sum(-1) + smooth) 
        loss = 1 - dice 
        return loss.mean()


class WeightedBCELoss(nn.Module):
    def __init__(self, chan_weights=(1., 2., 2.), reduction='mean', final_reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.reduction = reduction
        self.final_reduction = final_reduction
        self.chan_weights = torch.tensor(list(chan_weights))


    def forward(self, inputs, targets, channel_dim=1):
        inputs, targets = flatten_channels(inputs, targets, channel_dim)
        unreduced = self.bce(inputs, targets)
        if self.reduction == 'mean':
            channel_losses = unreduced.mean(-1) * self.chan_weights
        elif self.reduction == 'sum':
            channel_losses = unreduced.sum(-1) * self.chan_weights
        else:
            raise ValueError('reduction param must be mean or sum')
        if self.final_reduction == 'mean':
            loss = channel_losses.mean()
        elif self.final_reduction == 'sum':
            loss = channel_losses.sum()
        else:
            raise ValueError('final_reduction must be mean or sum')
        return loss



def flatten_channels(inputs, targets, channel_dim):
    '''
    Helper function to flatten inputs and targets for each channel

    E.g., (1, 3, 10, 256, 256) --> (3, 655360)

    Parameters
    ----------
    inputs: torch.Tensor
        U-net output
    targets: torch.Tensor
        Target labels
    channel_dim: int
        Which dim represents output channels? 
    '''
    order = [channel_dim, ]
    for i in range(len(inputs.shape)):
        if i != channel_dim:
            order.append(i)
    inputs = inputs.permute(*order)
    inputs = torch.flatten(inputs, start_dim=1)
    targets = targets.permute(*order)
    targets = torch.flatten(targets, start_dim=1)
    return inputs, targets
