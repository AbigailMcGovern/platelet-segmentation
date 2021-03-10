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
               image_paths=None, 
               labels_paths=None, 
               epochs=5, 
               lr=0.01, 
               train_data='load'):
    '''
    Train a basic U-Net on affinities data
    '''
    if train_data == 'get':
        xs, ys, ids = get_train_data(image_paths, labels_paths, out_dir)
    if train_data == 'load':
        xs, ys, ids = load_train_data(out_dir)
        print(len(xs))
    unet = UNet()
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
                if i % 25 == 24:
                    print(f'Epoch {e} - running loss: {running_loss / 25}')
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
    torch.save(checkpoint, path)


def save_output(y_hats, ids, out_dir):
    assert len(y_hats) == len(ids)
    for i in range(len(y_hats)):
        n = ids[i] + '_output.tif'
        p = os.path.join(out_dir, n)
        with TiffWriter(p) as tiff:
            tiff.write(y_hats[i].detach().numpy())

