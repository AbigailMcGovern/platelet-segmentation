import torch
import torch.nn as nn
import numpy as np
from skimage.morphology._util import _offsets_to_raveled_neighbors


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


class BCELossWithCentrenessPenalty(nn.Module):

    def __init__(
                 self, 
                 weight=0.1
                 ):
        super(BCELossWithCentrenessPenalty, self).__init__()
        self.weight = weight
        self.bce = nn.BCELoss()

    
    def forward(self, inputs, targets):
        loss = self.bce(inputs, targets)
        affinities = inputs[:3, ...].detach()
        centreness = targets[3, ...].detach()
        penalty = self.centreness_penalty(affinities, centreness)
        loss = loss.subtract(self.weight * penalty)
        return loss


    def centreness_penalty(self, affinities, centreness):
        '''
        Parameters
        ----------
        affinities: torch.Tensor
            shape (3, z, y, x)
        centreness: torch.Tensor
            shape (z, y, x)
        '''
        scores = []
        for i in range(affinities.shape[0]):
            aff = affinities[i].detach()
            score = self.bce(aff, centreness)
            scores.append(score)
        scores = torch.tensor(scores)
        score = scores.mean()
        return score



class CentroidLoss(nn.Module):

    def __init__(
                 self, 
                 selem=np.ones((3, 9, 9)), 
                 centre=(4, 4, 4), 
                 chan_weights=(1., 1., 1.), 
                 eta = 0.5,
                 phi = 0.5
                 ):
        super(WeightedBCELoss, self).__init__()
        self.selem = selem
        self.centre = centre
        self.shape = None
        self.BCELoss = WeightedBCELoss(chan_weights=chan_weights)
        self.eta = eta
        self.phi = phi


    def forward(
                self, 
                inputs, 
                targets, 
                selem=np.ones((3, 9, 9)), 
                centre=(1, 4, 4), 
                channel_dim=1
                ):
        
        # Prep, because flat is easier
        block_shape = [s for s in inputs.shape[-3:]]
        block_shape = tuple(shape)
        inputs, targets = flatten_channels(inputs, targets, self.channel_dim)

        # -----------------------
        # BCE Loss for Affinities
        # -----------------------
        # ???? pytorch will make this unexpectedly difficult I'm sure
        input_affs = inputs[:3] # pretty sure this won't work LOL
        target_affs = targets[:3]
        loss = self.BCELoss(input_affs, target_affs)

        # -------------------
        # Centroid Similarity
        # -------------------
        target_cent = targets[3].numpy()
        # penalise similarity of affinities to centroids - high score
        aff_penalties = []
        for i in range(3):
            aff_chan = inputs[i].numpy()
            penalty = self.centroid_similarity(aff_chan, target_cent)
        aff_penalties = np.array(aff_penalties)
        aff_penalty = aff_penalties.mean() * self.phi
        aff_penalties = torch.from_numpy(aff_penalty)
        # penalise difference of centroid output channel from centroids
        input_cent = inputs[3].numpy()
        cent_penalty = (1 - self.centroid_similarity(input_cent, target_cent)) * self.eta
        cent_penalty = torch.from_numpy(cent_penalty)
        # add penalties to 
        loss.add_(aff_penalty)
        loss.add_(cent_penalty)
        return loss



    def centroid_similarity(
                            self, 
                            inputs, 
                            targets,
                            shape 
                            ):
        """
        Metric for similarity to centroid. Computes a similarity based on
        inverse of normaised euclidian distance for every centroid neighbor.
        Neighbors are determined by a structing element (cell-ish dims)

        The output should be between 0-1 (therefore easily invertable)
        MAKE SURE THIS IS TRUE!!!

        Notes
        -----
        Currently uses a structing element to find centroid neighbors.
        Hoping to add an option for using segmentation to get neighbors. 

        """
        offsets = _offsets_to_raveled_neighbors(shape, 
                                                self.selem, 
                                                self.centre)

        euclid_dists = self.euclidian_distances()
        weights = euclid_dists - 1
        centroids = np.argwhere(np_targets == 1.)
        score = 0
        for c in centroids:
            max_ind = np.inputs.shape[-1]
            raveled_indices = c + offsets
            in_bounds_indices = np.array([idx for idx in raveled_indices \
                                            if idx >= 0 and idx < max_ind])
            neighbors = np_inputs[in_bounds_indices]
            weighted = neighbors * weights
            score += weighted.mean()
        return mean


    def euclidian_distances(self):
        '''
        Compute euclidian distances of each index from 
        '''
        selem_indices = np.stack(np.nonzero(self.selem), axis=-1)
        distances = []
        centre = np.array(self.centre)
        for ind in selem_indices:
            dist = np.linalg.norm(centre - ind)
            distances.append(dist)
        distances = np.array(distances)
        return distances / distances.max()


# helper function 
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