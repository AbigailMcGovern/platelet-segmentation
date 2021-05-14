from datetime import datetime
import os
import train
import torch.nn as nn


def run_experiment(experiments, image_paths, labels_paths, out_dir):
    for exp in experiments:
        exp_dir = os.path.join(out_dir, experiments[exp]['suffix'])
        _ = train.train_unet_get_labels(
                exp_dir, 
                image_paths, 
                labels_paths, 
                **experiments[exp]
            )  
    

def get_experiment_dict(custom_options):
    experiment = {
            'validation_prop' : 0.2, 
            'n_each' : 100, 
            'scale' : (4, 1, 1),
            'epochs' : 4,
            'lr' : .01,
            'loss_function' : 'BCELoss',
            'chan_weights' : None, 
            'weights' : None,
            'update_every' : 20, 
            'fork_channels' : None
        }
    for key in custom_options:
        experiment[key] = custom_options[key]
    # find the suffix
    name = experiment['name']
    if name is not None:
        end = '_' + name
    else:
        end = ''
    now = datetime.now()
    suffix = now.strftime("%y%m%d_%H%M%S") + end
    # add the suffix (lol, slash directory name ??!)
    # NB: it was initially used as a suffix for naming the loss data 
    # and was never renamed
    experiment['suffix'] = suffix
    if 'mask' in experiment['channels']:
        experiment['absolute_thresh'] = 0.5
    return experiment


# -----------------
# Experiments
# -----------------

affinities_exp = {
        0: get_experiment_dict({
            'name' : 'z-1s_y-1s_x-1s_c', 
            'channels' : ('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'centreness'), 
        }), 
        1: get_experiment_dict({
            'name' : 'z-1_y-1_x-1_c', 
            'channels' : ('z-1', 'y-1', 'x-1', 'centreness'), 
        }), 
        2: get_experiment_dict({
            'name' : 'z-1_y-1_x-1_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'centreness-log'), 
        }), 
        3: get_experiment_dict({
            'name' : 'z-1_y-1_x-1_c_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'centreness', 'centreness-log'), 
        }), 
        4: get_experiment_dict({
            'name' : 'z-1_y-1_y-2_x-1_x-2_c_cl', 
            'channels' : ('z-1', 'y-1', 'y-2', 'x-1', 'x-2', 'centreness', 'centreness-log'), 
        }),
        5: get_experiment_dict({
            'name' : 'z-1_z-2_y-1_y-2_x-1_x-2_c_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'centreness', 'centreness-log'), 
        }), 
        6: get_experiment_dict({
            'name' : 'z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
        }),
        7: get_experiment_dict({
            'name' : 'z-1s_y-1s_x-1s_c_cl', 
            'channels' : ('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'centreness', 'centreness-log'), 
        }), 
        8: get_experiment_dict({
            'name' : 'z-1_z-1s_y-1_y-1s_x-1_x-1s_c_cl', 
            'channels' : ('z-1', 'z-1-smooth', 'y-1', 'y-1-smooth', 'x-1', 'x-1-smooth', 'centreness', 'centreness-log'), 
        }),
        9: get_experiment_dict({
            'name' : 'z-1_z-2s_y-1_y-2s_x-1_x-2s_c_cl', 
            'channels' : ('z-1', 'z-2-smooth', 'y-1', 'y-2-smooth', 'x-1', 'x-2-smooth', 'centreness', 'centreness-log'), 
        })
    } 


affinities_exp_0 = {
        8: get_experiment_dict({
            'name' : 'z-1_z-1s_y-1_y-1s_x-1_x-1s_c_cl', 
            'channels' : ('z-1', 'z-1-smooth', 'y-1', 'y-1-smooth', 'x-1', 'x-1-smooth', 'centreness', 'centreness-log'), 
        }),
        9: get_experiment_dict({
            'name' : 'z-1_z-2s_y-1_y-2s_x-1_x-2s_c_cl', 
            'channels' : ('z-1', 'z-2-smooth', 'y-1', 'y-2-smooth', 'x-1', 'x-2-smooth', 'centreness', 'centreness-log'), 
        })
    } 


affinities_exp_1 = {
        3: get_experiment_dict({
            'name' : 'z-1_y-1_x-1_c_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'centreness', 'centreness-log'), 
            'fork_channels': (3, 2)
        }), 
        5: get_experiment_dict({
            'name' : 'z-1_z-2_y-1_y-2_x-1_x-2_c_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'centreness', 'centreness-log'),
            'fork_channels': (6, 2) 
        }), 
        6: get_experiment_dict({
            'name' : 'z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'fork_channels': (8, 2)
        })
}

mask_exp = {
        3: get_experiment_dict({
            'name' : 'z-1_y-1_x-1_m_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
        }), 
        5: get_experiment_dict({
            'name' : 'z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'mask', 'centreness-log'), 
        }), 
        6: get_experiment_dict({
            'name' : 'z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
            })
}


forked_exp = {
        3: get_experiment_dict({
            'name' : 'f3,2_z-1_y-1_x-1_m_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
            'fork_channels': (3, 2)
        }), 
        5: get_experiment_dict({
            'name' : 'f6,2_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'mask', 'centreness-log'),
            'fork_channels': (6, 2) 
        }), 
        6: get_experiment_dict({
            'name' : 'f8,2_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
            'fork_channels': (8, 2)
        })
}

# the following is an experiment with different something that approximates cirriculum learning
cirriculum_exp = {
        0: get_experiment_dict({
            'name' : 'EWBCE_uw0123_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]
            ]
        }),
        # some cirriculum
        1: get_experiment_dict({
            'name' : 'EWBCE_uw012-w3_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.]
            ]
        }),
        2: get_experiment_dict({
            'name' : 'EWBCE_uw01-w23_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.],
                [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.]
            ]
        }),
        3: get_experiment_dict({
            'name' : 'EWBCE_uw0-w123_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.],
                [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.],
                [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.]
            ]
        }),
    }


cirriculum_exp_0 = {
        0: get_experiment_dict({
            'name' : 'EWBCE_uw0123_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2.]
            ]
        }),
        # some cirriculum
        1: get_experiment_dict({
            'name' : 'EWBCE_uw012-w3_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [3., 1., 3., 1., 3., 1., 2., 2.]
            ]
        }),
        2: get_experiment_dict({
            'name' : 'EWBCE_uw01-w23_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [3., 1., 3., 1., 3., 1., 2., 2.],
                [3., 1., 3., 1., 3., 1., 2., 2.]
            ]
        }),
        3: get_experiment_dict({
            'name' : 'EWBCE_uw0-w123_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2.],
                [3., 1., 3., 1., 3., 1., 2., 2.],
                [3., 1., 3., 1., 3., 1., 2., 2.],
                [3., 1., 3., 1., 3., 1., 2., 2.]
            ]
        }),
    }


thresh_exp = {
    0: get_experiment_dict({
            'name' : 'thresh_z-1_y-1_x-1_c_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'centreness', 'centreness-log'), 
        }),
    1: get_experiment_dict({
            'name' : 'thresh_z-1_y-1_x-1_m_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
        }),
}

thresh_exp_0 = {
    0: get_experiment_dict({
            'name' : 'thresh_z-1_y-1_x-1_c_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'centreness', 'centroid-gauss'), 
        }),
    1: get_experiment_dict({
            'name' : 'thresh_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'), 
        }),
}

seed_exp = {
    0: get_experiment_dict({
            'name' : 'seed_z-1_y-1_x-1_m_c', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness'), 
        }),
    1: get_experiment_dict({
            'name' : 'seed_z-1_y-1_x-1_m_cl', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
        }),
    2: get_experiment_dict({
            'name' : 'seed_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'), 
        }),
}


affinities_exp_2 = {
    0: get_experiment_dict({
            'name' : 'aff_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')
        }), 
    1: get_experiment_dict({
            'name' : 'aff_z-1_z-2_y-1_y-2_x-1_x-2_m_centg', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'mask', 'centroid-gauss')
        }), 
    2: get_experiment_dict({
            'name' : 'aff_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_centg', 
            'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centroid-gauss')
        })
}

lsr_exp = {
    0: get_experiment_dict({
            'name' : 'lsr_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')
        }), 
    1: get_experiment_dict({
            'name' : 'lsr_z-1s_y-1s_x-1s_m_centg', 
            'channels' : ('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'mask', 'centroid-gauss')
        }), 
}

lsr_exp_mse = {
    0: get_experiment_dict({
            'name' : 'lsr-mse_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
            'loss_function' : 'MSELoss'
        }), 
    1: get_experiment_dict({
            'name' : 'lsr-mse_z-1s_y-1s_x-1s_m_centg', 
            'channels' : ('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'mask', 'centroid-gauss'),
            'loss_function' : 'MSELoss'
        }), 
}



loss_exp = {
    0: get_experiment_dict({
            'name' : 'loss-BCE_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
            'loss_function' : 'BCELoss'
        }),
    1: get_experiment_dict({
            'name' : 'loss-DICE_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
            'loss_function' : 'DiceLoss'
        }),
   # 2: get_experiment_dict({
    #        'name' : 'loss-DICE_z-1_y-1_x-1_m_centg', 
    #        'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
    #        'loss_function' : 'MSELoss'
    #    })
}

lr_exp = {
    0: get_experiment_dict({
            'name' : 'lr-05_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
            'lr' : 0.05
        }),
    1: get_experiment_dict({
            'name' : 'lr-01_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
            'lr' : 0.01
        }),
    2: get_experiment_dict({
            'name' : 'lr-005_z-1_y-1_x-1_m_centg', 
            'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
            'lr' : 0.005
        }),
}

if __name__ == '__main__':
    # the following is an experiment with different forms of cirriculum learning 
    weighted_exp = {
        # the following simply cuts to the chase and weighs harder tasks as more important
        3: get_experiment_dict({
            'name' : 'WBCE_1_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'loss_function' : 'WeightedBCE', 
            'chan_weights' : [2., 1., 2., 1., 1., 2., 1., 1., 1., 2.] # note / 14 not 12 as above
        }), 
        4: get_experiment_dict({
            'name' : 'WBCE_1_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'loss_function' : 'WeightedBCE', 
            'chan_weights' : [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.] # note / 10 
        })   
    }

    offsets_experiment = {
        1 : get_experiment_dict({
            'name' : 'f_BCE_z-1_y-1_x-1_oz_oy_ox', 
            'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x'), 
            'fork_channels': (3, 3), 
            'loss_function' : 'Channelwise',
            'losses' : [nn.BCELoss(), nn.MSELoss()], 
            'chan_losses' : [slice(0, 3), slice(3, 6)], 
            'chan_final_activations' : ['sigmoid', 'tanh']
        })
    }

    offsets_experiment_0 = {
        1 : get_experiment_dict({
            'name' : 'f_BCE_z-1_y-1_x-1_oz_oy_ox', 
            'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x'), 
            'loss_function' : 'Channelwise',
            'losses' : [nn.BCELoss(), nn.MSELoss()], 
            'chan_losses' : [slice(0, 3), slice(3, 6)], 
            'chan_final_activations' : ['sigmoid', 'sigmoid']
        })
    }

    offsets_experiment_1 = {
        1 : get_experiment_dict({
            'name' : 'f_BCE_z-1_y-1_x-1_oz_oy_ox', 
            'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x'), 
        })
    }

    offsets_experiment_2 = {
        0 : get_experiment_dict({
            'name' : 'BCE_z-1_y-1_x-1_oz_oy_ox_m', 
            'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
        }), 
        1 : get_experiment_dict({
            'name' : 'f6,1_BCE_z-1_y-1_x-1_oz_oy_ox_m', 
            'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
            'fork_channels' : (6, 1)
        }), 
        2 : get_experiment_dict({
            'name' : 'f3,4_BCE_z-1_y-1_x-1_oz_oy_ox_m', 
            'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
            'fork_channels' : (3, 4)
        }),
        3 : get_experiment_dict({
            'name' : 'BCE_oz_oy_ox_m', 
            'channels' : ('offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
        }),
        4 : get_experiment_dict({
            'name' : 'f3,1_BCE_oz_oy_ox_m', 
            'channels' : ('offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
            'fork_channels' : (3, 1)
        }),
    }

    offsets_experiment_3 = {
        0 : get_experiment_dict({
            'name' : 'cw3MSE,1BCE_oz_oy_ox_m', 
            'channels' : ('offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
            'loss_function' : 'Channelwise',
            'losses' : [nn.MSELoss(), nn.BCELoss()], 
            'chan_losses' : [slice(0, 3), slice(3, None)], 
        }),
    }

    # Directory for training data and network output 
    # data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    # Directory for training data and network output 
    data_dir = '/home/abigail/data/platelet-segmentation-training'
    # Path for original image volumes for which GT was generated
    image_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_image.zarr')] 
    # Path for GT labels volumes
    labels_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_labels.zarr')]
    # Run the experiments
    # there were others but they weren't used and the code is probs gone --_()_--

    # 20th April 2021 - DL machine
    #run_experiment(forked_exp, image_paths, labels_paths, data_dir)
    #run_experiment(affinities_exp, image_paths, labels_paths, data_dir)
    
    # 26th April 2021 - Macbook pro
    #run_experiment(offsets_experiment_1, image_paths, labels_paths, data_dir)

    # 30th April
    #run_experiment(offsets_experiment_2, image_paths, labels_paths, data_dir)
    #run_experiment(offsets_experiment_3, image_paths, labels_paths, data_dir)

    # 3rd May 2021 - DL machine
    #run_experiment(affinities_exp, image_paths, labels_paths, data_dir) # fixed data normalisation
    #run_experiment(affinities_exp_0, image_paths, labels_paths, data_dir) # rerun last two - mixed up names
    #run_experiment(forked_exp, image_paths, labels_paths, data_dir) # fixed forked unet
    #run_experiment(mask_exp, image_paths, labels_paths, data_dir) # added mask

    # 5th May 2021 - DL machine
    #run_experiment(cirriculum_exp, image_paths, labels_paths, data_dir)
    #run_experiment(thresh_exp, image_paths, labels_paths, data_dir)
    #run_experiment(seed_exp, image_paths, labels_paths, data_dir)

    # 6th May 2021 - DL machine
    #run_experiment(cirriculum_exp_0, image_paths, labels_paths, data_dir)
    #run_experiment(affinities_exp_2, image_paths, labels_paths, data_dir)
    #run_experiment(lsr_exp, image_paths, labels_paths, data_dir)

    # 7th May 2021 - DL machine
    #run_experiment(thresh_exp_0, image_paths, labels_paths, data_dir)

    # 12th May 2021 - DL machine
    #run_experiment(lr_exp, image_paths, labels_paths, data_dir)
    run_experiment(loss_exp, image_paths, labels_paths, data_dir)
    # rerun & make sure using BCE loss
    #run_experiment(lsr_exp, image_paths, labels_paths, data_dir)
    #run_experiment(affinities_exp_2, image_paths, labels_paths, data_dir)
    #run_experiment(seed_exp, image_paths, labels_paths, data_dir)
    #run_experiment(thresh_exp_0, image_paths, labels_paths, data_dir) 
    #run_experiment(lsr_exp_mse, image_paths, labels_paths, data_dir)