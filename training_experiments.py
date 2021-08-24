from datetime import datetime
import os
import train
from train_io import get_train_data
import torch.nn as nn
from pathlib import Path


def run_experiment(
    experiment_dict, 
    image_paths, 
    labels_paths, 
    out_dir, 
    *args, 
    **kwargs
    ):
    gtd_kwargs = experiment_dict['get_train_data']
    train_dict = get_train_data(image_paths, labels_paths, out_dir, **gtd_kwargs)
    unets = {}
    for key in train_dict.keys():
        train_kwargs = train_dict[key]
        train_kwargs.update(experiment_dict[key])
        unet, unet_path = train.train_unet(**train_kwargs)
        unets[key] = {'unet': unet, 'unet_path' : unet_path}
    upper_dir = Path(train_kwargs['out_dir']).parents[1]
    unet_path_log = upper_dir / 'unet_paths.txt'
    s = [unets[key]['unet_path'] for key in unets.keys()]
    s = str(s)
    with open(unet_path_log, 'a') as f:
        f.write(s)
    return unets


#def train_experiment(
 #   experiments, 
  #  out_dir, 
   # image_paths, 
#    labels_path
 #   exp, 
  #  ):
   # exp_dir = os.path.join(out_dir, experiments[exp]['suffix'])
    #    unet = train.train_unet_get_labels(
     #           exp_dir, 
      #          image_paths, 
       #         labels_paths, 
        #        **experiments[exp]) 
       # exp['unet'] = unet


def get_experiment_dict(
    channels_list,
    condition_names,
    conditions_list=None,
    name='train-unet' ,
    validation_prop=0.2, 
    n_each=100,
    scale=(4, 1, 1),
    **kwargs
    ):
    # get the kwargs for obtaining the training data
    experiment = {}
    experiment['get_train_data'] = {
       'validation_prop' : validation_prop, 
        'n_each' : n_each, 
        'scale' : scale, 
        'name' : name, 
        'channels' : {}
    }
    for i, nm in enumerate(condition_names):
        experiment['get_train_data']['channels'][nm] = channels_list[i]
    # get the kwargs for training under each condition
    for i in range(len(condition_names)):
        experiment[condition_names[i]] = {
            'scale' : scale,
            'epochs' : 4,
            'lr' : .01,
            'loss_function' : 'BCELoss',
            'chan_weights' : None, 
            'weights' : None,
            'update_every' : 20, 
            'fork_channels' : None
        }
        if conditions_list is not None:
            custom_kw = conditions_list[i]
            for key in custom_kw.keys():
                experiment[condition_names[i]][key] = custom_kw[key]
    if 'mask' in experiment['get_train_data']['channels']:
        experiment['get_train_data']['absolute_thresh'] = 0.5
    return experiment


# -----------------
# Experiments
# -----------------

lsr_exp = get_experiment_dict(
    [('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['z-1s_y-1s_x-1s_m_cl', 'z-1_y-1_x-1_m_cl'], 
    name='label-smoothing-reg-exp'
)

affinities_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'mask', 'centreness-log'),
     ('z-1', 'z-3', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log')], 
    ['z-1_y-1_x-1_m_cl', 'z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 'z-1_z-2_z-3_y-1_y-2_y-3_x-1_x-2_x-3_m_cl'], 
     name='affinities-exp'
)

thresh_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'centreness', 'centreness-log')], 
    ['z-1_y-1_x-1_m_cl', 'z-1_y-1_x-1_c_cl'], 
    name='threshold-exp'
)

forked_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['z-1_y-1_x-1_m_cl', 'f3,2_z-1_y-1_x-1_m_cl'], 
    [{}, {'fork_channels': (3, 2)}], 
    name ='forked-exp'
)

seed_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'),
     ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')],
    ['z-1_y-1_x-1_m_c', 'z-1_y-1_x-1_m_cl', 'z-1_y-1_x-1_m_cg'], 
    name='seed-exp'
)

loss_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')],
    ['BCE_z-1_y-1_x-1_m_cl', 'DICE_z-1_y-1_x-1_m_cl'], 
    [{'loss_function' : 'BCELoss'}, {'loss_function' : 'DICELoss'}], 
    name='loss-exp'
)

lr_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'),
     ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['lr0-05_z-1_y-1_x-1_m_cl', 'lr0-01_z-1_y-1_x-1_m_cl', 'lr0-005_z-1_y-1_x-1_m_cl'], 
    [{'lr' : 0.05}, {'lr' : 0.01}, {'lr' : 0.005}], 
    name='learning-rate-exp'
)

mini_exp = get_experiment_dict(
    [('z-1', 'y-1', 'x-1', 'mask', 'centreness-log')], 
    ['z-1_y-1_x-1_m_c'], 
    [{'epochs' : 2}], 
    n_each=50, 
    name='mini-train-unet'
)
#forked_exp = {
 #       3: get_experiment_dict({
  #          'name' : 'f3,2_z-1_y-1_x-1_m_cl', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
    #        'fork_channels': (3, 2)
     #   }), 
#        5: get_experiment_dict({
 #           'name' : 'f6,2_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
  #          'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'mask', 'centreness-log'),
   #         'fork_channels': (6, 2) 
    #    }), 
     #   6: get_experiment_dict({
      #      'name' : 'f8,2_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
       #     'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
        #    'fork_channels': (8, 2)
#        })
#}

# the following is an experiment with different something that approximates cirriculum learning
#cirriculum_exp = {
 #       0: get_experiment_dict({
  #          'name' : 'EWBCE_uw0123_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
   #         'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
    #        'loss_function' : 'EpochWeightedBCE', 
     #       'chan_weights' : [
      #          [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
       #         [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
        #        [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
         #       [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]
          #  ]
       # }),
        # some cirriculum
      #  1: get_experiment_dict({
       #     'name' : 'EWBCE_uw012-w3_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
        #    'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
         #   'loss_function' : 'EpochWeightedBCE', 
          #  'chan_weights' : [
           #     [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
            #    [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
             #   [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
              #  [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.]
     #       ]
      #  }),
       # 2: get_experiment_dict({
        #    'name' : 'EWBCE_uw01-w23_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
         #   'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
          #  'loss_function' : 'EpochWeightedBCE', 
           # 'chan_weights' : [
            #    [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
             #   [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
     #           [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.],
      #          [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.]
       #     ]
        #}),
#        3: get_experiment_dict({
 #           'name' : 'EWBCE_uw0-w123_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_cl', 
  #          'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centreness-log'), 
   #         'loss_function' : 'EpochWeightedBCE', 
    #        'chan_weights' : [
     #           [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
      #          [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.],
       #         [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.],
        #        [3., 1., 3., 2., 1., 3., 2., 1., 2., 2.]
         #   ]
  #      }),
   # }


#cirriculum_exp_0 = {
 #       0: get_experiment_dict({
  #          'name' : 'EWBCE_uw0123_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
   #         'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
    #        'loss_function' : 'EpochWeightedBCE', 
     #       'chan_weights' : [
      #          [2., 2., 2., 2., 2., 2., 2., 2.],
       #         [2., 2., 2., 2., 2., 2., 2., 2.],
        #        [2., 2., 2., 2., 2., 2., 2., 2.],
         #       [2., 2., 2., 2., 2., 2., 2., 2.]
          #  ]
   #     }),
        # some cirriculum
#        1: get_experiment_dict({
 #           'name' : 'EWBCE_uw012-w3_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
  #          'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
   #         'loss_function' : 'EpochWeightedBCE', 
    #        'chan_weights' : [
     #           [2., 2., 2., 2., 2., 2., 2., 2.],
      #          [2., 2., 2., 2., 2., 2., 2., 2.],
       #         [2., 2., 2., 2., 2., 2., 2., 2.],
        #        [3., 1., 3., 1., 3., 1., 2., 2.]
         #   ]
#        }),
 #       2: get_experiment_dict({
  #          'name' : 'EWBCE_uw01-w23_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
   #         'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
    #        'loss_function' : 'EpochWeightedBCE', 
     #       'chan_weights' : [
      #          [2., 2., 2., 2., 2., 2., 2., 2.],
       #         [2., 2., 2., 2., 2., 2., 2., 2.],
        #        [3., 1., 3., 1., 3., 1., 2., 2.],
         #       [3., 1., 3., 1., 3., 1., 2., 2.]
          #  ]
#        }),
 #       3: get_experiment_dict({
  #          'name' : 'EWBCE_uw0-w123_z-1_z-2_y-1_y-2_x-1_x-2_m_cl', 
   #         'channels' : ('z-1', 'z-2','y-1', 'y-2', 'x-1', 'x-2','mask', 'centreness-log'), 
    #        'loss_function' : 'EpochWeightedBCE', 
     #       'chan_weights' : [
      #          [2., 2., 2., 2., 2., 2., 2., 2.],
       #         [3., 1., 3., 1., 3., 1., 2., 2.],
        #        [3., 1., 3., 1., 3., 1., 2., 2.],
         #       [3., 1., 3., 1., 3., 1., 2., 2.]
          #  ]
#        }),
 #   }


#thresh_exp = {
 #   0: get_experiment_dict({
  ##          'name' : 'thresh_z-1_y-1_x-1_c_cl', 
    #        'channels' : ('z-1', 'y-1', 'x-1', 'centreness', 'centreness-log'), 
     #   }),
  #  1: get_experiment_dict({
   #         'name' : 'thresh_z-1_y-1_x-1_m_cl', 
    #        'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
     #   }),
#}

#thresh_exp_0 = {
 #   0: get_experiment_dict({
  #          'name' : 'thresh_z-1_y-1_x-1_c_centg', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'centreness', 'centroid-gauss'), 
    #    }),
#    1: get_experiment_dict({
 #           'name' : 'thresh_z-1_y-1_x-1_m_centg', 
  #          'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'), 
   #     }),
#}

#seed_exp = {
 #   0: get_experiment_dict({
  #          'name' : 'seed_z-1_y-1_x-1_m_c', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness'), 
    #    }),
#    1: get_experiment_dict({
 #           'name' : 'seed_z-1_y-1_x-1_m_cl', 
  #          'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centreness-log'), 
   #     }),
#    2: get_experiment_dict({
 #           'name' : 'seed_z-1_y-1_x-1_m_centg', 
  #          'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'), 
   #     }),
#}


#affinities_exp_2 = {
 #   0: get_experiment_dict({
  #          'name' : 'aff_z-1_y-1_x-1_m_centg', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')
    #    }), 
#    1: get_experiment_dict({
 #           'name' : 'aff_z-1_z-2_y-1_y-2_x-1_x-2_m_centg', 
 #           'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'x-1', 'x-2', 'mask', 'centroid-gauss')
  #      }), 
   # 2: get_experiment_dict({
    #        'name' : 'aff_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_m_centg', 
     #       'channels' : ('z-1', 'z-2', 'y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'mask', 'centroid-gauss')
      #  })
#}

#lsr_exp = {
 #   0: get_experiment_dict({
  #          'name' : 'lsr_z-1_y-1_x-1_m_centg', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss')
    #    }), 
    #1#: get_experiment_dict({
     #       'name' : 'lsr_z-1s_y-1s_x-1s_m_centg', 
      #      'channels' : ('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'mask', 'centroid-gauss')
       # }), 
#}

#lsr_exp_mse = {
 #   0: get_experiment_dict({
  #          'name' : 'lsr-mse_z-1_y-1_x-1_m_centg', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
    #        'loss_function' : 'MSELoss'
     #   }), 
#    1: get_experiment_dict({
 #           'name' : 'lsr-mse_z-1s_y-1s_x-1s_m_centg', 
  #          'channels' : ('z-1-smooth', 'y-1-smooth', 'x-1-smooth', 'mask', 'centroid-gauss'),
   #         'loss_function' : 'MSELoss'
    #    }), 
#}



#loss_exp = {
 #   0: get_experiment_dict({
  #          'name' : 'loss-BCE_z-1_y-1_x-1_m_centg', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
    #        'loss_function' : 'BCELoss'
     #   }),
#    1: get_experiment_dict({
 #           'name' : 'loss-DICE_z-1_y-1_x-1_m_centg', 
  #          'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
   #         'loss_function' : 'DiceLoss'
    #    }),
   # 2: get_experiment_dict({
    #        'name' : 'loss-DICE_z-1_y-1_x-1_m_centg', 
    #        'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
    #        'loss_function' : 'MSELoss'
    #    })
#}

#lr_exp = {
 #   0: get_experiment_dict({
  #          'name' : 'lr-05_z-1_y-1_x-1_m_centg', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
    #        'lr' : 0.05
     #   }),
#    1: get_experiment_dict({
 #           'name' : 'lr-01_z-1_y-1_x-1_m_centg', 
  #          'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
   #         'lr' : 0.01
    #    }),
#    2: get_experiment_dict({
 #           'name' : 'lr-005_z-1_y-1_x-1_m_centg', 
  #          'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'),
   #         'lr' : 0.005
    #    }),
#}

#basic = {
 #   1: get_experiment_dict({
  #          'name' : 'basic_z-1_y-1_x-1_m_centg', 
   #         'channels' : ('z-1', 'y-1', 'x-1', 'mask', 'centroid-gauss'), 
    #    }),
#}

if __name__ == '__main__':
    # the following is an experiment with different forms of cirriculum learning 
#    weighted_exp = {
 #       # the following simply cuts to the chase and weighs harder tasks as more important
  #      3: get_experiment_dict({
   #         'name' : 'WBCE_1_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
    #        'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
     #       'loss_function' : 'WeightedBCE', 
      #      'chan_weights' : [2., 1., 2., 1., 1., 2., 1., 1., 1., 2.] # note / 14 not 12 as above
       # }), 
#        4: get_experiment_dict({
 #           'name' : 'WBCE_1_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
  #          'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
   #         'loss_function' : 'WeightedBCE', 
    #        'chan_weights' : [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.] # note / 10 
     #   })   
#    }

#   offsets_experiment = {
 #       1 : get_experiment_dict({
  #          'name' : 'f_BCE_z-1_y-1_x-1_oz_oy_ox', 
   #         'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x'), 
    #        'fork_channels': (3, 3), 
     #       'loss_function' : 'Channelwise',
      #      'losses' : [nn.BCELoss(), nn.MSELoss()], 
       #     'chan_losses' : [slice(0, 3), slice(3, 6)], 
        #    'chan_final_activations' : ['sigmoid', 'tanh']
    #    })
   # }

#    offsets_experiment_0 = {
 #       1 : get_experiment_dict({
  #          'name' : 'f_BCE_z-1_y-1_x-1_oz_oy_ox', 
   #         'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x'), 
    #        'loss_function' : 'Channelwise',
     #       'losses' : [nn.BCELoss(), nn.MSELoss()], 
      #      'chan_losses' : [slice(0, 3), slice(3, 6)], 
       #     'chan_final_activations' : ['sigmoid', 'sigmoid']
#        })
 #   }

#    offsets_experiment_1 = {
 #       1 : get_experiment_dict({
  #          'name' : 'f_BCE_z-1_y-1_x-1_oz_oy_ox', 
   #         'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x'), 
    #    })
#    }

   # offsets_experiment_2 = {
    #    0 : get_experiment_dict({
     #       'name' : 'BCE_z-1_y-1_x-1_oz_oy_ox_m', 
      #      'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
       # }), 
#        1 : get_experiment_dict({
 #           'name' : 'f6,1_BCE_z-1_y-1_x-1_oz_oy_ox_m', 
  #          'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
   #         'fork_channels' : (6, 1)
    #    }), 
     #   2 : get_experiment_dict({
      #      'name' : 'f3,4_BCE_z-1_y-1_x-1_oz_oy_ox_m', 
       #     'channels' : ('z-1', 'y-1','x-1', 'offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
        #    'fork_channels' : (3, 4)
#        }),
 #       3 : get_experiment_dict({
  #          'name' : 'BCE_oz_oy_ox_m', 
   #         'channels' : ('offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
    #    }),
     #   4 : get_experiment_dict({
      #      'name' : 'f3,1_BCE_oz_oy_ox_m', 
       #     'channels' : ('offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
        #    'fork_channels' : (3, 1)
#        }),
 #   }

    #offsets_experiment_3 = {
     #   0 : get_experiment_dict({
      #      'name' : 'cw3MSE,1BCE_oz_oy_ox_m', 
       #     'channels' : ('offsets-z', 'offsets-y', 'offsets-x', 'mask'), 
        #    'loss_function' : 'Channelwise',
         #   'losses' : [nn.MSELoss(), nn.BCELoss()], 
          #  'chan_losses' : [slice(0, 3), slice(3, None)], 
    #    }),
    #}

    # Directory for training data and network output 
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    # Directory for training data and network output 
    # data_dir = '/home/abigail/data/platelet-segmentation-training'
    # Path for original image volumes for which GT was generated
    image_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_image.zarr')] 
    # Path for GT labels volumes
    labels_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_labels.zarr')]
    
    # New code
    out_dir = '/Users/amcg0011/Data/pia-tracking/cang_training/mini-train'
    unets = run_experiment(mini_exp, image_paths, labels_paths, out_dir)
    
    