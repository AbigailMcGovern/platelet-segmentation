from datetime import datetime
import os
import train

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
            'scale ' : (4, 1, 1),
            'epochs' : 4,
            'lr' : .01,
            'loss_function' : 'BCELoss',
            'chan_weights' : None, 
            'weights' : None,
            'update_every' : 20
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
    # add the suffix
    experiment['suffix'] = suffix
    return experiment


# -----------------
# Measuing outcomes
# -----------------


# segmentation
# metric for label similarity??
# examine relationship between channels?

if __name__ == '__main__':
    # Define experiments dictionary here:
    experiments = {
        #0: get_experiment_dict({
         #   'name' : 'z-1_y-1_x-1', 
          #  'channels' : ('z-1', 'y-1', 'x-1'), 
        #}), 
        #1: get_experiment_dict({
         #   'name' : 'z-1_y-1_x-1_c', 
          #  'channels' : ('z-1', 'y-1', 'x-1', 'centreness'), 
        #}), 
        #2: get_experiment_dict({
         #   'name' : 'z-1_y-1_x-1_cl', 
          #  'channels' : ('z-1', 'y-1', 'x-1', 'centreness-log'), 
        #}), 
        #3: get_experiment_dict({
           # 'name' : 'z-1_y-1_x-1_c_cl', 
           # 'channels' : ('z-1', 'y-1', 'x-1', 'centreness', 'centreness-log'), 
       # }), 
        4: get_experiment_dict({
            'name' : 'z-1_y-1_x-1__wBCE2-1-1', 
            'channels' : ('z-1', 'y-1', 'x-1'), 
            'loss_function' : 'WeightedBCE', 
            'chan_weights' : (2., 1., 1.) 
        }), 
         
    } 
    cirriculum_exp0 = {
        # moderate cirriculum
        0: get_experiment_dict({
            'name' : 'EWBCE_0_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [0., 3., 0., 1., 2., 0., 1., 2., 2., 1.],
                [1., 2., 1., 1., 1., 1., 1., 1., 2., 1.],
                [2., 1., 2., 1., 0., 2., 1., 0., 1., 2.],
                [3., 0., 3., 0., 0., 3., 0., 0., 0., 3.]
            ]
        })
    }
    cirriculum_exp1 = {
        # graded cirriculum
        0: get_experiment_dict({
            'name' : 'EWBCE_1_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.],
                [3., 1., 3., 2., 1., 3., 2., 1., 1., 3.],
                [3., 1., 3., 2., 1., 3., 2., 1., 1., 3.]
            ]
        })
    }
    # the following is an experiment with different forms of cirriculum learning 
    cirriculum_exp = {
        # moderate cirriculum
        0: get_experiment_dict({
            'name' : 'EWBCE_0_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [0., 3., 0., 1., 2., 0., 1., 2., 2., 1.],
                [1., 2., 1., 1., 1., 1., 1., 1., 2., 1.],
                [2., 1., 2., 1., 0., 2., 1., 0., 1., 2.],
                [3., 0., 3., 0., 0., 3., 0., 0., 0., 3.]
            ]
        }),
        # most extreme cirriculum
        1: get_experiment_dict({
            'name' : 'EWBCE_1_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [0., 3., 0., 0., 3., 0., 0., 3., 3., 0.],
                [1., 2., 0., 1., 2., 0., 1., 2., 2., 1.],
                [2., 1., 2., 1., 0., 2., 1., 0., 1., 2.],
                [3., 0., 3., 0., 0., 3., 0., 0., 0., 3.]
            ]
        }),
        # Most gradual cirriculum
        2: get_experiment_dict({
            'name' : 'EWBCE_1_z-1_z-2_y-1_y-2_y-3_x-1_x-2_x-3_c_cl', 
            'channels' : ('z-1', 'z-2','y-1', 'y-2', 'y-3', 'x-1', 'x-2', 'x-3', 'centreness', 'centreness-log'), 
            'loss_function' : 'EpochWeightedBCE', 
            'chan_weights' : [
                [1., 2., 0., 1., 2., 0., 1., 2., 2., 1.],
                [1., 2., 1., 1., 1., 1., 1., 1., 2., 1.],
                [2., 1., 1., 1., 1., 1., 1., 1., 1., 2.],
                [2., 1., 2., 1., 0., 2., 1., 0., 1., 2.]
            ]
        }),
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
    # Directory for training data and network output 
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    # Path for original image volumes for which GT was generated
    image_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_image.zarr')] 
    # Path for GT labels volumes
    labels_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_labels.zarr')]
    # Run the experiments
    #run_experiment(experiments, image_paths, labels_paths, data_dir)
    run_experiment(cirriculum_exp1, image_paths, labels_paths, data_dir)