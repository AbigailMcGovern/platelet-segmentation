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



if __name__ == '__main__':
    # Define experiments dictionary here:
    experiments = {
        0: get_experiment_dict({
            'name' : 'z-1_y-1_x-1', 
            'channels' : ('z-1', 'y-1', 'x-1'), 
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
            'name' : 'z-1_y-1_x-1__wBCE2-1-1', 
            'channels' : ('z-1', 'y-1', 'x-1'), 
            'loss_function' : 'WeightedBCE', 
            'chan_weights' : (2., 1., 1.) 
        }) # that's enough for now (lol, this is like 10 hours of compute)
    } 
    # Directory for training data and network output 
    data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    # Path for original image volumes for which GT was generated
    image_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_image.zarr')] 
    # Path for GT labels volumes
    labels_paths = [os.path.join(data_dir, '191113_IVMTR26_I3_E3_t58_cang_training_labels.zarr')]
    # Run the experiments
    run_experiment(experiments, image_paths, labels_paths, data_dir)