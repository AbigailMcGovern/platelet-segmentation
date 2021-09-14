import logging
import argparse
import os
from training_experiments import run_experiment
from datetime import datetime

now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")
LOG_NAME = f'/projects/rl54/segmentation/unet_training/training_experiments/{dt}_training-log.log'
logging.basicConfig(filename=LOG_NAME, encoding='utf-8', level=logging.DEBUG)

def get_files(dirs, ends='.zarr'):
    files = []
    for d in dirs:
        for sub in os.walk(d):
            if ends.endswith('.zarr'):
                if sub[0].endswith(ends):
                    files.append(sub[0])
            for fl in sub[2]:
                f = os.path.join(sub[0], fl)
                if f.endswith(ends):
                    files.append(f)
    return files

try:
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dirs', nargs='*', help='the directories in which to find data')
    p.add_argument('-s', '--savedir', help='directory into which output will be saved')
    h = 'type of experiment (mini, threshold, seed, affinities, lr, lsr, forked) or path to experiment JSON'
    p.add_argument('-e', '--experiment', help=h)
    args = p.parse_args()
    out_dir = args.savedir
    now = datetime.now()
    dt = now.strftime("%y%m%d_%H%M%S")
    LOG_NAME = os.path.join(out_dir, f'{dt}_training-log.log')
    logging.basicConfig(filename=LOG_NAME, encoding='utf-8', level=logging.DEBUG)
    dirs = args.dirs
    logging.debug(str(dirs))
    image_paths = get_files(dirs, ends='_image.zarr')
    logging.debug(str(image_paths))
    gt_paths = get_files(dirs, ends='_labels.zarr')
    logging.debug(str(gt_paths))
    logging.debug(out_dir)
    exp = args.experiment
    logging.debug(exp)
    assert isinstance(exp, str)
    if exp == 'basic':
        from training_experiments import basic_exp as experiment
    if exp == 'mini':
        from training_experiments import mini_exp as experiment
    elif exp == 'threshold':
        from training_experiments import thres_exp as experiment
    elif exp == 'seed':
        from training_experiments import seed_exp as experiment
    elif exp == 'affinities':
        from training_experiments import affinities_exp as experiment
    elif exp == 'lr':
        from training_experiments import lr_exp as experiment
    elif exp == 'lsr':
        from training_experiments import lsr_exp as experiment
    elif exp == 'forked':
        from training_experiments import forked_exp as experiment
    elif os.path.exists(exp) and exp.endswith('.json'):
        from training_experiments import get_experiment_dict
        import json
        with open(exp) as f:
            kwargs = json.load(f)
        experiment = get_experiment_dict(**kwargs)
    else:
        from training_experiments import basic_exp as experiment
    logging.debug(str(experiment))
    unets = run_experiment(experiment, image_paths, gt_paths, out_dir)
except:
    logging.exception('Got an exeption')
    raise
# add code for producing diagnostic and comparison plots