import argparse
import os
from training_experiments import run_experiment


p = argparse.ArgumentParser()
p.add_argument('-i', '--images', nargs='*', help='the image paths to process')
p.add_argument('-g', '--groundtruth', nargs='*', help='the ground truth paths to process')
p.add_argument('-s', '--savedir', help='directory into which output will be saved')
h = 'type of experiment (mini, threshold, seed, affinities, lr, lsr, forked) or path to experiment JSON'
p.add_argument('-e', '--experiment', help=h)
args = p.parse_args()
image_paths = args.images
gt_paths = args.groundtruth
out_dir = args.savedir
exp = args.experiment
assert isinstance(exp, str)
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
    from training_experiments import mini_exp as experiment

unets = run_experiment(experiment, image_paths, gt_paths, out_dir)

# add code for producing diagnostic and comparison plots