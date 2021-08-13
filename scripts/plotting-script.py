from training_experiments import seed_exp, cirriculum_exp_0, affinities_exp_2, \
     thresh_exp_0, lr_exp, loss_exp
from compare_output import segmentation_plots, get_experiment_seg_info


data_dir = '/Users/amcg0011/Data/pia-tracking/dl-results'

# Get info with which to find files
seed_info = get_experiment_seg_info(data_dir, seed_exp)
#cirr_info = get_experiment_seg_info(data_dir, cirriculum_exp_0)
affi_info = get_experiment_seg_info(data_dir, affinities_exp_2)
thre_info = get_experiment_seg_info(data_dir, thresh_exp_0)
lera_info = get_experiment_seg_info(data_dir, lr_exp)
loss_info = get_experiment_seg_info(data_dir, loss_exp)

# Get plots for each experiment
segmentation_plots(data_dir, seed_info, 'Seed Experiment', 'seed-exp')
#segmentation_plots(data_dir, seed_info, 'Cirriculum Experiment', 'cirriculum-exp')
segmentation_plots(data_dir, affi_info, 'Affinities Experiment', 'affinities-exp')
segmentation_plots(data_dir, thre_info, 'Thresholding Experiment', 'thresholding-exp')
segmentation_plots(data_dir, lera_info, 'Learning Rate Experiment', 'learning-rate-exp')
segmentation_plots(data_dir, loss_info, 'Loss Experiment', 'loss-exp')