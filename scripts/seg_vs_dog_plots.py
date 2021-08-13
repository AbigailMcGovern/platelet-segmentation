from plots import plot_experiment_APs, plot_experiment_no_diff, experiment_VI_plots
import os
# paths, names, title, out_dir, out_name
data_dir = '/Users/amcg0011/Data/pia-tracking/dl-results/210512_150843_seed_z-1_y-1_x-1_m_centg'
suffix = 'seed_z-1_y-1_x-1_m_centg'
out_dir = os.path.join(data_dir, 'DL-vs-Dog')
ap_paths = [os.path.join(data_dir, suffix + '_validation_AP.csv'), 
                os.path.join(data_dir, 'DoG-segmentation_average_precision.csv')]
nd_paths = [os.path.join(data_dir, 'seed_z-1_y-1_x-1_m_centg_validation_metrics.csv'), 
                os.path.join(data_dir, 'DoG-segmentation_metrics.csv')]
vi_paths = [
    os.path.join(data_dir, 'seed_z-1_y-1_x-1_m_centgvalidation_VI.csv'),
    os.path.join(data_dir, 'seed_z-1_y-1_x-1_m_centgvalidation_VI_DOG-seg.csv')
]
#plot_experiment_APs(ap_paths, ['DL', 'DoG'], 'Average precision: DL vs Dog', out_dir, 'AP_DL-vs-Dog')
#plot_experiment_no_diff(nd_paths, ['DL', 'DoG'], 'Number difference: DL vs Dog', out_dir, 'ND_DL-vs-Dog')
experiment_VI_plots(vi_paths, ['DL', 'DoG'], 'VI Subscores: DL vs DoG', 'VI_DL-vs-DoG', out_dir)