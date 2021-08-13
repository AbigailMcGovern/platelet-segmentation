import os
from os import path
import re
from segmentation import segment_from_directory
from plots import experiment_VI_plots, VI_plot, plot_experiment_APs, plot_experiment_no_diff

def segment_experiment(
        data_dir, 
        experiments,
        w_scale=None, 
        compactness=0.,
        display=True, 
    ):
    seg_info = get_experiment_seg_info(data_dir, experiments, 
                                        w_scale, compactness, 
                                        display)
    for key in seg_info.keys():
        nm = seg_info[key]['suffix']
        print(f'Segmenting {nm}')
        segment_from_directory(validation=False, **seg_info[key])
        segment_from_directory(validation=True, **seg_info[key])
    
    return seg_info
    
def segmentation_VI_plots(data_dir, seg_info, exp_name, out_name):
    vi_train_paths, vi_train_names = _get_VI_paths(data_dir, 
                                                   seg_info, 
                                                   validation=False)
    vi_val_paths, vi_val_names = _get_VI_paths(data_dir, 
                                               seg_info, 
                                               validation=True)
    for p in vi_train_paths:
        VI_plot(p, lab='_train')
    for p in vi_val_paths:
        VI_plot(p, lab='_val')
    out_dir = os.path.join(data_dir, out_name + '_VI_plots')
    experiment_VI_plots(vi_train_paths, 
                        vi_train_names, 
                        f'Training output VI Scores: {exp_name}', 
                        out_name + '_train', 
                        out_dir)
    experiment_VI_plots(vi_val_paths, 
                        vi_val_names, 
                        f'Test output VI Scores: {exp_name}', 
                        out_name + '_val', 
                        out_dir)


def segmentation_plots(data_dir, seg_info, exp_name, out_name):
    vi_paths, vi_names = _get_VI_paths(data_dir, 
                                               seg_info, 
                                               validation=True)
    ap_paths, ap_names = _get_AP_paths(data_dir, seg_info)
    nd_paths, nd_names = _get_IoU_paths(data_dir, seg_info)
    out_dir = os.path.join(data_dir, out_name)
    experiment_VI_plots(vi_paths, 
                        vi_names, 
                        f'Test output VI Scores: {exp_name}', 
                        out_name + '_val_VI', 
                        out_dir)
    plot_experiment_APs(ap_paths, 
                        ap_names, 
                        f'Average precision: {exp_name}', 
                        out_dir, 
                        out_name + '_val_AP')
    plot_experiment_no_diff(nd_paths, 
                            nd_names, 
                            f'Number difference: {exp_name}', 
                            out_dir, 
                            out_name + '_val_num-diff')



def get_experiment_seg_info(
        data_dir, 
        experiments, 
        w_scale=None, 
        compactness=0.,
        display=True,
        centroid_opt=('centreness-log', 'centreness', 'centroid-gauss'), 
        thresh_opt=('mask', 'centreness', 'centreness-log'),
        z_aff_opt=('z-1', 'z-1-smooth'),
        y_aff_opt=('y-1', 'y-1-smooth'), 
        x_aff_opt=('x-1', 'x-1-smooth'), 
        date='recent'
    ):
    seg_info = {}
    for key in experiments.keys():
        n = experiments[key]['name']
        regex = re.compile( r'\d{6}_\d{6}_' + n)
        files = os.listdir(data_dir)
        matches = []
        for f in files:
            mo = regex.search(f)
            if mo is not None:
                matches.append(mo[0])
        if date == 'recent':
            seg_dir = sorted(matches)[-1]
        else: # specific date range ??
            pass
        exp_dir = os.path.join(data_dir, seg_dir)
        if os.path.exists(exp_dir):
            seg_info[key] = {}
            # find the prefered channels for segmenting
            chans = experiments[key]['channels']
            cent_chan = _get_index(chans, centroid_opt)
            seg_info[key]['centroids_channel'] = cent_chan
            thresh_chan = _get_index(chans, thresh_opt)
            seg_info[key]['thresholding_channel'] = thresh_chan
            z_chan = _get_index(chans, z_aff_opt)
            y_chan = _get_index(chans, y_aff_opt)
            x_chan = _get_index(chans, x_aff_opt)
            seg_info[key]['affinities_channels'] = (z_chan, y_chan, x_chan)
            # FIX THIS!!! Need to use name and add a date/date range option
            # raise ValueError(f'path {exp_dir} does not exist')
            seg_info[key]['directory'] = exp_dir
            seg_info[key]['suffix'] = experiments[key]['name']
            seg_info[key]['scale'] = experiments[key]['scale']
            seg_info[key]['w_scale'] = w_scale
            seg_info[key]['compactness'] = compactness
            seg_info[key]['display'] = display
        else:
            print(f'path {exp_dir} does not exist')
    return seg_info
    

def _get_index(chans, opts):
    idx = []
    for opt in opts:
        for i, c in enumerate(chans):
            if c == opt:
                idx.append(i)
    if len(idx) == 0:
        raise ValueError(f'No channel in {chans} matches the options {opts}')
    return idx[0]


def _get_VI_paths(data_dir, seg_info, validation=False):
    if validation:
        s = 'validation_VI.csv'
    else:
        s = '_VI.csv'
    paths, names = _get_data_paths(data_dir, seg_info, s)
    return paths, names


def _get_AP_paths(data_dir, seg_info, validation=True):
    if validation:
        s = '_validation_AP.csv'
    else:
        s = '_test_AP.csv'
    paths, names = _get_data_paths(data_dir, seg_info, s)
    return paths, names
        

def _get_IoU_paths(data_dir, seg_info, validation=True):
    if validation:
        s = '_validation_metrics.csv'
    else:
        s = '_test_metrics.csv'
    paths, names = _get_data_paths(data_dir, seg_info, s)
    return paths, names


def _get_data_paths(data_dir, seg_info, s):
    paths = []
    names = []
    for key in seg_info.keys():
        dir_ = seg_info[key]['directory']
        suffix = seg_info[key]['suffix']
        p = os.path.join(dir_, suffix + s)
        if not os.path.exists(p):
            raise ValueError(f'Cannot find path {p}')
        paths.append(p)
        n = seg_info[key]['suffix']
        names.append(n)
    return paths, names


# ---------------------------
# DoG Segmentation Comparison
# ---------------------------

def compare_with_DoG(
        directory, 
        suffix,
        affinities_channels, 
        centroids_channel, 
        thresholding_channel, 
        dog_config
        ):
    #segment_from_directory(directory, suffix, affinities_channels, 
                     #       centroids_channel, thresholding_channel, 
                      #      validation=False, dog_config=dog_config)
    segment_from_directory(directory, suffix, affinities_channels, 
                            centroids_channel, thresholding_channel, 
                            validation=True, dog_config=dog_config, save=True)
    s = ('validation_VI', '_VI')
    # training data
    train_path = os.path.join(directory, suffix + s[1] + '.csv')
    train_path_dog = os.path.join(directory, suffix + s[1] + '_DOG-seg' + '.csv')
    vi_train_paths = [train_path, train_path_dog]
    vi_train_names = ['DL Segmentation', 'DoG Segmentation']
    experiment_VI_plots(vi_train_paths, 
                        vi_train_names, 
                        'Training output VI Scores: Comparison with DoG', 
                        'DoG-comparison_train', 
                        directory)
    # validation data
    val_path = os.path.join(directory, suffix + s[0] + '.csv')
    val_path_dog = os.path.join(directory, suffix + s[0] + '_DOG-seg' + '.csv')
    vi_val_paths = [val_path, val_path_dog]
    vi_val_names = ['DL Segmentation', 'DoG Segmentation']
    experiment_VI_plots(vi_val_paths, 
                        vi_val_names, 
                        'Test output VI Scores: Comparison with DoG', 
                        'DoG-comparison_val', 
                        directory)



# -----------
# Main Script
# -----------
if __name__ == '__main__':
    from training_experiments import affinities_exp, mask_exp, forked_exp, cirriculum_exp, \
        thresh_exp, seed_exp, cirriculum_exp_0, affinities_exp_2, lsr_exp, thresh_exp_0, \
            lr_exp, loss_exp, lsr_exp_mse
    data_dir = '/home/abigail/data/platelet-segmentation-training'

    # -----------------------
    # Experiment segmentation
    # -----------------------

    seg_info = segment_experiment(data_dir, lr_exp, display=False)
    seg_info = segment_experiment(data_dir, loss_exp, display=False)
    seg_info = segment_experiment(data_dir, thresh_exp_0, display=False)
    seg_info = segment_experiment(data_dir, seed_exp, display=False)
    seg_info = segment_experiment(data_dir, affinities_exp_2, display=False)
    seg_info = segment_experiment(data_dir, cirriculum_exp_0, display=False)

    # --------------------------
    # Experiment raincloud plots
    # --------------------------

    #seg_info = _get_experiment_seg_info(data_dir, loss_exp)
    #segmentation_VI_plots(data_dir, 
                  #        seg_info, 
                  #        'Loss experiment (13/05/21)', 
                  #        '210513_losses')

    
    # --------------------------------
    # Comparison with DoG Segmentation
    # --------------------------------

    #directory = os.path.join(data_dir, '210513_131426_loss-BCE_z-1_y-1_x-1_m_centg')
    #directory = os.path.join(data_dir, '210512_150843_seed_z-1_y-1_x-1_m_centg')
    #suffix = 'seed_z-1_y-1_x-1_m_centg'
    #affinities_channels = (0, 1, 2)
    #centroids_channel = 4
    #thresholding_channel = 3
    #dog_config = dict(
           # dog_sigma1 = 1.4,
           # dog_sigma2 = 1.7,
           # threshold = 0.15,
           # peak_min_dist = 3,
            #)
    #compare_with_DoG(directory, suffix, affinities_channels, 
            #         centroids_channel, thresholding_channel, 
            #         dog_config)


    # ---------------------
    # Get DoG vs DL Metrics
    # ---------------------
    #directory = os.path.join(data_dir, '210512_150843_seed_z-1_y-1_x-1_m_centg')
    #from helpers import get_dataset_segs
    #import umetrics
    #from segmentation import metrics_for_stack
    #gt, seg, dog = get_dataset_segs(directory)
    #gt_0 = gt[0].compute()
    #seg_0 = seg[0].compute()
    #dog_0 = dog[0].compute()
    #result = umetrics.calculate(gt_0, seg_0, strict=True, iou_threshold=0.5)
    #print(result.results)
    #result = umetrics.calculate(gt_0, dog_0, strict=True, iou_threshold=0.5)
    #print(result.results)
    #df_s, ap_s = metrics_for_stack(directory, 'z-1_y-1_x-1_m_centg', seg, gt)
    #df_d, ap_d = metrics_for_stack(directory, 'DoG-segmentation', dog, gt)
    #print(df_s['n_diff'].mean(), '+/-', df_s['n_diff'].sem(), ' vs ', df_d['n_diff'].mean(),'+/-', df_d['n_diff'].sem())

    # ----------------------
    # Plot average precision
    # ----------------------

    #import matplotlib.pyplot as plt
    #import pandas as pd
    #plt.rcParams.update({'font.size': 16})
    #plt.rcParams["figure.figsize"] = (10,10)
    #fig = plt.figure()
    #ap_s = pd.read_csv(os.path.join(directory, 'z-1_y-1_x-1_m_centg_average_precision.csv'))
    #ap_d = pd.read_csv(os.path.join(directory, 'DoG-segmentation_average_precision.csv'))
    #df_s = pd.read_csv(os.path.join(directory, 'z-1_y-1_x-1_m_centg_metrics.csv'))
    #df_d = pd.read_csv(os.path.join(directory, 'DoG-segmentation_metrics.csv'))
    #plt.plot(ap_s['threshold'].values, ap_s['average_precision'].values)
    #plt.plot(ap_d['threshold'].values, ap_d['average_precision'].values)
    #plt.xlabel('IoU threshold')
    #plt.ylabel('Average precision')
    #plt.legend(['DL', 'DoG'])
    #p = os.path.join(directory, 'Average_precision_plot.png')
    #fig.savefig(p)
    #plt.show()
    
    # -------------
    # Other metrics
    # -------------

    #thresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    #piap_s = []
    #piap_d = []
    #for t in thresh:
        #n = f't{t}_per_image_average_precision'
        #n = f't{t}_Jaccard'
        #n = f't{t}_IoU'
        #n = f't{t}_pixel_identity'
        #n = f't{t}_localization_error'
        #piap_s.append(df_s[n].mean())
        #piap_d.append(df_d[n].mean())
    #plt.plot(thresh, piap_s)
    #plt.plot(thresh, piap_d)
    #plt.show()

