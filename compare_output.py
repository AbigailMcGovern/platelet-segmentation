import os
import re
from segmentation import segment_from_directory
from plots import experiment_VI_plots, VI_plot

def segment_experiment(
        data_dir, 
        experiments,
        w_scale=None, 
        compactness=0.,
        display=True, 
    ):
    seg_info = _get_experiment_seg_info(data_dir, experiments, 
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
                                               validation=False)
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


def _get_experiment_seg_info(
        data_dir, 
        experiments, 
        w_scale,
        compactness, 
        display,
        centroid_opt=('centreness-log', 'centreness'), 
        thresh_opt=('centreness', 'centreness-log'),
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
    paths = []
    names = []
    for key in seg_info.keys():
        dir_ = seg_info[key]['directory']
        suffix = seg_info[key]['suffix']
        if validation:
            s = 'validation_VI.csv'
        else:
            s = '_VI.csv'
        p = os.path.join(dir_, suffix + s)
        if not os.path.exists(p):
            raise ValueError(f'Cannot find path {p}')
        paths.append(p)
        n = seg_info[key]['suffix']
        names.append(n)
    return paths, names


if __name__ == '__main__':
    from training_experiments import affinities_exp
    data_dir = '/home/abigail/data/platelet-segmentation-training'
    seg_info = segment_experiment(data_dir, affinities_exp)
    segmentation_VI_plots(data_dir, 
                          seg_info, 
                          'Affinities experiment (20/04/21)', 
                          '210420_affinities')
