import os
import re


LINE = '------------------------------------------------------------'

def get_files(
              data_dir, 
              x_regex=r'\d{6}_\d{6}_\d{1,3}_image.tif', 
              y_regex=r'\d{6}_\d{6}_\d{1,3}_labels.tif'
              ):
    '''
    '''
    files = os.listdir(data_dir)
    x_paths = get_paths(
                        data_dir, 
                        regex=x_regex, 
                        )
    y_paths = get_paths(
                        data_dir, 
                        regex=y_regex, 
                        )
    m = 'There is a mismatch in the number of images and training labels'
    assert len(x_paths) == len(y_paths), m
    return x_paths, y_paths


def get_paths(
              data_dir, 
              regex=r'\d{6}_\d{6}_\d{1,3}_output.tif'
              ):
    '''
    Awesome default for regex... just so lazy !!
    '''
    files = os.listdir(data_dir)
    paths = []
    pattern = re.compile(regex)
    for f in files:
        match = pattern.search(f)
        if match is not None:
            paths.append(os.path.join(data_dir, match[0]))
    return paths


def write_log(string, out_dir, log_name='log.txt'):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, log_name), 'a') as log:
        log.write(string + '\n')


def log_dir_or_None(log, out_dir):
    if log:
        return out_dir
    else:
        return None