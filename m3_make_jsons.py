import json
import os
from datetime import datetime
from pathlib import Path
now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")

s = '#!/bin/bash\n'
s = s + '#SBATCH --job-name=FirstSegmentationAttempt\n'
s = s + '#SBATCH --account=rl54\n'
s = s + '#SBATCH --time=01:00:00\n'
s = s + '#SBATCH --ntasks=1\n'
s = s + '#SBATCH --mem=16G\n'
s = s + '#SBATCH --cpus-per-task=1\n'
s = s + '#SBATCH --gres=gpu:P100:1\n'
s = s + '#SBATCH --partition=m3h\n'
s = s + '#SBATCH --mail-user=Abigail.McGovern1@monash.edu\n'
s = s + '#SBATCH --mail-type=ALL\n\n'
s = s + 'source /projects/rl54/Abi/miniconda/bin/activate\n'
s = s + 'conda activate dl-env\n'

def get_files(dirs, ends='.nd2'):
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

data_dir = '/projects/rl54/data'
out_dir = '/projects/rl54/results'
scratch_dir = '/fs03/rl54/'
unet_path = ''
batch_name = dt + '_initial-retrain'
save_dir = os.path.join(data_dir, batch_name)
os.makedirs(save_dir, exist_ok=True)
files = get_files(data_dir)
for f in files:
    info = {}
    info['out_dir'] = out_dir
    info['image_path'] = f
    info['scratch_dir'] = scratch_dir
    info['unet_path'] = unet_path
    info['batch_name'] = batch_name
    fn = Path(f).stem + '_seg-info.json'
    fp = os.path.join(save_dir, fn)
    jobn = Path(f).stem + '_job-script.sh'
    cmd = f'python /projects/rl54/segmentation/unet_training/platelet-segmentation/m3_segmentation_pipeline.py -i {fp}'
    with open(fp, 'w') as file_:
        json.dump(info, file_)