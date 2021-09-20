import json
import os
from datetime import datetime
from pathlib import Path
now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")


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


def write_bash_script(save_dir, batch_name, script_path, f):
    p = Path(f)
    name = p.stem
    job_name = batch_name + '_' + name
    s = '#!/bin/bash\n'
    s = s + f'#SBATCH --job-name={job_name}\n'
    s = s + '#SBATCH --account=rl54\n'
    s = s + '#SBATCH --time=00:30:00\n'
    s = s + '#SBATCH --ntasks=1\n'
    s = s + '#SBATCH --mem=16G\n'
    s = s + '#SBATCH --cpus-per-task=1\n'
    s = s + '#SBATCH --gres=gpu:P100:1\n'
    s = s + '#SBATCH --partition=m3h\n'
    s = s + '#SBATCH --mail-user=Abigail.McGovern1@monash.edu\n'
    s = s + '#SBATCH --mail-type=ALL\n\n'
    s = s + 'source /projects/rl54/Abi/miniconda/bin/activate\n'
    s = s + 'conda activate dl-env\n'
    json_name = Path(f).stem + '_seg-info.json'
    json_path = os.path.join(save_dir, json_name)
    s = s + f'python {script_path} -i {json_path}\n'
    # save script
    script_name = Path(f).stem + '_job-script.sh'
    script_path = os.path.join(save_dir, script_name)
    with open(script_path, 'w') as bs:
        bs.write(s)
    return json_path


def write_json(json_path, out_dir, image_path, scratch_dir, unet_path, batch_name):
    info = {}
    info['out_dir'] = out_dir
    info['image_path'] = image_path
    info['scratch_dir'] = scratch_dir
    info['unet_path'] = unet_path
    info['batch_name'] = batch_name
    with open(json_path, 'w') as file_:
        json.dump(info, file_, indent=4)

# -------------
# MASSIVE Paths
# -------------

data_dir = ['/projects/rl54/data',]
out_dir = '/projects/rl54/results'
scratch_dir = '/fs03/rl54/'


# ----------------
# DL Machine Paths
# ----------------

#data_dir = ['/home/abigail/data/plateseg-training/timeseries_seg',]
#out_dir = '/home/abigail/data/plateseg-training/timeseries_seg'
#scratch_dir = '/home/abigail/data/plateseg-training/timeseries_seg'


# -----------
# Other Paths
# -----------

# get platelet-trackinf directory location
dir_path = Path(__file__).parent.resolve()
# get unet path (in unets subdirectory in platelet-tracking)
unet_dir = dir_path / 'unets' 
unet_path = unet_dir / '211309_151717_unet_z-1_y-1_x-1_m_c.pt'
unet_path = str(unet_path)
# get script path
script_path = dir_path / 'm3_segmentation_pipeline.py'
script_path = str(script_path)
# batch_name --> save_dir for job scripts and jsons
batch_name = dt + '_seg-track'
save_dir = os.path.join(out_dir, batch_name)
os.makedirs(save_dir, exist_ok=True)


# -------
# COMPUTE
# -------
files = get_files(data_dir)
for f in files:
    json_path = write_bash_script(save_dir, batch_name, script_path, f)
    write_json(json_path, out_dir, f, scratch_dir, unet_path, batch_name)