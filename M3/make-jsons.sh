#!/bin/bash
#SBATCH --job-name=MakeJsonsAndScripts
#SBATCH --account=rl54
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=Abigail.McGovern1@monash.edu
#SBATCH --mail-type=ALL

source /projects/rl54/Abi/miniconda/bin/activate
conda activate dl-env
python /projects/rl54/segmentation/unet_training/platelet-segmentation/m3_make_jsons.py