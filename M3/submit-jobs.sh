#!/bin/bash

cd /projects/rl54/results/210920_141056_seg-track
for f in *.sh;  do sbatch ${f}; done;