#!/bin/bash

cd 
for f in *.sh;  do sbatch ${f}; done;