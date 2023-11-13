#!/bin/sh

## Sherlock deployment file that requests 4 cpu cores for 24 hours to run run_amp_normal_bayes_jit.py
#SBATCH --job-name=run_amp_normal_bayes_jit
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --error=run_amp_normal_bayes_jit.err
#SBATCH --output=run_amp_normal_bayes_jit.out


## Run the python script
python3 ./run_amp_normal_bayes_jit.py
