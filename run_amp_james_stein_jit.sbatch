#!/bin/sh

## Sherlock deployment file that requests 4 cpu cores for 24 hours to run run_amp_blocksoft_jit.py
#SBATCH --job-name=run_amp_blocksoft_jit
#SBATCH --partition=donoho
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --error=run_amp_blocksoft_jit.err
#SBATCH --output=run_amp_blocksoft_jit.out


## Run the python script
python3 ./run_amp_james_stein_jit.py
