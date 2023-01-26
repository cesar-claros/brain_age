#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu
#SBATCH --job-name=brain_age_prediction
#SBATCH --partition=cniel
#SBATCH --time=7-00:00:00
#SBATCH --mail-user='cesar@udel.edu'
#SBATCH --mail-type=ALL
#SBATCH --export=NONE
#SBATCH -D /work/cniel/sw/BrainAge/src
#UD_QUIET_JOB_SETUP=YES
#export UD_JOB_EXIT_FN_SIGNALS="SIGTERM EXIT"
#
# Add a TensorFlow container to the environment:
#
vpkg_devrequire intel-python/2022u1:python3
source activate /work/cniel/sw/BrainAge/venv
#
# Execute our Python script:
python main.py
# Synchronize runs in Wandb:
# cd /work/cniel/sw/BrainAge/
# wandb sync --include-synced --include-offline --sync-all