#!/bin/bash -l

#SBATCH --job-name=stack-lr01
#SBATCH -N 1
#SBATCH -n 1

#SBATCH --gres=gpu:1
#SBATCH -t 2-10:00:00

conda activate polybinder

python run.py -data_path "/home/marjanalbooyeh/logs/pps_rigid/2023-03-09-18:47:47/dataset" \
 -project "NN_multi_PPS" -group "10_pps" \
 -lr 0.1 -batch 128 -hidden_dim 64 -epochs 10000 -n_layer 3 -act_fn "Tanh" -mode "single" \
 -notes "appended input, dataset (neighbors) 1 row in data for all particles, num_compounds=10" \
 -inp_mode "append"

