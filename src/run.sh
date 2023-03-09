#!/bin/bash -l

#SBATCH --job-name=stack-lr01
#SBATCH -N 1
#SBATCH -n 1

#SBATCH --gres=gpu:1
#SBATCH -t 2-10:00:00

conda activate polybinder

python run.py -lr 0.1 -batch 128 -hidden_dim 64 -epochs 10000 -n_layer 3 -act_fn "Tanh" -mode "single" \
 -notes "stacked input: [[rel_pos, r], [riemann_log_map]], dataset (neighbors) 1 data for two particle" \
 -inp_mode "stack"

