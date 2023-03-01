#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 1

#SBATCH --gres=gpu:1
#SBATCH -t 2-10:00:00

conda activate polybinder

python run.py -lr 0.01 -batch 64 -hidden_dim 64 -epochs 20000 -n_layer 2 -mode "single" -notes "appended input: [rel_pos, r, riemann_log_map, sym_intrinsic_distance], dataset: 1 data per particle"
