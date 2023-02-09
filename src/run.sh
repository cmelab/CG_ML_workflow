#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 1

#SBATCH --gres=gpu:1
#SBATCH -t 2-10:00:00

conda activate polybinder

python run.py -lr 0.01 -batch 64 -hidden_dim 64 -epochs 10000 -n_layer 3 -mode "sweep" -project "NN_synthesized_Jan30"
