#!/bin/bash -l

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p volta
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
conda activate polybinder

python run.py -lr 0.01 -batch 128 -hidden_dim 256 -epochs 100 -n_layer 7

