#!/bin/bash
#
#SBATCH --job-name=dropout_1
#SBATCH -p gpu
#SBATCH --gres=gpu  #a100 #is the fastest one but can also just use gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 3:00:00
#SBATCH -o ../slurm/slurm.%j.out
#SBATCH -e ../slurm/slurm.%j.err
#
hostname; date

source ~/.bashrc
conda activate test2

module load cuda/12

#export LD_LIBRARY_PATH=/ceph/apps/ubuntu-20/packages/cuda/11.2.0_460.27.04/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#export XLA_FLAGS= --xla_gpu_cuda_data_dir=/ceph/apps/ubuntu-20/packages/cuda/11.2.0_460.27.04
#export CUDA_DIR=/ceph/apps/ubuntu-20/packages/cuda/11.2.0_460.27.04

#export TF_XLA_FLAGS=--tf_xla_auto_jit=2 run_tem.py

python train.py

