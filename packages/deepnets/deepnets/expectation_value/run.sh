#!/bin/bash
#SBATCH --job-name=exp_value
###SBATCH --output=/mnt/beegfs/workdir/rajah.nutakki/slurm_bin/%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=ndqm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rajah.nutakki@polytechnique.edu

#Command line arguments
DIRECTORY=$1
N_SAMPLES_PER_CHAIN=$2
N_CHAINS=$3
N_DISCARD_PER_CHAIN=$4

module purge
module load mambaforge
conda activate netket_sharding_local

export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=gpu

run_file=/mnt/beegfs/workdir/rajah.nutakki/repos/netket_pro/deepnets/expectation_value/run.py

python $run_file --directory=$DIRECTORY --net="ConvNext" --n_samples_per_chain=$N_SAMPLES_PER_CHAIN --n_chains=$N_CHAINS --n_discard_per_chain=$N_DISCARD_PER_CHAIN