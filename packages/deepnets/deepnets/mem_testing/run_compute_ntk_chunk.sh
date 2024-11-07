#!/bin/bash
#SBATCH --job-name=chunk_test
#SBATCH --output=/mnt/beegfs/workdir/rajah.nutakki/slurm_bin/%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_v100
#SBATCH --account=ndqm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rajah.nutakki@polytechnique.edu

module purge
module load mambaforge
conda activate netket_sharding_local

export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=gpu

python /mnt/beegfs/workdir/rajah.nutakki/repos/netket_pro/deepnets/mem_testing/compute_ntk_chunk.py