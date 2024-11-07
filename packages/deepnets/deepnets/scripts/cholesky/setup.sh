module purge
module load mambaforge
mamba create -n netket_sharding_local python=3.11
conda activate netket_sharding_local

pip install --upgrade pip
pip install --upgrade "jax[cuda]" 'nvidia-cudnn-cu12<9.4'
pip install mpi4py
pip install git+https://github.com/netket/netket
pip install -e /mnt/beegfs/workdir/rajah.nutakki/repos/netket_pro
pip install h5py

conda deactivate