#To set up netket with mpi to run on amd gpus
module purge
module load cpuarch/amd # load this package only if you are running on A100
module load gcc/12.2.0 anaconda-py3 openmpi/4.1.5
  
conda create -y --name amd_gpu python=3.11 
conda activate amd_gpu

pip install --upgrade pip

# Remove mpi4py and mpi4jax from build cache
pip cache remove mpi4py

pip install --upgrade "jax[cuda12]" "nvidia-cudnn-cu12<9.4"
pip install --upgrade mpi4py 
pip install --upgrade git+https://github.com/netket/netket
pip install h5py
pip install -e /gpfswork/rech/iqu/uvm91ap/repos/netket_pro