#!/bin/bash
conda create -n netket_sharding_local python=3.11 
conda activate netket_sharding_local

pip install --upgrade pip
pip install jax
pip install mpi4py
pip install git+https://github.com/netket/netket
pip install git+https://github.com/NeuralQXLab/netket_pro
pip install h5py