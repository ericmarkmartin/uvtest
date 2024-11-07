#!/bin/bash
#Run multi-process cpu sharding
conda activate netket_sharding_local
run_file=/Users/rajah.nutakki/repos/netket_pro/deepnets/optimization/run.py
test_file=/Users/rajah.nutakki/repos/netket_pro/deepnets/multi/test_checkpoint.py
#param_str='--L 6 --J 0.8 1.0 --n_blocks 1 --features 24 --expansion_factor=2 --downsample_factor=2 --kernel_width=3 --output_head=Vanilla --samples_per_rank=100 --chains_per_rank=10 --discard_fraction=0.0 --iters 50  --lr 0.01 --alpha 1 --diag_shift 0.01 --diag_shift_end 1 --r=1e-06 --chunk_size=250 --save_every=10 --symmetries=0 --symmetry_ramping=0 --momentum=0.9 --double_precision=1 --time_it=0 --show_progress=1 --checkpoint=0 --seed=280 --save_base /Users/rajah.nutakki/Desktop/work_dummy/test/'
export NETKET_MPI=0
export NETKET_EXPERIMENTAL_SHARDING=1
export JAX_PLATFORM_NAME=cpu
export MULTI_PROCESS_CPU=1
#mpirun -np 2 python $run_file --L 6 --J 0.8 1.0 --n_blocks 1 --features 24 --expansion_factor=2 --downsample_factor=2 --kernel_width=3 --output_head=Vanilla --samples_per_rank=100 --chains_per_rank=10 --discard_fraction=0.0 --iters 50  --lr 0.01 --alpha 1 --diag_shift 0.01 --diag_shift_end 1 --r=1e-06 --chunk_size=250 --save_every=10 --symmetries=0 --symmetry_ramping=0 --momentum=0.9 --double_precision=1 --time_it=0 --show_progress=1 --checkpoint=0 --seed=280 --save_base /Users/rajah.nutakki/Desktop/work_dummy/test/
mpirun -np 2 python $test_file