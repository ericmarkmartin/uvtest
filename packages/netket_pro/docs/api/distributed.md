# Distributed

This module contains several utilities to write algorithms that run on multiple GPUs and/or nodes.

The idea behind this module is that code written using `nkp.distributed` will correctly run both under MPI and sharding, as opposed to writing code only for one of the two paradigms.

Also, `nkp.distributed` will do nothing if you are not using more than 1 GPU, which limits the possible jax bugs arising, and simplifies serialization of variational states.

Examples about its usage can be mainly found in the MinSR/SRt and Infidelity optimisation drivers.

```{eval-rst}
.. currentmodule:: netket_pro.distributed

.. autosummary::
   :toctree: _generated/distributed
   :nosignatures:
   :recursive:

process_count
process_index
device_count
mode
broadcast_key
broadcast
allgather
pad_axis_for_sharding
reshard
_inspect

```

In general, try to use `nkp.distributed.process_count` and `nkp.distributed.process_index` when checking how many devices are available, instead of alternatives. 
Especially when doing serialization work. This will always work!

