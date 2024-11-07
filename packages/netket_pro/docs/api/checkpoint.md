# NetKet-Checkpoint

This is an extra package installed with ``NetKet-pro``, which is called ``NetKet-Checkpoint``.
It presents a simple API which re-implements the major NetKet drivers, hiding some complexities from the package
[orbax-checkpoint](https://orbax.readthedocs.io/en/latest/) used under the hood and from MPI.


If you want to use checkpointing, you should use the drivers in this package.
For a terse example, look at the examples in [this folder](https://github.com/NeuralQXLab/netket_pro/tree/main/examples/checkpoint),
where Ising1d will run with or without MPI, and the sharding example will run with sharding.

The usage is roughly as follows:

```python
import netket_checkpoint as nkc

...
gs = nkc.driver1.VMC(ha, op, variational_state=vs, preconditioner=sr)

log = nk.logging.JsonLog("test/test_r1", save_params=False)
ckpt = nkc.checkpoint.CheckpointManager(directory="/run1")
gs.run(
    n_iter=150,
    out=log,
    timeit=True,
    checkpointer=ckpt,
)

# You must create a new checkpointer for every new .run function call.
options = nkc.checkpoint.CheckpointManagerOptions(
    save_interval_steps=100, keep_period=20
)
ckpt = nkc.checkpoint.CheckpointManager(directory="/run2", options=options)
gs.run(
    n_iter=100,
    out=log,
    timeit=True,
    checkpointer=ckpt,
)

```

NetKet-Checkpoint works in principle with MPI and Sharding, though you must be careful that your code executes identically on all nodes.

## Checkpointers

This module includes the checkpointers

```{eval-rst}
.. currentmodule:: netket_checkpoint

.. autosummary::
   :toctree: _generated/checkpoint/checkpoint
   :nosignatures:
   :recursive:

   checkpoint.CheckpointManager
   checkpoint.CheckpointManagerOptions

```

## Drivers

The drivers with checkpointing are available in the driver subpackage of ``netket_checkpoint``. 
Below you find the documentation of the methods relevant to checkpointing that are added to those classes.

```{eval-rst}
.. currentmodule:: netket_checkpoint

.. autosummary::
   :toctree: _generated/checkpoint/driver
   :nosignatures:
   :recursive:

   driver1.AbstractVariationalDriver.run_checkpointed
   driver1.AbstractVariationalDriver._restore_checkpoint
   driver1.AbstractVariationalDriver._save_checkpoint

```
