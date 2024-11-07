# Driver

This module contains some _finished_ self standing drivers that can be used as standalone algorithms or as part of larger algorithms.


## Ground-state algorithms

This is a vendored copy of NetKet's {class}`~netket.experimental.driver.VMC_SRt` with the Kackzmarz/SPRING optimizer from [ArXiV:2401.10190](https://arxiv.org/pdf/2401.10190.pdf) that was prototyped by Dian in [netket/netket#1708](https://github.com/netket/netket/pull/1708).
It seems to not improve much.
There mainly for experimentation.

```{eval-rst}
.. currentmodule:: netket_pro.driver

.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   VMC_SRt
   VMC_SRt_ntk
```

## State Matching

Those drivers can be used to match states one to another using Infidelity or L2L1 optimization.
A version of those codes with state-matching using a Midpoint scheme is also provided.

```{eval-rst}
.. currentmodule:: netket_pro.driver

.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   InfidelityOptimizer
   Infidelity_SRt
   Infidelity_SRt_ntk
   MidpointInfidelityOptimizer
   MidpointInfidelity_SRt
   MidpointInfidelity_SRt_ntk
   MidpointL2L1Optimizer
   L2L1Optimizer


```

## Overall dynamics

Drivers to do the whole dynamics

```{eval-rst}
.. currentmodule:: netket_pro.driver

.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

PTVMC
```
