# Optimizer

This module contains some extra optimziers and preconditioners, and tools to work with them.


```{eval-rst}
.. currentmodule:: netket_pro

.. automodule:: netket_pro.optimizer
   :members:

```

## Preconditioners

```{eval-rst}
.. currentmodule:: netket_pro

.. automodule:: netket_pro.preconditioner
   :members:

```



## L-curve

This contains some tooling built by Luca to quickly look at the L-curve of an S- or K- matrix.


```{eval-rst}
.. currentmodule:: netket_pro.optimizer

.. autosummary::
   :toctree: _generated/optimizer
   :nosignatures:

   L_curve.locate_corner
   L_curve.lcurve_solver_srt

```

Note that the solver above can be used as a standard netket linear solver everywhere, and will autotune the diag shift.
