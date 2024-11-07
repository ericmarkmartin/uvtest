# Monkeypatch

This is an internal module that monkeypatches NetKet's classes (at the moment, mainly {class}`netket.vqs.MCState`) to add some useful methods to them necessary to implement some more complex algorithms.

You should not need to use this module directly when running simulations, but it might be very useful to implement some custom operators/observables or logic to work with Importance Sampling.

## Utilities

Those utilities can be used to add new features to an already defined class that you don't controll.


```{eval-rst}
.. currentmodule:: netket_pro.monkeypatch

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:
   :recursive:

	add_method
	attach_method
	attach_property
```

## New MCState methods

