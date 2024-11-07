(operator_api)=
# Operator

```{eval-rst}
.. currentmodule:: netket_pro.operator

```

This module vendors a few NetKet-compliant operators.

At the moment, those are just a few useful to compose a quantum circuit, and they all are jax-compatible.


```{eval-rst}
.. currentmodule:: netket_pro.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   Rx
   Ry
   Hadamard
```

## Fermionic operators

Specialized operators (much faster than the standard versions in NetKet).

```{eval-rst}
.. currentmodule:: netket_pro.operator

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   ParticleNumberConservingFermioperator2ndJax
   ParticleNumberConservingFermioperator2ndSpinJax
```