# NetKet Pro: Internal NetKet™ codes of the Quantum AI Group in Ecole Polytechnique

NetKet pro is the collection of codes developed by the group working on Neural-Network
methods for Quantum Many Body Physics (and beyond) at Ecole Polytechnique, in Paris/Saclay.

This code is not public, even though this documentation might be, for our own ease. 
It includes stuff we are working on, we worked on and we are experimenting with. 
Some of this might end up in NetKet one day, most of it won't.
If you see something you're interested in, get in touch.

In general this repository requires the latest release of NetKet, if not a recent commit from the master branch.


## Contents

Those are tools that can be used to run algorithms/research.

```{toctree}
:caption: NetKet Pro Reference Documentation
:maxdepth: 2

api/api
api/callbacks
api/driver
api/distributed
api/infidelity
api/jax
api/models
api/nn
api/operator
api/optimizer
api/sampler
api/testing
```

Here we have the updated version of MCState, with some additional methods related to 
sampling other distributions.
This also includes utilities to hack new functionalities onto NetKet (called monkeypatching).

```{toctree}
:caption: New NetKet functionalities
:maxdepth: 2

api/monkeypatch-vqs
api/monkeypatch
```

There is also a side package that reimplements the drivers while allowing for checkpointing.

```{toctree}
:caption: Additional 
:maxdepth: 2

api/checkpoint
```

And those are tools that can be used to implement new algorithms.

```{toctree}
:maxdepth: 2

```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
