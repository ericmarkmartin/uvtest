# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

print("\n---------------------------------------------", flush=True)
print("Initializing JAX distributed using GLOO...", flush=True)
import jax

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
jax.config.update("jax_cpu_enable_gloo_collectives", True)
jax.distributed.initialize(cluster_detection_method="mpi4py")

default_string = f"r{jax.process_index()}/{jax.process_count()} - "
print(default_string, jax.devices(), flush=True)
print(default_string, jax.local_devices(), flush=True)
print("---------------------------------------------\n", flush=True)

import netket as nk
import optax
from netket import experimental as nkx
import netket_checkpoint as nkc

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)

# Ising spin hamiltonian
ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=3, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer with a decreasing learning rate
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))

# SR
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=512, n_discard_per_chain=10, seed=1)

# Variational monte carlo driver with a variational state
gs = nkc.driver1.VMC_SRt(ha, op, variational_state=vs, diag_shift=0.01)

# Run the optimization for 500 iterations
log = nk.logging.JsonLog("/tmp/ckp22", save_params=False)

options = nkc.checkpoint.CheckpointManagerOptions(save_interval_steps=5, keep_period=20)
ckpt = nkc.checkpoint.CheckpointManager(directory="/tmp/ckp2", options=options)

gs.run_checkpointed(
    n_iter=200,
    out=log,
    timeit=True,
    checkpointer=ckpt,
)

gs = nkc.driver1.VMC_SRt(ha, op, variational_state=vs, diag_shift=0.01)
print(default_string, vs.expect(ha), vs.sampler_state.rng)
