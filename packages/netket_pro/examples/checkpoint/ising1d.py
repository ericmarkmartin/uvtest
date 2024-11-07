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

import netket as nk
import optax
from netket import experimental as nkx
import netket_checkpoint as nkc
import netket_pro as nkp


print("mode:", nkp.distributed.mode())

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)

# Ising spin hamiltonian
ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer with a decreasing learning rate
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))

# SR
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10, seed=1)

# Variational monte carlo driver with a variational state
# gs = nkc.driver1.VMC_SRt(ha, op, variational_state=vs, diag_shift=0.01)
gs = nkc.driver1.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Run the optimization for 500 iterations
log = nk.logging.JsonLog("test/test_r2", save_params=False)

options = nkc.checkpoint.CheckpointManagerOptions(save_interval_steps=5, keep_period=20)
ckpt = nkc.checkpoint.CheckpointManager(directory="/tmp/ckp2", options=options)

# gs._restore_checkpoint(ckpt, step=300)
# print("restore:", gs.state.expect(ha), flush=True)

gs.run_checkpointed(
    n_iter=300,
    out=log,
    timeit=True,
    checkpointer=ckpt,
)

gs = nkc.driver1.VMC_SRt(ha, op, variational_state=vs, diag_shift=0.01)

print(gs.state.expect(ha))

# You can also restore arbitrary states by running
state_step_200 = ckpt.restore_state(vs, step=200)
print("Energy at step 200 was : ", state_step_200.expect(ha).Mean)
print("From logger it is      : ", log.data["Energy"].Mean[200])
