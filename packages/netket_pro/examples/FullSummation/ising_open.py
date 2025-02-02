# Copyright 2020 The Netket Authors. - All Rights Reserved.
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
import netket_pro as nkp
import jax

# 1D Lattice
L = 3
gp = 0.3
Vp = 2.0

g = nk.graph.Hypercube(length=L, n_dim=2, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)

# The hamiltonian
ha = nk.operator.LocalOperator(hi)

# List of dissipative jump operators
j_ops = []

# Observables
obs_sx = nk.operator.LocalOperator(hi)
obs_sy = nk.operator.LocalOperator(hi, dtype=complex)
obs_sz = nk.operator.LocalOperator(hi)

for i in range(L):
    ha += (gp / 2.0) * nk.operator.spin.sigmax(hi, i)
    ha += (
        (Vp / 4.0)
        * nk.operator.spin.sigmaz(hi, i)
        * nk.operator.spin.sigmaz(hi, (i + 1) % L)
    )
    # sigma_{-} dissipation on every site
    j_ops.append(nk.operator.spin.sigmam(hi, i))
    obs_sx += nk.operator.spin.sigmax(hi, i)
    obs_sy += nk.operator.spin.sigmay(hi, i)
    obs_sz += nk.operator.spin.sigmaz(hi, i)


# Create the liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)
ldagl = lind.H @ lind

ma = nk.models.NDM(
    beta=1,
)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

vs = nkp.vqs.FullSumMixedState(hi, ma)
vs.init_parameters(jax.nn.initializers.normal(stddev=0.01))

ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)

obs = {"Sx": obs_sx, "Sy": obs_sy, "Sz": obs_sz}

out = nk.logging.JsonLog("test")
ss.run(n_iter=10, out=out, obs=obs)
