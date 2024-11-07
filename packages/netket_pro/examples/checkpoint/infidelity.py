import netket as nk
import netket_pro as nkp
import netket_checkpoint as nkc

# from matplotlib import pyplot as plt
import numpy as np


# Let's generate the ground state of the TFIM model. This
# technique also works with excited states.
g = nk.graph.Square(4)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)
sa = nk.sampler.MetropolisLocal(hi)
ha = nk.operator.Ising(hi, graph=g, h=1.0)

# Here is the ground state
egs, psi0 = nk.exact.lanczos_ed(ha, compute_eigenvectors=True)
psi0 = psi0[:, 0]

# Let's build the target variational state as a LogStateVector, which
# has 2^N parameters encoding the log-wavefunction
ma_target = nk.models.LogStateVector(hi)
vs_target = nk.vqs.MCState(sa, ma_target, seed=234)

# Compute the log-wavefunction. As some numbers can be negative, we
# convert the wavefunciton to complex beforehand
log_psi0 = np.log(psi0 + 0j)

# And fix the 'parameters' that represent the target state
vs_target.parameters = {
    "logstate": vs_target.parameters["logstate"].at[:].set(log_psi0)
}

# Now construct the variational state we are going to train
ma = nk.models.RBM()
# vs = nk.vqs.FullSumState(hi, ma)
vs = nk.vqs.MCState(
    sa,
    ma,
    seed=123,
)

# The infidelity operator, that can be used to compute the infidelity
inf = nkp.InfidelityOperator(vs_target)
vs.expect(inf)

# or just use the Infidelity optimisation driver
optimizer = nk.optimizer.Adam()
driver = nkc.driver1.InfidelityOptimizer(vs_target, optimizer, variational_state=vs)

log = nk.logging.JsonLog("test/run5", save_params=False)
ckpt = nkc.checkpoint.CheckpointManager(directory="/tmp/ckp4")
driver.run(
    300,
    out=log,
    checkpointer=ckpt,
)
