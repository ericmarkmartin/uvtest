import netket as nk
import netket_pro as nkp
import jax
import jax.numpy as jnp

L = 10
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0).to_jax_operator()
ma = nk.models.RBM(alpha=1, use_visible_bias=True, param_dtype=float)
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
op = nk.optimizer.Sgd(learning_rate=0.05)
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

hai = nkp.ShiftImportanceOperator(
    ha, epsilon=0.01, second_order=True, reweight_norm=True
)

qgt1 = nkp.importance_sampling.qgt.QGTJacobianDenseImportanceSampling(
    importance_operator=hai, chunk_size=16
)
sr = nk.optimizer.SR(qgt=qgt1, diag_shift=0.01, solver=nk.optimizer.solver.pinv_smooth)
gs = nk.VMC(hai, op, variational_state=vs, preconditioner=sr)

print("ED:", nk.exact.lanczos_ed(ha))

log = nk.logging.RuntimeLog()
gs.run(n_iter=1000, out=log, obs={"ha": ha})

# afun, avars = hai.get_log_importance(vs)
# ares = afun(avars, vs.samples)
# vs._samples_distributions[afun] = vs.samples
