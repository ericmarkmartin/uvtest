import pytest
import jax.numpy as jnp

from netket.operator.spin import sigmaz, sigmax
from netket.sampler import MetropolisLocal
from netket.models import RBMModPhase
from netket.hilbert import Spin
from netket.vqs import MCState
from netket.graph import Grid

from netket_pro.driver import MidpointInfidelity_SRt
from netket_pro.jumps.networks import Jastrow_zz_frozen
from netket_pro.optimizer.L_curve import lcurve_solver_srt, gcv_solver_srt
import flax.serialization as serialization

from optax import sgd
from copy import copy

from ..common import skipif_distributed


def _setup(data_path: str = None):
    L = 2
    g = Grid((L, L), pbc=True)
    hi = Spin(s=1 / 2, N=g.n_nodes)

    J = 1.0
    hc = 3.044 * J
    h = hc * 1 / 10

    H = sum([-J * sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges()])
    H += sum([-h * sigmax(hi, i) for i in g.nodes()])

    model = RBMModPhase(alpha=4)
    model = Jastrow_zz_frozen(model, param_dtype=complex)

    n_samples = 128
    sampler = MetropolisLocal(hilbert=hi, n_chains=16, sweep_size=hi.size // 2)

    psi = MCState(
        sampler=sampler,
        model=model,
        n_samples=n_samples,
        n_discard_per_chain=0,
    )

    with open(data_path, "rb") as file:
        psi = serialization.from_bytes(psi, file.read())
    return psi, H


auto_solvers = [
    pytest.param(lcurve_solver_srt, id="lcurve"),
    pytest.param(gcv_solver_srt, id="gcv"),
]


@skipif_distributed
@pytest.mark.parametrize("auto_solver", auto_solvers)
def test_automatic_regularization(datadir, auto_solver):
    vstate, H = _setup(datadir / "2x2_state.mpack")

    target = copy(vstate)
    target.replace_sampler_seed()

    B = 0.5 + 1j * 0.5
    dt = 0.05
    diag_shift = 0  # the diag_shift should be 0 when using the lcurve solver
    linear_solver_fn = auto_solver

    lr = 5e-2
    optimizer = sgd(learning_rate=lr)
    n_iter = 150

    te = MidpointInfidelity_SRt(
        target,
        optimizer,
        variational_state=vstate,
        H=H,
        dt=dt,
        B_num=B,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        cv_coeff=-0.5,
    )

    initial_infidelity = vstate.expect(te._I_op).mean.real

    te.run(
        n_iter=n_iter,
        show_progress=False,
    )

    final_infidelity = vstate.expect(te._I_op).mean.real
    assert not jnp.isnan(final_infidelity)
    assert not jnp.isinf(final_infidelity)
    assert final_infidelity < initial_infidelity
