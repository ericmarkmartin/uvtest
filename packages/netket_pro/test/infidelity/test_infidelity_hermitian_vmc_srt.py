import pytest
import netket as nk

import jax.numpy as jnp
import numpy as np

from netket.models import RBM, RBMModPhase
from netket.vqs import MCState
from netket.optimizer.solver import cholesky
from netket import VMC

from netket_pro.driver import VMC_SRt, VMC_SRt_ntk
from netket_pro.infidelity import InfidelityOperator


from optax import sgd
from functools import partial

from jax import clear_caches
from jax.tree_util import tree_map


tol = 1e-8

seed = 123456
L = 4
lattice = nk.graph.Chain(L, pbc=True)
n_samples = 512

machines = [
    pytest.param(RBM(param_dtype=jnp.float64), id="RBM(float64)"),
    pytest.param(RBM(param_dtype=jnp.complex128), id="RBM(complex128)"),
    pytest.param(RBMModPhase(), id="RBMModPhase"),
]


def _setup(
    *,
    machine,
):
    clear_caches()
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)

    sampler = nk.sampler.MetropolisLocal(
        hilbert=hi, n_chains=16, sweep_size=hi.size // 2
    )

    vstate = MCState(
        sampler=sampler,
        model=machine,
        n_samples=n_samples,
        n_discard_per_chain=0,
        seed=seed,
        sampler_seed=seed,
    )

    # Parameters of the ground state of the TFIM model L=3
    # Generated with
    # g = nk.graph.Hypercube(length=3, n_dim=1, pbc=True)
    # hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    # ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)
    # nk.exact.lanczos_ed(ha, compute_eigenvectors=True)[1].reshape(-1)
    # fmt: off
    gs = jnp.array(
        [
            0.07664074, 0.1767767, 0.1767767, 0.13529903, 0.1767767,
            0.57664074, 0.13529903, 0.1767767, 0.1767767, 0.13529903,
            0.57664074, 0.1767767, 0.13529903, 0.1767767, 0.1767767,
            0.07664074,
        ]
    )
    # fmt: on
    target_model = nk.models.LogStateVector(hi, param_dtype=float)
    tstate = MCState(
        sampler=sampler,
        model=target_model,
        n_samples=n_samples,
        n_discard_per_chain=0,
        seed=seed + 1,
        sampler_seed=seed + 1,
    )
    tstate.parameters = {"logstate": jnp.log(gs)}

    return vstate, tstate


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_hermitian_gradient(machine, B, cv_coeff):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 + B * 1j * H * 0.05) if B is not None else None

    lr = 1e-2
    optimizer = sgd(learning_rate=lr)
    diag_shift = 1e-2
    linear_solver_fn = cholesky

    # SR
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=True,
    )

    preconditioner = nk.optimizer.SR(
        diag_shift=diag_shift, holomorphic=False, solver=linear_solver_fn
    )

    te_sr = VMC(
        Iop,
        optimizer,
        variational_state=vstate,
        preconditioner=preconditioner,
    )

    update_sr = te_sr._forward_and_backward()

    # SRt
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=True,
    )

    te_srt = VMC_SRt(
        Iop,
        optimizer,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        variational_state=vstate,
    )

    update_srt = te_srt._forward_and_backward()

    # SRt_NTK
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=True,
    )

    te_srt_ntk = VMC_SRt_ntk(
        Iop,
        optimizer,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        variational_state=vstate,
    )
    update_srt_ntk = te_srt_ntk._forward_and_backward()

    # Check that SRt and NTK are consistent
    tree_map(
        partial(np.testing.assert_allclose, atol=tol, rtol=tol),
        update_srt,
        update_srt_ntk,
    )

    # CHECK SR vs SRt
    tree_map(
        partial(np.testing.assert_allclose, atol=tol, rtol=tol), update_sr, update_srt
    )

    # CHECK SR vs NTK
    tree_map(
        partial(np.testing.assert_allclose, atol=tol, rtol=tol),
        update_sr,
        update_srt_ntk,
    )
