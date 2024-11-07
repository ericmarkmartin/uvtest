import pytest
import netket as nk

import jax.numpy as jnp
import numpy as np

from functools import partial

from netket.models import RBM, RBMModPhase
from netket.vqs import MCState
from netket_pro.driver import InfidelityHOptimizer, Infidelity_H_SRt_ntk
from netket_pro import distributed

from optax import sgd
from netket.optimizer.solver import cholesky

from jax import clear_caches
from jax.tree_util import tree_map


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
    chunk_size=None,
):
    clear_caches()
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)

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
        chunk_size=chunk_size,
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
        chunk_size=chunk_size,
    )
    tstate.parameters = {"logstate": jnp.log(gs)}

    return vstate, tstate


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_sr_update(machine, B, cv_coeff):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 - B * 1j * H * 0.05) if B is not None else None

    diag_shift = 1e-2
    linear_solver_fn = cholesky
    optimizer = sgd(learning_rate=0.01)

    # SR update
    vstate, tstate = _setup(machine=machine)
    preconditioner = nk.optimizer.SR(
        solver=linear_solver_fn, diag_shift=diag_shift, holomorphic=False
    )

    te_sr = InfidelityHOptimizer(
        tstate,
        optimizer,
        variational_state=vstate,
        preconditioner=preconditioner,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
    )

    updates_sr = te_sr._forward_and_backward()

    # autoreg update
    vstate, tstate = _setup(machine=machine)

    te = Infidelity_H_SRt_ntk(
        tstate,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
    )

    updates = te._forward_and_backward()

    # CHECK
    tree_map(
        partial(np.testing.assert_allclose, atol=1e-10, rtol=1e-10),
        updates_sr,
        updates,
    )


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_sr_run(machine, B, cv_coeff):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 - B * 1j * H * 0.05) if B is not None else None

    diag_shift = 1e-2
    linear_solver_fn = cholesky
    optimizer = sgd(learning_rate=0.01)

    # SR update
    vstate_sr, tstate = _setup(machine=machine)
    preconditioner = nk.optimizer.SR(
        solver=linear_solver_fn, diag_shift=diag_shift, holomorphic=False
    )

    te_sr = InfidelityHOptimizer(
        tstate,
        optimizer,
        variational_state=vstate_sr,
        preconditioner=preconditioner,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
    )
    te_sr.run(10, show_progress=False)

    # autoreg update
    vstate, tstate = _setup(machine=machine)

    te = Infidelity_H_SRt_ntk(
        tstate,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        collect_quadratic_model=True,
    )
    te.run(10, show_progress=False)

    # CHECK
    tree_map(
        partial(np.testing.assert_allclose, atol=1e-10, rtol=1e-10),
        vstate.parameters,
        vstate_sr.parameters,
    )


@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_chunked(model, B, cv_coeff):
    """
    nk.driver.VMC_kernelSR must give **exactly** the same dynamics with and without chunking
    """
    n_iters = 5
    chunk_size = 64

    diag_shift = 1e-2
    linear_solver_fn = cholesky
    optimizer = sgd(learning_rate=0.01)

    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 - B * 1j * H * 0.01) if B is not None else None

    vstate, tstate = _setup(machine=model)
    te = Infidelity_H_SRt_ntk(
        tstate,
        optimizer,
        variational_state=vstate,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
    )
    logger = nk.logging.RuntimeLog()
    te.run(n_iter=n_iters, out=logger)

    vstate_chunked, tstate_chunked = _setup(machine=model, chunk_size=chunk_size)
    te_chunked = Infidelity_H_SRt_ntk(
        tstate_chunked,
        optimizer,
        variational_state=vstate_chunked,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        chunk_size=chunk_size,
    )
    logger_chunked = nk.logging.RuntimeLog()
    te_chunked.run(n_iter=n_iters, out=logger_chunked)

    # check same parameters
    tree_map(np.testing.assert_allclose, vstate.parameters, vstate_chunked.parameters)

    if distributed.process_index() == 0:
        infidelity = logger.data["Infidelity"]["Mean"]
        infidelity_chunked = logger_chunked.data["Infidelity"]["Mean"]

        np.testing.assert_allclose(infidelity, infidelity_chunked, atol=1e-10)
