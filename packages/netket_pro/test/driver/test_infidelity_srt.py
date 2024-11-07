import pytest
import netket as nk

import jax.numpy as jnp
import numpy as np

from functools import partial

from netket.models import RBM, RBMModPhase
from netket.vqs import MCState
from netket_pro.driver import InfidelityOptimizer, Infidelity_SRt
from netket.jax import tree_ravel, jacobian
from netket.optimizer.solver import cholesky

from netket_pro._src.driver.vmc_srt import _flatten_samples
from netket_pro.infidelity.overlap_UV.exact import _prepare

from optax import sgd

from jax import clear_caches
from jax.tree_util import tree_map
from flax import core as fcore


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
@pytest.mark.parametrize("proj_reg", [None, 1.0])
def test_srt_update(machine, B, cv_coeff, proj_reg):
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

    te_sr = InfidelityOptimizer(
        tstate,
        optimizer,
        variational_state=vstate,
        preconditioner=preconditioner,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        sample_Upsi=True,
        use_hermitian_gradient_estimator=True,
    )

    updates_sr = te_sr._forward_and_backward()

    # SRt update
    vstate, tstate = _setup(machine=machine)

    te_srt = Infidelity_SRt(
        tstate,
        optimizer,
        variational_state=vstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        proj_reg=proj_reg,
    )

    updates_srt = te_srt._forward_and_backward()

    # CHECK
    tree_map(
        partial(np.testing.assert_allclose, atol=1e-10, rtol=1e-10),
        updates_sr,
        updates_srt,
    )


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
@pytest.mark.parametrize("proj_reg", [None, 1.0])
def test_srt_optimization(machine, B, cv_coeff, proj_reg):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 - B * 1j * H * 0.05) if B is not None else None

    n_iter = 10
    diag_shift = 1e-2
    linear_solver_fn = cholesky
    optimizer = sgd(learning_rate=0.01)

    # SR optimization
    vstate_sr, tstate = _setup(machine=machine)
    preconditioner = nk.optimizer.SR(
        solver=linear_solver_fn, diag_shift=diag_shift, holomorphic=False
    )

    te_sr = InfidelityOptimizer(
        tstate,
        optimizer,
        variational_state=vstate_sr,
        preconditioner=preconditioner,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        sample_Upsi=True,
        use_hermitian_gradient_estimator=True,
    )

    te_sr.run(n_iter=n_iter)

    # SRt optimization
    vstate_srt, tstate = _setup(machine=machine)

    te_srt = Infidelity_SRt(
        tstate,
        optimizer,
        variational_state=vstate_srt,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        proj_reg=proj_reg,
    )

    te_srt.run(n_iter=n_iter)

    # CHECK
    tree_map(
        partial(np.testing.assert_allclose, atol=1e-10, rtol=1e-10),
        vstate_sr.parameters,
        vstate_srt.parameters,
    )


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_srt_loss(machine, B, cv_coeff):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 - B * 1j * H * 0.05) if B is not None else None

    diag_shift = 1e-2
    linear_solver_fn = cholesky
    optimizer = sgd(learning_rate=0.01)

    # SRt update
    vstate, tstate = _setup(machine=machine)

    te = Infidelity_SRt(
        tstate,
        optimizer,
        variational_state=vstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
    )

    te._forward_and_backward()

    expect_exact = vstate.expect(te._I_op).mean.real
    expect_from_te = te._loss_stats.mean
    np.testing.assert_almost_equal(
        expect_exact,
        expect_from_te,
    )


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_quadratic_model(machine, B, cv_coeff):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 - B * 1j * H * 0.05) if B is not None else None

    diag_shift = 1e-4
    linear_solver_fn = cholesky
    optimizer = sgd(learning_rate=0.01)

    # SRt update with quadratic model save
    vstate, tstate = _setup(machine=machine)

    te = Infidelity_SRt(
        tstate,
        optimizer,
        variational_state=vstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        collect_quadratic_model=True,
    )

    δ = te._forward_and_backward()
    δ, _ = tree_ravel(δ)

    linear_term_0 = te.info["linear_term"]
    quadratic_term_0 = te.info["quadratic_term"]

    # Compute quadratic model from scratch
    vstate, tstate = _setup(machine=machine)
    op = te._I_op

    ## linear term
    _, grad_h = vstate.expect_and_grad(op)
    grad_h, _ = tree_ravel(grad_h)
    linear_term_1 = δ.T.conj() @ grad_h

    np.testing.assert_allclose(linear_term_1.imag, 0.0, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        linear_term_1.real, linear_term_0, atol=1e-10, rtol=1e-10
    )

    ## quadratic term
    Vψ_logfun, Vψ_vars, Vψ_distribution = _prepare(
        vstate, op.V_state, extra_hash_data="V"
    )
    Vψ_samples = vstate.samples_distribution(
        Vψ_distribution,
        variables=Vψ_vars,
        resample_fraction=op.resample_fraction,
    )
    Vψ_samples = _flatten_samples(Vψ_samples)

    Vψ_model_state, Vψ_params = fcore.pop(Vψ_vars, "params")

    X = jacobian(
        Vψ_logfun,
        Vψ_params,
        Vψ_samples,
        Vψ_model_state,
        mode="holomorphic",
        dense=True,
        center=True,
        _sqrt_rescale=True,
    )
    S = X.conj().T @ X

    quadratic_term_1 = δ.T.conj() @ S @ δ

    np.testing.assert_allclose(quadratic_term_1.imag, 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(quadratic_term_1.real, quadratic_term_0, rtol=1e-6)
