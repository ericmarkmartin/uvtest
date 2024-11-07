import netket as nk
import numpy as np
import netket_pro as nkp

import jax
import optax
import pytest
import jax.numpy as jnp

from flax import core as fcore

from netket.models import RBM, RBMModPhase

from netket_pro.driver import InfidelityOptimizerNG
from netket_pro._src.driver.ngd.infidelity_estimators import (
    hermitian_estimators,
    mixed_estimators,
)
from netket_pro._src.driver.ngd.infidelity_kernels import infidelity_UV_kernel
from netket_pro._src.driver.ngd.sr_srt_common import _prepare_input
from netket_pro._src.driver.ngd.driver_abstract_ngd import _flatten_samples
from netket_pro._src.driver.ngd.driver_infidelity_ngd import to_jax_operator

from ..._finite_diff import same_derivatives


machines = [
    pytest.param(
        RBM(
            param_dtype=jnp.float64,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            hidden_bias_init=jax.nn.initializers.normal(stddev=0.02),
            use_visible_bias=False,
        ),
        id="RBM(float64)",
    ),
    pytest.param(
        RBM(
            param_dtype=jnp.complex128,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            hidden_bias_init=jax.nn.initializers.normal(stddev=0.02),
            use_visible_bias=False,
        ),
        id="RBM(complex128)",
    ),
    pytest.param(RBMModPhase(), id="RBMModPhase"),
]

estimators = [
    pytest.param(hermitian_estimators, id="hermitian"),
    pytest.param(mixed_estimators, id="mixed"),
]

transformations = [
    pytest.param((False, False), id="None"),
    pytest.param((True, False), id="U"),
    pytest.param((True, False), id="V"),
    pytest.param((True, True), id="UV"),
]


def _setup(*, machine=None, chunk_size=None):
    L = 4
    lattice = nk.graph.Chain(L, max_neighbor_order=2)
    Ns = lattice.n_nodes
    hi = nk.hilbert.Spin(s=1 / 2, N=Ns, inverted_ordering=False)

    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0)
    if nk.config.netket_experimental_sharding:
        H = H.to_jax_operator()

    dt = 0.01
    U = to_jax_operator((0.5 + 0.5j) * H * dt)
    V = to_jax_operator((0.5 - 0.5j) * H * dt)

    if machine is None:
        machine = RBM(
            param_dtype=jnp.complex128,
        )

    sampler = nk.sampler.MetropolisLocal(
        hilbert=hi, n_chains=16, sweep_size=hi.size // 2
    )

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=machine,
        n_samples=512,
        n_discard_per_chain=0,
        seed=0,
        sampler_seed=0,
        chunk_size=chunk_size,
    )

    gs = jnp.array(
        [
            0.07664074,
            0.1767767,
            0.1767767,
            0.13529903,
            0.1767767,
            0.57664074,
            0.13529903,
            0.1767767,
            0.1767767,
            0.13529903,
            0.57664074,
            0.1767767,
            0.13529903,
            0.1767767,
            0.1767767,
            0.07664074,
        ]
    )
    # fmt: on
    target_model = nk.models.LogStateVector(hi, param_dtype=float)
    tstate = nk.vqs.MCState(
        sampler=sampler,
        model=target_model,
        n_samples=512,
        n_discard_per_chain=0,
        seed=1,
        sampler_seed=1,
        chunk_size=chunk_size,
    )
    tstate.parameters = {"logstate": jnp.log(gs)}

    return U, V, vstate, tstate


@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("transformations", transformations)
@pytest.mark.parametrize("estimator", estimators)
def test_infidelity_MCState(model, transformations, estimator):
    r"""
    Test estimators for the loss and gradient of the infidelity.
    Compare within tollerance to the exact infidelity and its gradient.
    """
    U, V, vstate, tstate = _setup(machine=model)

    use_U, use_V = transformations
    U = U if use_U else None
    V = V if use_V else None

    vstate_fs = nk.vqs.FullSumState(
        hilbert=vstate.hilbert, model=vstate.model, variables=vstate.variables
    )
    tstate_fs = nk.vqs.FullSumState(
        hilbert=tstate.hilbert, model=tstate.model, variables=tstate.variables
    )

    op_exact = nkp.InfidelityOperator(tstate_fs, U=U, V=V)
    I_exact, I_grad_exact = vstate_fs.expect_and_grad(op_exact)
    I_exact = I_exact.mean
    I_grad_exact, _ = nk.jax.tree_ravel(I_grad_exact)

    afun, vars, σ, afun_t, vars_t, σ_t = infidelity_UV_kernel(
        vstate, tstate, U, V, resample_fraction=None
    )
    σ = _flatten_samples(σ)
    σ_t = _flatten_samples(σ_t)

    local_grad, local_loss = estimator(
        afun,
        vars,
        σ,
        afun_t,
        vars_t,
        σ_t,
        cv_coeff=-0.5,
    )

    I_mc = nk.stats.statistics(local_loss)
    I_mc_mean = np.asarray(I_mc.mean)
    err = 5 * np.asarray(I_mc.error_of_mean)

    np.testing.assert_allclose(I_exact.real, I_mc_mean.real, atol=err)

    model_state, params = fcore.pop(vars, "params")

    mode = nk.jax.jacobian_default_mode(
        afun,
        params,
        model_state,
        vstate.hilbert.random_state(jax.random.key(1), 3),
        warn=False,
    )

    ΔX = nk.jax.jacobian(
        afun,
        params,
        σ,
        model_state,
        mode=mode,
        dense=True,
        center=True,
        chunk_size=None,
    )
    ΔX, Δf = _prepare_input(ΔX, local_grad, mode=mode, e_mean=nk.stats.mean(local_grad))

    I_grad_mc = ΔX.T @ Δf

    params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params
    )
    if mode == "complex" and nk.jax.tree_leaf_iscomplex(params_structure):
        num_p = I_grad_mc.shape[-1] // 2
        I_grad_mc = I_grad_mc[:num_p] + 1j * I_grad_mc[num_p:]

    same_derivatives(I_grad_mc, I_grad_exact, rel_eps=1e-1, abs_eps=1e-1)


@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("transformations", transformations)
@pytest.mark.parametrize("estimator", ["hermitian", "mixed"])
@pytest.mark.parametrize("ntk", [True, False])
def test_cv_invariance(model, transformations, estimator, ntk):
    r"""
    Check that CV doesnt modify the evolution, only the loss estimator.
    """
    n_iters = 15
    diag_shift = 1e-3
    solver_fn = nk.optimizer.solver.cholesky
    optimizer = optax.sgd(0.01)

    U, V, vstate_nocv, tstate = _setup(machine=model)

    use_U, use_V = transformations
    U = U if use_U else None
    V = V if use_V else None

    gs = InfidelityOptimizerNG(
        target_state=tstate,
        optimizer=optimizer,
        variational_state=vstate_nocv,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode=None,
        use_ntk=ntk,
        U=U,
        V=V,
        cv_coeff=0,
        estimator=estimator,
    )
    gs.run(n_iters)

    _, _, vstate_cv, tstate = _setup(machine=model)
    gs = InfidelityOptimizerNG(
        target_state=tstate,
        optimizer=optimizer,
        variational_state=vstate_cv,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode=None,
        use_ntk=ntk,
        U=U,
        V=V,
        cv_coeff=-0.5,
        estimator=estimator,
    )
    gs.run(n_iters)

    # check same updates
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_cv.parameters, vstate_nocv.parameters
    )
