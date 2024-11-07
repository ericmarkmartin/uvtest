import netket as nk
import numpy as np
import netket_pro as nkp

import jax
import optax
import pytest
import jax.numpy as jnp


from netket.models import RBM, RBMModPhase

from netket_pro.driver import InfidelityOptimizerNG, VMC_NG

from ...common import skipif_distributed


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

drivers = [
    pytest.param(False, None, id="SR"),
    pytest.param(True, None, id="SRt"),
    pytest.param(True, "onthefly", id="SRt-onthefly"),
]

transformations = [
    pytest.param((False, False), id="None"),
    pytest.param((True, False), id="U"),
    pytest.param((True, False), id="V"),
    pytest.param((True, True), id="UV"),
]

estimators = [
    pytest.param("hermitian", id="hermitian"),
    pytest.param("mixed", id="mixed"),
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
    U = (0.5 + 0.5j) * H * dt
    V = (0.5 - 0.5j) * H * dt

    if machine is None:
        machine = RBM(
            param_dtype=jnp.complex128,
        )

    sampler = nk.sampler.MetropolisLocal(
        hilbert=hi, n_chains=16, sweep_size=hi.size // 2
    )

    opt = optax.sgd(learning_rate=0.035)

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

    return H, U, V, opt, vstate, tstate


@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("transformations", transformations)
@pytest.mark.parametrize("estimator", estimators)
@pytest.mark.parametrize("chunk_size", [None, 16, 128])
def test_SR_vs_SRt_infidelity(model, transformations, estimator, chunk_size):
    r"""
    Test that the SR and SRt give the same results for infidelity minimization. For SRt we check both 'jacobian' and 'onthefly' mode.
    We check both 'hermitian', and 'mixed' estimator, for all `U` and `V` combinations.
    """
    n_iters = 5
    diag_shift = 1e-3
    solver_fn = nk.optimizer.solver.cholesky

    _, U, V, opt, vstate_srt, tstate = _setup(machine=model, chunk_size=chunk_size)

    use_U, use_V = transformations
    U = U if use_U else None
    V = V if use_V else None

    gs = InfidelityOptimizerNG(
        target_state=tstate,
        optimizer=opt,
        variational_state=vstate_srt,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode=None,
        use_ntk=True,
        U=U,
        V=V,
        cv_coeff=-0.5,
        estimator=estimator,
        chunk_size_bwd=chunk_size,
        rloo=False,
    )
    logger_srt = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt)

    _, _, _, _, vstate_srt_onthefly, tstate = _setup(machine=model)
    gs = InfidelityOptimizerNG(
        target_state=tstate,
        optimizer=opt,
        variational_state=vstate_srt_onthefly,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode="onthefly",
        use_ntk=True,
        U=U,
        V=V,
        cv_coeff=-0.5,
        estimator=estimator,
        chunk_size_bwd=chunk_size,
        rloo=False,
    )
    logger_srt_onthefly = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt_onthefly)

    _, _, _, _, vstate_sr, tstate = _setup(machine=model)
    gs = InfidelityOptimizerNG(
        target_state=tstate,
        optimizer=opt,
        variational_state=vstate_sr,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode=None,
        use_ntk=False,
        U=U,
        V=V,
        cv_coeff=-0.5,
        estimator=estimator,
        chunk_size_bwd=chunk_size,
        rloo=False,
    )
    logger_sr = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_sr)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt.parameters, vstate_sr.parameters
    )

    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt_onthefly.parameters, vstate_sr.parameters
    )

    if nkp.distributed.process_index() == 0:
        infidelity_SRt = logger_srt.data["Infidelity"]["Mean"]
        infidelity_SRt_onthefly = logger_srt_onthefly.data["Infidelity"]["Mean"]
        infidelity_SR = logger_sr.data["Infidelity"]["Mean"]

        np.testing.assert_allclose(infidelity_SRt, infidelity_SR, atol=1e-10)
        np.testing.assert_allclose(infidelity_SRt_onthefly, infidelity_SR, atol=1e-10)


@pytest.mark.parametrize("model", machines)
def test_SR_vs_SRt_VMC(
    model,
):
    r"""
    Test that the SR and SRt give the same results for energy minimization. For SRt we check both 'jacobian' and 'onthefly' mode.
    We also compare against the base netket `nk.VMC` with preconditioner `nk.optimizer.SR`.
    """
    n_iters = 5
    diag_shift = 1e-3
    solver_fn = nk.optimizer.solver.cholesky

    H, _, _, opt, vstate_srt, _ = _setup(machine=model)

    gs = VMC_NG(
        hamiltonian=H,
        optimizer=opt,
        variational_state=vstate_srt,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode=None,
        use_ntk=True,
    )
    logger_srt = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt)

    _, _, _, _, vstate_srt_onthefly, _ = _setup(machine=model)
    gs = VMC_NG(
        hamiltonian=H,
        optimizer=opt,
        variational_state=vstate_srt_onthefly,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode="onthefly",
        use_ntk=True,
    )
    logger_srt_onthefly = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_srt_onthefly)

    _, _, _, _, vstate_sr, _ = _setup(machine=model)
    gs = VMC_NG(
        hamiltonian=H,
        optimizer=opt,
        variational_state=vstate_sr,
        diag_shift=diag_shift,
        linear_solver_fn=solver_fn,
        evaluation_mode=None,
        use_ntk=False,
    )
    logger_sr = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_sr)

    _, _, _, _, vstate_sr_vanilla, _ = _setup(machine=model)
    preconditioner = nk.optimizer.SR(
        diag_shift=diag_shift, solver=solver_fn, holomorphic=False
    )
    gs = nk.VMC(
        hamiltonian=H,
        optimizer=opt,
        variational_state=vstate_sr_vanilla,
        preconditioner=preconditioner,
    )
    logger_sr_vanilla = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger_sr_vanilla)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_sr_vanilla.parameters, vstate_sr.parameters
    )

    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt.parameters, vstate_sr.parameters
    )

    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate_srt_onthefly.parameters, vstate_sr.parameters
    )

    if nkp.distributed.process_index() == 0:
        energy_SRt = logger_srt.data["Energy"]["Mean"]
        energy_SRt_onthefly = logger_srt_onthefly.data["Energy"]["Mean"]
        energy_SR = logger_sr.data["Energy"]["Mean"]
        energy_SR_vanilla = logger_sr_vanilla.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy_SRt, energy_SR, atol=1e-10)
        np.testing.assert_allclose(energy_SRt_onthefly, energy_SR, atol=1e-10)
        np.testing.assert_allclose(energy_SR_vanilla, energy_SR, atol=1e-10)


@skipif_distributed
def test_SRt_constructor_errors():
    """
    Check on VMC_NG constructor errors
    """
    H, _, _, opt, vstate, _ = _setup()
    gs = VMC_NG(
        H,
        opt,
        variational_state=vstate,
        diag_shift=0.1,
    )
    assert gs.evaluation_mode == "complex"
    gs.run(1)

    with pytest.raises(ValueError):
        gs = VMC_NG(
            H, opt, variational_state=vstate, diag_shift=0.1, evaluation_mode="belin"
        )


@skipif_distributed
@pytest.mark.parametrize("use_ntk, jacobian_mode", drivers)
def test_SRt_schedules(use_ntk, jacobian_mode):
    """
    Check on VMC_NG accepts schedules
    """
    H, _, _, opt, vstate, _ = _setup()
    gs = VMC_NG(
        H,
        opt,
        variational_state=vstate,
        diag_shift=optax.linear_schedule(0.1, 0.001, 100),
        use_ntk=use_ntk,
        evaluation_mode=jacobian_mode,
    )
    gs.run(5)


@skipif_distributed
@pytest.mark.parametrize("use_ntk, jacobian_mode", drivers)
def test_SRt_supports_netket_solvers(use_ntk, jacobian_mode):
    """
    Check on VMC_NG supports netket solvers
    """
    H, _, _, opt, vstate, _ = _setup()
    gs = VMC_NG(
        H,
        opt,
        variational_state=vstate,
        diag_shift=optax.linear_schedule(0.1, 0.001, 100),
        linear_solver_fn=nk.optimizer.solver.pinv_smooth,
        use_ntk=use_ntk,
        evaluation_mode=jacobian_mode,
    )
    gs.run(5)


# WARNING: We spotted an instability of this code.
# If we run the test test_SRt_chunked for 500 iterations, the test fails when
# using momentum and comparing chunking vs no chunking. We are unsure about the origin of
# the instability. Most likely cause is numerical errors accumulating.


@pytest.mark.parametrize("model", machines)
@pytest.mark.parametrize("evaluation_mode", [None, "onthefly"])
@pytest.mark.parametrize("momentum", [None, 0.9])
@pytest.mark.parametrize("proj_reg", [None, 1.0])
def test_SRt_chunked(model, evaluation_mode, momentum, proj_reg):
    """
    Check on VMC_NG that we get **exactly** the same dynamics with and without chunking.
    We check this on the SRt driver, so that we can check it with and without momentum and projection regularization.
    """
    n_iters = 5
    diag_shift = 0.01
    chunk_size = 64
    solver_fn = nk.optimizer.solver.cholesky

    H, _, _, opt, vstate, _ = _setup(
        machine=model,
    )
    gs = VMC_NG(
        H,
        opt,
        variational_state=vstate,
        diag_shift=diag_shift,
        proj_reg=proj_reg,
        momentum=momentum,
        use_ntk=True,
        evaluation_mode=evaluation_mode,
        linear_solver_fn=solver_fn,
    )
    logger = nk.logging.RuntimeLog()
    gs.run(n_iter=n_iters, out=logger)

    H, _, _, opt, vstate_chunked, _ = _setup(machine=model, chunk_size=chunk_size)
    gs_chunked = VMC_NG(
        H,
        opt,
        variational_state=vstate_chunked,
        diag_shift=diag_shift,
        proj_reg=proj_reg,
        momentum=momentum,
        use_ntk=True,
        evaluation_mode=evaluation_mode,
        linear_solver_fn=solver_fn,
        chunk_size_bwd=chunk_size,
    )

    logger_chunked = nk.logging.RuntimeLog()
    gs_chunked.run(n_iter=n_iters, out=logger_chunked)

    # check same parameters
    jax.tree_util.tree_map(
        np.testing.assert_allclose, vstate.parameters, vstate_chunked.parameters
    )

    if nkp.distributed.process_index() == 0:
        energy = logger.data["Energy"]["Mean"]
        energy_chunked = logger_chunked.data["Energy"]["Mean"]

        np.testing.assert_allclose(energy, energy_chunked, atol=1e-10)
