import pytest
from pytest import approx
import numpy as np

import jax

import netket as nk
import netket_pro as nkp

from ..common import skipif_distributed
from ._exact import compute_exact_infidelity
from .._finite_diff import central_diff_grad, same_derivatives

N = 3
hi = nk.hilbert.Spin(0.5, N, inverted_ordering=False)
g = nk.graph.Chain(N, pbc=False)

SEED = 1234

operators = {}

operators["Hamiltonian"] = nk.operator.IsingJax(hi, graph=g, h=1.0)


def _setup(exact: bool = False, seed=SEED):
    n_samples = 1e6
    n_discard_per_chain = 1e3

    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
    ma = nk.models.RBM(alpha=1)

    if not exact:
        vs_t = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            seed=seed,
        )
        vs = nk.vqs.MCState(
            sampler=sa,
            model=ma,
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            seed=seed + 1,
        )
    else:
        vs_t = nk.vqs.FullSumState(hilbert=hi, model=ma, seed=seed)
        vs = nk.vqs.FullSumState(
            hilbert=hi,
            model=ma,
            seed=seed + 1,
        )

    return vs_t, vs


@pytest.mark.parametrize(
    "operator", [pytest.param(val, id=f"U={name}") for name, val in operators.items()]
)
def test_Midpoint_numerator_sameas_infidelity(operator):
    """This test verifies that if using only the numerator or denominator in the
    Midpoint infidelity operator, the result is the same as using the standard
    infidelity operator.
    """
    vs_t, vs = _setup(exact=True)

    dt = 0.5
    U_op = 1 - 1j * operator * dt
    U_dag_op = 1 + 1j * operator * dt

    I_standard = nkp.InfidelityOperator(
        vs_t,
        U=U_op,
        U_dagger=U_dag_op,
        cv_coeff=-0.5,
        is_unitary=False,
        sample_Upsi=True,
    )

    I_midpoint = nkp.midpoint.MidpointInfidelityOperator(
        vs_t,
        H=operator,
        dt=dt,
        B_num=1.0,
        B_den=None,
        cv_coeff=-0.5,
    )

    I_md, g_md = vs.expect_and_grad(I_midpoint)
    I_st, g_st = vs.expect_and_grad(I_standard)

    np.testing.assert_allclose(I_md.mean, I_st.mean)
    np.testing.assert_allclose(I_md.variance, I_st.variance)
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y), g_md, g_st)


@skipif_distributed
@pytest.mark.parametrize(
    "operator", [pytest.param(val, id=f"U={name}") for name, val in operators.items()]
)
@pytest.mark.parametrize(
    "B_num, B_den",
    [
        pytest.param(v1, v2, id=f"B_num={v1}_B_den={v2}")
        for (v1, v2) in [
            (1.0, -1.0),
            (1.0, None),
            (None, 1.0),
            (1.25 - 0.5j, 0.25 + 0.3j),
        ]
    ],
)
def test_Midpoint_FullSumState(operator, B_num, B_den):
    vs_exact_t, vs_exact = _setup(exact=True)
    H = operator
    dt = 0.5

    params, unravel = nk.jax.tree_ravel(vs_exact.parameters)

    def _infidelity_exact_fun(params, vstate):
        return compute_exact_infidelity(
            unravel(params), vstate, H, dt, B_num=B_num, B_den=B_den
        )

    I_exact = compute_exact_infidelity(
        vs_exact.parameters,
        vs_exact_t,
        H,
        dt,
        B_num=B_num,
        B_den=B_den,
    )

    I_grad_exact = central_diff_grad(
        _infidelity_exact_fun,
        params,
        1.0e-5,
        vs_exact_t,
    )

    I_op = nkp.midpoint.MidpointInfidelityOperator(
        vs_exact_t,
        H=H,
        dt=dt,
        B_num=B_num,
        B_den=B_den,
        cv_coeff=-0.5,
    )

    I_stat1 = vs_exact.expect(I_op)
    I_stat, I_grad = vs_exact.expect_and_grad(I_op)

    I1_mean = np.asarray(I_stat1.mean)
    I_mean = np.asarray(I_stat.mean)
    err = 5 * I_stat1.error_of_mean
    I_grad, _ = nk.jax.tree_ravel(I_grad)

    np.testing.assert_allclose(I_exact.real, I1_mean.real, atol=err)

    np.testing.assert_almost_equal(I1_mean.real, I_mean.real)
    np.testing.assert_almost_equal(I_stat1.variance, I_stat.variance)
    np.testing.assert_almost_equal(I_grad, I_grad_exact)


@pytest.mark.parametrize(
    "operator", [pytest.param(val, id=f"U={name}") for name, val in operators.items()]
)
@pytest.mark.parametrize(
    "B_num, B_den",
    [
        pytest.param(v1, v2, id=f"B_num={v1}_B_den={v2}")
        for (v1, v2) in [
            # (1.0, -1.0),
            (1.0, None),
            (None, 1.0),
            (1.25 - 0.5j, 0.25 + 0.3j),
        ]
    ],
)
def test_Midpoint_MCState(operator, B_num, B_den):
    vs_t, vs = _setup(exact=True)
    H = operator
    dt = 0.5

    params, unravel = nk.jax.tree_ravel(vs.parameters)

    def _infidelity_exact_fun(params, vstate):
        return compute_exact_infidelity(
            unravel(params), vstate, H, dt, B_num=B_num, B_den=B_den
        )

    I_exact = compute_exact_infidelity(
        vs.parameters,
        vs_t,
        H,
        dt,
        B_num=B_num,
        B_den=B_den,
    )

    I_grad_exact = central_diff_grad(
        _infidelity_exact_fun,
        params,
        1.0e-5,
        vs_t,
    )

    I_op = nkp.midpoint.MidpointInfidelityOperator(
        vs_t,
        H=H,
        dt=dt,
        B_num=B_num,
        B_den=B_den,
        cv_coeff=-0.5,
    )

    I_stat1 = vs.expect(I_op)
    I_stat, I_grad = vs.expect_and_grad(I_op)

    I1_mean = np.asarray(I_stat1.mean)
    I_mean = np.asarray(I_stat.mean)
    err = 5 * I_stat1.error_of_mean
    I_grad, _ = nk.jax.tree_ravel(I_grad)

    np.testing.assert_allclose(I_exact.real, I1_mean.real, atol=err)

    assert I1_mean.real == approx(I_mean.real, abs=1e-5)
    assert np.asarray(I_stat1.variance) == approx(np.asarray(I_stat.variance), abs=1e-5)

    same_derivatives(I_grad, I_grad_exact, rel_eps=5e-1)


@skipif_distributed
@pytest.mark.parametrize(
    "operator", [pytest.param(val, id=f"U={name}") for name, val in operators.items()]
)
@pytest.mark.parametrize(
    "B_num, B_den",
    [
        pytest.param(v1, v2, id=f"B_num={v1}_B_den={v2}")
        for (v1, v2) in [
            (1.0, None),
            (1.25 - 0.5j, 0.25 + 0.3j),
        ]
    ],
)
@pytest.mark.parametrize(
    "chunk_size",
    [pytest.param(chunk_size, id=f"{chunk_size=}") for chunk_size in [1000, 10000]],
)
def test_Infidelity_chunking_identical_result(operator, B_num, B_den, chunk_size):
    vs_t, vs = _setup(exact=True)
    H = operator
    dt = 0.5

    I_op = nkp.midpoint.MidpointInfidelityOperator(
        vs_t,
        H=H,
        dt=dt,
        B_num=B_num,
        B_den=B_den,
        cv_coeff=-0.5,
    )

    vs.chunk_size = None
    I_stat1 = vs.expect(I_op)
    I_stat, I_grad = vs.expect_and_grad(I_op)

    vs.chunk_size = chunk_size
    I_stat1_chunked = vs.expect(I_op)
    I_stat_chunked, I_grad_chunked = vs.expect_and_grad(I_op)

    jax.tree.map(
        lambda x, y: np.testing.assert_almost_equal(x, y), I_stat1, I_stat1_chunked
    )
    jax.tree.map(
        lambda x, y: np.testing.assert_almost_equal(x, y), I_stat, I_stat_chunked
    )
    jax.tree.map(lambda x, y: np.testing.assert_allclose(x, y), I_grad, I_grad_chunked)
