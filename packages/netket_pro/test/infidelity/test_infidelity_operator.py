import pytest
from pytest import approx
import numpy as np

import jax

import netket as nk
import netket_pro as nkp

from ._infidelity_exact import compute_exact_infidelity
from .._finite_diff import central_diff_grad, same_derivatives
from ..common import skipif_distributed


N = 3
hi = nk.hilbert.Spin(0.5, N, inverted_ordering=False)

SEED = 12345

operators = {}

operators["Identity"] = (None, None)

theta = 0.01

op = nkp.operator.Rx(hi, 0, theta)
operators["Rx"] = (op, None)

op = nkp.operator.Rx(hi, 0, theta).to_local_operator().to_pauli_strings()
op_d = nkp.operator.Rx(hi, 0, theta).H.to_local_operator().to_pauli_strings()
operators["PauliStringNumba"] = (op, op_d)
operators["PauliStringJax"] = (op.to_jax_operator(), op_d.to_jax_operator())


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
    "sample_Upsi",
    [
        pytest.param(False, id="sample_Upsi=False"),
        pytest.param(True, id="sample_Upsi=True"),
    ],
)
@pytest.mark.parametrize(
    "UUdag", [pytest.param(val, id=f"U={name}") for name, val in operators.items()]
)
def test_Infidelity_FullSumState(sample_Upsi, UUdag):
    vs_exact_t, vs_exact = _setup(exact=True)

    U, U_dag = UUdag

    params, unravel = nk.jax.tree_ravel(vs_exact.parameters)

    def _infidelity_exact_fun(params, vstate, U):
        return compute_exact_infidelity(unravel(params), vstate, U)

    I_exact = compute_exact_infidelity(
        vs_exact.parameters,
        vs_exact_t,
        U,
    )

    I_grad_exact = central_diff_grad(
        _infidelity_exact_fun, params, 1.0e-5, vs_exact_t, U
    )

    I_op = nkp.InfidelityOperator(
        vs_exact_t,
        U=U,
        U_dagger=U_dag,
        sample_Upsi=sample_Upsi,
        cv_coeff=-0.5,
        is_unitary=True,
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
    "sample_Upsi",
    [
        pytest.param(False, id="sample_Upsi=False"),
        pytest.param(True, id="sample_Upsi=True"),
    ],
)
@pytest.mark.parametrize(
    "UUdag", [pytest.param(val, id=f"U={name}") for name, val in operators.items()]
)
def test_Infidelity_MCState(sample_Upsi, UUdag):
    vs_t, vs = _setup(exact=False)
    vs_exact_t, vs_exact = _setup(exact=True)

    U, U_dag = UUdag

    params, unravel = nk.jax.tree_ravel(vs.parameters)

    I_op_exact = nkp.InfidelityOperator(
        vs_exact_t,
        U=U,
        U_dagger=U_dag,
        sample_Upsi=sample_Upsi,
        cv_coeff=-0.5,
        is_unitary=True,
    )
    I_exact, I_grad_exact = vs_exact.expect_and_grad(I_op_exact)
    I_exact = I_exact.mean
    I_grad_exact, _ = nk.jax.tree_ravel(I_grad_exact)

    I_op = nkp.InfidelityOperator(
        vs_t,
        U=U,
        U_dagger=U_dag,
        sample_Upsi=sample_Upsi,
        cv_coeff=-0.5,
        is_unitary=True,
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
    "sample_Upsi",
    [
        pytest.param(False, id="sample_Upsi=False"),
        pytest.param(True, id="sample_Upsi=True"),
    ],
)
@pytest.mark.parametrize(
    "UUdag", [pytest.param(val, id=f"U={name}") for name, val in operators.items()]
)
@pytest.mark.parametrize(
    "chunk_size",
    [pytest.param(chunk_size, id=f"{chunk_size=}") for chunk_size in [1000, 10000]],
)
def test_Infidelity_chunking_identical_result(sample_Upsi, UUdag, chunk_size):
    vs_t, vs = _setup(exact=False)

    U, U_dag = UUdag

    I_op = nkp.InfidelityOperator(
        vs_t,
        U=U,
        U_dagger=U_dag,
        sample_Upsi=sample_Upsi,
        cv_coeff=-0.5,
        is_unitary=True,
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
