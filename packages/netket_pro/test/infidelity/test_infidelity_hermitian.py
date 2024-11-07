# TOADD: test for complex parameters and complex output

import pytest
import netket as nk

import jax.numpy as jnp
import numpy as np

from flax import linen as nn

from netket.models import RBM, RBMModPhase
from netket.jax import HashablePartial
from netket.vqs import MCState
from netket.vqs.mc import get_local_kernel_arguments
from netket.jax import tree_ravel, jacobian
from netket.utils import mpi

from netket_pro.infidelity import InfidelityOperator
from netket_pro import distributed

from jax import clear_caches
from jax.tree_util import tree_map

from ..common import skipif_distributed

seed = 123456
L = 4
lattice = nk.graph.Chain(4, pbc=True)
n_samples = 512

machine_complexRBM = RBM(param_dtype=jnp.complex128)
machine_RBM_modphase = RBMModPhase()
machines = [machine_RBM_modphase, machine_complexRBM]


def _setup(
    *,
    machine,
):
    clear_caches()
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)

    sampler = nk.sampler.MetropolisLocal(
        hilbert=hi, n_chains_per_rank=16, sweep_size=hi.size // 2
    )

    vstate = MCState(
        sampler=sampler,
        model=machine,
        n_samples=n_samples,
        n_discard_per_chain=0,
        seed=seed,
        sampler_seed=seed,
    )

    tstate = MCState(
        sampler=sampler,
        model=machine,
        n_samples=n_samples,
        n_discard_per_chain=0,
        seed=seed + 1,
        sampler_seed=seed + 1,
    )

    return vstate, tstate


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_hermitian_gradient(machine, B, cv_coeff):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 + B * 1j * H * 0.05) if B is not None else None

    # HERMITIAN GRADIENT
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=True,
    )

    _, grad_hermitian = vstate.expect_and_grad(Iop)
    grad_hermitian_ravelled, _ = tree_ravel(grad_hermitian)

    # EXPLICIT JACOBIAN CALCULATION
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=False,  # doesn't matter
    )

    ΔJx = jacobian(
        vstate._apply_fun,
        vstate.parameters,
        vstate.samples.reshape(-1, vstate.samples.shape[-1]),
        vstate.model_state,
        mode="complex",
        center=True,
        dense=True,
        _sqrt_rescale=False,
    )

    def to_complex(x):
        return x[:, 0] + 1j * x[:, 1]

    ΔJx = tree_map(to_complex, ΔJx)

    eps = vstate.local_estimators(Iop) * 2
    assert eps.shape == vstate.samples.shape[:-1]

    eps = eps.reshape(-1)
    eps -= jnp.mean(eps, axis=0)

    grad_explicit, _ = mpi.mpi_sum_jax(ΔJx.conj().T @ eps / vstate.n_samples)
    grad_explicit = grad_explicit.real

    # if complex parameters, recompose them
    if grad_explicit.size == grad_hermitian_ravelled.size * 2:
        grad_explicit = (
            grad_explicit[: grad_hermitian_ravelled.size]
            + 1j * grad_explicit[grad_hermitian_ravelled.size :]
        )

    np.testing.assert_allclose(
        grad_explicit,
        grad_hermitian_ravelled,
    )


@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_hermitian_expectation(B, cv_coeff):
    machine = machine_RBM_modphase
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    U = (1 + B * 1j * H * 0.05) if B is not None else None

    # HERMITIAN AVERAGE
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=True,
    )

    # when calculating the hermitian gradient, the expectation value is calculated differently
    mean_hermitian, _ = vstate.expect_and_grad(Iop)
    mean_hermitian = mean_hermitian.mean.real

    # NON-HERMITIAN AVERAGE
    # The expectation value of the infidelity is the same, but calculated differently in the two cases
    # The expectation used in the non-hermitian gradient and vstate.expect(Iop) are the same
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=None,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=False,
    )

    mean_non_hermitian = vstate.expect(Iop).mean.real

    np.testing.assert_allclose(
        mean_non_hermitian, mean_hermitian, atol=1e-13, rtol=1e-13
    )


@skipif_distributed
@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_local_estimator_samples(machine, B, cv_coeff):
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=lattice, h=1.0, J=1.0)
    V = (
        (1 + B * 1j * H * 0.05) if B is not None else None
    )  # V is applied to the variational state

    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=None,
        V=V,
        cv_coeff=cv_coeff,
        use_hermitian_gradient_estimator=True,
    )

    samples_from_kernel, _ = get_local_kernel_arguments(vstate, Iop)
    samples_from_kernel = samples_from_kernel.reshape(-1, samples_from_kernel.shape[-1])

    if V is None:
        key = vstate.model
    else:
        distribution_keys = vstate._samples_distributions.keys()
        key = [k for k in distribution_keys if isinstance(k, HashablePartial)]

        np.testing.assert_equal(len(key), 1)
        key = key[0]

    samples_from_distribution = vstate.samples_distribution(key)
    samples_from_distribution = samples_from_distribution.reshape(
        -1, samples_from_distribution.shape[-1]
    )

    if distributed.mode() == "sharding":
        samples_from_kernel, token = distributed.allgather(samples_from_kernel)
        samples_from_distribution, token = distributed.allgather(
            samples_from_distribution, token=token
        )

    np.testing.assert_array_equal(samples_from_kernel, samples_from_distribution)


@skipif_distributed
@pytest.mark.parametrize("Bu", [None, 1])
@pytest.mark.parametrize("Bv", [None, 1])
@pytest.mark.parametrize("chunk_size", [32])
def test_estimator_chunking(Bu, Bv, chunk_size):
    """
    This test checks that we never call the module with a batch size
    larger than chunk size. The chunk size must be chosen such that
    it is larger than the nunber of sampling chains (16) * max_conn_size
    of the operator used (in this case an hamiltonian with max conn=2).

    This checks that also chunking and no chunking give the same values.
    """

    class RBMAssert(nn.Module):
        chunk_size: int = None

        @nn.compact
        def __call__(self, x):
            assert x.shape[-1] == lattice.n_nodes
            assert x.ndim in (1, 2)
            # Check that we call this with size not larger than chunk size
            if x.ndim == 2 and chunk_size is not None:
                assert x.shape[0] <= chunk_size
            return RBM()(x)

    machine = RBMAssert(chunk_size=chunk_size)

    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)
    H = (
        nk.operator.spin.sigmax(hi, 0) + nk.operator.spin.sigmay(hi, 1)
    ).to_jax_operator()

    U = H if Bu is not None else None
    V = H if Bv is not None else None

    # HERMITIAN GRADIENT
    vstate, tstate = _setup(machine=machine)

    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=V,
        cv_coeff=None,
        use_hermitian_gradient_estimator=True,
    )

    iloc1 = vstate.local_estimators(Iop, chunk_size=chunk_size)

    # HERMITIAN GRADIENT
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=V,
        cv_coeff=None,
        use_hermitian_gradient_estimator=True,
    )

    tstate.chunk_size = chunk_size
    vstate.chunk_size = chunk_size
    iloc2 = vstate.local_estimators(Iop)
    i_mean = vstate.expect(Iop)  # noqa: F841

    np.testing.assert_allclose(iloc1, iloc2)
    # To be added later
    # np.testing.assert_allclose(jnp.mean(iloc1).real, i_mean.Mean)

    # No chunking
    machine = RBMAssert(chunk_size=None)
    vstate, tstate = _setup(machine=machine)
    Iop = InfidelityOperator(
        tstate,
        U=U,
        V=V,
        cv_coeff=None,
        use_hermitian_gradient_estimator=True,
    )

    tstate.chunk_size = chunk_size
    vstate.chunk_size = chunk_size
    iloc_no_chunking = vstate.local_estimators(Iop)

    np.testing.assert_allclose(iloc1, iloc_no_chunking)
