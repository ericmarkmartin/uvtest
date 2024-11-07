# TOADD: test for complex parameters and complex output

import pytest
import netket as nk

import jax.numpy as jnp
import numpy as np

from netket.models import RBM, RBMModPhase
from netket.jax import HashablePartial
from netket.vqs import MCState
from netket.vqs.mc import get_local_kernel_arguments
from netket.jax import tree_ravel, jacobian

from netket_pro.infidelity import InfidelityOperator
from netket_pro import distributed

from jax import clear_caches
from jax.tree_util import tree_map

from ..common import skipif_mpi

seed = 123456
L = 4
lattice = nk.graph.Chain(L, pbc=True)
n_samples = 512

machine_complexRBM = RBM(param_dtype=jnp.complex128)
machine_RBM_modphase = RBMModPhase()
machines = [
    machine_RBM_modphase,
]


def _setup(
    *,
    machine,
):
    clear_caches()
    hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes)

    sampler = nk.sampler.MetropolisLocal(
        hilbert=hi, n_chains_per_rank=n_samples, sweep_size=hi.size // 2
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
    eps -= jnp.mean(eps, axis=0)

    grad_explicit = ΔJx.conj().T @ eps / vstate.n_samples
    grad_explicit = jnp.squeeze(grad_explicit.real)

    np.testing.assert_allclose(
        grad_explicit, grad_hermitian_ravelled, atol=1e-13, rtol=1e-13
    )


@pytest.mark.parametrize("machine", machines)
@pytest.mark.parametrize("B", [None, 0.5 + 0.5j])
@pytest.mark.parametrize("cv_coeff", [None, -0.5])
def test_hermitian_expectation(machine, B, cv_coeff):
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


@skipif_mpi
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
