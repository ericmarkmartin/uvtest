import pytest

import numpy as np

import netket as nk

from netket_pro import distributed

from ..common import skipif_mpi, skipif_sharding

SEED = 2148364


@skipif_mpi
@skipif_sharding
def test_chunk_size_api():
    hi = nk.hilbert.Spin(0.5, 6)
    sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=10)
    model = nk.models.RBM()

    vs = nk.vqs.MCState(sampler, model, n_samples=100)

    # Check standard interface
    if nk.utils.mpi.n_nodes > 1:
        assert vs.samples.shape == (sampler.n_chains_per_rank, 10, hi.size)
        assert vs.sampler_state is not None
    else:
        assert vs.samples.shape == (10, 10, hi.size)
        assert vs.sampler_state is not None
    if distributed.mode() == "sharding":
        a, token = distributed.allgather(vs.samples[:, -1, :])
        b, token = distributed.allgather(vs.sampler_state.σ)
        np.testing.assert_allclose(a, b)
    else:
        np.testing.assert_allclose(vs.samples[:, -1, :], vs.sampler_state.σ)

    # check samples match with overridden interface
    np.testing.assert_allclose(vs.samples, vs.samples_distribution())
    np.testing.assert_allclose(
        vs.samples_distribution(vs.model), vs.samples_distribution()
    )
    # check samples updated
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            vs.samples_distribution(vs.model), vs.sample_distribution()
        )

    # Check that resample fraction works
    assert vs.resample_fraction is None
    vs.resample_fraction = 0
    assert vs.resample_fraction == 0.10
    # must reset after setting the resample fraction such that it stores the samples
    vs.reset()
    samples_old = vs.samples
    vs.reset()
    new_samples = vs.samples
    np.testing.assert_allclose(samples_old[:, 1:, :], new_samples[:, :9, :])

    # samples_distr does not resample
    new_new_samples = vs.samples_distribution()
    np.testing.assert_allclose(new_samples, new_new_samples)

    # Check that it adds to the same
    new_new_samples = vs.sample_distribution()

    np.testing.assert_allclose(new_samples[:, 1:, :], new_new_samples[:, :9, :])
