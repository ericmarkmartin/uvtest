import pytest

import jax
import jax.numpy as jnp
import numpy as np

import netket_pro as nkp

from ..common import skipif_mpi, skipif_sharding


@skipif_sharding
@skipif_mpi
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("k", [0, -1])
def test_vec_to_tril(dtype, k):
    nv = 6
    il = jnp.tril_indices(nv, k=k)

    npars = len(np.tril_indices(nv, k=k)[0])
    params = jax.random.normal(jax.random.key(1), (npars,), dtype)

    tril_mat = jnp.zeros((nv, nv), dtype=dtype)
    tril_mat = tril_mat.at[il].set(params)

    # if cpu, interept
    interpret = jax.devices()[0].device_kind == "cpu"

    tril_mat_pallas = nkp.jax.vec_to_tril(params, nv, k=k, interpret=interpret)
    assert tril_mat_pallas.dtype == params.dtype
    np.testing.assert_allclose(tril_mat, tril_mat_pallas)
