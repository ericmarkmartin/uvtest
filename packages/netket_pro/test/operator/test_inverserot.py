import netket as nk
import jax.numpy as jnp
import numpy as np

import netket_pro as nkp

from ..common import skipif_mpi, skipif_sharding

pytestmark = [skipif_mpi, skipif_sharding]


def test_rotations():
    N = 4
    hi = nk.hilbert.Spin(0.5, N, inverted_ordering=False)
    rx = nkp.operator.Rx(hi, 0, 0.3)
    rx_dagger = rx.H
    rx_dagger_ = nkp.operator.Rx(hi, 0, -0.3)

    np.testing.assert_allclose(rx_dagger.to_dense(), rx_dagger_.to_dense(), atol=1e-12)
    np.testing.assert_allclose(
        rx_dagger.to_dense() @ rx.to_dense(),
        jnp.identity(rx.to_dense().shape[0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rx_dagger_.to_dense() @ rx.to_dense(),
        jnp.identity(rx.to_dense().shape[0]),
        atol=1e-12,
    )
