import pytest
import numpy as np

import jax

import netket as nk
import netket_pro as nkp

from ..common import skipif_mpi, skipif_sharding


@skipif_mpi
@skipif_sharding
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_jastrow_gpu_fast(dtype):
    hi = nk.hilbert.Spin(0.5, 5)

    jastrow = nk.models.Jastrow(param_dtype=dtype)
    jastrow_fast = nkp.models.JastrowGPUFast(
        param_dtype=dtype,
        _interpret_debug_slow=jax.devices()[0].device_kind
        == "cpu",  # on cpu test interpret mode
    )

    x = hi.random_state(jax.random.key(1), 10, dtype=np.float32)

    pars_key = jax.random.key(1)
    pars = jastrow.init(
        pars_key,
        x,
    )
    pars_fast = jastrow_fast.init(
        pars_key,
        x,
    )

    np.testing.assert_allclose(pars["params"]["kernel"], pars_fast["params"]["kernel"])
    assert pars_fast["params"]["kernel"].dtype == dtype

    y = jastrow.apply(pars, x)
    y_fast = jastrow_fast.apply(pars_fast, x)

    np.testing.assert_allclose(y, y_fast)
