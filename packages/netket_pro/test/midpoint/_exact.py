import jax.numpy as jnp
from scipy import sparse


def compute_exact_infidelity(params_new, vstate, H, dt, B_num=None, B_den=None):
    hi = vstate.hilbert
    params_old = vstate.parameters
    state_old = vstate.to_array()

    vstate.parameters = params_new
    state_new = vstate.to_array()
    vstate.parameters = params_old

    x = -1j * H * dt

    U_num = (
        (1 + B_num * x).to_sparse()
        if B_num is not None
        else sparse.eye(hi.n_states, format="csc")
    )
    U_den = (
        (1 + B_den * x).to_sparse()
        if B_den is not None
        else sparse.eye(hi.n_states, format="csc")
    )

    Upsi = U_den @ state_new
    Vphi = U_num @ state_old

    Upsi_norm = jnp.sum(jnp.abs(Upsi) ** 2)
    Vphi_norm = jnp.sum(jnp.abs(Vphi) ** 2)

    return 1 - jnp.absolute(Vphi.conj().T @ Upsi) ** 2 / (Upsi_norm * Vphi_norm)
