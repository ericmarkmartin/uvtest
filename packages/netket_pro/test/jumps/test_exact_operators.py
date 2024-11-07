import pytest
import copy

import numpy as np
from scipy.sparse.linalg import expm_multiply

import netket as nk
from netket_pro import jumps as nkj

from netket.operator.spin import sigmaz


@pytest.mark.parametrize("A1", [1.0, 0.5, 0.5 - 0.1j, 0.5 + 0.1j])
def test_jastrow_zz_module(A1):
    g = nk.graph.Chain(2, pbc=False)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes)
    Hd = sum(sigmaz(hi, i) * sigmaz(hi, j) for i, j in g.edges())

    model = nk.models.RBM(alpha=1, param_dtype=complex)
    model = nkj.networks.Jastrow_zz_frozen(model, param_dtype=complex)

    vs = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=6),
        model=model,
        n_samples=500,
        n_discard_per_chain=0,
    )

    # exact
    Hd_sparse_diag = Hd.to_sparse()
    psi_vec = vs.to_array()
    psi1_exact = expm_multiply(-1j * Hd_sparse_diag * A1, psi_vec)
    psi1_exact = psi1_exact / np.linalg.norm(psi1_exact)

    # Approximate
    vs_Upsi = copy.copy(vs)
    vs_Upsi.variables = vs_Upsi.model.apply_zz(vs_Upsi.variables, Hd, scale=A1)
    psi1_var = vs_Upsi.to_array()  # this is already normalised

    np.testing.assert_allclose(psi1_exact, psi1_var)
