import netket as nk
import netket_pro as nkp

from ..common import skipif_mpi


@skipif_mpi
def test_exchange_excitation_rule():
    g = nk.graph.Chain(4, pbc=False)
    hilbert = nk.hilbert.Spin(1.0, N=g.n_nodes, total_sz=0)

    # with n_exchange=1, the rule should be equivalent to the local rule and is not ergodic
    rule = nkp.sampler.rules.HoppingRule(graph=g, n_exchange=1)
    is_ok, _ = nkp.testing.verify_metropolis_rule(
        hilbert, rule, n_samples=100000, seed=1234, n_repeats=10, sweep_size=4
    )
    assert not is_ok

    rule = nkp.sampler.rules.HoppingRule(graph=g, n_exchange=2)
    is_ok, _ = nkp.testing.verify_metropolis_rule(
        hilbert, rule, n_samples=100000, seed=1234, n_repeats=10, sweep_size=4
    )
    assert is_ok
