import netket as nk
import netket_pro as nkp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import numpy as np

# Set the parameters
N = 10
Γ = -1.0
γ = 1.0

dt = 0.05
tf = 15.0
ts = jnp.arange(0, tf, dt)
T = 8

# Create the Hilbert space and the variational states |ψ⟩ and |ϕ⟩
hi = nk.hilbert.Spin(0.5, N, inverted_ordering=False)
g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)

sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
model = nk.models.RBM(alpha=1, param_dtype=complex)

if False:
    phi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)
    psi = nk.vqs.MCState(sampler=sampler, model=model, n_samples=1000)
else:
    phi = nk.vqs.FullSumState(
        hi,
        model,
    )
    psi = nk.vqs.FullSumState(
        hi,
        model,
    )

# Choose the number of iterations, the learning rate and the optimizer
n_iter = 1000
lr = 0.01
optimizer = nk.optimizer.Adam(learning_rate=lr)

# Prepare the initial state \bigotimes_{i=}^{N} |+⟩_i
with open("initial_state.mpack", "rb") as file:
    phi.variables = flax.serialization.from_bytes(phi.variables, file.read())

with open("initial_state.mpack", "rb") as file:
    psi.variables = flax.serialization.from_bytes(psi.variables, file.read())

# Instantiate the observable to monitor
obs = sum([nk.operator.spin.sigmaz(hi, i) for i in range(N)]) / N

ha_X = sum(nk.operator.spin.sigmax(hi, i) for i in range(hi.size))


# Function doing the adiabtic dynamics with Trotterized p-tVMC
def Trotter_adiabatic(phi, optimizer, psi, γ, Γ, ts, n_iter, obs=None, log=None):
    if log is None:
        log = {}
    if obs is not None:
        log["obs"] = []

    for t in ts:
        print(f"Time t = {t}: ")
        print("##########################################")

        # Calculate the time-dependent couplings
        if t < T:
            γt = γ * t / T
            Γt = Γ * (1 - t / T)
        else:
            γt = γ * (1 - (t - T) / T)
            Γt = Γ * (t - T) / T

        # Z diagonal term
        params = flax.core.unfreeze(phi.parameters)
        params["visible_bias"] = params["visible_bias"] + 1j * γt * dt / 2
        psi.parameters = params
        phi.parameters = params

        # Create the X-rotations
        # Uxs.append(nkp.operator.Rx(hi, i, 2 * dt * Γt))
        # Uxs_dagger.append(nkp.operator.Rx(hi, i, -2 * dt * Γt))

        # X terms
        te = nkp.driver.MidpointL2L1Optimizer(
            phi,
            optimizer,
            H=ha_X,
            dt=dt * Γt,
            variational_state=psi,
            cv_coeff=-0.5,
        )
        te.run(n_iter=n_iter)
        phi.parameters = psi.parameters

        # Z diagonal term
        params = flax.core.unfreeze(phi.parameters)
        params["visible_bias"] = params["visible_bias"] + 1j * γt * dt / 2
        psi.parameters = params
        phi.parameters = params

        if obs is not None:
            log["obs"].append(psi.expect(obs))

        print("##########################################")
        print("\n")

    if obs is not None:
        return psi, log

    else:
        return psi


# Run the evolution
obs_dict = {}
psi, obs_dict = Trotter_adiabatic(
    phi,
    optimizer,
    psi,
    γ,
    Γ,
    ts,
    n_iter=n_iter,
    obs=obs,
    log=obs_dict,
)

obs_mean = np.array([x.mean for x in obs_dict["obs"]])
obs_error = np.array([x.error_of_mean for x in obs_dict["obs"]])

# Plot the results
ts = ts[: len(obs_mean)]
fig = plt.figure(figsize=(8, 8))
plt.errorbar(ts, obs_mean, obs_error)
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle \sigma_i^z \rangle$")
plt.legend()
plt.tight_layout()
plt.savefig("adiabatic_sweeping.pdf", bbox_inches="tight")
plt.show()
