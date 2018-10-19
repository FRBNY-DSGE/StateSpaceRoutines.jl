"""
```
carter_kohn_smoother(y, T, R, C, Q, Z, D, E, s_0, P_0;
    Nt0 = 0, draw_states = true)

carter_kohn_smoother(regime_indices, y, Ts, Rs, Cs, Qs,
    Zs, Ds, Es, s_0, P_0; Nt0 = 0, draw_states = true)
```
This program is a simulation smoother based on Carter and Kohn's \"On Gibbs
Sampling for State Space Models\" (Biometrika, 1994). It recursively samples
from the conditional distribution of time-t states given the full set of
observables and states from time t+1 to time T. Unlike the Durbin-Koopman
simulation smoother, this one does rely on inverting potentially singular
matrices using the Moore-Penrose pseudoinverse.

The state space is augmented with shocks (see `?augment_states_with_shocks`),
and the augmented state space is Kalman filtered and smoothed. Finally, the
smoothed states and shocks are indexed out of the augmented state vectors.

The original state space (before augmenting with shocks) is given by:

```
s_{t+1} = C + T*s_t + R*ϵ_t    (transition equation)
y_t     = D + Z*s_t + u_t      (measurement equation)

ϵ_t ∼ N(0, Q)
u_t ∼ N(0, E)
Cov(ϵ_t, u_t) = 0
```

### Inputs

- `y`: `Ny` x `Nt` matrix containing data `y_1, ... , y_T`
- `s_0`: `Ns` x 1 initial state vector
- `P_0`: `Ns` x `Ns` initial state covariance matrix

**Method 1 only:** state-space system matrices `T`, `R`, `C`, `Q`, `Z`,
`D`, `E`. See `?kalman_filter`
pp
**Method 2 only:** `regime_indices` and system matrices for each regime `Ts`,
`Rs`, `Cs`, `Qs`, `Zs`, `Ds`, `Es`. See `?kalman_filter`

where:

- `Nt`: number of periods for which we have data
- `Ns`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `Nt0`: if greater than 0, the returned smoothed states and shocks matrices
  will be shorter by that number of columns (taken from the beginning)
- `draw_states`: indicates whether to draw smoothed states from the distribution
  `N(s_{t|T}, P_{t|T})` or to use the mean `s_{t|T}` (reducing to the Hamilton
  smoother)

### Outputs

- `s_smth`: `Ns` x `Nt` matrix of smoothed states `s_{t|T}`
- `ϵ_smth`: `Ne` x `Nt` matrix of smoothed shocks `ϵ_{t|T}`
"""
function carter_kohn_smoother(y::AbstractArray{Union{S, Missing}},
    T::Matrix{S}, R::Matrix{S}, C::Vector{S},
    Q::Matrix{S}, Z::Matrix{S}, D::Vector{S}, E::Matrix{S},
    s_0::Vector{S}, P_0::Matrix{S};
    Nt0::Int = 0, draw_states::Bool = true) where {S<:AbstractFloat}

    Nt = size(y, 2)
    carter_kohn_smoother(AbstractRange{Int}[1:Nt], y, Matrix{S}[T], Matrix{S}[R], Vector{S}[C],
        Matrix{S}[Q], Matrix{S}[Z], Vector{S}[D], Matrix{S}[E], s_0, P_0;
        Nt0 = Nt0, draw_states = draw_states)
end

function carter_kohn_smoother(regime_indices::Vector{AbstractRange{Int}}, y::AbstractArray{Union{S, Missing}},
    Ts::Vector{Matrix{S}}, Rs::Vector{Matrix{S}}, Cs::Vector{Vector{S}}, Qs::Vector{Matrix{S}},
    Zs::Vector{Matrix{S}}, Ds::Vector{Vector{S}}, Es::Vector{Matrix{S}},
    s_0::Vector{S}, P_0::Matrix{S};
    Nt0::Int = 0, draw_states::Bool = true) where {S<:AbstractFloat}

    # Dimensions
    Nt = size(y,     2) # number of periods of data
    Ns = size(Ts[1], 1) # number of states
    Ne = size(Rs[1], 2) # number of shocks

    # Augment state space with shocks
    Ts, Rs, Cs, Zs, s_0, P_0 =
        augment_states_with_shocks(regime_indices, Ts, Rs, Cs, Qs, Zs, s_0, P_0)

    # Kalman filter stacked states and shocks stil_t
    _, stil_pred, Ptil_pred, stil_filt, Ptil_filt, _, _, _, _ =
        kalman_filter(regime_indices, y, Ts, Rs, Cs, Qs, Zs, Ds, Es, s_0, P_0,
                      outputs = [:pred, :filt])

    # Smooth the stacked states and shocks recursively, starting at t = T-1 and
    # going backwards
    stil_smth = copy(stil_filt)

    for i = length(regime_indices):-1:1
        # Get state-space system matrices for this regime
        T = Ts[i]

        for t in reverse(regime_indices[i])
            if t == Nt
                μ = stil_filt[:, end]
                Σ = Ptil_filt[:, :, end]
            else
                J = Ptil_filt[:, :, t] * T' * pinv(Ptil_pred[:, :, t+1])
                μ = stil_filt[:, t] + J*(stil_smth[:, t+1] - stil_pred[:, t+1]) # stil_{t|T}
                Σ = Ptil_filt[:, :, t] - J*T*Ptil_filt[:, :, t]                 # Ptil_{t|T}
            end

            # Draw stil_t ∼ N(stil_{t|T}, Ptil_{t|T})
            stil_smth[:, t] = if draw_states
                U, eig, _ = svd(Σ)
                μ + U * diagm(0 => (sqrt.(eig))) * randn(Ns+Ne)
            else
                μ
            end
        end
    end

    # Index out states and shocks
    s_smth = stil_smth[1:Ns,     :] # s_{t|T}
    ϵ_smth = stil_smth[Ns+1:end, :] # ϵ_{t|T}

    # Trim the presample if needed
    if Nt0 > 0
        insample = Nt0+1:Nt
        s_smth = s_smth[:, insample]
        ϵ_smth = ϵ_smth[:, insample]
    end

    return s_smth, ϵ_smth
end
