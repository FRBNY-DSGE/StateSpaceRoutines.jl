"""
```
hamilton_smoother(data, TTT, RRR, CCC, QQ, ZZ, DD, EE, z0, P0;
    n_presample_periods = 0)

hamilton_smoother(regime_indices, data, TTTs, RRRs, CCCs,
    QQs, ZZs, DDs, EEs, z0, P0; n_presample_periods = 0)
```

This is a Kalman smoothing program based on the treatment in James Hamilton's
\"Time Series Analysis\". Unlike the Koopman smoother, this one does rely on
inverting potentially singular matrices using the Moore-Penrose
pseudoinverse.

The state space is augmented with shocks (see `?augment_states_with_shocks`),
and the augmented state space is Kalman filtered and smoothed. Finally, the
smoothed states and shocks are indexed out of the augmented state vectors.

The original state space (before augmenting with shocks) is given by:

```
z_{t+1} = CCC + TTT*z_t + RRR*ϵ_t    (transition equation)
y_t     = DD  + ZZ*z_t  + η_t        (measurement equation)

ϵ_t ∼ N(0, QQ)
η_t ∼ N(0, EE)
Cov(ϵ_t, η_t) = 0
```

### Inputs

- `data`: `Ny` x `T` matrix containing data `y(1), ... , y(T)`
- `z0`: `Nz` x 1 initial state vector
- `P0`: `Nz` x `Nz` initial state covariance matrix
- `pred`: `Nz` x `T` matrix of one-step predicted state vectors `z_{t|t-1}`
  (from the Kalman filter)
- `vpred`: `Nz` x `Nz` x `T` array of mean squared errors `P_{t|t-1}` of
  predicted state vectors

**Method 1 only:** state-space system matrices `TTT`, `RRR`, `CCC`, `QQ`, `ZZ`,
`DD`, `EE`. See `?kalman_filter`

**Method 2 only:** `regime_indices` and system matrices for each regime `TTTs`,
`RRRs`, `CCCs`, `QQs`, `ZZs`, `DDs`, `EEs`. See `?kalman_filter`

where:

- `T`: number of periods for which we have data
- `Nz`: number of states
- `Ne`: number of shocks
- `Ny`: number of observables

### Keyword Arguments

- `n_presample_periods`: if greater than 0, the returned smoothed states and
  shocks matrices will be shorter by that number of columns (taken from the
  beginning)

### Outputs

- `smoothed_states`: `Nz` x `T` matrix of smoothed states `z_{t|T}`
- `smoothed_shocks`: `Ne` x `T` matrix of smoothed shocks `ϵ_{t|T}`
"""
function hamilton_smoother{S<:AbstractFloat}(data::Matrix{S},
    TTT::Matrix{S}, RRR::Matrix{S}, CCC::Vector{S},
    QQ::Matrix{S}, ZZ::Matrix{S}, DD::Vector{S}, EE::Matrix{S},
    z0::Vector{S}, P0::Matrix{S}; n_presample_periods::Int = 0)

    T = size(data, 2)
    regime_indices = Range{Int64}[1:T]

    hamilton_smoother(regime_indices, data, Matrix{S}[TTT], Matrix{S}[RRR], Vector{S}[CCC],
        Matrix{S}[QQ], Matrix{S}[ZZ], Vector{S}[DD], Matrix{S}[EE], z0, P0;
        n_presample_periods = n_presample_periods)
end

function hamilton_smoother{S<:AbstractFloat}(regime_indices::Vector{Range{Int64}},
    data::Matrix{S}, TTTs::Vector{Matrix{S}}, RRRs::Vector{Matrix{S}}, CCCs::Vector{Vector{S}},
    QQs::Vector{Matrix{S}}, ZZs::Vector{Matrix{S}}, DDs::Vector{Vector{S}}, EEs::Vector{Matrix{S}},
    z0::Vector{S} = Vector{S}(), P0::Matrix{S} = Matrix{S}();
    n_presample_periods::Int = 0)

    # Dimensions
    n_regimes = length(regime_indices)
    T  = size(data,    2) # number of periods of data
    Nz = size(TTTs[1], 1) # number of states
    Ne = size(RRRs[1], 2) # number of shocks

    # Augment state space with shocks
    TTTs, RRRs, CCCs, ZZs, z0, P0 =
        augment_states_with_shocks(regime_indices, TTTs, RRRs, CCCs, QQs, ZZs, z0, P0)

    # Kalman filter stacked states and shocks
    _, pred, vpred, filt, vfilt, _ =
        kalman_filter(regime_indices, data, TTTs, RRRs, CCCs, QQs, ZZs, DDs, EEs, z0, P0)

    # Smooth the stacked states and shocks recursively, starting at t = T-1 and
    # going backwards
    augmented_smoothed_states = copy(filt)

    for i = n_regimes:-1:1
        regime_periods = regime_indices[i]

        # The smoothed state in t = T is the same as the filtered state
        if i == n_regimes
            regime_periods = regime_periods[1:end-1]
        end

        # Get state-space system matrices for this regime
        TTT = TTTs[i]

        for t in reverse(regime_periods)
            J = vfilt[:, :, t] * TTT' * pinv(vpred[:, :, t+1])
            augmented_smoothed_states[:, t] = filt[:, t] + J*(augmented_smoothed_states[:, t+1] - pred[:, t+1])
        end
    end

    # Index out states and shocks
    smoothed_states = augmented_smoothed_states[1:Nz, :]
    smoothed_shocks = augmented_smoothed_states[Nz+1:end, :]

    # Trim the presample if needed
    if n_presample_periods > 0
        mainsample_periods = n_presample_periods+1:T

        smoothed_states = smoothed_states[:, mainsample_periods]
        smoothed_shocks = smoothed_states[:, mainsample_periods]
    end

    return smoothed_states, smoothed_shocks
end